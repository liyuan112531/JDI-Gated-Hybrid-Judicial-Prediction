# -*- coding: utf-8 -*-
"""
model_library — 共享模型工具库（自原 Stage 4 Phase 1 冻结配置抽取）
==================================================================

为 04_ultimate_gated_model.py / 05_generate_paper_tables.py 提供：
  - build_X           判前特征矩阵（案件级，索引 = case_id）
  - make_model        冻结默认参数模型工厂
  - build_pipeline    中位数填补（+按需标准化）+ 模型
  - select_rule_fast  A/C 类项目训练折内规则标定
  - metrics           案件级指标（MAE / RMSE / MdAE / R² / acc@5% / acc@10%）

本模块为 import-safe：不读取任何数据文件、不执行训练。
所有超参数与 configs/ 冻结策略一致，禁止在调用方覆盖。
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

# TabPFN 在 CPU 上默认拒绝 >1000 样本；本项目最大分项训练样本约 1600，
# 故开启官方提供的开关（不改变模型行为，仅解除 CPU 大样本保护）
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

RS = 42

# ---------------------------------------------------------------------------
# 冻结配置（import 时只读 configs，不读数据）
# ---------------------------------------------------------------------------
policy = yaml.safe_load((CONFIGS / "experiment_policy.yaml").read_text(encoding="utf-8"))
_proj = json.loads((CONFIGS / "project_strategy_config.json").read_text(encoding="utf-8"))

A_ITEMS = list(_proj["rule_based"])
B_ITEMS = list(_proj["ml_based"])
C_ITEMS = list(_proj["small_sample"])


# ===========================================================================
# 特征工程
# ===========================================================================
def build_X(case: pd.DataFrame, encoding: str = "ordinal") -> pd.DataFrame:
    """构建判前特征矩阵（案件级，索引 = case_id）"""
    items = list(policy["data"]["items"])
    X = pd.DataFrame(index=case["case_id"].values)
    X.index.name = "case_id"

    # 基础司法特征
    X["year"] = pd.to_numeric(case["year"], errors="coerce").values
    X["liability_ratio_clean"] = case["liability_ratio_clean"].values
    X["liability_missing"] = case["liability_missing"].values
    X["disability_abnormal"] = case["disability_abnormal"].values
    X["disability_missing"] = case["disability_missing"].values

    # 伤残等级编码
    dlv = pd.to_numeric(case["disability_level_clean"], errors="coerce")
    if encoding == "ordinal":
        X["disability_ordinal"] = dlv.values
    elif encoding == "onehot":
        for v in range(0, 11):
            X[f"disability_{v}"] = (dlv == v).astype(float).values
    elif encoding == "group":
        # 司法分组：0=无伤残；1=重度（1-6 级）；2=轻度（7-10 级）
        g = pd.Series(np.nan, index=case.index)
        g[dlv == 0] = 0.0
        g[(dlv >= 1) & (dlv <= 6)] = 1.0
        g[(dlv >= 7) & (dlv <= 10)] = 2.0
        X["disability_g0_none"] = (g == 0).astype(float).values
        X["disability_g1_severe"] = (g == 1).astype(float).values
        X["disability_g2_mild"] = (g == 2).astype(float).values
    else:
        raise ValueError(encoding)

    # 审理程序（固定类别集合，不涉及标签）
    proc = case["procedure_type"].astype(str)
    for p in ["一审", "二审", "再审", "其他"]:
        X[f"proc_{p}"] = (proc == p).astype(float).values

    # 16 项请求金额 + 结构特征
    for it in items:
        X[f"{it}_claim"] = case[f"{it}_claim"].values
    X["claim_total_calc"] = case["claim_total_calc"].values
    X["n_nonzero_claims"] = case["n_nonzero_claims"].values
    for it in items:
        X[f"claim_share__{it}"] = case[f"claim_share__{it}"].values
    for it in items:
        X[f"log_claim__{it}"] = case[f"log_claim__{it}"].values
    return X


# ===========================================================================
# 模型池
# ===========================================================================
def make_model(name: str):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=RS) if "random_state" in Ridge().get_params() \
            else Ridge(alpha=1.0)
    if name == "ElasticNet":
        return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RS, max_iter=10000)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=RS, n_jobs=4, verbosity=0)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=RS, n_jobs=4, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_state=RS, verbose=0, allow_writing_files=False)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=RS)
    if name == "SVR":
        return SVR()
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=RS)
    if name == "TabPFN":
        from tabpfn import TabPFNRegressor
        try:
            return TabPFNRegressor(device="cpu", random_state=RS)
        except TypeError:
            return TabPFNRegressor(device="cpu")
    raise ValueError(name)


SCALED = {"LinearRegression", "Ridge", "ElasticNet", "SVR", "MLP"}


def build_pipeline(name: str) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if name in SCALED:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", make_model(name)))
    return Pipeline(steps)


# ===========================================================================
# 规则标定（A/C 类）—— 训练折内标定
# ===========================================================================
def select_rule_fast(train_it: pd.DataFrame, item: str, n_inner: int = 3):
    """训练折内规则标定（精简版：候选集不含重复项）"""
    from rule_engine.rule_base import CapRule, GlobalMedianRule, QuantilePriorRule
    from rule_engine.rule_generic import RULE_CANDIDATES
    ids = train_it["case_id"].unique()
    rng = np.random.default_rng(RS)
    ids_s = rng.permutation(ids)
    parts = np.array_split(ids_s, n_inner)
    cands = RULE_CANDIDATES.get(item, [])
    best = None
    for name, cls, params in cands:
        errs = []
        for k in range(n_inner):
            te_ids = set(parts[k].tolist())
            tr_ids = set(ids_s.tolist()) - te_ids
            tr_sub = train_it[train_it["case_id"].isin(tr_ids)]
            te_sub = train_it[train_it["case_id"].isin(te_ids)]
            if len(tr_sub) == 0 or len(te_sub) == 0:
                continue
            r = cls(item, **params)
            r.fit(tr_sub)
            p = np.asarray(r.predict_frame(te_sub, item), dtype=float)
            errs.append(float(np.mean(np.abs(p - te_sub["support_rate_label"].to_numpy()))))
        if not errs:
            continue
        sc = float(np.mean(errs))
        if best is None or sc < best[0] - 1e-12:
            best = (sc, name, cls, params)
    if best is None:
        return None
    rule = best[2](item, **best[3])
    rule.fit(train_it)
    return rule


# ===========================================================================
# 指标
# ===========================================================================
def metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    if len(y) == 0:
        return {}
    err = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))), "MdAE": float(np.median(np.abs(err))),
            "R2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
            "within_5pct": float(np.mean(np.abs(err) <= 0.05)),
            "within_10pct": float(np.mean(np.abs(err) <= 0.10))}
