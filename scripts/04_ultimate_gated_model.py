# -*- coding: utf-8 -*-
"""
04 — Ultimate SOTA: 异质分项最优模型 + 认知不确定性动态门控（最终主模型）
==========================================================================

目标：在异质分项最优模型组合（12 LightGBM + 4 CatBoost）上施加
      基于分位数认知不确定性的实例级动态门控（Epistemic Uncertainty Gating）。

协议（与论文发表版一致）：
  - 切分：strata = year | procedure_type | SR_total 五分位，
    案件级 RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)；
  - 规则：训练折内规则标定（select_rule_fast / select_rule_generic）；
  - JDI：训练折内重算（compute_jdi_fold），非 02 冻结值；
  - gamma=0 行即 JDI 静态连续融合（Soft Hybrid），gamma=1.0 为论文最终主模型。

分项异质三头分位数回归（Stage 5 冻结指派，已硬编码于下方 PER_ITEM）：
  LightGBM : mean=objective'regression'；q10/q90=objective'quantile', alpha=0.1/0.9
  CatBoost : mean=loss_function'RMSE'； q10/q90=loss_function'Quantile:alpha=0.1/0.9'

动态门控：delta = clip(q90-q10, 0, None)；
          w_dyn = w_i * exp(-gamma * delta)；
          SR_item = clip(w_dyn*SR_rule + (1-w_dyn)*pred_mean, 0, 1)；
          案件级 SR_total = Σ(claim*SR)/Σclaim，再 clip[0,1]。

输出：results/stage10_ultimate_sota/
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti TC",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[1]
CONFIGS = BASE / "configs"
S1 = BASE / "results" / "stage1"
S2D = BASE / "results" / "stage2"
OUT = BASE / "results" / "stage10_ultimate_sota"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# 复用共享模型库（build_X / make_model / select_rule_fast，冻结配置与原 Stage 4 一致）
sys.path.insert(0, str(BASE / "src"))
from model_library import build_X, make_model, select_rule_fast  # noqa: E402
from rule_engine.rule_base import (CapRule, GlobalMedianRule,  # noqa: E402
                                   QuantilePriorRule)

# ===========================================================================
# 冻结超参数（Stage 5 扫描 + Stage 9/10 门控扫描选定的最佳设定，直接写死）
# ===========================================================================
# 分项模型指派（Stage 5 逐 item 交叉验证选定；未列出的项目默认 LightGBM）
PER_ITEM = {
    "lost_income": "CatBoost",
    "nursing": "CatBoost",
    "property_loss": "CatBoost",
    "transport": "CatBoost",
    "mental_solatium": "LightGBM",
    "nutrition": "LightGBM",
    "other": "LightGBM",
}
# gamma 网格：gamma=1.0 为论文最终主模型（Stage 9/10 扫描选定）；
# gamma=0.0（JDI 静态连续融合，Soft Hybrid）仅作为对照行保留。
GAMMA_LIST = [0.0, 1.0]
GAMMA_FINAL = 1.0


def log(m: str) -> None:
    print(f"[04 {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ===========================================================================
# 0. 配置
# ===========================================================================
policy = yaml.safe_load((CONFIGS / "experiment_policy.yaml").read_text(encoding="utf-8"))
RS = int(policy["global"]["random_state"])
N_SPLITS = int(policy["global"]["n_splits"])
N_REPEATS = int(policy["global"]["n_repeats"])
ALL_ITEMS = list(policy["data"]["items"])

strategy = json.loads((CONFIGS / "project_strategy_config.json").read_text(encoding="utf-8"))
A_ITEMS = list(strategy["rule_based"])
C_ITEMS = list(strategy["small_sample"])

WEIGHTS = json.loads((S2D / "jdi_weight_config.json").read_text(encoding="utf-8"))["weights"]
POSITIVE = {"p_sr_eq_1", "p_sr_eq_0", "SR_mode_density_bin10", "liability_consistency"}
N_BINS = 10

log("=" * 76)
log("04: Ultimate SOTA — 异质模型 + 认知不确定性动态门控")
log(f"  协议：{N_SPLITS} 折 x {N_REPEATS} 重复，rs={RS}（发表版口径）")
log(f"  分项模型指派（冻结）：{PER_ITEM}；其余 LightGBM")
log(f"  gamma 网格：{GAMMA_LIST}（最终主模型 gamma={GAMMA_FINAL}）")
log("=" * 76)

# ===========================================================================
# 1. 数据与特征（65 维判前特征）
# ===========================================================================
case = pd.read_csv(S1 / "dataset_case_level.csv", encoding="utf-8-sig")
item_all = pd.read_csv(S1 / "dataset_item_level.csv", encoding="utf-8-sig",
                       usecols=["case_id", "item_name", "claim_amount", "award_amount",
                                "support_rate_label", "claim_positive"])
valid = item_all[item_all["claim_positive"] == 1].reset_index(drop=True)

def disability_severity_score(level):
    lv = pd.to_numeric(level, errors="coerce")
    s = pd.Series(np.nan, index=lv.index)
    s[lv == 0] = 0.0
    for k in range(1, 11):
        s[lv == k] = round(1.1 - 0.1 * k, 2)
    return s


def add_semantic(case):
    X = X_orig.copy()
    X["disability_severity_score"] = disability_severity_score(
        case["disability_level_clean"]).values
    sc = [c for c in case.columns if c.startswith("claim_share__")]
    P = np.where(np.isfinite(case[sc].to_numpy(float)), case[sc].to_numpy(float), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(P > 0, np.log(P), 0.0)
    X["claim_entropy"] = -np.sum(P * lp, axis=1) / np.log(P.shape[1])
    Ps = np.sort(P, axis=1)[:, ::-1]
    X["top1_claim_share"] = Ps[:, 0]
    X["top3_claim_share"] = Ps[:, :3].sum(axis=1)
    return X


X_orig = build_X(case, "ordinal")
X_aug = add_semantic(case)

BASE_JUD = list(X_orig.columns)
SEM_COLS = ["disability_severity_score", "claim_entropy", "top1_claim_share",
            "top3_claim_share"]
FEATURE_COLS = BASE_JUD + SEM_COLS + ["claim_amount_relative_ratio"]
log(f"  特征：{X_orig.shape[1]} 判前 + {len(SEM_COLS)} 语义 + 1 折内 ratio = {len(FEATURE_COLS)} 维")

# ===========================================================================
# 2. 折内规则标定
# ===========================================================================
def select_rule_generic(train_it, item, n_inner=3):
    cands = [(f"cap_q{q}", CapRule, {"q": q}) for q in (0.5, 0.75, 0.9, 0.95)]
    cands += [("global_median", GlobalMedianRule, {}),
              ("quantile_prior3", QuantilePriorRule, {"n_bins": 3})]
    ids = train_it["case_id"].unique()
    rng = np.random.default_rng(RS)
    ids_s = rng.permutation(ids)
    parts = np.array_split(ids_s, n_inner)
    best = None
    for name, cls, params in cands:
        errs = []
        for k in range(n_inner):
            te_ids = set(parts[k].tolist())
            tr_ids = set(ids_s.tolist()) - te_ids
            a = train_it[train_it["case_id"].isin(tr_ids)]
            b = train_it[train_it["case_id"].isin(te_ids)]
            if len(a) == 0 or len(b) == 0:
                continue
            r = cls(item, **params)
            r.fit(a)
            p = np.asarray(r.predict_frame(b, item), dtype=float)
            errs.append(float(np.mean(np.abs(p - b["support_rate_label"].to_numpy()))))
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


def rule_predict(tr_it, te_it, item):
    fn = select_rule_fast if (item in A_ITEMS or item in C_ITEMS) else select_rule_generic
    r = fn(tr_it, item)
    if r is None:
        return np.full(len(te_it), float(np.nanmedian(tr_it["support_rate_label"].values)))
    return np.clip(np.asarray(r.predict_frame(te_it, item), dtype=float), 0, 1)


# ===========================================================================
# 3. 折内 JDI
# ===========================================================================
def shannon_entropy(x, n_bins=N_BINS):
    h, _ = np.histogram(x, bins=n_bins, range=(0.0, 1.0))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mode_density(x, n_bins=N_BINS):
    h, _ = np.histogram(x, bins=n_bins, range=(0.0, 1.0))
    return float(h.max() / h.sum())


def compute_jdi_fold(tv, items):
    rows = []
    for it in items:
        sub = tv[tv["item_name"] == it]
        sr = sub["support_rate_label"].to_numpy(float)
        if len(sr) < 5:
            rows.append({"item_name": it, "JDI": np.nan})
            continue
        p1 = float(np.mean(np.abs(sr - 1.0) <= 1e-6))
        p0 = float(np.mean(np.abs(sr) <= 1e-6))
        std = float(np.std(sr, ddof=1))
        q75, q25 = np.percentile(sr, [75, 25])
        liab = (sub["liability_ratio_clean"].to_numpy(float)
                if "liability_ratio_clean" in sub.columns else np.full(len(sr), np.nan))
        m = ~np.isnan(liab)
        lc = abs(float(stats.spearmanr(liab[m], sr[m])[0])) \
            if (m.sum() >= 30 and np.std(liab[m]) > 0 and np.std(sr[m]) > 0) else np.nan
        rows.append({"item_name": it, "p_sr_eq_1": p1, "p_sr_eq_0": p0, "std_SR": std,
                     "IQR_SR": float(q75 - q25),
                     "SR_entropy_norm": shannon_entropy(sr) / np.log2(N_BINS),
                     "SR_mode_density_bin10": mode_density(sr),
                     "liability_consistency": lc,
                     "liability_reliable": int(not np.isnan(lc))})
    df = pd.DataFrame(rows).set_index("item_name")
    score = pd.Series(0.0, index=df.index)
    for col, w in WEIGHTS.items():
        x = df[col].astype(float).copy()
        if col == "liability_consistency":
            x = x.where(df["liability_reliable"] == 1, other=np.nan).fillna(0.5)
        xn = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() \
            else pd.Series(0.5, index=x.index)
        xn = xn.fillna(0.5)
        score += w * (xn if col in POSITIVE else 1.0 - xn)
    df["JDI"] = score * 100
    return df


# ===========================================================================
# 4. 异质三头模型工厂
# ===========================================================================
def make_head(item: str, head: str):
    """按冻结指派构建 mean/q10/q90 三头。所有头目标均为分位数/平方误差族。"""
    name = PER_ITEM.get(item, "LightGBM")
    alpha = 0.1 if head == "q10" else 0.9
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        if head == "mean":
            return make_model("LightGBM")  # objective 默认 regression(L2)
        return LGBMRegressor(objective="quantile", alpha=alpha,
                             random_state=RS, n_jobs=4, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        if head == "mean":
            return make_model("CatBoost")  # 默认 loss_function='RMSE'（平方误差）
        return CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}",
                                 random_state=RS, verbose=0, allow_writing_files=False)
    raise ValueError(f"未知模型指派：{name}")


def build_pipe(item: str, head: str) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", make_head(item, head))])


# ===========================================================================
# 5. CV 切分（发表版口径）
# ===========================================================================
sr_bin = pd.qcut(case["sr_total_label"].rank(method="first"), 5, labels=False)
strata = (case["year"].astype(str) + "|" + case["procedure_type"].astype(str)
          + "|" + sr_bin.astype(str))
case_ids = case["case_id"].values
claim_total_by_case = case.set_index("case_id")["claim_total_calc"]
sr_true_by_case = case.set_index("case_id")["sr_total_label"]
splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                      random_state=RS).split(case, strata))
log(f"  分层组合数：{strata.nunique()}；折组合数：{len(splits)}")

# ===========================================================================
# 6. 主循环：折内规则 + 三头异质分位数 + 门控输入
# ===========================================================================
pred_rows = []       # 分项级 OOF：mean/q10/q90/rule/w（含折内中位数回退信息）
fold_stats = {}      # (rep, fold) -> {"item_median": {...}, "gmean": float}
t0 = time.time()

for i, (tr, te) in enumerate(splits):
    rep, fold = i // N_SPLITS, i % N_SPLITS
    train_cases = set(case_ids[tr].tolist())
    test_cases = case_ids[te]
    train_valid = valid[valid["case_id"].isin(train_cases)]
    item_median = train_valid.groupby("item_name")["support_rate_label"].median().to_dict()
    gmean = float(train_valid["support_rate_label"].mean())
    fold_stats[(rep, fold)] = {"item_median": item_median, "gmean": gmean}

    P_rule = {}
    for it in ALL_ITEMS:
        tr_it = train_valid[train_valid["item_name"] == it]
        te_it = valid[(valid["item_name"] == it) & (valid["case_id"].isin(test_cases))]
        if len(te_it) == 0 or len(tr_it) < 5:
            continue
        P_rule[it] = pd.Series(rule_predict(tr_it, te_it, it), index=te_it["case_id"].values)

    preds = {}   # it -> {"mean": Series, "q10": Series, "q90": Series}
    for it in ALL_ITEMS:
        tr_it = train_valid[train_valid["item_name"] == it]
        te_it = valid[(valid["item_name"] == it) & (valid["case_id"].isin(test_cases))]
        if len(te_it) == 0 or len(tr_it) < 5:
            continue
        y_tr = tr_it["support_rate_label"].values
        med = float(np.nanmedian(tr_it["claim_amount"].values))
        Xtr = X_aug.loc[tr_it["case_id"].values].copy()
        Xte = X_aug.loc[te_it["case_id"].values].copy()
        Xtr["claim_amount_relative_ratio"] = (tr_it["claim_amount"].values / med
                                              if med > 0 else 0.0)
        Xte["claim_amount_relative_ratio"] = (te_it["claim_amount"].values / med
                                              if med > 0 else 0.0)
        d = {}
        for head in ("mean", "q10", "q90"):
            p = build_pipe(it, head)
            p.fit(Xtr[FEATURE_COLS], y_tr)
            d[head] = np.asarray(p.predict(Xte[FEATURE_COLS]), dtype=float)
        # mean 头：分项级 clip[0,1]
        preds[it] = {"mean": np.clip(d["mean"], 0, 1),
                     "q10": d["q10"], "q90": d["q90"]}

    # 折内 JDI -> 静态权重
    jdi = compute_jdi_fold(train_valid, ALL_ITEMS)["JDI"]
    w = (jdi - jdi.min()) / (jdi.max() - jdi.min()) if jdi.max() > jdi.min() \
        else pd.Series(0.5, index=jdi.index)

    for it in ALL_ITEMS:
        if it not in preds:
            continue
        te_it = valid[(valid["item_name"] == it) & (valid["case_id"].isin(test_cases))]
        for j, cid in enumerate(te_it["case_id"].values):
            pred_rows.append({
                "repeat": rep, "fold": fold, "case_id": cid, "item_name": it,
                "claim_amount": float(te_it["claim_amount"].values[j]),
                "sr_true": float(te_it["support_rate_label"].values[j]),
                "rule": float(P_rule[it].values[j]),
                "mean": float(preds[it]["mean"][j]),
                "q10": float(preds[it]["q10"][j]),
                "q90": float(preds[it]["q90"][j]),
                "w": float(w.get(it, 0.5)),
            })

    done = i + 1
    if done % 3 == 0 or done == len(splits):
        log(f"  折进度 {done}/{len(splits)}（耗时 {(time.time() - t0) / 60:.1f} 分钟）")

pred = pd.DataFrame(pred_rows)
assert len(pred) == N_REPEATS * len(valid), \
    f"OOF 行数异常：{len(pred)} != {N_REPEATS} x {len(valid)}"
pred["delta_uncert"] = np.clip(pred["q90"].to_numpy(float) - pred["q10"].to_numpy(float),
                               0, None)
log(f"  OOF 分项预测：{pred.shape}")
log(f"  delta_uncert：mean={pred['delta_uncert'].mean():.4f}, "
    f"median={pred['delta_uncert'].median():.4f}, p90={pred['delta_uncert'].quantile(0.9):.4f}")

# ===========================================================================
# 7. 门控融合 + 案件级聚合（含中位数回退）
# ===========================================================================
def assemble_from_item(sr_hat_item: pd.DataFrame, system: str) -> pd.DataFrame:
    """对给定分项级 sr_hat（含 repeat/fold/case_id/item_name）聚合到案件级。"""
    out = []
    for (rep, fold), g in sr_hat_item.groupby(["repeat", "fold"]):
        fs = fold_stats[(int(rep), int(fold))]
        item_median, gmean = fs["item_median"], fs["gmean"]
        test_cases = case_ids[_fold_test_mask[(int(rep), int(fold))]]
        mp = {it: gg.set_index("case_id")["sr_hat"] for it, gg in g.groupby("item_name")}
        tot = pd.Series(0.0, index=pd.Index(test_cases, name="case_id"))
        for it in ALL_ITEMS:
            te_it = valid[(valid["item_name"] == it) & (valid["case_id"].isin(set(test_cases)))]
            if len(te_it) == 0:
                continue
            s = mp.get(it)
            if s is None:
                sr_hat = np.full(len(te_it), item_median.get(it, gmean))
            else:
                v = s.reindex(te_it["case_id"].values).to_numpy()
                sr_hat = np.where(np.isnan(v), item_median.get(it, gmean), v)
            aw = pd.Series(np.clip(sr_hat, 0, 1) * te_it["claim_amount"].values,
                           index=te_it["case_id"].values)
            tot = tot.add(aw.groupby(level=0).sum(), fill_value=0.0)
        denom = claim_total_by_case.reindex(tot.index).values
        sr_case = np.clip(tot.values / np.where(denom > 0, denom, np.nan), 0, 1)
        tv = sr_true_by_case.reindex(tot.index).values
        for cid, pv, t in zip(tot.index, sr_case, tv):
            out.append({"system": system, "repeat": rep,
                        "case_id": cid, "sr_pred": float(pv), "sr_true": float(t)})
    return pd.DataFrame(out)


def aggregate_gamma(gamma: float) -> pd.DataFrame:
    """对给定 gamma 生成案件级 OOF。"""
    w_arr = pred["w"].to_numpy(float)
    rule_arr = pred["rule"].to_numpy(float)
    mean_arr = pred["mean"].to_numpy(float)
    delta_arr = pred["delta_uncert"].to_numpy(float)
    if gamma == 0.0:
        sr_item = np.clip(w_arr * rule_arr + (1.0 - w_arr) * mean_arr, 0.0, 1.0)
    else:
        w_dyn = w_arr * np.exp(-gamma * delta_arr)
        sr_item = np.clip(w_dyn * rule_arr + (1.0 - w_dyn) * mean_arr, 0.0, 1.0)
    work = pred[["repeat", "fold", "case_id", "item_name", "claim_amount"]].copy()
    work["sr_hat"] = sr_item
    return assemble_from_item(work, f"gated_gamma_{gamma:g}")


# 预生成每折测试掩码（与 splits 枚举一致）
_fold_test_mask = {}
for i, (tr, te) in enumerate(splits):
    rep, fold = i // N_SPLITS, i % N_SPLITS
    _fold_test_mask[(rep, fold)] = te

# Full ML 基线：mean 头直接裁剪聚合（无融合）
full_ml_work = pred[["repeat", "fold", "case_id", "item_name", "claim_amount"]].copy()
full_ml_work["sr_hat"] = pred["mean"].to_numpy(float)

case_oof = {"REF_Full_ML": assemble_from_item(full_ml_work, "REF_Full_ML")}
for g in GAMMA_LIST:
    case_oof[f"gated_gamma_{g:g}"] = aggregate_gamma(g)
    log(f"  gamma={g:g} 案件级聚合完成")

# ===========================================================================
# 8. 指标汇总
# ===========================================================================
def metrics_case(g: pd.DataFrame) -> dict:
    y, p = g["sr_true"].to_numpy(float), g["sr_pred"].to_numpy(float)
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(ss_res / len(y))),
            "MdAE": float(np.median(np.abs(err))),
            "R2": 1.0 - ss_res / ss_tot,
            "acc@5%": float(np.mean(np.abs(err) <= 0.05)),
            "acc@10%": float(np.mean(np.abs(err) <= 0.10)),
            "n_cases": int(len(y))}


by_repeat = []
for name, g in case_oof.items():
    for rep, sub in g.groupby("repeat"):
        m = metrics_case(sub)
        m.update({"system": name, "repeat": int(rep)})
        by_repeat.append(m)
by_repeat = pd.DataFrame(by_repeat)[
    ["system", "repeat", "MAE", "RMSE", "MdAE", "R2", "acc@5%", "acc@10%", "n_cases"]]

order = ["REF_Full_ML"] + [f"gated_gamma_{g:g}" for g in GAMMA_LIST]
summary = by_repeat.groupby("system").agg(
    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
    R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    acc5_mean=("acc@5%", "mean"), acc10_mean=("acc@10%", "mean"),
).reindex(order)

# 内部参照系（同一 OOF 上计算，无外部文件依赖）：
#   FULL_REF = REF_Full_ML（异质模型无融合）；SOFT_REF = gamma=0（JDI 静态连续融合）
r_full = summary.loc["REF_Full_ML"]
r_soft = summary.loc["gated_gamma_0"]
FULL_REF = {"MAE": float(r_full["MAE_mean"]), "RMSE": float(r_full["RMSE_mean"]),
            "R2": float(r_full["R2_mean"])}
SOFT_REF = {"MAE": float(r_soft["MAE_mean"]), "RMSE": float(r_soft["RMSE_mean"]),
            "R2": float(r_soft["R2_mean"])}

# ===========================================================================
# 9. 主模型判定（最终模型 gamma=1.0：MAE <= Soft Hybrid 且 R2 > Full ML）
# ===========================================================================
r_final = summary.loc[f"gated_gamma_{GAMMA_FINAL:g}"]
final_hits = (r_final["MAE_mean"] <= SOFT_REF["MAE"]
              and r_final["R2_mean"] > FULL_REF["R2"])

# ===========================================================================
# 10. 产物保存
# ===========================================================================
by_repeat.to_csv(OUT / "stage10_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUT / "stage10_summary.csv", encoding="utf-8-sig")
try:
    pred.to_parquet(OUT / "stage10_oof_item_predictions.parquet", index=False)
    pd.concat(case_oof.values()).to_parquet(OUT / "stage10_oof_case_predictions.parquet",
                                            index=False)
except Exception as e:
    log(f"  parquet 不可用（{type(e).__name__}），改存 CSV")
    pred.to_csv(OUT / "stage10_oof_item_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(case_oof.values()).to_csv(OUT / "stage10_oof_case_predictions.csv",
                                        index=False, encoding="utf-8-sig")

config_out = {
    "stage": "04_ultimate_gated_model",
    "protocol": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "random_state": RS,
                 "strata": "year|procedure_type|sr_total_quintile (发表版口径)"},
    "hetero_assignment": PER_ITEM,
    "heads": {"LightGBM": {"mean": "objective='regression'",
                           "q10/q90": "objective='quantile', alpha=0.1/0.9"},
              "CatBoost": {"mean": "loss_function='RMSE'",
                           "q10/q90": "loss_function='Quantile:alpha=0.1/0.9'"}},
    "uncertainty": "delta_uncert = clip(q90 - q10, 0, None)",
    "dynamic_weight": "w_dynamic = w_i(fold-internal minmax JDI) * exp(-gamma * delta)",
    "gamma_list": GAMMA_LIST,
    "gamma_final": GAMMA_FINAL,
    "internal_references": {"Full ML (no fusion)": FULL_REF,
                            "Soft Hybrid (gamma=0)": SOFT_REF},
    "notes": [
        "规则与 JDI 均为训练折内计算，非 03 OOF / 02 冻结值",
        "分项模型指派与 gamma=1.0 均为冻结的最佳设定（原 Stage 5 / Stage 9-10 扫描选定）",
        "gamma 网格在原开发数据上扫描选定，正式结论需独立数据确认",
    ],
}
(OUT / "stage10_config.json").write_text(
    json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 图 1：MAE-R² 平面 ----
fig, ax = plt.subplots(figsize=(7.0, 5.6))
xs = [summary.loc[f"gated_gamma_{g:g}", "R2_mean"] for g in GAMMA_LIST]
ys = [summary.loc[f"gated_gamma_{g:g}", "MAE_mean"] for g in GAMMA_LIST]
ax.plot(xs, ys, "o-", color="#C0392B", lw=1.5, ms=6, label="Ultimate Gated (异质模型)")
for g, x, y in zip(GAMMA_LIST, xs, ys):
    ax.annotate(f"γ={g:g}", (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
ax.scatter([FULL_REF["R2"]], [FULL_REF["MAE"]], marker="s", s=100, color="#8C8C8C",
           zorder=5, label="Full ML (no fusion)")
ax.scatter([SOFT_REF["R2"]], [SOFT_REF["MAE"]], marker="^", s=100, color="#1F3A5F",
           zorder=5, label="Soft Hybrid (γ=0)")
ax.axhline(SOFT_REF["MAE"], color="#1F3A5F", ls=":", lw=0.9, alpha=0.7)
ax.axvline(FULL_REF["R2"], color="#8C8C8C", ls=":", lw=0.9, alpha=0.7)
ax.set_xlabel("案件级 R²")
ax.set_ylabel("案件级 MAE")
ax.set_title("Ultimate Gated Hybrid — MAE–R² 平面\n"
             "（目标区域：MAE ≤ Soft Hybrid 且 R² > Full ML 的左上象限）")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig1_ultimate_pareto.png", dpi=160)
plt.close(fig)

# ---- Markdown 报告 ----
md = [
    "# 04 — Ultimate SOTA（异质模型 + 认知不确定性动态门控）",
    "",
    f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"- 协议：{N_SPLITS} 折 x {N_REPEATS} 重复（rs={RS}，发表版口径）",
    f"- 最终主模型：gamma={GAMMA_FINAL}（冻结设定）",
    "",
    "| Model / Setting | MAE | RMSE | R2 |",
    "|---|---:|---:|---:|",
    f"| Full ML (Hetero, no fusion) | {FULL_REF['MAE']:.4f} | {FULL_REF['RMSE']:.4f} | {FULL_REF['R2']:.4f} |",
    f"| Soft Hybrid (JDI static, gamma=0) | {SOFT_REF['MAE']:.4f} | {SOFT_REF['RMSE']:.4f} | {SOFT_REF['R2']:.4f} |",
]
for g in GAMMA_LIST:
    r = summary.loc[f"gated_gamma_{g:g}"]
    md.append(f"| Gated Hybrid (gamma={g:g}) | {r['MAE_mean']:.4f} ± {r['MAE_std']:.4f} "
              f"| {r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f} "
              f"| {r['R2_mean']:.4f} ± {r['R2_std']:.4f} |")
md += ["",
       f"**主模型判定（gamma={GAMMA_FINAL}: MAE ≤ {SOFT_REF['MAE']:.4f} 且 "
       f"R² > {FULL_REF['R2']:.4f}）：{'PASS' if final_hits else '未达成'}**",
       "",
       "## 口径说明",
       "- 规则标定、JDI 权重、异质三头模型均在同一 5x3 CV 协议下训练折内计算；",
       "- REF_Full_ML / gamma=0 参照系与最终模型共享同一 OOF 基础预测，差异完全归因于融合机制；",
       "- gamma 于原开发数据上扫描选定并冻结为 1.0，正式结论需独立数据确认。",
       ]
(OUT / "stage10_comparison.md").write_text("\n".join(md), encoding="utf-8")

# ===========================================================================
# 11. 控制台输出
# ===========================================================================
print("\n" + "=" * 96)
print("04 最终对比（案件级 SR_total，3 repeats 均值 ± std；统一 5x3 CV 协议）")
print("=" * 96)
print(f"| {'Model / Setting':<40} | {'MAE':>17} | {'RMSE':>17} | {'R2':>17} |")
print(f"|{'-' * 42}|{'-' * 19}|{'-' * 19}|{'-' * 19}|")
print(f"| {'Full ML (Hetero, no fusion)':<40} "
      f"| {FULL_REF['MAE']:>9.4f} {'':>7} | {FULL_REF['RMSE']:>9.4f} {'':>7} "
      f"| {FULL_REF['R2']:>9.4f} {'':>7} |")
print(f"| {'Soft Hybrid (JDI static, gamma=0)':<40} "
      f"| {SOFT_REF['MAE']:>9.4f} {'':>7} | {SOFT_REF['RMSE']:>9.4f} {'':>7} "
      f"| {SOFT_REF['R2']:>9.4f} {'':>7} |")
for g in GAMMA_LIST:
    r = summary.loc[f"gated_gamma_{g:g}"]
    print(f"| {'Gated Hybrid (gamma=' + format(g, 'g') + ')':<40} "
          f"| {r['MAE_mean']:>9.4f} ± {r['MAE_std']:.4f} "
          f"| {r['RMSE_mean']:>9.4f} ± {r['RMSE_std']:.4f} "
          f"| {r['R2_mean']:>9.4f} ± {r['R2_std']:.4f} |")
print("=" * 96)
print(f"\n主模型判定（gamma={GAMMA_FINAL}: MAE ≤ {SOFT_REF['MAE']:.4f} 且 "
      f"R² > {FULL_REF['R2']:.4f}）：{'PASS' if final_hits else '未达成'}")
print(f"\n产物目录：{OUT}")

if final_hits:
    print(f"\n🏆 FINAL MODEL SECURED (gamma={GAMMA_FINAL:g}, "
          f"MAE={r_final['MAE_mean']:.4f}, RMSE={r_final['RMSE_mean']:.4f}, "
          f"R2={r_final['R2_mean']:.4f})")
else:
    print("\n门控相对内部参照系的优势未完全达成：详见 stage10_comparison.md 与 fig1_ultimate_pareto.png。")

log("04 完成。")
