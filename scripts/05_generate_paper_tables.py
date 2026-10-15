# -*- coding: utf-8 -*-
"""
05 — 论文实验综合更新（配合主模型定稿：Gated Hybrid, gamma=1.0）
==========================================================================

指标体系锁定：MAE / RMSE / R² / MdAE（四大金刚）

任务 A（论文表 10）：分组误差分析——主模型案件级 OOF 按四个维度分组，
    每组报告 MAE / RMSE / MdAE（不含 R²）。
任务 B（论文表 5）：逐级提升来源分解——统一 5折x3重复 pooled 口径，
    S0 Plain LGBM -> S1 Structured LGBM -> Full ML(异质) -> Soft Hybrid(g=0)
    -> Gated Hybrid(g=1)，含逐步 ΔMAE / ΔR²。
    注：S0/S1 两个 LightGBM 基线在本仓库内以同一 CV 协议重算
    （原开发仓库的公平比较产物不随开源发布）；
    Full ML / Soft Hybrid / Gated 三级直接读取 04 的案件级 OOF。
任务 C（论文表 9）：跨年份外推——2022 训练 / 2023 测试，主模型核心逻辑
    （异质三头分位数 + 折内规则/JDI + gamma=1 门控）在时间划分上的完整重演，
    对比 Full ML / Soft Hybrid / Gated / 全局均值 / 规则系统。

输出：results/stage12_paper_updates/
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[1]
S1 = BASE / "results" / "stage1"
S2D = BASE / "results" / "stage2"
S10 = BASE / "results" / "stage10_ultimate_sota"
OUT = BASE / "results" / "stage12_paper_updates"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "src"))
from model_library import build_X, make_model, select_rule_fast  # noqa: E402
from rule_engine.rule_base import (CapRule, GlobalMedianRule,  # noqa: E402
                                   QuantilePriorRule)

policy = yaml.safe_load((BASE / "configs" / "experiment_policy.yaml").read_text(encoding="utf-8"))
RS = int(policy["global"]["random_state"])
ALL_ITEMS = list(policy["data"]["items"])
strategy = json.loads((BASE / "configs" / "project_strategy_config.json").read_text(encoding="utf-8"))
A_ITEMS = list(strategy["rule_based"])
C_ITEMS = list(strategy["small_sample"])
WEIGHTS = json.loads((S2D / "jdi_weight_config.json").read_text(encoding="utf-8"))["weights"]
POSITIVE = {"p_sr_eq_1", "p_sr_eq_0", "SR_mode_density_bin10", "liability_consistency"}
N_BINS = 10
N_SPLITS = int(policy["global"]["n_splits"])
N_REPEATS = int(policy["global"]["n_repeats"])
GAMMA = 1.0

# ===========================================================================
# 冻结分项模型指派（与 04_ultimate_gated_model.py 顶部完全一致）
# ===========================================================================
PER_ITEM = {
    "lost_income": "CatBoost",
    "nursing": "CatBoost",
    "property_loss": "CatBoost",
    "transport": "CatBoost",
    "mental_solatium": "LightGBM",
    "nutrition": "LightGBM",
    "other": "LightGBM",
}


def log(m: str) -> None:
    print(f"[05 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return {"n": len(y), "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "MdAE": float(np.median(np.abs(err))),
            "R2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan}


log("=" * 76)
log("05: 论文实验综合更新（主模型 = Gated Hybrid gamma=1.0）")
log("=" * 76)

case = pd.read_csv(S1 / "dataset_case_level.csv", encoding="utf-8-sig")
item_all = pd.read_csv(S1 / "dataset_item_level.csv", encoding="utf-8-sig",
                       usecols=["case_id", "item_name", "claim_amount", "award_amount",
                                "support_rate_label", "claim_positive"])
valid = item_all[item_all["claim_positive"] == 1].reset_index(drop=True)

# ===========================================================================
# 任务 A：分组误差分析（主模型 gamma=1.0，案件级 OOF x 案件特征）
# ===========================================================================
log("--- 任务 A：分组误差分析 ---")
try:
    oof10 = pd.read_parquet(S10 / "stage10_oof_case_predictions.parquet")
except Exception:
    oof10 = pd.read_csv(S10 / "stage10_oof_case_predictions.csv", encoding="utf-8-sig")
g1 = oof10[oof10["system"] == f"gated_gamma_{GAMMA:g}"][
    ["repeat", "case_id", "sr_pred", "sr_true"]] \
    .merge(case[["case_id", "disability_level_clean", "claim_total_calc",
                 "mental_solatium_claim", "year"]], on="case_id", how="left")
assert len(g1) == N_REPEATS * len(case), \
    f"g1 行数异常：{len(g1)} != {N_REPEATS} x {len(case)}"

g1["disability_group"] = np.select(
    [pd.to_numeric(g1["disability_level_clean"], errors="coerce") == 0,
     pd.to_numeric(g1["disability_level_clean"], errors="coerce").between(1, 6),
     pd.to_numeric(g1["disability_level_clean"], errors="coerce").between(7, 10)],
    ["无伤残(0)", "重度(1-6级)", "轻度(7-10级)"], default="缺失/异常")
g1["claim_quartile"] = pd.qcut(g1["claim_total_calc"].rank(method="first"), 4,
                               labels=["Q1(最小)", "Q2", "Q3", "Q4(最大)"])
g1["mental_claimed"] = np.where(g1["mental_solatium_claim"] > 0, "已主张", "未主张")
g1["sr_interval"] = pd.cut(g1["sr_true"], bins=[-1e-9, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9],
                           labels=["[0,0.2]", "(0.2,0.4]", "(0.4,0.6]", "(0.6,0.8]", "(0.8,1.0]"])


DISABILITY_ORDER = ["无伤残(0)", "轻度(7-10级)", "重度(1-6级)", "缺失/异常"]


def group_metrics(df: pd.DataFrame, by: str, order=None) -> pd.DataFrame:
    rows = []
    for gval, sub in df.groupby(by, observed=True):
        m = metrics(sub["sr_true"], sub["sr_pred"])
        rows.append({"分组维度": by, "分组": gval, "n": m["n"], "MAE": m["MAE"],
                     "RMSE": m["RMSE"], "MdAE": m["MdAE"]})
    out = pd.DataFrame(rows)
    if order is not None:
        out["_ord"] = out["分组"].map({g: i for i, g in enumerate(order)})
        out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return out


# 首行基准：全部袋外样本（5折 x 3重复 pooled OOF）
overall_m = metrics(g1["sr_true"], g1["sr_pred"])
baseline_row = pd.DataFrame([{"分组维度": "overall",
                              "分组": "全部袋外样本（基准）/ Overall baseline",
                              "n": overall_m["n"], "MAE": overall_m["MAE"],
                              "RMSE": overall_m["RMSE"], "MdAE": overall_m["MdAE"]}])

subgroup = pd.concat(
    [baseline_row]
    + [group_metrics(g1, c, order=DISABILITY_ORDER if c == "disability_group" else None)
       for c in ["disability_group", "claim_quartile", "mental_claimed", "sr_interval"]],
    ignore_index=True)
# 显示格式：样本量 "n (pct%)"；指标四位小数（论文表 10 口径）
n_total = overall_m["n"]
subgroup["n"] = subgroup["n"].astype(int).map(
    lambda k: f"{k:,} ({k / n_total * 100:.2f}%)")
for col in ("MAE", "RMSE", "MdAE"):
    subgroup[col] = subgroup[col].map(lambda v: f"{v:.4f}")
subgroup.to_csv(OUT / "table10_subgroup_error_analysis.csv", index=False, encoding="utf-8-sig")
log(f"  分组表：{subgroup.shape}（基准行 + 4 维度），已保存 table10_subgroup_error_analysis.csv")

# ===========================================================================
# 任务 B：逐级提升来源分解（pooled 口径）
#   S0 / S1 两级在同一 CV 协议下于本仓库内重算（均匀 LightGBM mean 头）；
#   Full ML / Soft Hybrid(g=0) / Gated Hybrid(g=1) 读取 04 案件级 OOF。
# ===========================================================================
log("--- 任务 B：逐级提升来源分解（S0/S1 基线重算） ---")

# --- 与 04 完全一致的特征与切分 ---
def disability_severity_score(level):
    lv = pd.to_numeric(level, errors="coerce")
    s = pd.Series(np.nan, index=lv.index)
    s[lv == 0] = 0.0
    for k in range(1, 11):
        s[lv == k] = round(1.1 - 0.1 * k, 2)
    return s


X_orig = build_X(case, "ordinal")
X_aug = X_orig.copy()
X_aug["disability_severity_score"] = disability_severity_score(
    case["disability_level_clean"]).values
sc = [c for c in case.columns if c.startswith("claim_share__")]
P_mat = np.where(np.isfinite(case[sc].to_numpy(float)), case[sc].to_numpy(float), 0.0)
with np.errstate(divide="ignore", invalid="ignore"):
    lp = np.where(P_mat > 0, np.log(P_mat), 0.0)
X_aug["claim_entropy"] = -np.sum(P_mat * lp, axis=1) / np.log(P_mat.shape[1])
Ps = np.sort(P_mat, axis=1)[:, ::-1]
X_aug["top1_claim_share"] = Ps[:, 0]
X_aug["top3_claim_share"] = Ps[:, :3].sum(axis=1)
FEATURE_COLS_FULL = list(X_orig.columns) + ["disability_severity_score", "claim_entropy",
                                            "top1_claim_share", "top3_claim_share",
                                            "claim_amount_relative_ratio"]
FEATURE_COLS_PLAIN = list(X_orig.columns)   # S0：仅基础判前司法特征

sr_bin = pd.qcut(case["sr_total_label"].rank(method="first"), 5, labels=False)
strata = (case["year"].astype(str) + "|" + case["procedure_type"].astype(str)
          + "|" + sr_bin.astype(str))
case_ids = case["case_id"].values
claim_total_by_case = case.set_index("case_id")["claim_total_calc"]
sr_true_by_case = case.set_index("case_id")["sr_total_label"]
splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                      random_state=RS).split(case, strata))


def assemble_case(sr_hat_item: pd.DataFrame, fold_test_mask: dict,
                  fold_medians: dict) -> pd.DataFrame:
    """分项级 sr_hat -> 案件级 SR_total（含折内中位数回退）。"""
    out = []
    for (rep, fold), g in sr_hat_item.groupby(["repeat", "fold"]):
        fs = fold_medians[(int(rep), int(fold))]
        item_median, gmean = fs["item_median"], fs["gmean"]
        test_cases = case_ids[fold_test_mask[(int(rep), int(fold))]]
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
            out.append({"system": "", "repeat": rep,
                        "case_id": cid, "sr_pred": float(pv), "sr_true": float(t)})
    return pd.DataFrame(out)


def run_lgbm_baseline(feature_cols: list, system: str) -> pd.DataFrame:
    """均匀 LightGBM（mean 头）基线，协议与 04 完全一致。"""
    fold_test_mask, fold_medians = {}, {}
    rows = []
    for i, (tr, te) in enumerate(splits):
        rep, fold = i // N_SPLITS, i % N_SPLITS
        train_cases = set(case_ids[tr].tolist())
        test_cases = case_ids[te]
        train_valid = valid[valid["case_id"].isin(train_cases)]
        fold_medians[(rep, fold)] = {
            "item_median": train_valid.groupby("item_name")["support_rate_label"].median().to_dict(),
            "gmean": float(train_valid["support_rate_label"].mean())}
        fold_test_mask[(rep, fold)] = te
        for it in ALL_ITEMS:
            tr_it = train_valid[train_valid["item_name"] == it]
            te_it = valid[(valid["item_name"] == it) & (valid["case_id"].isin(test_cases))]
            if len(te_it) == 0 or len(tr_it) < 5:
                continue
            y_tr = tr_it["support_rate_label"].values
            med = float(np.nanmedian(tr_it["claim_amount"].values))
            Xtr = X_aug.loc[tr_it["case_id"].values].copy()
            Xte = X_aug.loc[te_it["case_id"].values].copy()
            if "claim_amount_relative_ratio" in feature_cols:
                Xtr["claim_amount_relative_ratio"] = (tr_it["claim_amount"].values / med
                                                      if med > 0 else 0.0)
                Xte["claim_amount_relative_ratio"] = (te_it["claim_amount"].values / med
                                                      if med > 0 else 0.0)
            p = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("model", make_model("LightGBM"))])
            p.fit(Xtr[feature_cols], y_tr)
            pr = np.clip(np.asarray(p.predict(Xte[feature_cols]), dtype=float), 0, 1)
            for cid, v in zip(te_it["case_id"].values, pr):
                rows.append({"repeat": rep, "fold": fold, "case_id": cid,
                             "item_name": it, "sr_hat": float(v)})
        if (i + 1) % 5 == 0:
            log(f"    [{system}] 折进度 {i + 1}/{len(splits)}")
    item_df = pd.DataFrame(rows)
    res = assemble_case(item_df, fold_test_mask, fold_medians)
    res["system"] = system
    return res


s0 = run_lgbm_baseline(FEATURE_COLS_PLAIN, "S0_plain_LightGBM")
log(f"  S0 Plain LightGBM 重算完成：{len(s0)} 行")
s1 = run_lgbm_baseline(FEATURE_COLS_FULL, "S1_structured_LightGBM")
log(f"  S1 Structured LightGBM 重算完成：{len(s1)} 行")

# --- 从 04 案件级 OOF 取 Full ML / Soft Hybrid / Gated（pooled = 全 repeats 合并） ---
def pooled_from_oof(system: str) -> dict:
    sub = oof10[oof10["system"] == system]
    return metrics(sub["sr_true"], sub["sr_pred"])


g1_pooled = metrics(g1["sr_true"], g1["sr_pred"])

path_defs = [
    ("S0 Plain LightGBM（起点）", metrics(s0["sr_true"], s0["sr_pred"])),
    ("S1 Structured LightGBM（+司法语义特征）", metrics(s1["sr_true"], s1["sr_pred"])),
    ("Full ML（+分项异质建模）", pooled_from_oof("REF_Full_ML")),
    ("Soft Hybrid（+JDI 静态连续融合, γ=0）", pooled_from_oof("gated_gamma_0")),
    ("Gated Hybrid（+不确定性门控, γ=1, 最终主模型）", g1_pooled),
]

rows_b = []
for i, (name, m) in enumerate(path_defs):
    d_mae = m["MAE"] - path_defs[i - 1][1]["MAE"] if i > 0 else np.nan
    d_r2 = m["R2"] - path_defs[i - 1][1]["R2"] if i > 0 else np.nan
    rows_b.append({"步骤": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                   "R2": m["R2"], "MdAE": m["MdAE"],
                   "ΔMAE(相对上一步)": d_mae, "ΔR2(相对上一步)": d_r2})
path_b = pd.DataFrame(rows_b)
path_b.to_csv(OUT / "table5_improvement_path.csv", index=False, encoding="utf-8-sig")
log(f"  提升路径表：{path_b.shape}，已保存 table5_improvement_path.csv")

# ===========================================================================
# 任务 C：跨年份外推（2022 训练 -> 2023 测试，主模型核心逻辑重演）
# ===========================================================================
log("--- 任务 C：跨年份外推验证 ---")


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
        sc_ = float(np.mean(errs))
        if best is None or sc_ < best[0] - 1e-12:
            best = (sc_, name, cls, params)
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


def make_head(item: str, head: str):
    name = PER_ITEM.get(item, "LightGBM")
    alpha = 0.1 if head == "q10" else 0.9
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        if head == "mean":
            return make_model("LightGBM")
        return LGBMRegressor(objective="quantile", alpha=alpha,
                             random_state=RS, n_jobs=4, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        if head == "mean":
            return make_model("CatBoost")
        return CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}",
                                 random_state=RS, verbose=0, allow_writing_files=False)
    raise ValueError(name)


# ---- 时间划分 ----
year_num = pd.to_numeric(case["year"], errors="coerce")
train_mask = (year_num == 2022).values
test_mask = (year_num == 2023).values
train_cases = set(case["case_id"].values[train_mask])
test_cases = case["case_id"].values[test_mask]
train_valid = valid[valid["case_id"].isin(train_cases)]
log(f"  训练 2022：{int(train_mask.sum())} 案；测试 2023：{int(test_mask.sum())} 案")

item_median = train_valid.groupby("item_name")["support_rate_label"].median().to_dict()
gmean = float(train_valid["support_rate_label"].mean())

P_rule, preds = {}, {}
t0 = time.time()
for it in ALL_ITEMS:
    tr_it = train_valid[train_valid["item_name"] == it]
    te_it = valid[(valid["item_name"] == it) & (valid["case_id"].isin(set(test_cases)))]
    if len(te_it) == 0 or len(tr_it) < 5:
        continue
    P_rule[it] = pd.Series(rule_predict(tr_it, te_it, it), index=te_it["case_id"].values)
    y_tr = tr_it["support_rate_label"].values
    med = float(np.nanmedian(tr_it["claim_amount"].values))
    Xtr = X_aug.loc[tr_it["case_id"].values].copy()
    Xte = X_aug.loc[te_it["case_id"].values].copy()
    Xtr["claim_amount_relative_ratio"] = (tr_it["claim_amount"].values / med if med > 0 else 0.0)
    Xte["claim_amount_relative_ratio"] = (te_it["claim_amount"].values / med if med > 0 else 0.0)
    d = {}
    for head in ("mean", "q10", "q90"):
        p = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("model", make_head(it, head))])
        p.fit(Xtr[FEATURE_COLS_FULL], y_tr)
        d[head] = np.asarray(p.predict(Xte[FEATURE_COLS_FULL]), dtype=float)
    preds[it] = {"mean": np.clip(d["mean"], 0, 1), "q10": d["q10"], "q90": d["q90"],
                 "case_ids": te_it["case_id"].values,
                 "claim": te_it["claim_amount"].values}
log(f"  三头异质模型训练+推理完成（{time.time() - t0:.0f}s）")

jdi = compute_jdi_fold(train_valid, ALL_ITEMS)["JDI"]
w_map = (jdi - jdi.min()) / (jdi.max() - jdi.min()) if jdi.max() > jdi.min() \
    else pd.Series(0.5, index=jdi.index)


def assemble_temporal(sr_by_item: dict) -> pd.Series:
    tot = pd.Series(0.0, index=pd.Index(test_cases, name="case_id"))
    for it in ALL_ITEMS:
        te_it = valid[(valid["item_name"] == it) & (valid["case_id"].isin(set(test_cases)))]
        if len(te_it) == 0:
            continue
        s = sr_by_item.get(it)
        if s is None:
            sr_hat = np.full(len(te_it), item_median.get(it, gmean))
        else:
            v = s.reindex(te_it["case_id"].values).to_numpy()
            sr_hat = np.where(np.isnan(v), item_median.get(it, gmean), v)
        aw = pd.Series(np.clip(sr_hat, 0, 1) * te_it["claim_amount"].values,
                       index=te_it["case_id"].values)
        tot = tot.add(aw.groupby(level=0).sum(), fill_value=0.0)
    denom = claim_total_by_case.reindex(tot.index).values
    return pd.Series(np.clip(tot.values / np.where(denom > 0, denom, np.nan), 0, 1),
                     index=tot.index)


# 各系统分项预测
sr_items = {}
sr_items["B1_Global_Mean(全局均值)"] = {it: pd.Series(gmean, index=preds[it]["case_ids"])
                                       for it in preds}
rule_sys = {}
for it in preds:
    if it in A_ITEMS or it in C_ITEMS:
        rule_sys[it] = P_rule[it]
    else:
        rule_sys[it] = pd.Series(item_median.get(it, gmean), index=preds[it]["case_ids"])
sr_items["Rule_System(规则系统)"] = rule_sys
sr_items["Full_ML_异质_2022重训"] = {it: pd.Series(preds[it]["mean"], index=preds[it]["case_ids"])
                                    for it in preds}
for gamma, label in [(0.0, "Soft_Hybrid_静态γ=0"), (GAMMA, f"Gated_Hybrid_γ={GAMMA:g}")]:
    fused = {}
    for it in preds:
        w_it = float(w_map.get(it, 0.5))
        mean_ = preds[it]["mean"]
        rule_ = P_rule[it].values
        delta_ = np.clip(preds[it]["q90"] - preds[it]["q10"], 0, None)
        w_dyn = w_it * np.exp(-gamma * delta_)
        sr = np.clip(w_dyn * rule_ + (1.0 - w_dyn) * mean_, 0, 1)
        fused[it] = pd.Series(sr, index=preds[it]["case_ids"])
    sr_items[label] = fused

rows_c = []
for sysname, mp in sr_items.items():
    pr = assemble_temporal(mp)
    tv = sr_true_by_case.reindex(pr.index).values
    m = metrics(tv, pr.values)
    rows_c.append({"System": sysname, "n_test": m["n"], "MAE": m["MAE"],
                   "RMSE": m["RMSE"], "R2": m["R2"], "MdAE": m["MdAE"]})
table_c = pd.DataFrame(rows_c)
table_c.to_csv(OUT / "table9_temporal_validation_2022to2023.csv", index=False,
               encoding="utf-8-sig")
log(f"  时间外推表：{table_c.shape}，已保存 table9_temporal_validation_2022to2023.csv")

# ===========================================================================
# 控制台输出（论文可直接复制的 Markdown 表）
# ===========================================================================
print("\n" + "=" * 100)
print("任务 B —— 逐级提升来源分解（论文表 5；5折x3重复 pooled 口径）")
print("=" * 100)
print(path_b.to_markdown(index=False, floatfmt=".4f"))

print("\n" + "=" * 100)
print(f"任务 C —— 跨年份外推验证（论文表 9；2022 训练 -> 2023 测试，n={int(test_mask.sum())}）")
print("=" * 100)
print(table_c.to_markdown(index=False, floatfmt=".4f"))

print("\n" + "=" * 100)
print("任务 A —— 分组误差分析摘要（论文表 10；完整表见 CSV）")
print("=" * 100)
print(subgroup.to_markdown(index=False, floatfmt=".4f"))

config_out = {
    "stage": "05_generate_paper_tables",
    "main_model": "Gated Hybrid (gamma=1.0)",
    "metrics_locked": ["MAE", "RMSE", "R2", "MdAE"],
    "task_a": "主模型案件级 OOF 分组误差（4 维度）",
    "task_b": "提升路径（S0/S1 本仓库重算 + 04 案件级 OOF pooled）",
    "task_c": {"split": "year==2022 train / year==2023 test",
               "protocol": "主模型核心逻辑重演：折内规则/JDI + 异质三头 + gamma=1 门控",
               "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())},
}
(OUT / "stage12_config.json").write_text(
    json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")
log("05 完成。全部产物见 results/stage12_paper_updates/")
