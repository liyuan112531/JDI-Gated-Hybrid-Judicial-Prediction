# -*- coding: utf-8 -*-
"""
Stage 3 —— 规则引擎与强基线体系

本阶段禁止训练任何机器学习模型（仅使用规则计算与训练折内统计估计）。

内容：
  1. A 类（5 项）规则预测
  2. C 类（4 项）Rule-informed fallback
  3. 四条强基线（Global mean / Item median / liability_ratio / claim × liability_ratio）
  4. 5 折 × 3 次重复的案件级交叉验证评估（所有统计量仅训练折内计算）
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rule_engine import (A_CLASS_RULES, C_CLASS_FALLBACK,  # noqa: E402
                         build_candidates, build_rule)

CONFIGS = ROOT / "configs"
S1 = ROOT / "results" / "stage1"
S2 = ROOT / "results" / "stage2"
OUT = ROOT / "results" / "stage3"
REPORTS = ROOT / "results" / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

NOW = datetime.now().isoformat(timespec="seconds")


def log(m: str) -> None:
    print(m, flush=True)


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    log(f"[saved] results/stage3/{name}  rows={len(df)}")


# ===========================================================================
# 一、读取配置与数据
# ===========================================================================
log("=" * 72)
log("一、读取配置与数据")
log("=" * 72)

policy = yaml.safe_load((CONFIGS / "experiment_policy.yaml").read_text(encoding="utf-8"))
proj = json.loads((CONFIGS / "project_strategy_config.json").read_text(encoding="utf-8"))

A_ITEMS = list(proj["rule_based"])
B_ITEMS = list(proj["ml_based"])
C_ITEMS = list(proj["small_sample"])
ALL_ITEMS = list(policy["data"]["items"])
N_SPLITS = int(policy["global"]["n_splits"])
N_REPEATS = int(policy["global"]["n_repeats"])
RS = int(policy["global"]["random_state"])

log(f"  A 类（规则）   ：{A_ITEMS}")
log(f"  B 类（ML）     ：{B_ITEMS}")
log(f"  C 类（fallback）：{C_ITEMS}")
log(f"  CV：{N_SPLITS} 折 × {N_REPEATS} 次重复，random_state={RS}")

case = pd.read_csv(S1 / "dataset_case_level.csv", encoding="utf-8-sig")
item_all = pd.read_csv(
    S1 / "dataset_item_level.csv", encoding="utf-8-sig",
    usecols=["case_id", "item_name", "claim_amount", "award_amount",
             "support_rate_label", "claim_positive",
             "liability_ratio_clean", "disability_level_clean"])
valid = item_all[item_all["claim_positive"] == 1].copy().reset_index(drop=True)
log(f"  案件级 {case.shape}；分项有效样本 {len(valid)}")

# 分层变量：year × sr_total 十分位 × liability_missing
sr_bin = pd.qcut(case["sr_total_label"].rank(method="first"), 5, labels=False)
strata = (case["year"].astype(str) + "|" + sr_bin.astype(str)
          + "|" + case["liability_missing"].astype(str))
log(f"  分层变量唯一组合数：{strata.nunique()}")

# ===========================================================================
# 二-七、交叉验证评估
# ===========================================================================
log("=" * 72)
log("二-七、规则与基线的交叉验证评估")
log("=" * 72)

def select_rule(train_it: pd.DataFrame, item: str, n_inner: int = 3, rs: int = 42):
    """
    规则标定（rule selection）：在训练折内部做 n_inner 折交叉验证，
    从候选规则集中选出 MAE 最低者，再用完整训练折重新拟合。

    注意：只接触训练折数据；这是规则参数标定，不是机器学习模型训练。
    """
    ids = train_it["case_id"].unique()
    rng = np.random.default_rng(rs)
    ids_shuffled = rng.permutation(ids)
    parts = np.array_split(ids_shuffled, n_inner)
    best = None
    for name, cls, params in build_candidates(item):
        errs = []
        for k in range(n_inner):
            te_ids = set(parts[k].tolist())
            tr_ids = set(ids_shuffled.tolist()) - te_ids
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
        score = float(np.mean(errs))
        if best is None or score < best[0] - 1e-12:
            best = (score, name, cls, params)
    if best is None:
        return None, "none", np.nan
    rule = best[2](item, **best[3])
    rule.fit(train_it)
    return rule, best[1], best[0]


cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RS)
splits = list(cv.split(case, strata))
log(f"  总折数：{len(splits)}")

oof_item_rows = []
oof_case_rows = []
disab_fold_rows = []
fit_log_rows = []
consistency_checks = []

for i, (tr, te) in enumerate(splits):
    rep, fold = i // N_SPLITS, i % N_SPLITS
    train_ids = set(case["case_id"].iloc[tr])
    test_ids = set(case["case_id"].iloc[te])
    train_valid = valid[valid["case_id"].isin(train_ids)]
    test_valid = valid[valid["case_id"].isin(test_ids)].copy()

    # ---- A 类规则 + C 类 fallback（法律先验规则 & 训练折标定规则）----
    pred_rule: dict = {}
    pred_rule_sel: dict = {}
    for it in A_ITEMS + C_ITEMS:
        tr_it = train_valid[train_valid["item_name"] == it]
        te_it = test_valid[test_valid["item_name"] == it]
        if len(tr_it) == 0 or len(te_it) == 0:
            continue
        # (a) 法律先验规则
        rule = build_rule(it)
        rule.fit(tr_it)
        p = rule.predict_frame(te_it, it)
        # 统一接口一致性校验（向量化 vs predict_sr）
        for si in te_it.index[:3]:
            single = rule.predict_sr(te_it.loc[si], it)
            vec = p[list(te_it.index).index(si)]
            consistency_checks.append(abs(float(single) - float(vec)) < 1e-9)
        pred_rule[it] = pd.Series(np.asarray(p, dtype=float), index=te_it.index)
        fit_log_rows.append({"repeat": rep, "fold": fold, "item": it,
                             "variant": "legal_prior", "rule": rule.name,
                             "n_train": len(tr_it), "inner_mae": np.nan,
                             "fit_info": json.dumps(rule.fit_info, ensure_ascii=False)})
        # (b) 训练折内标定的规则（候选集内部交叉验证选择）
        sel_rule, sel_name, sel_score = select_rule(tr_it, it)
        if sel_rule is not None:
            pred_rule_sel[it] = pd.Series(
                np.asarray(sel_rule.predict_frame(te_it, it), dtype=float), index=te_it.index)
            fit_log_rows.append({"repeat": rep, "fold": fold, "item": it,
                                 "variant": "selected", "rule": sel_name,
                                 "n_train": len(tr_it), "inner_mae": round(float(sel_score), 6),
                                 "fit_info": json.dumps(sel_rule.fit_info, ensure_ascii=False)})
        if it == "disability_comp":
            for lvl, med in rule.table.items():
                sub = tr_it[pd.to_numeric(tr_it["disability_level_clean"], errors="coerce") == lvl]
                disab_fold_rows.append({
                    "repeat": rep, "fold": fold, "level": float(lvl),
                    "n_train": int(rule.group_n.get(lvl, len(sub))),
                    "median_SR_fold": round(float(med), 6),
                    "mean_SR_fold": round(float(sub["support_rate_label"].mean()), 6),
                    "p_sr_eq_1_fold": round(float((sub["support_rate_label"] >= 1 - 1e-4).mean()), 6),
                })

    # ---- 四条强基线 ----
    g_mean = float(train_valid["support_rate_label"].mean())
    item_med = train_valid.groupby("item_name")["support_rate_label"].median().to_dict()
    case_sr_med = float(case.loc[case["case_id"].isin(train_ids), "sr_total_label"].median())

    tv = test_valid
    b1 = np.full(len(tv), g_mean, dtype=float)
    med_map = tv["item_name"].map(item_med).fillna(g_mean).to_numpy(dtype=float)
    b2 = np.clip(med_map, 0, 1)
    lr = tv["liability_ratio_clean"].to_numpy(dtype=float)
    lr_clip = np.where(np.isnan(lr), np.nan, np.clip(lr, 0, 1))
    b3 = np.where(np.isnan(lr_clip), med_map, lr_clip)        # 缺失 -> 分项中位数
    b4 = np.where(np.isnan(lr_clip), case_sr_med, lr_clip)    # 缺失 -> 案件级 SR_total 中位数

    tmp = tv[["case_id", "item_name", "claim_amount", "award_amount",
              "support_rate_label"]].copy()
    tmp["repeat"] = rep
    tmp["fold"] = fold
    tmp["pred_rule"] = np.nan
    tmp["pred_rule_sel"] = np.nan
    for it, s in pred_rule.items():
        tmp.loc[s.index, "pred_rule"] = s.values
    for it, s in pred_rule_sel.items():
        tmp.loc[s.index, "pred_rule_sel"] = s.values
    tmp["pred_b1"] = b1
    tmp["pred_b2"] = b2
    tmp["pred_b3"] = b3
    tmp["pred_b4"] = b4
    oof_item_rows.append(tmp)

    # ---- 案件级组装 ----
    tmp["pred_sys"] = tmp["pred_rule_sel"].where(tmp["pred_rule_sel"].notna(), tmp["pred_b2"])
    tmp["award_hat_sys"] = tmp["claim_amount"] * tmp["pred_sys"]
    tmp["pred_sys_legal"] = tmp["pred_rule"].where(tmp["pred_rule"].notna(), tmp["pred_b2"])
    tmp["award_hat_legal"] = tmp["claim_amount"] * tmp["pred_sys_legal"]
    g = tmp.groupby("case_id").agg(claim_sum=("claim_amount", "sum"),
                                   award_hat=("award_hat_sys", "sum"),
                                   award_hat_legal=("award_hat_legal", "sum"))
    sr_sys_full = (g["award_hat"] / g["claim_sum"]).clip(0, 1)
    sr_sys_legal = (g["award_hat_legal"] / g["claim_sum"]).clip(0, 1)

    ac_mask = tmp["item_name"].isin(A_ITEMS + C_ITEMS)
    ac = tmp[ac_mask]
    gac = ac.groupby("case_id").agg(claim=("claim_amount", "sum"),
                                    award_hat=("award_hat_sys", "sum"),
                                    award_true=("award_amount", "sum"))
    sr_ac_pred = (gac["award_hat"] / gac["claim"]).clip(0, 1)
    sr_ac_true = gac["award_true"] / gac["claim"]

    for b in ("b1", "b2", "b3", "b4"):
        tmp[f"award_{b}"] = tmp["claim_amount"] * tmp[f"pred_{b}"]
    gb = tmp.groupby("case_id")[[f"award_{b}" for b in ("b1", "b2", "b3", "b4")]].sum()

    sr_true_full = case.set_index("case_id")["sr_total_label"].reindex(g.index)
    case_out = pd.DataFrame({
        "case_id": g.index, "repeat": rep, "fold": fold,
        "claim_total": g["claim_sum"].values,
        "sr_true": sr_true_full.values,
        "sr_pred_rule_full": sr_sys_full.values,
        "sr_pred_rulelegal_full": sr_sys_legal.values,
        "sr_pred_rule_ac": sr_ac_pred.reindex(g.index).values,
        "sr_true_ac": sr_ac_true.reindex(g.index).values,
    })
    for b in ("b1", "b2", "b3", "b4"):
        case_out[f"sr_pred_{b}"] = (gb[f"award_{b}"] / g["claim_sum"]).clip(0, 1).values
    oof_case_rows.append(case_out)

log(f"  统一接口一致性校验（predict_sr vs predict_frame）："
    f"{sum(consistency_checks)}/{len(consistency_checks)} 通过")

oof_item = pd.concat(oof_item_rows, ignore_index=True)
oof_case = pd.concat(oof_case_rows, ignore_index=True)
log(f"  分项级 OOF 预测：{len(oof_item)} 行；案件级 OOF：{len(oof_case)} 行")

# ===========================================================================
# 八、统一评价
# ===========================================================================
log("=" * 72)
log("八、统一评价")
log("=" * 72)


def calc_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MdAE": np.nan,
                "R2": np.nan, "within_5pct": np.nan, "within_10pct": np.nan}
    err = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MdAE": float(np.median(np.abs(err))),
        "R2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
        "within_5pct": float(np.mean(np.abs(err) <= 0.05)),
        "within_10pct": float(np.mean(np.abs(err) <= 0.10)),
    }


def metrics_with_repeat(df: pd.DataFrame, true_col: str, pred_col: str) -> dict:
    """pooled 指标 + 按 repeat 的 MAE 均值±标准差"""
    base = calc_metrics(df[true_col].to_numpy(), df[pred_col].to_numpy())
    maes = []
    for rep, sub in df.groupby("repeat"):
        mm = calc_metrics(sub[true_col].to_numpy(), sub[pred_col].to_numpy())
        if mm["n"] > 0:
            maes.append(mm["MAE"])
    base["MAE_mean_over_repeats"] = float(np.mean(maes)) if maes else np.nan
    base["MAE_std_over_repeats"] = float(np.std(maes, ddof=1)) if len(maes) > 1 else np.nan
    return base


SYSTEMS_ITEM = ["Rule_legal", "Rule_selected", "B1_global_mean", "B2_item_median",
                "B3_liability_SR", "B4_claim_x_liability"]
PRED_COL = {"Rule_legal": "pred_rule", "Rule_selected": "pred_rule_sel",
            "B1_global_mean": "pred_b1", "B2_item_median": "pred_b2",
            "B3_liability_SR": "pred_b3", "B4_claim_x_liability": "pred_b4"}
ITEM_CLASS = {**{i: "A" for i in A_ITEMS}, **{i: "B" for i in B_ITEMS},
              **{i: "C" for i in C_ITEMS}}

# --- rule_metrics.csv（系统 × 项目） ---
rm_rows = []
for sysname in SYSTEMS_ITEM:
    for it in ALL_ITEMS:
        sub = oof_item[oof_item["item_name"] == it]
        if sub.empty:
            continue
        pred = sub[PRED_COL[sysname]]
        if pred.isna().all():
            rm_rows.append({"system": sysname, "item_name": it, "item_class": ITEM_CLASS[it],
                            "n": 0, "MAE": np.nan, "RMSE": np.nan, "MdAE": np.nan,
                            "R2": np.nan, "within_5pct": np.nan, "within_10pct": np.nan,
                            "MAE_mean_over_repeats": np.nan, "MAE_std_over_repeats": np.nan})
            continue
        sub = sub.assign(_p=pred)
        m = metrics_with_repeat(sub, "support_rate_label", "_p")
        rm_rows.append({"system": sysname, "item_name": it, "item_class": ITEM_CLASS[it], **m})
rule_metrics = pd.DataFrame(rm_rows)
save_csv(rule_metrics, "rule_metrics.csv")

# --- baseline_metrics.csv（基线 × 口径） ---
bm_rows = []
scopes = {"ALL": ALL_ITEMS, "A_CLASS": A_ITEMS, "B_CLASS": B_ITEMS, "C_CLASS": C_ITEMS}
for sysname in ["B1_global_mean", "B2_item_median", "B3_liability_SR", "B4_claim_x_liability"]:
    for sname, items in scopes.items():
        sub = oof_item[oof_item["item_name"].isin(items)]
        m = metrics_with_repeat(sub, "support_rate_label", PRED_COL[sysname])
        bm_rows.append({"system": sysname, "level": "item", "scope": sname, **m})
    m = metrics_with_repeat(oof_case, "sr_true", f"sr_pred_{sysname.split('_')[0].lower()}")
    bm_rows.append({"system": sysname, "level": "case", "scope": "SR_total", **m})
baseline_metrics = pd.DataFrame(bm_rows)
save_csv(baseline_metrics, "baseline_metrics.csv")

# --- case_level_rule_metrics.csv ---
cm_rows = []
case_systems = {
    "Rule_system_full(selected)": ("sr_true", "sr_pred_rule_full"),
    "Rule_system_full(legal_prior)": ("sr_true", "sr_pred_rulelegal_full"),
    "Rule_system_AC_subset(selected)": ("sr_true_ac", "sr_pred_rule_ac"),
    "B1_global_mean": ("sr_true", "sr_pred_b1"),
    "B2_item_median": ("sr_true", "sr_pred_b2"),
    "B3_liability_SR": ("sr_true", "sr_pred_b3"),
    "B4_claim_x_liability": ("sr_true", "sr_pred_b4"),
}
for sname, (tc, pc) in case_systems.items():
    m = metrics_with_repeat(oof_case, tc, pc)
    cm_rows.append({"system": sname, **m})
case_metrics = pd.DataFrame(cm_rows)
save_csv(case_metrics, "case_level_rule_metrics.csv")

log("  案件级指标（主指标 = SR_total MAE）：")
for _, r in case_metrics.iterrows():
    log(f"    {r['system']:<38} MAE={r['MAE']:.4f} ± {r['MAE_std_over_repeats']:.4f}  "
        f"RMSE={r['RMSE']:.4f}  MdAE={r['MdAE']:.4f}  R2={r['R2']:.4f}  "
        f"within5%={r['within_5pct']:.3f}")

# --- 规则分项误差上界 ---
log("  A/C 类规则分项误差（Stage 4 需超越的门槛）：")
for it in A_ITEMS + C_ITEMS:
    lg = rule_metrics[(rule_metrics["system"] == "Rule_legal") &
                      (rule_metrics["item_name"] == it)].iloc[0]
    sl = rule_metrics[(rule_metrics["system"] == "Rule_selected") &
                      (rule_metrics["item_name"] == it)].iloc[0]
    b = rule_metrics[(rule_metrics["system"] == "B2_item_median") &
                     (rule_metrics["item_name"] == it)].iloc[0]
    log(f"    {it:<18} legal={lg['MAE']:.4f}  selected={sl['MAE']:.4f}  "
        f"B2中位数={b['MAE']:.4f}  -> {'标定规则更优' if sl['MAE'] < b['MAE'] - 1e-9 else '与中位数基线持平'}")

# --- 输出预测明细 ---
save_csv(oof_item, "rule_predictions.csv")

# ===========================================================================
# 五、disability_comp 分层表
# ===========================================================================
log("=" * 72)
log("五、disability_comp 伤残等级分层规则表")
log("=" * 72)

sub_all = valid[valid["item_name"] == "disability_comp"].copy()
sub_all["level"] = pd.to_numeric(sub_all["disability_level_clean"], errors="coerce")
grp = sub_all.groupby("level")
drows = []
fold_df = pd.DataFrame(disab_fold_rows)
for lvl, s in grp:
    fm = fold_df[fold_df["level"] == float(lvl)]["median_SR_fold"] if not fold_df.empty else pd.Series(dtype=float)
    drows.append({
        "disability_level": float(lvl),
        "n_total": int(len(s)),
        "mean_SR": round(float(s["support_rate_label"].mean()), 6),
        "median_SR": round(float(s["support_rate_label"].median()), 6),
        "std_SR": round(float(s["support_rate_label"].std(ddof=1)), 6) if len(s) > 1 else np.nan,
        "p_sr_eq_1": round(float((s["support_rate_label"] >= 1 - 1e-4).mean()), 6),
        "median_claim": round(float(s["claim_amount"].median()), 2),
        "median_SR_fold_mean": round(float(fm.mean()), 6) if len(fm) else np.nan,
        "median_SR_fold_std": round(float(fm.std(ddof=1)), 6) if len(fm) > 1 else np.nan,
        "n_folds_used": int(len(fm)),
        "rule_note": "训练折内该等级样本 < 5 例时不建表，回落全局中位数",
    })
# 缺失等级行
miss = sub_all[sub_all["level"].isna()]
if len(miss):
    drows.append({"disability_level": np.nan, "n_total": int(len(miss)),
                  "mean_SR": round(float(miss["support_rate_label"].mean()), 6),
                  "median_SR": round(float(miss["support_rate_label"].median()), 6),
                  "std_SR": np.nan, "p_sr_eq_1": np.nan,
                  "median_claim": round(float(miss["claim_amount"].median()), 2),
                  "median_SR_fold_mean": np.nan, "median_SR_fold_std": np.nan,
                  "n_folds_used": 0, "rule_note": "等级缺失 → 回落全局中位数"})
disab_table = pd.DataFrame(drows).sort_values("disability_level")
save_csv(disab_table, "disability_rule_table.csv")

# ===========================================================================
# 六、C 类 fallback 策略表
# ===========================================================================
log("=" * 72)
log("六、C 类 Rule-informed fallback")
log("=" * 72)
fb_rows = []
for it, cfg in C_CLASS_FALLBACK.items():
    rm = rule_metrics[(rule_metrics["system"] == "Rule_selected") &
                      (rule_metrics["item_name"] == it)]
    lg = rule_metrics[(rule_metrics["system"] == "Rule_legal") &
                      (rule_metrics["item_name"] == it)]
    b2 = rule_metrics[(rule_metrics["system"] == "B2_item_median") &
                      (rule_metrics["item_name"] == it)]
    fb_rows.append({
        "item_name": it,
        "sample_size": int(valid[valid["item_name"] == it].shape[0]),
        "fallback_priority": cfg["priority"],
        "fallback_type": cfg["rule_class"].__name__,
        "formula": cfg["formula"],
        "legal_basis": cfg["legal_basis"],
        "reason": cfg["reason"],
        "params": json.dumps(cfg.get("params", {}), ensure_ascii=False),
        "rule_MAE_selected": round(float(rm["MAE"].iloc[0]), 6) if len(rm) else np.nan,
        "rule_MAE_legal_prior": round(float(lg["MAE"].iloc[0]), 6) if len(lg) else np.nan,
        "baseline_B2_MAE": round(float(b2["MAE"].iloc[0]), 6) if len(b2) else np.nan,
        "prohibited": "禁止直接进入主要模型排名；禁止删除；统计量仅训练折内计算",
        "ablation_stage": 6,
    })
fb = pd.DataFrame(fb_rows)
save_csv(fb, "fallback_strategy.csv")
for _, r in fb.iterrows():
    log(f"    {r['item_name']:<16} {r['fallback_priority']} {r['fallback_type']:<20} "
        f"selected MAE={r['rule_MAE_selected']:.4f}（legal {r['rule_MAE_legal_prior']:.4f}，"
        f"B2 {r['baseline_B2_MAE']:.4f}）")

save_csv(pd.DataFrame(fit_log_rows), "rule_fit_log.csv")

# ===========================================================================
# 十、阶段报告
# ===========================================================================
log("=" * 72)
log("十、生成阶段报告")
log("=" * 72)


def fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


best_baseline = case_metrics[case_metrics["system"].str.startswith("B")].sort_values("MAE").iloc[0]
rule_full = case_metrics.iloc[0]
rule_ac = case_metrics.iloc[1]

lines = [
    "# Stage 3 规则引擎与强基线体系 —— 阶段报告",
    "",
    f"生成时间：{NOW}  ",
    "阶段约束：**未训练任何机器学习模型**；仅使用规则计算与训练折内统计估计。",
    "",
    "## 1. 规则设计",
    "",
    "规则引擎位于 `src/rule_engine/`，统一接口 `predict_sr(case, item) -> predicted_SR`，"
    "所有统计量在 `fit()` 阶段仅接触训练折数据。",
    "",
    "| 项目 | 类别 | 规则类 | 计算逻辑 | 司法依据 |",
    "|---|---|---|---|---|",
    "| medical | A | CapRule(q=0.90) | SR = min(1, R/claim)，R = 训练折 SR=1 样本 claim 的 90 分位 | 凭据据实（票据+病历） |",
    "| hospital_meal | A | CapRule(q=0.90) | 同上 | 固定日标准 × 住院天数 |",
    "| disability_comp | A | StratifiedMedianRule | 按伤残等级取训练折 SR 中位数 | 法定公式（收入 × 年限 × 伤残系数） |",
    "| assistive_device | A | CapRule(q=0.75) | SR = min(1, R/claim) | 普通适用器具标准 + 配制机构意见 |",
    "| appraisal | A | CapRule(q=0.95) | SR = min(1, R/claim) | 凭鉴定费票据据实 |",
    "| death_comp | C-P1 | CapRule(q=0.50) | SR = min(1, R/claim)，R = 训练折 SR=1 样本 claim 中位数 | 人均可支配收入 × 法定年限 |",
    "| funeral | C-P1 | CapRule(q=0.50) | 同上 | 职工月平均工资 × 6 个月 |",
    "| accommodation | C-P2 | QuantilePriorRule(3 档) | 请求金额分位档 × 训练折该档 SR 中位数（clip[0,1]） | 凭据 + 合理性酌定 |",
    "| future_medical | C-P2 | QuantilePriorRule(3 档) | 同上 | 需鉴定结论或待实际发生 |",
    "",
    f"统一接口一致性校验：`predict_sr` 与向量化 `predict_frame` 结果比对 "
    f"**{sum(consistency_checks)}/{len(consistency_checks)}** 完全一致。",
    "",
    "### 1.1 残疾赔偿金伤残等级分层表",
    "",
    "| 伤残等级 | 样本数 | SR 中位数 | SR 均值 | P(SR=1) | 折间中位数均值 | 折间标准差 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for _, r in disab_table.iterrows():
    lv = "缺失" if np.isnan(r["disability_level"]) else f"{r['disability_level']:g}"
    lines.append(f"| {lv} | {int(r['n_total'])} | {fmt(r['median_SR'])} | {fmt(r['mean_SR'])} | "
                 f"{fmt(r['p_sr_eq_1'])} | {fmt(r['median_SR_fold_mean'])} | "
                 f"{fmt(r['median_SR_fold_std'])} |")
lines += [
    "",
    "### 1.2 C 类 fallback 策略",
    "",
    "| 项目 | 优先级 | 类型 | 公式 | 标定规则 MAE | 法律先验规则 MAE | 中位数基线 MAE |",
    "|---|---|---|---|---:|---:|---:|",
]
for _, r in fb.iterrows():
    lines.append(f"| {r['item_name']} | {r['fallback_priority']} | {r['fallback_type']} | "
                 f"{r['formula']} | {fmt(r['rule_MAE_selected'])} | "
                 f"{fmt(r['rule_MAE_legal_prior'])} | {fmt(r['baseline_B2_MAE'])} |")
lines += [
    "",
    "## 2. 基线结果",
    "",
    "四条强基线定义：",
    "",
    "| 基线 | 定义 | 责任比例缺失时 |",
    "|---|---|---|",
    "| B1 Global mean SR | 训练折全部有效样本 SR 均值（所有项目同一常数） | 不受影响 |",
    "| B2 Item median SR | 训练折该分项 SR 中位数 | 不受影响 |",
    "| B3 liability_ratio SR | SR̂ = liability_ratio_clean | 回落到训练折该分项 SR 中位数 |",
    "| B4 claim × liability_ratio | Â = claim × liability_ratio（金额法口径） | 回落到训练折案件级 SR_total 中位数 |",
    "",
    "> 说明：B3 与 B4 在责任比例**可用**时等价（同一案件的责任比例相同）；"
    "差异体现在缺失回退策略上（B3 用分项中位数、B4 用案件级 SR_total 中位数），"
    "由于责任比例 81.8% 缺失，二者实际表现会有明显区别。",
    "",
    "### 2.1 案件级指标（主指标 = SR_total MAE）",
    "",
    "| 系统 | n | MAE | MAE(折间±std) | RMSE | MdAE | R² | ≤5% | ≤10% |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for _, r in case_metrics.iterrows():
    lines.append(
        f"| {r['system']} | {int(r['n'])} | **{fmt(r['MAE'])}** | "
        f"{fmt(r['MAE_mean_over_repeats'])} ± {fmt(r['MAE_std_over_repeats'])} | "
        f"{fmt(r['RMSE'])} | {fmt(r['MdAE'])} | {fmt(r['R2'])} | "
        f"{fmt(r['within_5pct'], 3)} | {fmt(r['within_10pct'], 3)} |")
lines += [
    "",
    "### 2.2 基线分项级指标（按项目类别）",
    "",
    "| 基线 | 口径 | MAE | RMSE | MdAE | R² | ≤5% | ≤10% |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for _, r in baseline_metrics.iterrows():
    lines.append(f"| {r['system']} | {r['scope']} | {fmt(r['MAE'])} | {fmt(r['RMSE'])} | "
                 f"{fmt(r['MdAE'])} | {fmt(r['R2'])} | {fmt(r['within_5pct'], 3)} | "
                 f"{fmt(r['within_10pct'], 3)} |")
lines += [
    "",
    "## 3. 规则误差上界",
    "",
    "| 项目 | 类别 | 法律先验规则 MAE | 训练折标定规则 MAE | B2 中位数 MAE | B1 全局均值 MAE | 标定规则是否优于 B2 |",
    "|---|---|---:|---:|---:|---:|---|",
]
for it in A_ITEMS + C_ITEMS:
    lg = rule_metrics[(rule_metrics["system"] == "Rule_legal") & (rule_metrics["item_name"] == it)]
    sl = rule_metrics[(rule_metrics["system"] == "Rule_selected") &
                      (rule_metrics["item_name"] == it)]
    b2 = rule_metrics[(rule_metrics["system"] == "B2_item_median") &
                      (rule_metrics["item_name"] == it)]
    b1 = rule_metrics[(rule_metrics["system"] == "B1_global_mean") &
                      (rule_metrics["item_name"] == it)]
    if sl.empty:
        continue
    lgr, slr, b2r, b1r = lg.iloc[0], sl.iloc[0], b2.iloc[0], b1.iloc[0]
    win = "是" if slr["MAE"] < b2r["MAE"] - 1e-9 else "否（退化为中位数基线）"
    lines.append(f"| {it} | {ITEM_CLASS[it]} | {fmt(lgr['MAE'])} | {fmt(slr['MAE'])} | "
                 f"{fmt(b2r['MAE'])} | {fmt(b1r['MAE'])} | {win} |")
lines += [
    "",
    f"**案件级误差上界**：完整口径规则系统 SR_total MAE = **{fmt(rule_full['MAE'])}**"
    f"（± {fmt(rule_full['MAE_std_over_repeats'])}）；"
    f"A/C 子集口径 MAE = **{fmt(rule_ac['MAE'])}**。",
    f"**最强基线**：{best_baseline['system']}，MAE = **{fmt(best_baseline['MAE'])}**。",
    "",
    "该 MAE 即为 Stage 4 机器学习模型必须超越的门槛："
    "若模型在案件级 SR_total 上不能显著低于该值，则「数据驱动预测」在本数据集上不成立。",
    "",
    "## 4. 哪些项目仍存在预测空间",
    "",
    "| 项目 | 类别 | 支持率标准差 | 中位数基线 MAE | 预测空间判断 |",
    "|---|---|---:|---:|---|",
]
jdi = pd.read_csv(S2 / "JDI_ranking.csv", encoding="utf-8-sig")
for it in B_ITEMS:
    b2 = rule_metrics[(rule_metrics["system"] == "B2_item_median") &
                      (rule_metrics["item_name"] == it)].iloc[0]
    std = float(jdi.loc[jdi["item_name"] == it, "std_SR"].iloc[0])
    room = "大（裁量空间大，常数基线解释力弱）" if std > 0.33 else "中等"
    lines.append(f"| {it} | B | {std:.4f} | {fmt(b2['MAE'])} | {room} |")
lines += [
    "",
    f"- **B 类 7 项**（{', '.join(B_ITEMS)}）无规则可依，JDI 低、支持率标准差大，"
    "中位数基线 MAE 高，**是机器学习的主要战场**；",
    "- 其中 `mental_solatium`（精神损害抚慰金）支持率标准差 0.381、均值 0.432，"
    "既非全额支持也非全额驳回，是典型的比例裁量项目；",
    "- `lost_income`、`transport` 的 JDI 排名末两位（12.85 / 17.46），预测空间最大；",
    "- A 类项目若规则已优于中位数基线，则模型需以规则为基准继续改进才有意义。",
    "",
    "## 5. 进入 Stage 4 的必要性",
    "",
    "1. **规则无法覆盖 B 类项目**：7 个 ML 类项目无计算公式，只能用常数/中位数兜底，"
    "其分项 MAE 显著高于 A 类（见上表），构成案件级误差的主要来源。",
    f"2. **案件级仍有可观误差**：规则系统（B 类用中位数填充）SR_total MAE = "
    f"{fmt(rule_full['MAE'])}，最强基线 MAE = {fmt(best_baseline['MAE'])}，"
    "距离「可用」的司法预测精度仍有明显差距。",
    "3. **判前特征尚未被利用**：规则与基线几乎只用到了请求金额与责任比例；"
    "伤残等级、请求结构（占比、非零项数）、案件年份与审理程序等判前特征"
    "尚未参与预测，机器学习可系统利用这些非线性与交互信息。",
    "4. **任务形式尚未判定**：任务 A（金额预测）与任务 B（支持率预测）孰优"
    "必须在同协议下用实验裁决，这是 Stage 4 的核心设计。",
    "5. **必须验证混合框架的增益**：只有当 ML 在 B 类项目上显著优于中位数基线，"
    "且叠加到案件级后使 SR_total MAE 低于纯规则/纯基线系统时，"
    "「规则 + 数据驱动」混合框架的立论才成立。",
    "",
    "## 6. 本阶段产物",
    "",
    "| 文件 | 说明 | 规模 |",
    "|---|---|---|",
    f"| `results/stage3/rule_predictions.csv` | 分项级 OOF 预测（含规则与 4 条基线） | {len(oof_item)} 行 |",
    f"| `results/stage3/baseline_metrics.csv` | 基线分项级 + 案件级指标 | {len(baseline_metrics)} 行 |",
    f"| `results/stage3/rule_metrics.csv` | 系统 × 项目 指标 | {len(rule_metrics)} 行 |",
    f"| `results/stage3/case_level_rule_metrics.csv` | 案件级各系统指标 | {len(case_metrics)} 行 |",
    f"| `results/stage3/disability_rule_table.csv` | 伤残等级分层规则表 | {len(disab_table)} 行 |",
    f"| `results/stage3/fallback_strategy.csv` | C 类 fallback 策略 | {len(fb)} 行 |",
    f"| `results/stage3/rule_fit_log.csv` | 每折规则拟合日志（可复现） | {len(fit_log_rows)} 行 |",
    "| `src/rule_engine/*.py` | 规则引擎包 | 5 个模块 |",
    "| `src/rule_engine/definitions/*.md` | A 类 5 项规则定义 | 5 份 |",
    "",
    "> 本阶段未训练任何机器学习模型，已完成 Stage 3 全部任务并停止。",
]
(REPORTS / "stage3_summary.md").write_text("\n".join(lines), encoding="utf-8")
log(f"[saved] results/reports/stage3_summary.md")

log("=" * 72)
log("Stage 3 完成（未训练任何模型）。")
log("=" * 72)
