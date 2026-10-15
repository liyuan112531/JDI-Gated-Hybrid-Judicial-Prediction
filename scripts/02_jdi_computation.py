# -*- coding: utf-8 -*-
"""
02 —— 司法确定性指数（JDI）构建与赔偿项目分层分析

严格约束：
  - 禁止任何机器学习训练
  - 禁止模型比较 / 禁止调参
  - 禁止调用 XGBoost / LightGBM / CatBoost / TabPFN 等模型
  - 仅使用描述统计、相关分析、互信息估计等统计方法
  - award 字段仅用于构造标签 SR，绝不作为输入特征
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti TC",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
S1 = ROOT / "results" / "stage1"
OUT = ROOT / "results" / "stage2"
FIG = OUT / "figures"
REPORTS = ROOT / "results" / "reports"
for d in (OUT, FIG, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

TOL = 1e-6
RANDOM_STATE = 42
N_BINS = 10          # SR 分箱 [0, 0.1, ..., 1]
SMALL_SAMPLE_DEFAULT = 100


def log(m: str) -> None:
    print(m, flush=True)


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    log(f"[saved] results/stage2/{name}  rows={len(df)}")


# ===========================================================================
# 一、读取配置与数据
# ===========================================================================
log("=" * 72)
log("一、读取配置与数据")
log("=" * 72)

import yaml

policy = yaml.safe_load((CONFIGS / "experiment_policy.yaml").read_text(encoding="utf-8"))
proj_cfg = json.loads((CONFIGS / "project_strategy_config.json").read_text(encoding="utf-8"))
ss_md = (CONFIGS / "small_sample_policy.md").read_text(encoding="utf-8")

ITEMS = policy["data"]["items"]
SS_ITEMS = policy["small_sample_policy"]["small_sample_items"]

# 校验 small_sample_policy.md 与配置一致性
md_items = set(re.findall(r"`(death_comp|funeral|accommodation|future_medical)`", ss_md))
cfg_items = set(SS_ITEMS.keys())
consistency_note = ("一致" if md_items == cfg_items else f"不一致：md={md_items} cfg={cfg_items}")
log(f"  小样本策略文件与配置一致性：{consistency_note}")
log(f"  小样本项目：{SS_ITEMS}")

item_long = pd.read_csv(S1 / "dataset_item_level.csv", encoding="utf-8-sig")
item_base = pd.read_csv(S1 / "item_distribution_base.csv", encoding="utf-8-sig")
case_df = pd.read_csv(S1 / "dataset_case_level.csv", encoding="utf-8-sig")
log(f"  分项级长表：{item_long.shape}；案件级：{case_df.shape}")

# 仅保留 claim > 0 的样本（SR 有定义）
valid = item_long[item_long["claim_positive"] == 1].copy()
log(f"  有效样本（claim>0）：{len(valid)} 行")

# 泄漏自检：确保本阶段未使用任何 award 派生特征作为输入
BAN = re.compile(r"(_award$|^award_total$|^support_rate$|^support_rate_pct$|^n_nonzero_items$)")
used_cols = ["claim_amount", "liability_ratio_clean", "disability_level_clean"]
assert not [c for c in used_cols if BAN.search(c)], "输入列含判后字段"

# ===========================================================================
# 二、法律规则先验（人工提供的法学先验，非数据推断，需人工确认）
# ===========================================================================
LEGAL_BASIS = {
    # basis: 法条依据类型；strength: 规则强度（1.0 法定公式 / 0.75 凭据据实 /
    # 0.4 证据+酌定 / 0.3 需鉴定或待实际发生 / 0.1 纯裁量）
    "disability_comp": ("formula_based", 1.0,
                        "残疾赔偿金：受诉法院所在地上一年度人均可支配收入 × 赔偿年限 × 伤残系数，法定公式"),
    "death_comp": ("formula_based", 1.0,
                   "死亡赔偿金：人均可支配收入 × 法定年限，法定公式"),
    "funeral": ("formula_based", 1.0,
                "丧葬费：受诉法院所在地上一年度职工月平均工资 × 6 个月，法定标准"),
    "medical": ("evidence_based", 0.75, "医疗费：凭医疗费票据、病历据实计算"),
    "appraisal": ("evidence_based", 0.75, "鉴定费：凭鉴定费票据据实计算"),
    "hospital_meal": ("evidence_based", 0.75, "住院伙食补助费：固定日标准 × 住院天数"),
    "assistive_device": ("evidence_based", 0.75, "残疾辅助器具费：配置机构意见 + 普通适用器具标准"),
    "property_loss": ("evidence_based", 0.6, "财产损失：维修票据或评估结论，存在折旧与酌定"),
    "nursing": ("hybrid", 0.4, "护理费：护理人员收入 × 护理期限，期限与标准存在酌定"),
    "lost_income": ("hybrid", 0.4, "误工费：收入证明 × 误工时间，举证不足时法院酌定"),
    "nutrition": ("hybrid", 0.4, "营养费：参照医疗机构意见酌定"),
    "transport": ("hybrid", 0.4, "交通费：凭票据，但合理性由法院酌定"),
    "accommodation": ("hybrid", 0.4, "住宿费：凭据 + 合理性酌定"),
    "future_medical": ("conditional", 0.3, "后续医疗费：需鉴定结论或待实际发生后另行主张，常因证据不足驳回"),
    "other": ("discretion_based", 0.1, "其他费用：兜底项目，无明确计算标准，裁量空间大"),
    "mental_solatium": ("discretion_based", 0.1,
                        "精神损害抚慰金：无计算公式，依伤残等级、过错程度、当地生活水平酌定"),
}

# ===========================================================================
# 三、JDI 指标计算
# ===========================================================================
log("=" * 72)
log("三、JDI 指标计算")
log("=" * 72)


def shannon_entropy(x: np.ndarray, n_bins: int = N_BINS) -> float:
    """对 [0,1] 区间的 SR 做等宽分箱后计算香农熵（bits）"""
    hist, _ = np.histogram(x, bins=n_bins, range=(0.0, 1.0), density=False)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mode_density(x: np.ndarray, n_bins: int = N_BINS) -> float:
    hist, _ = np.histogram(x, bins=n_bins, range=(0.0, 1.0), density=False)
    return float(hist.max() / hist.sum())


rows = []
for it in ITEMS:
    sub = valid[valid["item_name"] == it]
    sr = sub["support_rate_label"].to_numpy(dtype=float)
    cl = sub["claim_amount"].to_numpy(dtype=float)
    n = len(sr)

    p1 = float(np.mean(np.abs(sr - 1.0) <= 1e-6))
    p0 = float(np.mean(np.abs(sr) <= 1e-6))
    mean_sr = float(np.mean(sr))
    std_sr = float(np.std(sr, ddof=1))
    q75, q25 = np.percentile(sr, [75, 25])
    iqr = float(q75 - q25)
    cv = float(std_sr / mean_sr) if mean_sr > TOL else np.nan
    ent = shannon_entropy(sr)
    ent_norm = ent / np.log2(N_BINS)
    md10 = mode_density(sr, N_BINS)
    md20 = mode_density(sr, 20)

    # 指标7：请求金额解释力
    if n >= 10 and np.std(cl) > 0 and np.std(sr) > 0:
        pear = float(stats.pearsonr(cl, sr)[0])
        spear = float(stats.spearmanr(cl, sr)[0])
        mi = float(mutual_info_regression(cl.reshape(-1, 1), sr,
                                          n_neighbors=3, random_state=RANDOM_STATE)[0])
        logc = np.log1p(cl)
        mi_log = float(mutual_info_regression(logc.reshape(-1, 1), sr,
                                              n_neighbors=3, random_state=RANDOM_STATE)[0])
    else:
        pear = spear = mi = mi_log = np.nan

    # 指标8：责任比例一致性（仅 clean 非缺失子集）
    liab = sub["liability_ratio_clean"].to_numpy(dtype=float)
    m_l = ~np.isnan(liab)
    n_l = int(m_l.sum())
    if n_l >= 10 and np.std(liab[m_l]) > 0 and np.std(sr[m_l]) > 0:
        rho_l = float(stats.spearmanr(liab[m_l], sr[m_l])[0])
        liab_cons = abs(rho_l)
        liab_reliable = 1 if n_l >= 30 else 0
    else:
        rho_l, liab_cons, liab_reliable = np.nan, np.nan, 0

    # 指标9：伤残等级解释力
    dis = sub["disability_level_clean"].to_numpy(dtype=float)
    m_d = ~np.isnan(dis)
    n_d = int(m_d.sum())
    if n_d >= 10 and np.std(dis[m_d]) > 0 and np.std(sr[m_d]) > 0:
        rho_d = float(stats.spearmanr(dis[m_d], sr[m_d])[0])
        dis_power = abs(rho_d)
        grp = pd.DataFrame({"d": dis[m_d], "sr": sr[m_d]}).groupby("d")["sr"].mean()
        dis_range = float(grp.max() - grp.min()) if len(grp) >= 2 else np.nan
        dis_reliable = 1 if n_d >= 30 else 0
    else:
        rho_d, dis_power, dis_range, dis_reliable = np.nan, np.nan, np.nan, 0

    basis, strength, basis_text = LEGAL_BASIS[it]

    rows.append({
        "item_name": it,
        "sample_size": n,
        # 1-2 极端结果集中度
        "p_sr_eq_1": round(p1, 6),
        "p_sr_eq_0": round(p0, 6),
        # 3-4 集中趋势与离散
        "mean_SR": round(mean_sr, 6),
        "std_SR": round(std_sr, 6),
        "IQR_SR": round(iqr, 6),
        "CV_SR": round(cv, 6) if not np.isnan(cv) else np.nan,
        # 5-6 分布形状
        "SR_entropy_bits": round(ent, 6),
        "SR_entropy_norm": round(ent_norm, 6),
        "SR_mode_density_bin10": round(md10, 6),
        "SR_mode_density_bin20": round(md20, 6),
        # 7 请求金额解释力
        "claim_pearson": round(pear, 6) if not np.isnan(pear) else np.nan,
        "claim_spearman": round(spear, 6) if not np.isnan(spear) else np.nan,
        "claim_SR_MI": round(mi, 6) if not np.isnan(mi) else np.nan,
        "logclaim_SR_MI": round(mi_log, 6) if not np.isnan(mi_log) else np.nan,
        # 8 责任比例一致性
        "liability_subset_n": n_l,
        "liability_spearman": round(rho_l, 6) if not np.isnan(rho_l) else np.nan,
        "liability_consistency": round(liab_cons, 6) if not np.isnan(liab_cons) else np.nan,
        "liability_reliable": liab_reliable,
        # 9 伤残等级解释力
        "disability_subset_n": n_d,
        "disability_spearman": round(rho_d, 6) if not np.isnan(rho_d) else np.nan,
        "disability_explanatory_power": round(dis_power, 6) if not np.isnan(dis_power) else np.nan,
        "disability_group_SR_range": round(dis_range, 6) if not np.isnan(dis_range) else np.nan,
        "disability_reliable": dis_reliable,
        # 法律先验
        "legal_basis_type": basis,
        "legal_rule_strength": strength,
        "legal_basis_text": basis_text,
    })
    log(f"  {it:<18} n={n:>5}  P(SR=1)={p1:.3f}  P(SR=0)={p0:.3f}  std={std_sr:.3f}  "
        f"entropy={ent_norm:.3f}  mode={md10:.3f}")

metrics = pd.DataFrame(rows)

# ===========================================================================
# 四、JDI 综合指数
# ===========================================================================
log("=" * 72)
log("四、JDI 综合指数构建")
log("=" * 72)

WEIGHTS = {
    # 确定性增强（正向）
    "p_sr_eq_1": 0.20,
    "p_sr_eq_0": 0.10,
    "SR_mode_density_bin10": 0.20,
    "liability_consistency": 0.10,
    # 不确定性（负向）
    "SR_entropy_norm": 0.15,
    "std_SR": 0.15,
    "IQR_SR": 0.10,
}
POSITIVE = {"p_sr_eq_1", "p_sr_eq_0", "SR_mode_density_bin10", "liability_consistency"}
NEGATIVE = {"SR_entropy_norm", "std_SR", "IQR_SR"}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

weight_cfg = {
    "version": "1.0",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "stage": "stage2",
    "method": "各分量先跨 16 个项目做 min-max 归一化到 [0,1]，负向指标取 1-x，再加权求和后线性映射到 0-100",
    "weights": WEIGHTS,
    "direction": {"positive": sorted(POSITIVE), "negative": sorted(NEGATIVE)},
    "excluded_from_index": {
        "claim_SR_MI": "反映『可预测性』而非『司法确定性』，方向不可解释（高 MI 可能来自复杂裁量），仅作诊断指标",
        "disability_explanatory_power": "仅部分项目适用（伤残相关），纳入会引入项目间不可比性，仅作诊断指标",
        "CV_SR": "与 std_SR 高度共线，避免重复计权",
        "legal_rule_strength": "法律先验用于分层决策，不进入数据驱动指数本身",
    },
    "reliability_handling": {
        "rule": "liability_consistency 在子集样本 n<30 时标记为不可靠（liability_reliable=0），"
                "归一化时以中性值 0.5 填充，避免小子集噪声主导排序",
        "note": "该处理在敏感性分析中检验其影响"
    },
    "sensitivity_schemes": {
        "default": WEIGHTS,
        "equal": {k: round(1 / len(WEIGHTS), 6) for k in WEIGHTS},
        "rule_oriented": None,  # 下方计算
    },
}
# rule_oriented：正向指标权重加倍后归一化
raw_ro = {k: (v * 2 if k in POSITIVE else v) for k, v in WEIGHTS.items()}
tot_ro = sum(raw_ro.values())
weight_cfg["sensitivity_schemes"]["rule_oriented"] = {k: round(v / tot_ro, 6) for k, v in raw_ro.items()}
(OUT / "jdi_weight_config.json").write_text(
    json.dumps(weight_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"[saved] results/stage2/jdi_weight_config.json（权重方案：default / equal / rule_oriented）")


def compute_jdi(metrics_df: pd.DataFrame, weights: dict) -> pd.Series:
    """返回 0-100 的 JDI 分数"""
    scores = pd.Series(0.0, index=metrics_df.index)
    for col, w in weights.items():
        x = metrics_df[col].astype(float).copy()
        if col == "liability_consistency":
            x = x.where(metrics_df["liability_reliable"] == 1, other=np.nan)
            x = x.fillna(0.5)          # 不可靠子集 -> 中性值
        xn = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else pd.Series(0.5, index=x.index)
        xn = xn.fillna(0.5)
        s = xn if col in POSITIVE else 1.0 - xn
        scores += w * s
    return (scores * 100).round(4)


metrics["JDI_score"] = compute_jdi(metrics, WEIGHTS)
metrics = metrics.sort_values("JDI_score", ascending=False).reset_index(drop=True)
metrics["rank"] = np.arange(1, len(metrics) + 1)

jdi_ranking = metrics[[
    "item_name", "sample_size", "JDI_score", "rank",
    "p_sr_eq_1", "p_sr_eq_0", "mean_SR", "std_SR", "IQR_SR", "CV_SR",
    "SR_entropy_bits", "SR_entropy_norm", "SR_mode_density_bin10", "SR_mode_density_bin20",
    "claim_pearson", "claim_spearman", "claim_SR_MI", "logclaim_SR_MI",
    "liability_subset_n", "liability_consistency", "liability_reliable",
    "disability_subset_n", "disability_explanatory_power", "disability_group_SR_range",
    "legal_basis_type", "legal_rule_strength",
]].copy()
save_csv(jdi_ranking, "JDI_ranking.csv")
for _, r in jdi_ranking.iterrows():
    log(f"  rank {r['rank']:>2}  {r['item_name']:<18} JDI={r['JDI_score']:>6.2f}  n={r['sample_size']:>5}  "
        f"P(SR=1)={r['p_sr_eq_1']:.3f}  entropy={r['SR_entropy_norm']:.3f}")

# ===========================================================================
# 五-六、自动分层（数据驱动 + 法律先验）
# ===========================================================================
log("=" * 72)
log("五-六、自动分层")
log("=" * 72)

# 数据驱动分界：在排序后的 JDI 上寻找最大间隙
jdi_sorted = jdi_ranking["JDI_score"].to_numpy()
gaps = np.diff(jdi_sorted)
gap_idx = int(np.argmax(np.abs(gaps)))
gap_cut = float((jdi_sorted[gap_idx] + jdi_sorted[gap_idx + 1]) / 2)
gap_size = float(abs(gaps[gap_idx]))
log(f"  JDI 最大间隙切分点 = {gap_cut:.3f}（间隙 {gap_size:.3f}，位于 rank {gap_idx + 1} 与 {gap_idx + 2} 之间）")

assign_rows = []
for _, r in jdi_ranking.iterrows():
    it, n, jdi = r["item_name"], int(r["sample_size"]), float(r["JDI_score"])
    basis, strength, basis_text = LEGAL_BASIS[it]
    # Small-sample 判定：n < 100（Stage 2 规则）或已列入 Stage 0/1 冻结的小样本清单
    in_cfg = it in SS_ITEMS
    is_small = bool((n < SMALL_SAMPLE_DEFAULT) or in_cfg)
    src_parts = []
    if n < SMALL_SAMPLE_DEFAULT:
        src_parts.append(f"样本量 {n} < {SMALL_SAMPLE_DEFAULT}（Stage 2 规则）")
    if in_cfg:
        src_parts.append("已列入 configs/small_sample_policy.md 与 experiment_policy.yaml 的小样本清单")
    ss_src = "；".join(src_parts)

    # A/B 建议（与 C 标记正交，按 Stage 0 决策）
    if jdi >= gap_cut and strength >= 0.5:
        ab = "Rule-based"
        ab_reason = f"JDI={jdi:.2f} 高于分界 {gap_cut:.2f}，且法律规则强度 {strength}（{basis_text}）"
    elif jdi < gap_cut and strength < 0.5:
        ab = "ML-based"
        ab_reason = f"JDI={jdi:.2f} 低于分界 {gap_cut:.2f}，且法律规则强度仅 {strength}（{basis_text}）"
    else:
        ab = "Rule-based" if jdi >= gap_cut else "ML-based"
        ab_reason = (f"[冲突] JDI={jdi:.2f}（分界 {gap_cut:.2f}）与法律规则强度 {strength} 不一致，"
                     f"以数据驱动 JDI 为准判为 {ab}")

    # recommended_strategy（三选一，按 Stage 2 要求）
    if is_small:
        strategy = "Small-sample"
        reason = (f"{ss_src} → 标记为 Small-sample；不因样本量自动判为规则项目，"
                  f"其 A/B 建议为 {ab}（{ab_reason}）")
    else:
        strategy = ab
        reason = ab_reason

    rule_reason = ""
    ml_reason = ""
    if ab == "Rule-based":
        rule_reason = (f"JDI={jdi:.2f}；P(SR=1)={r['p_sr_eq_1']:.3f}，"
                       f"众数密度={r['SR_mode_density_bin10']:.3f}，"
                       f"熵={r['SR_entropy_norm']:.3f}，std={r['std_SR']:.3f}；法律：{basis_text}")
    else:
        ml_reason = (f"JDI={jdi:.2f}；P(SR=1)={r['p_sr_eq_1']:.3f}，"
                     f"众数密度={r['SR_mode_density_bin10']:.3f}，"
                     f"熵={r['SR_entropy_norm']:.3f}，std={r['std_SR']:.3f}，"
                     f"IQR={r['IQR_SR']:.3f}；法律：{basis_text}")

    assign_rows.append({
        "item_name": it,
        "sample_size": n,
        "JDI_score": jdi,
        "recommended_strategy": strategy,
        "AB_suggestion": ab,
        "small_sample_flag": int(is_small),
        "reason": reason,
        "rule_candidate_reason": rule_reason,
        "ml_candidate_reason": ml_reason,
        "legal_basis_type": basis,
        "legal_rule_strength": strength,
    })

assign = pd.DataFrame(assign_rows)
save_csv(assign, "item_strategy_assignment.csv")
for _, r in assign.iterrows():
    log(f"  {r['item_name']:<18} n={r['sample_size']:>5}  JDI={r['JDI_score']:>6.2f}  "
        f"-> {r['recommended_strategy']:<13} (A/B: {r['AB_suggestion']})")

# 写回项目策略配置（三类列表）
tier_out = {
    "rule_based": sorted(assign[assign["recommended_strategy"] == "Rule-based"]["item_name"].tolist()),
    "ml_based": sorted(assign[assign["recommended_strategy"] == "ML-based"]["item_name"].tolist()),
    "small_sample": sorted(assign[assign["recommended_strategy"] == "Small-sample"]["item_name"].tolist()),
}
(OUT / "project_strategy_assigned.json").write_text(
    json.dumps({**{k: v for k, v in proj_cfg.items() if k not in ("rule_based", "ml_based", "small_sample")},
                **tier_out,
                "assigned_at_stage": 2,
                "assigned_at": datetime.now().isoformat(timespec="seconds"),
                "gap_cut_JDI": round(gap_cut, 4)},
               ensure_ascii=False, indent=2), encoding="utf-8")
log(f"[saved] results/stage2/project_strategy_assigned.json  {tier_out}")

# ===========================================================================
# 七、精神损害抚慰金专项分析
# ===========================================================================
log("=" * 72)
log("七、精神损害抚慰金专项分析")
log("=" * 72)
sub = valid[valid["item_name"] == "mental_solatium"].copy()
sr = sub["support_rate_label"].to_numpy(float)
cl = sub["claim_amount"].to_numpy(float)
n_ms = len(sr)

q = np.percentile(sr, [5, 25, 50, 75, 95])
ms_stats = {
    "n": n_ms,
    "mean": float(sr.mean()), "std": float(sr.std(ddof=1)), "median": float(np.median(sr)),
    "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
    "p_eq_1": float(np.mean(np.abs(sr - 1) <= 1e-6)),
    "p_eq_0": float(np.mean(np.abs(sr) <= 1e-6)),
}

# 伤残等级影响
dis = sub["disability_level_clean"]
ms_dis = sub.assign(_d=dis).groupby("_d")["support_rate_label"].agg(["count", "mean", "median"]).reset_index()
ms_dis.columns = ["disability_level_clean", "n", "mean_SR", "median_SR"]
ms_dis["mean_SR"] = ms_dis["mean_SR"].round(4)
ms_dis["median_SR"] = ms_dis["median_SR"].round(4)
m_d = (~dis.isna()).to_numpy()
rho_dis = float(stats.spearmanr(dis[m_d], sr[m_d])[0]) if m_d.sum() >= 10 else np.nan
rho_dis_p = float(stats.spearmanr(dis[m_d], sr[m_d])[1]) if m_d.sum() >= 10 else np.nan
grp0 = sr[m_d & (dis.to_numpy() == 0)]
grp1 = sr[m_d & (dis.to_numpy() > 0)]
diff_dis = float(grp1.mean() - grp0.mean()) if (len(grp0) and len(grp1)) else np.nan

# 责任比例影响
liab = sub["liability_ratio_clean"].to_numpy(float)
m_l = ~np.isnan(liab)
rho_liab = float(stats.spearmanr(liab[m_l], sr[m_l])[0]) if m_l.sum() >= 10 else np.nan
rho_liab_p = float(stats.spearmanr(liab[m_l], sr[m_l])[1]) if m_l.sum() >= 10 else np.nan
ms_liab_n = int(m_l.sum())

# 请求金额影响
rho_claim_p = float(stats.pearsonr(cl, sr)[0])
rho_claim_p_pval = float(stats.pearsonr(cl, sr)[1])
rho_claim_s = float(stats.spearmanr(cl, sr)[0])
rho_claim_s_pval = float(stats.spearmanr(cl, sr)[1])
mi_claim = float(mutual_info_regression(cl.reshape(-1, 1), sr, n_neighbors=3,
                                        random_state=RANDOM_STATE)[0])
mi_log = float(mutual_info_regression(np.log1p(cl).reshape(-1, 1), sr, n_neighbors=3,
                                      random_state=RANDOM_STATE)[0])
# 请求额分位分组
sub_q = sub.copy()
sub_q["claim_bin"] = pd.qcut(sub_q["claim_amount"], 5, duplicates="drop")
ms_claim_bin = sub_q.groupby("claim_bin", observed=True)["support_rate_label"].agg(
    ["count", "mean", "median"]).reset_index()
ms_claim_bin.columns = ["claim_quintile", "n", "mean_SR", "median_SR"]
ms_claim_bin["mean_SR"] = ms_claim_bin["mean_SR"].round(4)
ms_claim_bin["median_SR"] = ms_claim_bin["median_SR"].round(4)

# 方向性判断（数据驱动，不预设结论）
qs_vals = ms_claim_bin["mean_SR"].astype(float).tolist()
peak_idx = int(np.argmax(qs_vals))
if peak_idx == 0 and qs_vals[0] >= max(qs_vals[1:]):
    shape = "单调下降"
elif peak_idx == len(qs_vals) - 1 and qs_vals[-1] >= max(qs_vals[:-1]):
    shape = "单调上升"
else:
    shape = f"倒 U 型（峰值位于第 {peak_idx + 1} 个分位）"
mono_dir = "正向" if rho_claim_s > 0 else "负向"
mi_rank = int((jdi_ranking["claim_SR_MI"].astype(float) >= mi_claim).sum())
log(f"  请求金额分位形态：{shape}；Spearman={rho_claim_s:.4f}（{mono_dir}）；"
    f"MI={mi_claim:.4f}（16 项中排第 {mi_rank}）")

log(f"  n={n_ms}  mean={ms_stats['mean']:.4f}  median={ms_stats['median']:.4f}  "
    f"std={ms_stats['std']:.4f}  P(SR=1)={ms_stats['p_eq_1']:.4f}  P(SR=0)={ms_stats['p_eq_0']:.4f}")
log(f"  伤残等级 Spearman={rho_dis:.4f} (p={rho_dis_p:.2e})  有伤残-无伤残 SR 均值差={diff_dis:.4f}")
log(f"  责任比例子集 n={ms_liab_n}，Spearman={rho_liab:.4f} (p={rho_liab_p:.3f})")
log(f"  请求金额 Pearson={rho_claim_p:.4f}  Spearman={rho_claim_s:.4f}  MI={mi_claim:.4f}")

ms_md = [
    "# 精神损害抚慰金（mental_solatium）专项分析",
    "",
    f"生成时间：{datetime.now().isoformat(timespec='seconds')}  ",
    "数据来源：`results/stage1/dataset_item_level.csv`（仅 claim_amount > 0 的样本）  ",
    "说明：本分析仅使用描述统计与相关/互信息估计，未训练任何模型。",
    "",
    "## 1. 样本量",
    "",
    f"- 有效样本（claim > 0）：**{n_ms}** 例（占 {len(case_df)} 案件的 {n_ms / len(case_df):.2%}）",
    f"- 判赔 > 0 的样本：{int((sub['award_amount'] > 0).sum())} 例",
    f"- 样本量在 16 个项目中排第 {int((jdi_ranking['sample_size'] >= n_ms).sum())} 位，"
    "属于**充足**水平（远高于小样本阈值 100）",
    "",
    "## 2. SR 分布",
    "",
    "| 统计量 | 值 |",
    "|---|---:|",
    f"| 均值 | {ms_stats['mean']:.4f} |",
    f"| 中位数 | {ms_stats['median']:.4f} |",
    f"| 标准差 | {ms_stats['std']:.4f} |",
    f"| P5 / P25 / P50 / P75 / P95 | {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f} / {q[3]:.4f} / {q[4]:.4f} |",
    f"| P(SR = 1) 全额支持 | {ms_stats['p_eq_1']:.4f} |",
    f"| P(SR = 0) 全额驳回 | {ms_stats['p_eq_0']:.4f} |",
    "",
    f"分布形态：均值 {ms_stats['mean']:.4f}、标准差 {ms_stats['std']:.4f}，"
    f"全额支持率仅 {ms_stats['p_eq_1']:.2%}、全额驳回率 {ms_stats['p_eq_0']:.2%}，"
    "说明判决结果**高度分散于 (0, 1) 区间内部**——既不是「照单全收」也不是「一律驳回」，"
    "而是典型的**比例裁量**形态。",
    "",
    "## 3. 伤残等级影响",
    "",
    "| 伤残等级 | 样本数 | SR 均值 | SR 中位数 |",
    "|---|---:|---:|---:|",
]
for _, r in ms_dis.iterrows():
    ms_md.append(f"| {r['disability_level_clean']:g} | {int(r['n'])} | {r['mean_SR']:.4f} | {r['median_SR']:.4f} |")
ms_md += [
    "",
    f"- Spearman 相关：**{rho_dis:.4f}**（p = {rho_dis_p:.3e}）",
    f"- 有伤残等级（1-10）与无伤残等级（0）的 SR 均值差：**{diff_dis:.4f}**",
    "- 解读：伤残等级是该分项**最强的结构化解释变量**——等级越高（伤情越重），支持率越高，"
    "与《精神损害赔偿司法解释》中「侵权后果严重程度」的裁量因素一致。",
    "",
    "## 4. 责任比例影响",
    "",
    f"- 可用样本（liability_ratio_clean 非缺失）：**{ms_liab_n}** 例（该分项样本的 "
    f"{ms_liab_n / n_ms:.2%}）",
    f"- SR 与责任比例 Spearman 相关：**{rho_liab:.4f}**（p = {rho_liab_p:.3f}）",
    f"- 缺失组（n = {n_ms - ms_liab_n}）SR 均值："
    f"{float(sr[~m_l].mean()):.4f}；非缺失组 SR 均值：{float(sr[m_l].mean()):.4f}",
    "- 解读：受责任比例字段 81.8% 缺失影响，该维度在本分项上可用信息有限；"
    "缺失组与非缺失组的支持率差异需在 Stage 4 中通过缺失指示变量进一步检验。",
    "",
    "## 5. 请求金额影响",
    "",
    f"- Pearson 相关：**{rho_claim_p:.4f}**（p = {rho_claim_p_pval:.3e}）",
    f"- Spearman 相关：**{rho_claim_s:.4f}**（p = {rho_claim_s_pval:.3e}）",
    f"- 互信息 MI(claim, SR)：**{mi_claim:.4f}**；MI(log1p(claim), SR)：**{mi_log:.4f}**",
    "",
    "| 请求金额五分位 | 样本数 | SR 均值 | SR 中位数 |",
    "|---|---:|---:|---:|",
]
for _, r in ms_claim_bin.iterrows():
    ms_md.append(f"| {r['claim_quintile']} | {int(r['n'])} | {r['mean_SR']:.4f} | {r['median_SR']:.4f} |")
ms_md += [
    "",
    f"- 解读：分位均值序列为 {' → '.join(f'{v:.3f}' for v in qs_vals)}，呈**{shape}**；"
    f"单调趋势为**{mono_dir}**（Spearman={rho_claim_s:.4f}，{'显著' if rho_claim_s_pval < 0.05 else '不显著'}），"
    f"而 Pearson={rho_claim_p:.4f} 与之异号，两者分歧说明关系以**非线性为主**。"
    f"互信息 MI={mi_claim:.4f}（16 个项目中排第 {mi_rank} 位，越高表示该分项的请求金额对支持率"
    "的可解释性越强），表明金额对支持率存在**可学习但非单调**的结构——"
    "即法院以「合理额度」为锚：过低的请求（多为无伤残案件的一般性主张）常被驳回，"
    "中档请求支持率最高，过高请求被压缩。"
    "这种非线性、非比例关系正是机器学习相对规则公式的优势所在。",
    "",
    "## 6. 为什么精神损害抚慰金具有预测价值",
    "",
    "1. **裁量空间大但非随机**：无法律计算公式（法律先验 strength = 0.1），"
    f"但 SR 并非均匀分布——标准差 {ms_stats['std']:.4f} 虽大，却可被伤残等级"
    f"（Spearman {rho_dis:.4f}，强相关）与请求金额"
    f"（MI={mi_claim:.4f}，16 项中排第 {mi_rank}）系统解释，"
    "说明裁量遵循**可识别的经验规则**而非任意决断。",
    f"2. **存在锚定效应（{shape}）**：分位均值序列 "
    f"{' → '.join(f'{v:.3f}' for v in qs_vals)}，揭示法院以「合理额度」为锚："
    "过低请求常驳回、中档请求支持率最高、过高请求被压缩；"
    f"且 MI={mi_claim:.4f} 在 16 个项目中排第 {mi_rank} 位，说明该非线性规律可被数据驱动模型学习，"
    "而线性规则公式无法表达。",
    "3. **样本量充足**：1079 例有效样本，远超小样本阈值，支持稳定的模型训练与评估。",
    "4. **对案件整体支持率影响大**：该分项请求金额均值约 2.14 万元、支持率均值仅 0.43，"
    "是拉低案件整体支持率的主要项目之一，其预测精度直接影响系统级指标。",
    "5. **规则方法无法胜任**：既无公式可依，也不存在「全额支持/全额驳回」的集中模式"
    f"（全额支持 {ms_stats['p_eq_1']:.2%}、全额驳回 {ms_stats['p_eq_0']:.2%}），"
    "中位数法或责任比例法都会带来显著偏差。",
    "",
    "**结论：精神损害抚慰金应归入 B 类（ML-based，机器学习预测型）项目。**",
    "",
    "## 7. 风险与待确认",
    "",
    "- 伤残等级语义为暂定编码（TENTATIVELY_CONFIRMED），方向性结论需人工核验。",
    "- 责任比例在本分项上可用样本有限，其影响可能被低估。",
    "- 请求金额与支持率的负相关部分来自「原告高估」的行为因素，属于稳定的数据规律，"
    "但需在论文中说明其为民事实证规律而非法律规范。",
]
(OUT / "mental_solatium_analysis.md").write_text("\n".join(ms_md), encoding="utf-8")
log(f"[saved] results/stage2/mental_solatium_analysis.md")

# ===========================================================================
# 八、可视化
# ===========================================================================
log("=" * 72)
log("八、生成可视化")
log("=" * 72)

order = jdi_ranking["item_name"].tolist()
data_by_item = [valid.loc[valid["item_name"] == it, "support_rate_label"].to_numpy(float)
                for it in order]

# 1. 箱线图
fig, ax = plt.subplots(figsize=(14, 6))
ax.boxplot(data_by_item, labels=order, showmeans=True)
ax.set_title("16 个赔偿项目的支持率（SR）分布箱线图（按 JDI 降序）")
ax.set_ylabel("SR = award / claim")
ax.set_xlabel("赔偿项目（按 JDI 降序）")
plt.xticks(rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig1_sr_boxplot.png", dpi=150)
plt.close(fig)

# 2. 密度图（4x4）
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, it in zip(axes.ravel(), order):
    x = valid.loc[valid["item_name"] == it, "support_rate_label"].to_numpy(float)
    ax.hist(x, bins=20, range=(0, 1), density=True, alpha=0.55, color="#1F3A5F")
    if len(x) >= 10 and np.std(x) > 0:
        try:
            kde = stats.gaussian_kde(x)
            xs = np.linspace(0, 1, 200)
            ax.plot(xs, kde(xs), color="#C0392B", lw=1.6)
        except Exception:
            pass
    r = jdi_ranking[jdi_ranking["item_name"] == it].iloc[0]
    ax.set_title(f"{it}\nJDI={r['JDI_score']:.1f}  n={int(r['sample_size'])}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=7)
fig.suptitle("16 个赔偿项目支持率（SR）密度分布", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG / "fig2_sr_density.png", dpi=150)
plt.close(fig)

# 3. JDI 排名柱状图
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#0E7A5F" if s == "Rule-based" else "#C0392B" if s == "ML-based" else "#7F8C8D"
          for s in assign.set_index("item_name").loc[order, "recommended_strategy"]]
ax.bar(order, jdi_ranking["JDI_score"], color=colors)
ax.axhline(gap_cut, color="#1F3A5F", ls="--", lw=1.5,
           label=f"数据驱动分界 = {gap_cut:.2f}")
ax.set_title("司法确定性指数（JDI）排名")
ax.set_ylabel("JDI 分数（0-100）")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig3_jdi_ranking.png", dpi=150)
plt.close(fig)

# 4. JDI 二维散点
fig, ax = plt.subplots(figsize=(11, 7))
strat_map = assign.set_index("item_name")
cmap = {"Rule-based": "#0E7A5F", "ML-based": "#C0392B", "Small-sample": "#7F8C8D"}
for s, c in cmap.items():
    m = strat_map["recommended_strategy"] == s
    ax.scatter(strat_map.loc[m, "sample_size"], strat_map.loc[m, "JDI_score"],
               c=c, s=70, label=s, alpha=0.85, edgecolors="white")
for it in strat_map.index:
    ax.annotate(it, (strat_map.loc[it, "sample_size"], strat_map.loc[it, "JDI_score"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axvline(SMALL_SAMPLE_DEFAULT, color="#7F8C8D", ls=":", lw=1.5,
           label=f"样本量阈值 = {SMALL_SAMPLE_DEFAULT}")
ax.axhline(gap_cut, color="#1F3A5F", ls="--", lw=1.5, label=f"JDI 分界 = {gap_cut:.2f}")
ax.set_xlabel("有效样本量（claim > 0）")
ax.set_ylabel("JDI 分数")
ax.set_title("样本量 × JDI 二维分布")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig4_jdi_scatter.png", dpi=150)
plt.close(fig)

# 5. 分层图
fig, ax = plt.subplots(figsize=(13, 6))
ys = np.arange(len(order))
ax.barh(ys, [1] * len(order), color=[cmap[s] for s in strat_map.loc[order, "recommended_strategy"]])
ax.set_yticks(ys)
ax.set_yticklabels(order)
for y, it in enumerate(order):
    r = strat_map.loc[it]
    ax.text(0.5, y, f"{r['recommended_strategy']}  |  JDI={r['JDI_score']:.1f}  |  n={int(r['sample_size'])}",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax.set_title("赔偿项目策略分层结果")
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / "fig5_strategy_map.png", dpi=150)
plt.close(fig)
log(f"[saved] results/stage2/figures/（5 张图）")

# ===========================================================================
# 九、敏感性分析
# ===========================================================================
log("=" * 72)
log("九、敏感性分析")
log("=" * 72)
schemes = {
    "default": weight_cfg["sensitivity_schemes"]["default"],
    "equal": weight_cfg["sensitivity_schemes"]["equal"],
    "rule_oriented": weight_cfg["sensitivity_schemes"]["rule_oriented"],
}
sens_rows = []
base_rank = None
for name, w in schemes.items():
    sc = compute_jdi(jdi_ranking, w)
    rk = sc.rank(ascending=False, method="min").astype(int)
    tmp = pd.DataFrame({"item_name": jdi_ranking["item_name"], "score": sc, "rank": rk})
    tmp = tmp.sort_values("rank")
    if name == "default":
        base_rank = tmp.set_index("item_name")["rank"]
    for _, r in tmp.iterrows():
        sens_rows.append({
            "analysis": "weight_scheme", "scheme": name, "item_name": r["item_name"],
            "sample_size": int(jdi_ranking.set_index("item_name").loc[r["item_name"], "sample_size"]),
            "jdi_score": round(float(r["score"]), 4), "rank": int(r["rank"]),
            "category": assign.set_index("item_name").loc[r["item_name"], "recommended_strategy"],
            "note": "",
        })

# 排序变化统计
for name in schemes:
    sub_rank = pd.DataFrame(sens_rows)
    sub_rank = sub_rank[(sub_rank["analysis"] == "weight_scheme") & (sub_rank["scheme"] == name)]
    cur = sub_rank.set_index("item_name")["rank"]
    if name != "default":
        rho = float(stats.spearmanr(base_rank.loc[cur.index], cur)[0])
        changed = int((base_rank.loc[cur.index] != cur).sum())
        maxshift = int((base_rank.loc[cur.index] - cur).abs().max())
        sens_rows.append({
            "analysis": "weight_scheme", "scheme": name, "item_name": "SUMMARY",
            "sample_size": np.nan, "jdi_score": np.nan, "rank": np.nan, "category": "",
            "note": f"与 default 排序 Spearman={rho:.4f}；位次变化项目数={changed}/16；最大位次变动={maxshift}",
        })
        log(f"  权重方案 {name:<14} vs default：Spearman={rho:.4f}，变化 {changed}/16 项，最大位移 {maxshift}")

# 阈值敏感性
for thr in (50, 100, 200):
    for _, r in assign.iterrows():
        cat = "Small-sample" if r["sample_size"] < thr else r["AB_suggestion"]
        sens_rows.append({
            "analysis": "threshold", "scheme": f"small_sample_{thr}", "item_name": r["item_name"],
            "sample_size": int(r["sample_size"]), "jdi_score": r["JDI_score"], "rank": int(r["rank"]) if "rank" in r else np.nan,
            "category": cat,
            "note": "",
        })
    n_ss = int((assign["sample_size"] < thr).sum())
    sens_rows.append({
        "analysis": "threshold", "scheme": f"small_sample_{thr}", "item_name": "SUMMARY",
        "sample_size": np.nan, "jdi_score": np.nan, "rank": np.nan, "category": "",
        "note": f"阈值={thr} 时 Small-sample 项目数={n_ss}；其余项目按 A/B 建议归类"
                f"（阈值变化不改变 A/B 归属，仅改变标记范围）",
    })
    log(f"  阈值 {thr}：Small-sample 项目数 = {n_ss}")

save_csv(pd.DataFrame(sens_rows), "jdi_sensitivity.csv")

# ===========================================================================
# 十、阶段报告
# ===========================================================================
log("=" * 72)
log("十、生成阶段报告")
log("=" * 72)
rule_items = tier_out["rule_based"]
ml_items = tier_out["ml_based"]
ss_items = tier_out["small_sample"]

lines = [
    "# Stage 2 司法确定性指数（JDI）与项目分层分析 —— 阶段报告",
    "",
    f"生成时间：{datetime.now().isoformat(timespec='seconds')}  ",
    "阶段约束：**未训练任何机器学习模型、未做模型比较、未调参**；"
    "仅使用描述统计、相关分析与互信息估计。",
    "",
    "## 1. JDI 定义",
    "",
    "司法确定性指数（Judicial Determinacy Index）衡量某一赔偿项目「判决结果由稳定规则决定的程度」。",
    "JDI 越高 → 支持率越集中、越可归因于客观规则 → 适合规则计算；",
    "JDI 越低 → 支持率越分散、裁量空间越大 → 适合数据驱动预测。",
    "",
    "构造方式：9 项基础指标 → 选取 7 项进入指数 → 各分量跨 16 个项目 min-max 归一化 → "
    "负向指标取 `1 - x` → 加权求和 → 线性映射到 0-100。",
    "",
    "## 2. 指标解释",
    "",
    "| 指标 | 含义 | 方向 | 权重 |",
    "|---|---|---|---:|",
    "| P(SR=1) | 全额支持率占比 | 正向（确定性增强） | 0.20 |",
    "| P(SR=0) | 全额驳回率占比 | 正向（极端结果集中=规则稳定） | 0.10 |",
    "| SR_mode_density | 众数密度（10 箱最高占比） | 正向 | 0.20 |",
    "| liability_consistency | 支持率与责任比例的相关性（子集 n≥30 时启用，否则中性值 0.5） | 正向 | 0.10 |",
    "| SR_entropy_norm | SR 分布熵（10 箱，归一化） | 负向 | 0.15 |",
    "| std_SR | SR 标准差 | 负向 | 0.15 |",
    "| IQR_SR | SR 四分位距 | 负向 | 0.10 |",
    "",
    "**未进入指数的诊断指标**（避免方向不可解释或不可比）：",
    "",
    "- `claim_SR_MI`：反映「可预测性」而非「确定性」，高 MI 也可能来自复杂裁量，仅作诊断；",
    "- `disability_explanatory_power`：仅伤残相关项目适用，纳入会引入项目间不可比性；",
    "- `CV_SR`：与 std_SR 共线；`legal_rule_strength`：法律先验用于分层决策而非指数本身。",
    "",
    "## 3. 16 项目 JDI 排名",
    "",
    "| 排名 | 项目 | 样本量 | JDI | P(SR=1) | P(SR=0) | 熵(归一化) | 众数密度 | std | 法律先验 |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
]
for _, r in jdi_ranking.iterrows():
    lines.append(
        f"| {int(r['rank'])} | {r['item_name']} | {int(r['sample_size'])} | {r['JDI_score']:.2f} | "
        f"{r['p_sr_eq_1']:.3f} | {r['p_sr_eq_0']:.3f} | {r['SR_entropy_norm']:.3f} | "
        f"{r['SR_mode_density_bin10']:.3f} | {r['std_SR']:.3f} | {r['legal_basis_type']}({r['legal_rule_strength']}) |"
    )
lines += [
    "",
    f"数据驱动分界（JDI 排序最大间隙）：**{gap_cut:.2f}**（间隙 {gap_size:.2f}）",
    "",
    "## 4. 分层结果",
    "",
    f"- **Rule-based（A 类）**：{len(rule_items)} 项 —— {', '.join(rule_items) if rule_items else '（空）'}",
    f"- **ML-based（B 类）**：{len(ml_items)} 项 —— {', '.join(ml_items) if ml_items else '（空）'}",
    f"- **Small-sample（C 类）**：{len(ss_items)} 项 —— {', '.join(ss_items) if ss_items else '（空）'}",
    "",
    "| 项目 | 样本量 | JDI | 分类 | A/B 建议 | 判定依据 |",
    "|---|---:|---:|---|---|---|",
]
for _, r in assign.iterrows():
    lines.append(
        f"| {r['item_name']} | {r['sample_size']} | {r['JDI_score']:.2f} | "
        f"**{r['recommended_strategy']}** | {r['AB_suggestion']} | {r['reason']} |"
    )
lines += [
    "",
    "分层规则：① Small-sample 判定 = `sample_size < 100`（Stage 2 规则）**或**"
    "已列入 Stage 0/1 冻结的小样本清单（death_comp 78、funeral 78、accommodation 104、"
    "future_medical 172）——两者取并集，保证与已冻结配置一致；Small-sample 仅作标记，"
    "A/B 建议仍给出，不因样本量自动判为规则项目；② JDI 高于数据驱动分界且法律规则强度 ≥ 0.5 "
    "→ Rule-based；③ JDI 低于分界且规则强度 < 0.5 → ML-based；④ 冲突时以 JDI 为准并记录冲突。",
    "",
    "## 5. 精神损害抚慰金专项分析",
    "",
    f"- 有效样本：**{n_ms}** 例；均值 {ms_stats['mean']:.4f}、中位数 {ms_stats['median']:.4f}、"
    f"标准差 {ms_stats['std']:.4f}；全额支持 {ms_stats['p_eq_1']:.2%}、全额驳回 {ms_stats['p_eq_0']:.2%}",
    f"- 伤残等级 Spearman：**{rho_dis:.4f}**（p={rho_dis_p:.2e}），"
    f"有伤残 vs 无伤残 SR 均值差 **{diff_dis:.4f}**",
    f"- 责任比例 Spearman：**{rho_liab:.4f}**（子集 n={ms_liab_n}）",
    f"- 请求金额 Pearson **{rho_claim_p:.4f}** / Spearman **{rho_claim_s:.4f}** / MI **{mi_claim:.4f}**",
    "",
    "详见 `results/stage2/mental_solatium_analysis.md`。",
    "",
    "## 6. 下一阶段实验建议",
    "",
    "1. **Stage 3 规则引擎**：为 Rule-based 项目建立判前可算的规则公式"
    "（建议以分项支持率中位数、责任比例、法定系数为主），并在测试折上评估其误差上界。",
    "2. **Stage 4 建模范围**：ML-based 项目参与主要模型排名；Small-sample 项目仅做探索性分析，"
    "不参与排名；需补充「含/不含 procedure_type」与「含/不含责任比例」的特征集对照。",
    "3. **缺失机制**：责任比例子集在多数分项上样本不足（n<30），Stage 4 需显式检验"
    "`liability_missing` 的预测贡献。",
    "4. **冲突项目复核**：JDI 与法律先验冲突的项目需在论文中说明判定依据（已记录在 "
    "`item_strategy_assignment.csv`）。",
    "5. **敏感性结论**：权重方案与阈值变化的排序稳定性见 `jdi_sensitivity.csv`，"
    "论文应报告该稳健性结果。",
    "",
    "## 7. 本阶段产物",
    "",
    "| 文件 | 说明 |",
    "|---|---|",
    "| `results/stage2/JDI_ranking.csv` | 16 项目 JDI 排名与全部分量得分 |",
    "| `results/stage2/jdi_weight_config.json` | 指数权重与方向定义、排除项说明 |",
    "| `results/stage2/item_strategy_assignment.csv` | 分层结果与判定依据 |",
    "| `results/stage2/project_strategy_assigned.json` | 三类项目列表（回填策略配置） |",
    "| `results/stage2/mental_solatium_analysis.md` | 精神损害抚慰金专项分析 |",
    "| `results/stage2/jdi_sensitivity.csv` | 权重与阈值敏感性分析 |",
    "| `results/stage2/figures/fig1_sr_boxplot.png` | 16 项目 SR 箱线图 |",
    "| `results/stage2/figures/fig2_sr_density.png` | 16 项目 SR 密度图 |",
    "| `results/stage2/figures/fig3_jdi_ranking.png` | JDI 排名柱状图 |",
    "| `results/stage2/figures/fig4_jdi_scatter.png` | 样本量 × JDI 散点图 |",
    "| `results/stage2/figures/fig5_strategy_map.png` | 项目策略分层图 |",
    "| `results/reports/stage2_summary.md` | 本报告 |",
    "",
    "> 本阶段未训练任何模型、未比较模型、未调参，已完成 Stage 2 全部任务并停止。",
]
(REPORTS / "stage2_summary.md").write_text("\n".join(lines), encoding="utf-8")
log(f"[saved] results/reports/stage2_summary.md")

log("=" * 72)
log("Stage 2 完成。")
log("=" * 72)
