# -*- coding: utf-8 -*-
"""
01 — Pre-judgment Feature Engineering & Label Construction
==========================================================

输入：data/dataset_anonymized.csv（已脱敏，n=2380）
输出：results/stage1/dataset_case_level.csv          案件级数据集
      results/stage1/dataset_item_level.csv          分项级长表（2380 × 16 项目）
      results/stage1/item_distribution_base.csv      项目级统计基础表（02 使用）
      results/stage1/stage1_leakage_check.json       数据泄漏自动检查
      results/reports/01_feature_engineering_report.md

严格约束（与原 Stage 1 冻结协议一致）：
  - 禁止训练任何模型 / 禁止模型比较 / 禁止调参
  - 只做：判前特征构建 / 标签构造 / 案件级-分项级转换 / 数据泄漏检查

所有处理规则来自 configs/ 下已冻结配置，禁止在代码中覆盖策略。
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MAIN_CSV = DATA / "dataset_anonymized.csv"
OUT = ROOT / "results" / "stage1"
REPORTS = ROOT / "results" / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

TOL = 1e-6
RATE_TOL = 1e-4


def log(msg: str) -> None:
    print(msg, flush=True)


def save_csv(df: pd.DataFrame, name: str) -> None:
    p = OUT / name
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"[saved] results/stage1/{name}  rows={len(df)}  cols={df.shape[1]}")


# ===========================================================================
# 一、读取配置（禁止自行修改策略）
# ===========================================================================
log("=" * 72)
log("一、读取配置")
log("=" * 72)

policy = yaml.safe_load((CONFIGS / "experiment_policy.yaml").read_text(encoding="utf-8"))
liab_cfg = json.loads((CONFIGS / "liability_strategy.json").read_text(encoding="utf-8"))
disab_cfg = json.loads((CONFIGS / "disability_strategy.json").read_text(encoding="utf-8"))

BLACKLIST_CFG = CONFIGS / "feature_blacklist.json"
if not BLACKLIST_CFG.exists():
    src = ROOT / "results" / "stage1" / "feature_blacklist.json"
    shutil.copy(src, BLACKLIST_CFG)
    log("[note] configs/feature_blacklist.json 不存在，已从 results/stage1/ 复制一份")
black_cfg = json.loads(BLACKLIST_CFG.read_text(encoding="utf-8"))

log(f"  experiment_policy.yaml  sections={len(policy)}")
log(f"  liability_strategy.json  main={liab_cfg['main_strategy']} "
    f"sensitivity={liab_cfg['sensitivity_strategy']}")
log(f"  disability_strategy.json status={disab_cfg['encoding_status']}")
log(f"  feature_blacklist.json   blacklist={black_cfg['blacklist']}")

ITEMS = policy["data"]["items"]
BLACKLIST = black_cfg["blacklist"]
RANDOM_STATE = policy["global"]["random_state"]

# ===========================================================================
# 二、读取脱敏数据（只读，不修改）
# ===========================================================================
log("=" * 72)
log("二、读取脱敏数据")
log("=" * 72)
df = pd.read_csv(MAIN_CSV, encoding="utf-8-sig")
N = len(df)
CLAIM_COLS = [f"{i}_claim" for i in ITEMS]
AWARD_COLS = [f"{i}_award" for i in ITEMS]
log(f"  dataset_anonymized.csv：{df.shape[0]} 行 × {df.shape[1]} 列（只读，已脱敏）")

# 年份 / 审理程序直接来自脱敏表中的宏观类别列（原始 file_name 标识已在脱敏阶段剔除）
year_parsed = pd.to_numeric(df["year"], errors="coerce")
procedure_type = df["procedure_type"].astype(object).where(df["procedure_type"].notna(), np.nan)
n_proc_missing = int(pd.isna(procedure_type).sum())
log(f"  year 缺失={int(year_parsed.isna().sum())}  procedure_type 缺失={n_proc_missing}")

# ===========================================================================
# 三、责任比例特征（严格按 liability_strategy.json）
# ===========================================================================
log("=" * 72)
log("三、构造责任比例特征")
log("=" * 72)
lr_raw = pd.to_numeric(df["liability_ratio"], errors="coerce")
lr_clean = lr_raw.where(lr_raw != 0, other=np.nan)      # raw == 0 -> NaN；其余保持
lr_missing = lr_clean.isna().astype(int)

log(f"  raw  ：非缺失 {int(lr_raw.notna().sum())}，其中 0 值 {int((lr_raw == 0).sum())}，"
    f"正值 {int((lr_raw > 0).sum())}，NaN {int(lr_raw.isna().sum())}")
log(f"  clean：有效 {int(lr_clean.notna().sum())}，NaN {int(lr_clean.isna().sum())}")
log(f"  liability_missing=1：{int(lr_missing.sum())}  "
    f"（{lr_missing.mean():.4%}），未删除任何样本")

sr_orig = pd.to_numeric(df["support_rate"], errors="coerce")
rows = [
    ("raw_distribution", "n_rows", N),
    ("raw_distribution", "n_notnull", int(lr_raw.notna().sum())),
    ("raw_distribution", "n_nan", int(lr_raw.isna().sum())),
    ("raw_distribution", "n_zero", int((lr_raw == 0).sum())),
    ("raw_distribution", "n_positive", int((lr_raw > 0).sum())),
    ("share_zero", round(float((lr_raw == 0).mean()), 6)),
    ("min_positive", float(lr_raw[lr_raw > 0].min())),
    ("max", float(lr_raw.max())),
    ("mean_positive", round(float(lr_raw[lr_raw > 0].mean()), 6)),
    ("median_positive", float(lr_raw[lr_raw > 0].median())),
    ("clean_distribution", "n_valid", int(lr_clean.notna().sum())),
    ("clean_distribution", "n_missing", int(lr_clean.isna().sum())),
    ("clean_distribution", "missing_rate", round(float(lr_clean.isna().mean()), 6)),
    ("clean_distribution", "mean", round(float(lr_clean.mean()), 6)),
    ("clean_distribution", "median", float(lr_clean.median())),
    ("clean_distribution", "std", round(float(lr_clean.std()), 6)),
    ("missing_group_sr_total", "n", int(lr_missing.sum())),
    ("missing_group_sr_total", "sr_total_mean", round(float(sr_orig[lr_missing == 1].mean()), 6)),
    ("missing_group_sr_total", "sr_total_median", round(float(sr_orig[lr_missing == 1].median()), 6)),
    ("missing_group_sr_total", "sr_total_std", round(float(sr_orig[lr_missing == 1].std()), 6)),
    ("missing_group_sr_total", "sr_total_p10",
     round(float(np.nanpercentile(sr_orig[lr_missing == 1], 10)), 6)),
    ("missing_group_sr_total", "sr_total_p90",
     round(float(np.nanpercentile(sr_orig[lr_missing == 1], 90)), 6)),
    ("nonmissing_group_sr_total", "n", int((lr_missing == 0).sum())),
    ("nonmissing_group_sr_total", "sr_total_mean", round(float(sr_orig[lr_missing == 0].mean()), 6)),
    ("nonmissing_group_sr_total", "sr_total_median", round(float(sr_orig[lr_missing == 0].median()), 6)),
    ("nonmissing_group_sr_total", "sr_total_std", round(float(sr_orig[lr_missing == 0].std()), 6)),
    ("policy", "main_strategy", liab_cfg["main_strategy"]),
    ("policy", "sensitivity_strategy", liab_cfg["sensitivity_strategy"]),
    ("policy", "imputation_status", "NOT_PERFORMED（Stage 1 禁止提前填补，仅保留 NaN + 缺失指示）"),
    ("policy", "sample_deletion", "NONE（禁止删除缺失样本）"),
]
save_csv(pd.DataFrame(rows, columns=["section", "key", "value"]), "liability_stage1_report.csv")

# ===========================================================================
# 四、伤残等级特征（严格按 disability_strategy.json）
# ===========================================================================
log("=" * 72)
log("四、构造伤残等级特征")
log("=" * 72)
dl_raw = pd.to_numeric(df["disability_level"], errors="coerce")
dl_clean = dl_raw.where(~dl_raw.isin([25.0]), other=np.nan)   # 25 -> NaN；0 与 1-10 保留
dl_abnormal = (dl_raw == 25.0).astype(int)
dl_missing = dl_clean.isna().astype(int)
log(f"  raw：{dl_raw.value_counts(dropna=False).sort_index().to_dict()}")
log(f"  clean：有效 {int(dl_clean.notna().sum())}，NaN {int(dl_clean.isna().sum())}；"
    f"abnormal=1：{int(dl_abnormal.sum())}；未删除任何样本")

rows = [("raw_distribution", f"value_{v if pd.notna(v) else 'NaN'}", int(c))
        for v, c in dl_raw.value_counts(dropna=False).sort_index().items()]
rows += [
    ("raw_distribution", "n_rows", N),
    ("raw_distribution", "n_nan", int(dl_raw.isna().sum())),
    ("raw_distribution", "n_25_abnormal", int((dl_raw == 25.0).sum())),
    ("clean_distribution", "n_valid", int(dl_clean.notna().sum())),
    ("clean_distribution", "n_missing", int(dl_clean.isna().sum())),
    ("clean_distribution", "missing_rate", round(float(dl_clean.isna().mean()), 6)),
    ("clean_distribution", "share_level_0", round(float((dl_clean == 0).mean()), 6)),
    ("clean_distribution", "share_level_1_10", round(float(((dl_clean >= 1) & (dl_clean <= 10)).mean()), 6)),
    ("derived", "disability_abnormal_n", int(dl_abnormal.sum())),
    ("derived", "disability_missing_n", int(dl_missing.sum())),
    ("policy", "encoding_status", disab_cfg["encoding_status"]),
    ("policy", "encoding_confirmed",
     json.dumps(disab_cfg["confirmed_encoding"], ensure_ascii=False)),
    ("policy", "dual_encoding_required", str(disab_cfg.get("dual_encoding_required", True))),
    ("policy", "sample_deletion", "NONE（禁止删除异常样本）"),
]
save_csv(pd.DataFrame(rows, columns=["section", "key", "value"]), "disability_stage1_report.csv")

# ===========================================================================
# 五、案件级数据集构建
# ===========================================================================
log("=" * 72)
log("五、案件级数据集构建")
log("=" * 72)

claim_mat = df[CLAIM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
award_mat = df[AWARD_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

claim_total_calc = claim_mat.sum(axis=1)
claim_total_orig = pd.to_numeric(df["claim_total"], errors="coerce")
claim_diff = (claim_total_calc - claim_total_orig).abs()
claim_consistent = claim_diff <= np.maximum(1.0, claim_total_calc.abs() * RATE_TOL)
n_nonzero_claims = (claim_mat > 0).sum(axis=1).astype(int)

case_ids = [f"C{i:04d}" for i in range(N)]
case = pd.DataFrame({
    "case_id": case_ids,
    "year": year_parsed.astype("Int64"),
    "procedure_type": pd.Series(procedure_type, index=df.index).astype("object"),
    "liability_ratio_raw": lr_raw,
    "liability_ratio_clean": lr_clean,
    "liability_missing": lr_missing.astype(int),
    "disability_level_raw": dl_raw,
    "disability_level_clean": dl_clean,
    "disability_abnormal": dl_abnormal.astype(int),
    "disability_missing": dl_missing.astype(int),
})
for c in CLAIM_COLS:
    case[c] = claim_mat[c].values
case["claim_total_calc"] = claim_total_calc.values
case["claim_total_orig"] = claim_total_orig.values
case["claim_total_diff"] = claim_diff.values
case["claim_total_consistency"] = claim_consistent.astype(int).values
case["n_nonzero_claims"] = n_nonzero_claims.values

for it in ITEMS:
    case[f"claim_share__{it}"] = np.where(
        claim_total_calc > 0, claim_mat[f"{it}_claim"] / claim_total_calc.replace(0, np.nan), 0.0
    )
for it in ITEMS:
    case[f"log_claim__{it}"] = np.log1p(claim_mat[f"{it}_claim"])

# 案件级标签（预测目标，非特征）
award_total_calc = award_mat.sum(axis=1)
sr_total_label = np.where(claim_total_calc > 0, award_total_calc / claim_total_calc.replace(0, np.nan), np.nan)
case["sr_total_label"] = sr_total_label

log(f"  案件级数据集：{case.shape[0]} 行 × {case.shape[1]} 列")
log(f"  claim_total 一致性：一致 {int(claim_consistent.sum())} / {N}，"
    f"最大差异 {claim_diff.max():.4f}")
log(f"  n_nonzero_claims：min={int(n_nonzero_claims.min())} max={int(n_nonzero_claims.max())} "
    f"mean={n_nonzero_claims.mean():.2f}")

# ===========================================================================
# 六、分项级长表构建
# ===========================================================================
log("=" * 72)
log("六、分项级长表构建")
log("=" * 72)

case_feature_block = case.drop(columns=["sr_total_label"])   # 不含案件级标签（分项级标签单独给出）

blocks = []
for it in ITEMS:
    cl = claim_mat[f"{it}_claim"]
    aw = award_mat[f"{it}_award"]
    sr_i = np.where(cl > 0, aw / cl.where(cl > 0), np.nan)
    b = pd.DataFrame({
        "case_id": case_ids,
        "item_name": it,
        "claim_amount": cl.values,
        "award_amount": aw.values,
        "support_rate_label": sr_i,
        "claim_positive": (cl > 0).astype(int).values,
        "item_claim_share": np.where(
            claim_total_calc > 0, cl / claim_total_calc.replace(0, np.nan), 0.0),
        "item_log_claim": np.log1p(cl).values,
    })
    blocks.append(b)
item_long = pd.concat(blocks, ignore_index=True)
item_long = item_long.merge(case_feature_block, on="case_id", how="left")
log(f"  分项级长表：{item_long.shape[0]} 行 × {item_long.shape[1]} 列 "
    f"（{N} 案件 × {len(ITEMS)} 项目 = {N * len(ITEMS)}）")

# ---- 支持率校验 ----
log("=" * 72)
log("七、支持率标签校验")
log("=" * 72)
valid = item_long["claim_positive"] == 1
sr = item_long["support_rate_label"]
n_valid = int(valid.sum())
n_lt0 = int((sr[valid] < -TOL).sum())
n_gt1 = int((sr[valid] > 1 + TOL).sum())
n_nan = int(sr[valid].isna().sum())
log(f"  有效标签（claim>0）：{n_valid}")
log(f"  SR < 0：{n_lt0}   SR > 1：{n_gt1}   有效样本内 NaN：{n_nan}")

val_rows = [
    {"scope": "summary", "item_name": "ALL", "case_id": "", "claim_amount": np.nan,
     "award_amount": np.nan, "support_rate_label": round(float(sr[valid].mean()), 6),
     "issue_type": f"n_valid={n_valid}|n_lt0={n_lt0}|n_gt1={n_gt1}|"
                   f"min={round(float(sr[valid].min()), 6)}|max={round(float(sr[valid].max()), 6)}"},
]
for it in ITEMS:
    m = (item_long["item_name"] == it) & valid
    s = item_long.loc[m, "support_rate_label"]
    val_rows.append({
        "scope": "summary", "item_name": it, "case_id": "", "claim_amount": np.nan,
        "award_amount": np.nan,
        "support_rate_label": round(float(s.mean()), 6) if len(s) else np.nan,
        "issue_type": f"n={len(s)}|n_lt0={int((s < -TOL).sum())}|n_gt1={int((s > 1 + TOL).sum())}|"
                      f"min={round(float(s.min()), 6) if len(s) else 'NA'}|"
                      f"max={round(float(s.max()), 6) if len(s) else 'NA'}",
    })
bad = item_long[valid & ((sr < -TOL) | (sr > 1 + TOL))]
for _, r in bad.iterrows():
    val_rows.append({
        "scope": "anomaly", "item_name": r["item_name"], "case_id": r["case_id"],
        "claim_amount": r["claim_amount"], "award_amount": r["award_amount"],
        "support_rate_label": r["support_rate_label"],
        "issue_type": "SR_OUT_OF_RANGE",
    })
if bad.empty and n_lt0 == 0 and n_gt1 == 0:
    val_rows.append({
        "scope": "anomaly", "item_name": "ALL", "case_id": "", "claim_amount": np.nan,
        "award_amount": np.nan, "support_rate_label": np.nan,
        "issue_type": "NO_OUT_OF_RANGE_RECORD（未自动删除任何记录）",
    })
save_csv(pd.DataFrame(val_rows), "sr_validation.csv")

# ===========================================================================
# 八、案件级支持率标签校验
# ===========================================================================
log("=" * 72)
log("八、案件级支持率标签校验")
log("=" * 72)
diff_sr = (pd.Series(sr_total_label) - sr_orig).abs()
consistent_sr = diff_sr <= RATE_TOL
case_sr = pd.DataFrame({
    "case_id": case_ids,
    "claim_total_calc": claim_total_calc.values,
    "award_total_calc": award_total_calc.values,
    "sr_total_label": sr_total_label,
    "support_rate_original": sr_orig.values,
    "abs_diff": diff_sr.values,
    "is_consistent": consistent_sr.astype(int).values,
})
save_csv(case_sr, "case_sr_validation.csv")
log(f"  与原始 support_rate 一致：{int(consistent_sr.sum())} / {N}，"
    f"最大差异 {float(diff_sr.max()):.8f}")

# ===========================================================================
# 九、项目级统计基础表（为 02 JDI 计算准备）
# ===========================================================================
log("=" * 72)
log("九、项目级统计基础表")
log("=" * 72)
rows = []
for it in ITEMS:
    m = item_long["item_name"] == it
    cl = item_long.loc[m, "claim_amount"]
    aw = item_long.loc[m, "award_amount"]
    sr_i = item_long.loc[m, "support_rate_label"]
    pos = cl > 0
    srv = sr_i[pos]
    rows.append({
        "item_name": it,
        "claim_positive_count": int(pos.sum()),
        "award_positive_count": int((aw > 0).sum()),
        "mean_claim": round(float(cl[pos].mean()), 2) if pos.any() else np.nan,
        "mean_award": round(float(aw[pos].mean()), 2) if pos.any() else np.nan,
        "mean_SR": round(float(srv.mean()), 6) if len(srv) else np.nan,
        "std_SR": round(float(srv.std()), 6) if len(srv) else np.nan,
        "median_SR": round(float(srv.median()), 6) if len(srv) else np.nan,
        "zero_SR_ratio": round(float((srv.abs() <= TOL).mean()), 6) if len(srv) else np.nan,
    })
item_base = pd.DataFrame(rows)
save_csv(item_base, "item_distribution_base.csv")
for _, r in item_base.iterrows():
    log(f"  {r['item_name']:<18} n={int(r['claim_positive_count']):>5}  "
        f"mean_SR={r['mean_SR']:.4f}  std_SR={r['std_SR']:.4f}  "
        f"median_SR={r['median_SR']:.4f}  zero_ratio={r['zero_SR_ratio']:.4f}")

# ===========================================================================
# 十、字段审计表（feature_schema_stage1.csv）
# ===========================================================================
log("=" * 72)
log("十、建立字段审计表")
log("=" * 72)

PROVENANCE = {}
for it in ITEMS:
    PROVENANCE[f"{it}_claim"] = [f"{it}_claim"]
    PROVENANCE[f"claim_share__{it}"] = [f"{it}_claim"] + CLAIM_COLS
    PROVENANCE[f"log_claim__{it}"] = [f"{it}_claim"]
PROVENANCE["case_id"] = []
PROVENANCE["year"] = ["year"]
PROVENANCE["procedure_type"] = ["procedure_type"]
PROVENANCE["liability_ratio_raw"] = ["liability_ratio"]
PROVENANCE["liability_ratio_clean"] = ["liability_ratio"]
PROVENANCE["liability_missing"] = ["liability_ratio"]
PROVENANCE["disability_level_raw"] = ["disability_level"]
PROVENANCE["disability_level_clean"] = ["disability_level"]
PROVENANCE["disability_abnormal"] = ["disability_level"]
PROVENANCE["disability_missing"] = ["disability_level"]
PROVENANCE["claim_total_calc"] = CLAIM_COLS
PROVENANCE["claim_total_orig"] = ["claim_total"]
PROVENANCE["claim_total_diff"] = ["claim_total"] + CLAIM_COLS
PROVENANCE["claim_total_consistency"] = ["claim_total"] + CLAIM_COLS
PROVENANCE["n_nonzero_claims"] = CLAIM_COLS
PROVENANCE["sr_total_label"] = AWARD_COLS + CLAIM_COLS
PROVENANCE["award_amount"] = ["*_award"]
PROVENANCE["support_rate_label"] = ["*_award"]

FEATURE_ALLOWED = {
    "year": True, "procedure_type": True,
    "liability_ratio_clean": True, "liability_missing": True,
    "disability_level_clean": True, "disability_abnormal": True, "disability_missing": True,
    "claim_total_calc": True,
    **{f"{it}_claim": True for it in ITEMS},
    **{f"claim_share__{it}": True for it in ITEMS},
    **{f"log_claim__{it}": True for it in ITEMS},
    "n_nonzero_claims": True,
}
REASON = {
    "year": "判前元数据（裁判年份）",
    "procedure_type": "判前元数据（一审/二审/再审/其他）",
    "liability_ratio_clean": "判前司法特征（raw==0 视为缺失编码转 NaN）",
    "liability_missing": "责任比例缺失指示，保留缺失机制信息",
    "disability_level_clean": "判前司法特征（25 视为异常转 NaN）",
    "disability_abnormal": "伤残等级异常值指示",
    "disability_missing": "伤残等级缺失指示（原始缺失 + 异常转换）",
    "claim_total_calc": "请求总额（由 16 项 claim 重算，判前可得）",
    "n_nonzero_claims": "自建非零 claim 项数，替代判后字段 n_nonzero_items",
}

schema_rows = []


def add_schema(field, source_type, category, allowed, reason):
    schema_rows.append({
        "field_name": field,
        "source_type": source_type,
        "category": category,
        "allowed_as_feature": allowed,
        "reason": reason,
    })


# (1) 判前输入特征 / 辅助字段：案件级数据集列
for c in case.columns:
    if c in ("sr_total_label",):
        add_schema(c, "derived(label)", "预测标签", False,
                   "案件级支持率标签 = sum(award)/sum(claim)，最终系统评价目标，禁止入模")
        continue
    if c in FEATURE_ALLOWED:
        add_schema(c, "raw" if c in df.columns else "derived",
                   "判前输入特征", FEATURE_ALLOWED[c], REASON.get(c, "判前可得"))
    else:
        add_schema(c, "raw/derived", "辅助分析字段", False,
                   "标识/冗余/一致性校验字段，不进入模型")

# (2) 分项级长表特有列
add_schema("item_name", "derived", "辅助分析字段", False, "分项标识")
add_schema("claim_amount", "raw", "判前输入特征", True, "分项请求金额（判前可得）")
add_schema("award_amount", "raw", "预测标签", False,
           "分项实际判赔金额，label_only，禁止入模")
add_schema("support_rate_label", "derived(label)", "预测标签", False,
           "SR_i = award_i / claim_i（仅 claim_i>0 定义），label_only")
add_schema("claim_positive", "derived", "辅助分析字段", False,
           "样本有效性标记（claim>0 才有标签），非特征")
add_schema("item_claim_share", "derived", "判前输入特征", True, "该分项请求占请求总额比例")
add_schema("item_log_claim", "derived", "判前输入特征", True, "log1p(分项请求金额)")

# (3) 禁止字段（原始表中的判后/废弃字段）
for c in AWARD_COLS:
    add_schema(c, "raw", "禁止字段", False, "label_only：分项实际判赔金额，判后信息")
for c in ["award_total", "support_rate", "support_rate_pct", "n_nonzero_items", "text_len"]:
    add_schema(c, "raw", "禁止字段", False, {
        "award_total": "判赔总额，判后信息",
        "support_rate": "案件级实际支持率，判后信息（最终评价标签）",
        "support_rate_pct": "同上（百分数形式）",
        "n_nonzero_items": "经审计 100% 等于非零 award 项数，判后泄漏字段（脱敏时已剔除）",
        "text_len": "全空列，candidate_drop（脱敏时已剔除）",
    }[c])

schema = pd.DataFrame(schema_rows)
save_csv(schema, "feature_schema_stage1.csv")
log(f"  判前输入特征：{int((schema['category'] == '判前输入特征').sum())}  "
    f"预测标签：{int((schema['category'] == '预测标签').sum())}  "
    f"禁止字段：{int((schema['category'] == '禁止字段').sum())}  "
    f"辅助分析：{int((schema['category'] == '辅助分析字段').sum())}")

# ===========================================================================
# 十一、数据泄漏自动检查
# ===========================================================================
log("=" * 72)
log("十一、数据泄漏自动检查")
log("=" * 72)

allowed_features = [r["field_name"] for r in schema_rows
                    if r["allowed_as_feature"] and r["category"] == "判前输入特征"]
# 案件级实际使用的特征列
case_feature_cols = [c for c in case.columns if c in allowed_features]
item_feature_cols = [c for c in item_long.columns if c in allowed_features]

BAN_PAT = re.compile(r"(_award$|^award_total$|^support_rate$|^support_rate_pct$"
                     r"|^n_nonzero_items$|^text_len$)")

c1_hits = {"case_level": [c for c in case_feature_cols if BAN_PAT.search(c)],
           "item_level": [c for c in item_feature_cols if BAN_PAT.search(c)]}
c1_pass = (not c1_hits["case_level"]) and (not c1_hits["item_level"])

c2_bad = {f: src for f, src in PROVENANCE.items()
          if f in allowed_features and any(("award" in s) or ("support_rate" in s) for s in src)}
c2_pass = not c2_bad


def build_case_features(src_df: pd.DataFrame) -> pd.DataFrame:
    """仅依赖判前字段重建特征块（用于扰动检验）"""
    cm = src_df[CLAIM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tot = cm.sum(axis=1)
    out = pd.DataFrame({"case_id": [f"C{i:04d}" for i in range(len(src_df))]})
    out["year"] = pd.to_numeric(src_df["year"], errors="coerce")
    out["procedure_type"] = src_df["procedure_type"].values
    r = pd.to_numeric(src_df["liability_ratio"], errors="coerce")
    out["liability_ratio_clean"] = r.where(r != 0, other=np.nan)
    out["liability_missing"] = out["liability_ratio_clean"].isna().astype(int)
    d = pd.to_numeric(src_df["disability_level"], errors="coerce")
    dc = d.where(~d.isin([25.0]), other=np.nan)
    out["disability_level_clean"] = dc
    out["disability_abnormal"] = (d == 25.0).astype(int)
    out["disability_missing"] = dc.isna().astype(int)
    for c in CLAIM_COLS:
        out[c] = cm[c].values
    out["claim_total_calc"] = tot.values
    out["n_nonzero_claims"] = (cm > 0).sum(axis=1).astype(int).values
    for it in ITEMS:
        out[f"claim_share__{it}"] = np.where(tot > 0, cm[f"{it}_claim"] / tot.replace(0, np.nan), 0.0)
    for it in ITEMS:
        out[f"log_claim__{it}"] = np.log1p(cm[f"{it}_claim"])
    return out


X_ref = build_case_features(df)
rng = np.random.default_rng(RANDOM_STATE)
df_pert = df.copy()
for c in AWARD_COLS + ["award_total", "support_rate", "support_rate_pct", "n_nonzero_items"]:
    if c in df_pert.columns:
        df_pert[c] = rng.uniform(0.0, 2.0, size=len(df_pert))
X_pert = build_case_features(df_pert)
c3_pass = bool(X_ref.equals(X_pert))
c3_detail = {
    "method": "将所有 *_award / award_total / support_rate / support_rate_pct / n_nonzero_items "
              "随机扰动后重建特征矩阵，若与原始特征矩阵完全相同，则特征不依赖任何判后信息",
    "shape": list(X_ref.shape),
    "identical": c3_pass,
}

c4_label_in_features = [c for c in ["sr_total_label", "award_amount", "support_rate_label"]
                        if c in case_feature_cols or c in item_feature_cols]
c4_pass = not c4_label_in_features

checks = [
    {"id": "C1_blacklist_column_scan", "name": "特征列黑名单扫描",
     "status": "PASS" if c1_pass else "FAIL",
     "detail": {"banned_pattern": BAN_PAT.pattern, "hits": c1_hits,
                "n_case_feature_cols": len(case_feature_cols),
                "n_item_feature_cols": len(item_feature_cols)}},
    {"id": "C2_provenance_declaration", "name": "特征来源声明检查",
     "status": "PASS" if c2_pass else "FAIL",
     "detail": {"violations": c2_bad,
                "rule": "任何允许入模的特征，其来源字段不得包含 award / support_rate"}},
    {"id": "C3_award_perturbation_invariance", "name": "判后字段扰动不变性检验",
     "status": "PASS" if c3_pass else "FAIL",
     "detail": c3_detail},
    {"id": "C4_label_feature_separation", "name": "标签与特征分离检查",
     "status": "PASS" if c4_pass else "FAIL",
     "detail": {"label_columns_found_in_features": c4_label_in_features}},
]
overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"

leak = {
    "stage": "01_feature_engineering",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "random_state": RANDOM_STATE,
    "policy_source": ["configs/experiment_policy.yaml",
                      "configs/liability_strategy.json",
                      "configs/disability_strategy.json",
                      "configs/feature_blacklist.json"],
    "n_case_level_feature_columns": len(case_feature_cols),
    "n_item_level_feature_columns": len(item_feature_cols),
    "case_level_feature_columns": case_feature_cols,
    "checks": checks,
    "overall": overall,
    "action_if_fail": "停止并报告，不得进入 02",
}
p = OUT / "stage1_leakage_check.json"
p.write_text(json.dumps(leak, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"[saved] results/stage1/stage1_leakage_check.json  overall={overall}")
for c in checks:
    log(f"  {c['id']:<34} {c['status']}")

if overall == "FAIL":
    raise SystemExit("[STOP] 数据泄漏检查未通过，按配置停止，不得进入 02。")

# ===========================================================================
# 保存数据集
# ===========================================================================
save_csv(case, "dataset_case_level.csv")
save_csv(item_long, "dataset_item_level.csv")

# ===========================================================================
# 十二、阶段报告
# ===========================================================================
log("=" * 72)
log("十二、生成阶段报告")
log("=" * 72)

n_valid_labels = int((item_long["claim_positive"] == 1).sum())
n_case_features = len(case_feature_cols)

lines = [
    "# 01 判前特征工程与标签构建 —— 阶段报告",
    "",
    f"生成时间：{datetime.now().isoformat(timespec='seconds')}  ",
    "阶段约束：**禁止训练模型 / 禁止模型比较 / 禁止调参**；仅完成特征构建、标签构造、"
    "案件级与分项级转换、数据泄漏检查。",
    "",
    "## 1. 数据转换流程",
    "",
    "```",
    f"data/dataset_anonymized.csv  ({N} 行 × {df.shape[1]} 列，只读，已脱敏)",
    "        |",
    "        |-- 读取配置：experiment_policy.yaml / liability_strategy.json /",
    "        |              disability_strategy.json / feature_blacklist.json",
    "        |",
    "        |-- 案件基础信息：case_id / year / procedure_type（脱敏表宏观类别列）",
    "        |",
    "        |-- 责任比例：raw -> clean(raw==0 -> NaN) -> missing 指示（不删除、不填补）",
    "        |-- 伤残等级：raw -> clean(25 -> NaN) -> abnormal / missing 指示（不删除）",
    "        |",
    "        |-- 请求结构特征：claim_total_calc / n_nonzero_claims / claim_share_i / log_claim_i",
    "        |",
    "        |-- 案件级数据集 dataset_case_level.csv",
    "        |-- 分项级长表 dataset_item_level.csv（案件 × 16 项目）",
    "        |-- 标签：support_rate_label（分项级） / sr_total_label（案件级）",
    "        |",
    "        +-- 泄漏检查（4 项）-> 全部 PASS",
    "```",
    "",
    "## 2. 最终案件级特征数量",
    "",
    f"- 案件级数据集：`dataset_case_level.csv`，**{case.shape[0]} 行 × {case.shape[1]} 列**",
    f"- 可入模的判前特征列：**{n_case_features}** 列",
    "  - 案件基础信息：2（`year`、`procedure_type`）",
    "  - 判前司法特征：5（`liability_ratio_clean`、`liability_missing`、"
    "`disability_level_clean`、`disability_abnormal`、`disability_missing`）",
    "  - 16 项请求金额：16",
    "  - 请求结构特征：18（`claim_total_calc`、`n_nonzero_claims`、16 项 `claim_share_i`）"
    " + 16 项 `log_claim_i`",
    f"- 分项级可入模特征列：**{len(item_feature_cols)}** 列（含案件级特征 + "
    "`claim_amount` / `item_claim_share` / `item_log_claim`）",
    "",
    "## 3. 最终分项级样本数量",
    "",
    f"- 分项级长表：`dataset_item_level.csv`，**{item_long.shape[0]} 行 × {item_long.shape[1]} 列**",
    f"- 构成：{N} 案件 × {len(ITEMS)} 项目 = {N * len(ITEMS)} 行（**全部保留**，未删除任何项目或样本）",
    f"- 其中存在有效支持率标签（claim > 0）的行：**{n_valid_labels}** 行"
    f"（占长表 {n_valid_labels / item_long.shape[0]:.2%}）",
    "",
    "## 4. 有效预测项目样本量",
    "",
    "| 项目 | claim>0 | mean_SR | std_SR | median_SR | zero_SR_ratio |",
    "|---|---:|---:|---:|---:|---:|",
]
for _, r in item_base.iterrows():
    lines.append(
        f"| {r['item_name']} | {int(r['claim_positive_count'])} | {r['mean_SR']:.4f} | "
        f"{r['std_SR']:.4f} | {r['median_SR']:.4f} | {r['zero_SR_ratio']:.4f} |"
    )
lines += [
    "",
    f"- 小样本项目（`small_sample = true`）：death_comp / funeral / accommodation / future_medical"
    "（按 `configs/small_sample_policy.md`，仅探索性分析，不参与主要模型排名）",
    "",
    "## 5. 缺失处理情况",
    "",
    "| 字段 | 缺失数 | 缺失率 | 处理方式 |",
    "|---|---:|---:|---|",
    f"| `liability_ratio_clean` | {int(lr_clean.isna().sum())} | "
    f"{lr_clean.isna().mean():.2%} | **不删除、不填补**；生成 `liability_missing` 指示变量，"
    "填补推迟到 Pipeline 内按训练折拟合 |",
    f"| `disability_level_clean` | {int(dl_clean.isna().sum())} | "
    f"{dl_clean.isna().mean():.2%} | 不删除；生成 `disability_missing` / `disability_abnormal` |",
    f"| `procedure_type` | {n_proc_missing} | {n_proc_missing / N:.2%} | 留空并记录 |",
    "",
    "## 6. 标签定义",
    "",
    "- 分项级：`support_rate_label = award_i / claim_i`，**仅当 `claim_i > 0` 时定义**，"
    "其余为 NaN（不参与训练，案件聚合时按 0 计入）。",
    "- 案件级：`sr_total_label = sum(award_i) / sum(claim_i)`，为最终系统评价目标。",
    f"- 取值范围校验：0 ≤ SR ≤ 1，越界记录 **{n_lt0 + n_gt1}** 条（`sr_validation.csv`）。",
    f"- 与原始 `support_rate` 一致性：{int(consistent_sr.sum())}/{N} 条一致，"
    f"最大差异 {float(diff_sr.max()):.8f}（`case_sr_validation.csv`）。",
    "",
    "## 7. 泄漏检查结果",
    "",
    "| 检查项 | 结果 |",
    "|---|---|",
]
for c in checks:
    lines.append(f"| {c['id']}（{c['name']}） | **{c['status']}** |")
lines += [
    "",
    f"**总体结论：{overall}**",
    "",
    "- C1：特征列中不存在 `*_award`、`award_total`、`support_rate`、`support_rate_pct`、"
    "`n_nonzero_items`、`text_len`。",
    "- C2：所有允许入模特征的来源声明均不含判后字段。",
    "- C3：将所有判后字段随机扰动后重建特征矩阵，结果与原始特征矩阵**完全一致**，"
    "证明特征不依赖任何判后信息。",
    "- C4：标签列（`sr_total_label`、`award_amount`、`support_rate_label`）均未出现在特征集合中。",
    "",
    "## 8. 本阶段产物",
    "",
    "| 文件 | 说明 | 规模 |",
    "|---|---|---|",
    f"| `results/stage1/feature_schema_stage1.csv` | 全字段审计表 | {len(schema)} 行 |",
    f"| `results/stage1/liability_stage1_report.csv` | 责任比例统计 | 30 行 |",
    f"| `results/stage1/disability_stage1_report.csv` | 伤残等级特征统计 | {len(rows)} 行 |",
    f"| `results/stage1/dataset_case_level.csv` | 案件级数据集 | {case.shape[0]} × {case.shape[1]} |",
    f"| `results/stage1/dataset_item_level.csv` | 分项级长表 | {item_long.shape[0]} × {item_long.shape[1]} |",
    f"| `results/stage1/sr_validation.csv` | 分项支持率校验 | {len(val_rows)} 行 |",
    f"| `results/stage1/case_sr_validation.csv` | 案件级支持率校验 | {N} 行 |",
    f"| `results/stage1/item_distribution_base.csv` | 项目级统计基础表（02 用） | {len(item_base)} 行 |",
    "| `results/stage1/stage1_leakage_check.json` | 泄漏检查结果 | 4 项检查 |",
    "",
    "> 本阶段未训练任何模型、未做模型比较、未调参。",
]
(REPORTS / "01_feature_engineering_report.md").write_text("\n".join(lines), encoding="utf-8")
log("[saved] results/reports/01_feature_engineering_report.md")

log("=" * 72)
log("01 完成（未训练模型、未比较、未调参）。")
log("=" * 72)
