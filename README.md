# Judicial Determinacy-Guided Epistemic Gating Framework for Traffic Accident Compensation Prediction

**司法确定性指数驱动的认知不确定性门控融合框架 —— 交通事故赔偿支持率预测（开源复现包）**

[中文说明](#中文说明) | [English](#english)

---

<a id="english"></a>

## Overview

This repository is the minimal, self-contained release accompanying the paper
*"Judicial Determinacy-Guided Epistemic Gating Framework for Traffic Accident
Compensation Prediction"*.

The task is reframed from direct award-amount regression to **support-rate
prediction**: for each of 16 statutory compensation items (medical, nursing,
lost income, disability compensation, mental solatium, death compensation,
etc.), the model predicts the ratio `SR = award / claim` (clipped to [0, 1]),
and case-level support rate is aggregated as the claim-weighted combination.

The final model — **Gated Hybrid** — combines three components:

1. **Judicial Determinacy Index (JDI)** — a per-item, statistically constructed
   index (0–100) measuring how "rule-determined" each compensation item is
   (full-award rate, zero-award rate, SR dispersion/entropy, mode density,
   liability consistency). JDI weights a *rule-based prior* against an
   *ML prediction* in a **soft continuous fusion**:
   `SR_item = w_i · SR_rule + (1 − w_i) · SR_ml`.
2. **Heterogeneous per-item heads** — each item uses its cross-validated best
   model (12 LightGBM + 4 CatBoost, assignment frozen and hard-coded), with
   three quantile heads: `mean`, `q10`, `q90`.
3. **Epistemic uncertainty gating** — the inter-quantile width
   `delta = clip(q90 − q10, 0, None)` proxies per-instance epistemic
   uncertainty. The fusion weight decays exponentially with uncertainty:
   `w_dyn = w_i · exp(−γ · delta)` with the frozen optimal **γ = 1.0**.
   When the model is uncertain, the prediction falls back toward the
   deterministic rule/judicial prior.

Evaluation protocol: case-level `RepeatedStratifiedKFold(5 folds × 3 repeats,
random_state=42)` stratified by `year × procedure_type × support-rate quintile`;
all rules, JDI weights and models are fitted strictly inside training folds.
Metrics: **MAE / RMSE / R² / MdAE** at the case level.

## Data

- `data/dataset_anonymized.csv` — **Anonymized dataset (n = 2380)** of Chinese
  traffic-accident compensation first/second-instance civil judgments (2022–2023,
  single region). All direct identifiers have been removed:
  case numbers, party names, lawyers, court names, document URLs/paths and
  license plates are **not included**. What remains:
  - macro categories: `case_id` (sequential surrogate), `year`, `procedure_type`;
  - cleaned judicial fields: `liability_ratio`, `disability_level`;
  - 16 × claim / award amounts, totals, `support_rate`.
- The rule engine's legal priors live in `src/rule_engine/definitions/`.
- **No raw judgment documents are distributed with this release.**

## Usage

Requirements: Python ≥ 3.10.

```bash
pip install -r requirements.txt

# 1) feature engineering + label construction + leakage checks
python scripts/01_feature_engineering.py

# 2) Judicial Determinacy Index (JDI) construction & item stratification
python scripts/02_jdi_computation.py

# 3) rule engine calibration & strong baselines (no ML)
python scripts/03_rule_engine_calibration.py

# 4) final model: heterogeneous quantile heads + epistemic gating (gamma=1.0)
python scripts/04_ultimate_gated_model.py

# 5) paper tables: Table 4/5/9/10 (metrics, improvement path, temporal validation, subgroups)
python scripts/05_generate_paper_tables.py
```

Run the scripts **in order 01 → 05**. Each script writes to `results/<stage>/`
and reads only from `data/`, `configs/` and previous stages' outputs.

**Frozen reference outputs**: `results/` ships with this repository and contains
the frozen benchmark outputs corresponding to the paper's final version
(OOF prediction parquet, metric CSV/JSON and paper table CSV). Reviewers can
verify the reported numbers directly against these files without re-running
the full 15-fold cross-validation; re-running `scripts/01`–`05` will locally
reproduce and update the files in that directory.

### Reproduced outputs

| Script | Key outputs |
|---|---|
| `01` | `results/stage1/dataset_case_level.csv`, `dataset_item_level.csv`, leakage check (4/4 PASS) |
| `02` | `results/stage2/jdi_weight_config.json`, `JDI_ranking.csv`, item strategy assignment |
| `03` | `results/stage3/` rule & baseline metrics (case-level CV) |
| `04` | `results/stage10_ultimate_sota/` — case-level OOF, `stage10_summary.csv` (MAE / RMSE / R² / MdAE of the final gated model vs Full-ML and Soft-Hybrid references), Pareto figure |
| `05` | `results/stage12_paper_updates/` — `table5_improvement_path.csv`, `table9_temporal_validation_2022to2023.csv` (2022 train → 2023 test), `table10_subgroup_error_analysis.csv` |

Notes:

- Table 5's first two levels (`S0 Plain LightGBM`, `S1 Structured LightGBM`)
  are recomputed inside this repository under the identical CV protocol; the
  remaining three levels read the case-level OOF produced by `04`.
- `04` hard-codes the frozen best settings: per-item model assignment
  (Stage-5 cross-validated) and `GAMMA_FINAL = 1.0`.

## Repository layout

```
GitHub_Release_Traffic_Compensation/
├── data/
│   └── dataset_anonymized.csv        # anonymized dataset (n=2380)
├── configs/                          # frozen experiment policies & strategies
├── src/
│   ├── rule_engine/                  # rule engine package (A-class rules, C-class fallback)
│   └── model_library.py              # shared frozen model factory / feature builder
├── scripts/
│   ├── 01_feature_engineering.py
│   ├── 02_jdi_computation.py
│   ├── 03_rule_engine_calibration.py
│   ├── 04_ultimate_gated_model.py
│   └── 05_generate_paper_tables.py
├── tests/
│   └── test_reproducibility.py       # fast sanity checks (data, leakage, OOF metrics)
└── results/                          # frozen reference outputs (shipped)
```

## License & citation

Released for academic reproducibility. If you use this code or dataset, please
cite the corresponding paper.

### BibTeX

```bibtex
@article{gu2026judicial,
  title   = {Judicial Support Rate Prediction for Traffic Accident Damage
             Compensation: A Rule--Machine Learning Continuous Fusion Framework
             Driven by the Judicial Determinacy Index},
  author  = {Gu, Liyuan and Miao, Mingyue and Huang, Lihua and Zhao, Fangxia},
  year    = {2026},
  note    = {Code and data: v1.0.0 (this repository)}
}
```

### Version

This release is **v1.0.0** (released 2026-10-15), matching the paper's
accepted/final version. See `CITATION.cff` for machine-readable citation
metadata and `LICENSE` for the layered code (MIT) / data (CC BY 4.0) terms.

---

<a id="中文说明"></a>

## 中文说明

本仓库是论文《司法确定性指数驱动的门控连续融合模型（交通事故赔偿支持率预测）》的
最简开源复现包。

- **任务重构**：由"判赔金额回归"重构为"支持率预测"——对 16 个法定赔偿分项预测
  `SR = 判赔 / 请求`（裁剪到 [0,1]），案件级支持率按请求额加权聚合。
- **JDI（司法确定性指数）**：逐分项统计构建的 0–100 指数（全额支持率、零支持率、
  支持率离散度/熵、众数密度、责任比例一致性），度量该分项受法律规则"确定"的程度，
  用于在规则先验与 ML 预测之间做静态连续融合。
- **认知不确定性门控**：以三头分位数回归的 `q90 − q10` 宽度作为实例级认知不确定性
  代理，融合权重按 `w_dyn = w_i · exp(−γ · delta)` 衰减；γ = 1.0 为冻结的最佳设定。
  模型不确定时自动回退到司法规则先验。
- **协议**：案件级 `RepeatedStratifiedKFold(5 折 × 3 重复, rs=42)`，分层 =
  `年份 × 审理程序 × 支持率五分位`；规则/JDI/模型全部训练折内拟合；指标 =
  案件级 **MAE / RMSE / R² / MdAE**。
- **数据脱敏**：`data/dataset_anonymized.csv`（n=2380）已删除案号、当事人、代理人、
  法院、文书路径/URL、车牌等全部直接标识符，仅保留宏观类别、清洗后的司法字段、
  分项请求/判赔金额与支持率。原始判决文书不随本仓库分发。

按顺序运行 `scripts/01 → 05` 即可一键复现论文中的指标与 Table 4 / 5 / 9 / 10。
