# 01 判前特征工程与标签构建 —— 阶段报告

生成时间：2026-10-10T16:50:28  
阶段约束：**禁止训练模型 / 禁止模型比较 / 禁止调参**；仅完成特征构建、标签构造、案件级与分项级转换、数据泄漏检查。

## 1. 数据转换流程

```
data/dataset_anonymized.csv  (2380 行 × 41 列，只读，已脱敏)
        |
        |-- 读取配置：experiment_policy.yaml / liability_strategy.json /
        |              disability_strategy.json / feature_blacklist.json
        |
        |-- 案件基础信息：case_id / year / procedure_type（脱敏表宏观类别列）
        |
        |-- 责任比例：raw -> clean(raw==0 -> NaN) -> missing 指示（不删除、不填补）
        |-- 伤残等级：raw -> clean(25 -> NaN) -> abnormal / missing 指示（不删除）
        |
        |-- 请求结构特征：claim_total_calc / n_nonzero_claims / claim_share_i / log_claim_i
        |
        |-- 案件级数据集 dataset_case_level.csv
        |-- 分项级长表 dataset_item_level.csv（案件 × 16 项目）
        |-- 标签：support_rate_label（分项级） / sr_total_label（案件级）
        |
        +-- 泄漏检查（4 项）-> 全部 PASS
```

## 2. 最终案件级特征数量

- 案件级数据集：`dataset_case_level.csv`，**2380 行 × 64 列**
- 可入模的判前特征列：**57** 列
  - 案件基础信息：2（`year`、`procedure_type`）
  - 判前司法特征：5（`liability_ratio_clean`、`liability_missing`、`disability_level_clean`、`disability_abnormal`、`disability_missing`）
  - 16 项请求金额：16
  - 请求结构特征：18（`claim_total_calc`、`n_nonzero_claims`、16 项 `claim_share_i`） + 16 项 `log_claim_i`
- 分项级可入模特征列：**60** 列（含案件级特征 + `claim_amount` / `item_claim_share` / `item_log_claim`）

## 3. 最终分项级样本数量

- 分项级长表：`dataset_item_level.csv`，**38080 行 × 70 列**
- 构成：2380 案件 × 16 项目 = 38080 行（**全部保留**，未删除任何项目或样本）
- 其中存在有效支持率标签（claim > 0）的行：**13377** 行（占长表 35.13%）

## 4. 有效预测项目样本量

| 项目 | claim>0 | mean_SR | std_SR | median_SR | zero_SR_ratio |
|---|---:|---:|---:|---:|---:|
| medical | 1511 | 0.8999 | 0.2500 | 1.0000 | 0.0390 |
| future_medical | 172 | 0.1629 | 0.3629 | 0.0000 | 0.8256 |
| hospital_meal | 967 | 0.9304 | 0.2081 | 1.0000 | 0.0300 |
| nutrition | 1281 | 0.6645 | 0.3252 | 0.6667 | 0.0453 |
| nursing | 1245 | 0.6829 | 0.3005 | 0.7500 | 0.0538 |
| lost_income | 1411 | 0.5209 | 0.3569 | 0.5395 | 0.1807 |
| transport | 1598 | 0.5907 | 0.3319 | 0.5684 | 0.0676 |
| accommodation | 104 | 0.2829 | 0.4310 | 0.0000 | 0.6635 |
| disability_comp | 614 | 0.9214 | 0.1785 | 1.0000 | 0.0212 |
| assistive_device | 528 | 0.8656 | 0.2945 | 1.0000 | 0.0625 |
| death_comp | 78 | 0.8780 | 0.2542 | 1.0000 | 0.0513 |
| funeral | 78 | 0.8628 | 0.2421 | 0.9697 | 0.0385 |
| mental_solatium | 1079 | 0.4319 | 0.3812 | 0.5000 | 0.3188 |
| property_loss | 1500 | 0.6294 | 0.3590 | 0.6933 | 0.1100 |
| appraisal | 843 | 0.9356 | 0.2338 | 1.0000 | 0.0546 |
| other | 368 | 0.3392 | 0.4525 | 0.0000 | 0.5815 |

- 小样本项目（`small_sample = true`）：death_comp / funeral / accommodation / future_medical（按 `configs/small_sample_policy.md`，仅探索性分析，不参与主要模型排名）

## 5. 缺失处理情况

| 字段 | 缺失数 | 缺失率 | 处理方式 |
|---|---:|---:|---|
| `liability_ratio_clean` | 1946 | 81.76% | **不删除、不填补**；生成 `liability_missing` 指示变量，填补推迟到 Pipeline 内按训练折拟合 |
| `disability_level_clean` | 2 | 0.08% | 不删除；生成 `disability_missing` / `disability_abnormal` |
| `procedure_type` | 0 | 0.00% | 留空并记录 |

## 6. 标签定义

- 分项级：`support_rate_label = award_i / claim_i`，**仅当 `claim_i > 0` 时定义**，其余为 NaN（不参与训练，案件聚合时按 0 计入）。
- 案件级：`sr_total_label = sum(award_i) / sum(claim_i)`，为最终系统评价目标。
- 取值范围校验：0 ≤ SR ≤ 1，越界记录 **0** 条（`sr_validation.csv`）。
- 与原始 `support_rate` 一致性：2380/2380 条一致，最大差异 0.00000050（`case_sr_validation.csv`）。

## 7. 泄漏检查结果

| 检查项 | 结果 |
|---|---|
| C1_blacklist_column_scan（特征列黑名单扫描） | **PASS** |
| C2_provenance_declaration（特征来源声明检查） | **PASS** |
| C3_award_perturbation_invariance（判后字段扰动不变性检验） | **PASS** |
| C4_label_feature_separation（标签与特征分离检查） | **PASS** |

**总体结论：PASS**

- C1：特征列中不存在 `*_award`、`award_total`、`support_rate`、`support_rate_pct`、`n_nonzero_items`、`text_len`。
- C2：所有允许入模特征的来源声明均不含判后字段。
- C3：将所有判后字段随机扰动后重建特征矩阵，结果与原始特征矩阵**完全一致**，证明特征不依赖任何判后信息。
- C4：标签列（`sr_total_label`、`award_amount`、`support_rate_label`）均未出现在特征集合中。

## 8. 本阶段产物

| 文件 | 说明 | 规模 |
|---|---|---|
| `results/stage1/feature_schema_stage1.csv` | 全字段审计表 | 92 行 |
| `results/stage1/liability_stage1_report.csv` | 责任比例统计 | 30 行 |
| `results/stage1/disability_stage1_report.csv` | 伤残等级特征统计 | 16 行 |
| `results/stage1/dataset_case_level.csv` | 案件级数据集 | 2380 × 64 |
| `results/stage1/dataset_item_level.csv` | 分项级长表 | 38080 × 70 |
| `results/stage1/sr_validation.csv` | 分项支持率校验 | 18 行 |
| `results/stage1/case_sr_validation.csv` | 案件级支持率校验 | 2380 行 |
| `results/stage1/item_distribution_base.csv` | 项目级统计基础表（02 用） | 16 行 |
| `results/stage1/stage1_leakage_check.json` | 泄漏检查结果 | 4 项检查 |

> 本阶段未训练任何模型、未做模型比较、未调参。