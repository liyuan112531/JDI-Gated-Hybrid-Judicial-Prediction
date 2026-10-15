# Stage 2 司法确定性指数（JDI）与项目分层分析 —— 阶段报告

生成时间：2026-10-10T16:50:39  
阶段约束：**未训练任何机器学习模型、未做模型比较、未调参**；仅使用描述统计、相关分析与互信息估计。

## 1. JDI 定义

司法确定性指数（Judicial Determinacy Index）衡量某一赔偿项目「判决结果由稳定规则决定的程度」。
JDI 越高 → 支持率越集中、越可归因于客观规则 → 适合规则计算；
JDI 越低 → 支持率越分散、裁量空间越大 → 适合数据驱动预测。

构造方式：9 项基础指标 → 选取 7 项进入指数 → 各分量跨 16 个项目 min-max 归一化 → 负向指标取 `1 - x` → 加权求和 → 线性映射到 0-100。

## 2. 指标解释

| 指标 | 含义 | 方向 | 权重 |
|---|---|---|---:|
| P(SR=1) | 全额支持率占比 | 正向（确定性增强） | 0.20 |
| P(SR=0) | 全额驳回率占比 | 正向（极端结果集中=规则稳定） | 0.10 |
| SR_mode_density | 众数密度（10 箱最高占比） | 正向 | 0.20 |
| liability_consistency | 支持率与责任比例的相关性（子集 n≥30 时启用，否则中性值 0.5） | 正向 | 0.10 |
| SR_entropy_norm | SR 分布熵（10 箱，归一化） | 负向 | 0.15 |
| std_SR | SR 标准差 | 负向 | 0.15 |
| IQR_SR | SR 四分位距 | 负向 | 0.10 |

**未进入指数的诊断指标**（避免方向不可解释或不可比）：

- `claim_SR_MI`：反映「可预测性」而非「确定性」，高 MI 也可能来自复杂裁量，仅作诊断；
- `disability_explanatory_power`：仅伤残相关项目适用，纳入会引入项目间不可比性；
- `CV_SR`：与 std_SR 共线；`legal_rule_strength`：法律先验用于分层决策而非指数本身。

## 3. 16 项目 JDI 排名

| 排名 | 项目 | 样本量 | JDI | P(SR=1) | P(SR=0) | 熵(归一化) | 众数密度 | std | 法律先验 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | appraisal | 843 | 78.49 | 0.909 | 0.055 | 0.156 | 0.922 | 0.234 | evidence_based(0.75) |
| 2 | hospital_meal | 967 | 73.12 | 0.819 | 0.030 | 0.269 | 0.874 | 0.208 | evidence_based(0.75) |
| 3 | disability_comp | 614 | 67.30 | 0.559 | 0.021 | 0.315 | 0.839 | 0.178 | formula_based(1.0) |
| 4 | medical | 1511 | 62.59 | 0.594 | 0.039 | 0.347 | 0.830 | 0.250 | evidence_based(0.75) |
| 5 | death_comp | 78 | 62.41 | 0.577 | 0.051 | 0.402 | 0.769 | 0.254 | formula_based(1.0) |
| 6 | assistive_device | 528 | 62.21 | 0.725 | 0.062 | 0.411 | 0.782 | 0.295 | evidence_based(0.75) |
| 7 | future_medical | 172 | 57.30 | 0.134 | 0.826 | 0.234 | 0.826 | 0.363 | conditional(0.3) |
| 8 | funeral | 78 | 51.05 | 0.474 | 0.038 | 0.539 | 0.641 | 0.242 | formula_based(1.0) |
| 9 | accommodation | 104 | 45.73 | 0.231 | 0.663 | 0.445 | 0.663 | 0.431 | hybrid(0.4) |
| 10 | other | 368 | 31.33 | 0.283 | 0.582 | 0.468 | 0.611 | 0.452 | discretion_based(0.1) |
| 11 | nutrition | 1281 | 25.82 | 0.375 | 0.045 | 0.826 | 0.393 | 0.325 | hybrid(0.4) |
| 12 | nursing | 1245 | 22.35 | 0.230 | 0.054 | 0.891 | 0.316 | 0.301 | hybrid(0.4) |
| 13 | property_loss | 1500 | 21.44 | 0.322 | 0.110 | 0.870 | 0.373 | 0.359 | evidence_based(0.6) |
| 14 | mental_solatium | 1079 | 18.12 | 0.207 | 0.319 | 0.792 | 0.331 | 0.381 | discretion_based(0.1) |
| 15 | transport | 1598 | 17.46 | 0.273 | 0.068 | 0.897 | 0.298 | 0.332 | hybrid(0.4) |
| 16 | lost_income | 1411 | 12.85 | 0.166 | 0.181 | 0.945 | 0.218 | 0.357 | hybrid(0.4) |

数据驱动分界（JDI 排序最大间隙）：**38.53**（间隙 14.40）

## 4. 分层结果

- **Rule-based（A 类）**：5 项 —— appraisal, assistive_device, disability_comp, hospital_meal, medical
- **ML-based（B 类）**：7 项 —— lost_income, mental_solatium, nursing, nutrition, other, property_loss, transport
- **Small-sample（C 类）**：4 项 —— accommodation, death_comp, funeral, future_medical

| 项目 | 样本量 | JDI | 分类 | A/B 建议 | 判定依据 |
|---|---:|---:|---|---|---|
| appraisal | 843 | 78.49 | **Rule-based** | Rule-based | JDI=78.49 高于分界 38.53，且法律规则强度 0.75（鉴定费：凭鉴定费票据据实计算） |
| hospital_meal | 967 | 73.12 | **Rule-based** | Rule-based | JDI=73.12 高于分界 38.53，且法律规则强度 0.75（住院伙食补助费：固定日标准 × 住院天数） |
| disability_comp | 614 | 67.30 | **Rule-based** | Rule-based | JDI=67.30 高于分界 38.53，且法律规则强度 1.0（残疾赔偿金：受诉法院所在地上一年度人均可支配收入 × 赔偿年限 × 伤残系数，法定公式） |
| medical | 1511 | 62.59 | **Rule-based** | Rule-based | JDI=62.59 高于分界 38.53，且法律规则强度 0.75（医疗费：凭医疗费票据、病历据实计算） |
| death_comp | 78 | 62.41 | **Small-sample** | Rule-based | 样本量 78 < 100（Stage 2 规则）；已列入 configs/small_sample_policy.md 与 experiment_policy.yaml 的小样本清单 → 标记为 Small-sample；不因样本量自动判为规则项目，其 A/B 建议为 Rule-based（JDI=62.41 高于分界 38.53，且法律规则强度 1.0（死亡赔偿金：人均可支配收入 × 法定年限，法定公式）） |
| assistive_device | 528 | 62.21 | **Rule-based** | Rule-based | JDI=62.21 高于分界 38.53，且法律规则强度 0.75（残疾辅助器具费：配置机构意见 + 普通适用器具标准） |
| future_medical | 172 | 57.30 | **Small-sample** | Rule-based | 已列入 configs/small_sample_policy.md 与 experiment_policy.yaml 的小样本清单 → 标记为 Small-sample；不因样本量自动判为规则项目，其 A/B 建议为 Rule-based（[冲突] JDI=57.30（分界 38.53）与法律规则强度 0.3 不一致，以数据驱动 JDI 为准判为 Rule-based） |
| funeral | 78 | 51.05 | **Small-sample** | Rule-based | 样本量 78 < 100（Stage 2 规则）；已列入 configs/small_sample_policy.md 与 experiment_policy.yaml 的小样本清单 → 标记为 Small-sample；不因样本量自动判为规则项目，其 A/B 建议为 Rule-based（JDI=51.05 高于分界 38.53，且法律规则强度 1.0（丧葬费：受诉法院所在地上一年度职工月平均工资 × 6 个月，法定标准）） |
| accommodation | 104 | 45.73 | **Small-sample** | Rule-based | 已列入 configs/small_sample_policy.md 与 experiment_policy.yaml 的小样本清单 → 标记为 Small-sample；不因样本量自动判为规则项目，其 A/B 建议为 Rule-based（[冲突] JDI=45.73（分界 38.53）与法律规则强度 0.4 不一致，以数据驱动 JDI 为准判为 Rule-based） |
| other | 368 | 31.33 | **ML-based** | ML-based | JDI=31.33 低于分界 38.53，且法律规则强度仅 0.1（其他费用：兜底项目，无明确计算标准，裁量空间大） |
| nutrition | 1281 | 25.82 | **ML-based** | ML-based | JDI=25.82 低于分界 38.53，且法律规则强度仅 0.4（营养费：参照医疗机构意见酌定） |
| nursing | 1245 | 22.35 | **ML-based** | ML-based | JDI=22.35 低于分界 38.53，且法律规则强度仅 0.4（护理费：护理人员收入 × 护理期限，期限与标准存在酌定） |
| property_loss | 1500 | 21.44 | **ML-based** | ML-based | [冲突] JDI=21.44（分界 38.53）与法律规则强度 0.6 不一致，以数据驱动 JDI 为准判为 ML-based |
| mental_solatium | 1079 | 18.12 | **ML-based** | ML-based | JDI=18.12 低于分界 38.53，且法律规则强度仅 0.1（精神损害抚慰金：无计算公式，依伤残等级、过错程度、当地生活水平酌定） |
| transport | 1598 | 17.46 | **ML-based** | ML-based | JDI=17.46 低于分界 38.53，且法律规则强度仅 0.4（交通费：凭票据，但合理性由法院酌定） |
| lost_income | 1411 | 12.85 | **ML-based** | ML-based | JDI=12.85 低于分界 38.53，且法律规则强度仅 0.4（误工费：收入证明 × 误工时间，举证不足时法院酌定） |

分层规则：① Small-sample 判定 = `sample_size < 100`（Stage 2 规则）**或**已列入 Stage 0/1 冻结的小样本清单（death_comp 78、funeral 78、accommodation 104、future_medical 172）——两者取并集，保证与已冻结配置一致；Small-sample 仅作标记，A/B 建议仍给出，不因样本量自动判为规则项目；② JDI 高于数据驱动分界且法律规则强度 ≥ 0.5 → Rule-based；③ JDI 低于分界且规则强度 < 0.5 → ML-based；④ 冲突时以 JDI 为准并记录冲突。

## 5. 精神损害抚慰金专项分析

- 有效样本：**1079** 例；均值 0.4319、中位数 0.5000、标准差 0.3812；全额支持 20.67%、全额驳回 31.88%
- 伤残等级 Spearman：**0.6478**（p=2.74e-129），有伤残 vs 无伤残 SR 均值差 **0.5104**
- 责任比例 Spearman：**0.0036**（子集 n=266）
- 请求金额 Pearson **-0.0717** / Spearman **0.1390** / MI **0.4513**

详见 `results/stage2/mental_solatium_analysis.md`。

## 6. 下一阶段实验建议

1. **Stage 3 规则引擎**：为 Rule-based 项目建立判前可算的规则公式（建议以分项支持率中位数、责任比例、法定系数为主），并在测试折上评估其误差上界。
2. **Stage 4 建模范围**：ML-based 项目参与主要模型排名；Small-sample 项目仅做探索性分析，不参与排名；需补充「含/不含 procedure_type」与「含/不含责任比例」的特征集对照。
3. **缺失机制**：责任比例子集在多数分项上样本不足（n<30），Stage 4 需显式检验`liability_missing` 的预测贡献。
4. **冲突项目复核**：JDI 与法律先验冲突的项目需在论文中说明判定依据（已记录在 `item_strategy_assignment.csv`）。
5. **敏感性结论**：权重方案与阈值变化的排序稳定性见 `jdi_sensitivity.csv`，论文应报告该稳健性结果。

## 7. 本阶段产物

| 文件 | 说明 |
|---|---|
| `results/stage2/JDI_ranking.csv` | 16 项目 JDI 排名与全部分量得分 |
| `results/stage2/jdi_weight_config.json` | 指数权重与方向定义、排除项说明 |
| `results/stage2/item_strategy_assignment.csv` | 分层结果与判定依据 |
| `results/stage2/project_strategy_assigned.json` | 三类项目列表（回填策略配置） |
| `results/stage2/mental_solatium_analysis.md` | 精神损害抚慰金专项分析 |
| `results/stage2/jdi_sensitivity.csv` | 权重与阈值敏感性分析 |
| `results/stage2/figures/fig1_sr_boxplot.png` | 16 项目 SR 箱线图 |
| `results/stage2/figures/fig2_sr_density.png` | 16 项目 SR 密度图 |
| `results/stage2/figures/fig3_jdi_ranking.png` | JDI 排名柱状图 |
| `results/stage2/figures/fig4_jdi_scatter.png` | 样本量 × JDI 散点图 |
| `results/stage2/figures/fig5_strategy_map.png` | 项目策略分层图 |
| `results/reports/stage2_summary.md` | 本报告 |

> 本阶段未训练任何模型、未比较模型、未调参，已完成 Stage 2 全部任务并停止。