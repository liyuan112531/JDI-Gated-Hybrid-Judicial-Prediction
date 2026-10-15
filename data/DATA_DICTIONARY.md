# 数据字典与脱敏说明（Data Dictionary & De-identification Notes）

- 适用文件：`data/dataset_anonymized.csv`（n = 2,380，UTF-8-SIG 编码）
- 版本：1.0.0（2026-10-15，与 CITATION.cff 同步）
- 数据许可：CC BY 4.0（见根目录 `LICENSE`）

---

## 1. 数据来源与范围

- **来源**：北京市 2022–2023 年 2,380 份生效的机动车交通事故责任纠纷民事判决书
  （一审/二审），经结构化抽取形成本表。
- **时间分布**：2022 年 2,074 份、2023 年 346 份。
- **原始文书**：判决书原始文本（docx）**不随本仓库分发**。本表仅保留统计
  结构化字段，不含可回溯定位具体文书的任何信息。

## 2. 脱敏与隐私保护协议

在结构化过程中，以下直接标识符已被**全部移除**，且不以任何形式保留在本仓库中：

| 已移除项 | 说明 |
|---|---|
| 案号 | 如 "(2022)京0105民初xxxx号" |
| 当事人姓名 | 原告、被告、第三人 |
| 代理人信息 | 律师、法律工作者 |
| 具体基层法院名称 | 仅保留宏观地区（单一地区：北京） |
| 文书 URL / 文件路径 / SHA256 指纹 | 原始文书的可定位线索 |
| 车牌号、事故地点 | 事故细节标识 |

保留的 `case_id` 为**顺序代理编号**（C0000…C2379），与原始案号无任何
可逆对应关系。逐案保留的金额与比例属于裁判文书依法公开的事实信息，
经上述脱敏处理后，个体再识别风险可忽略。

## 3. 字段分类总表

### 3.1 标识与基础元数据（判前可得）

| 字段 | 类型 | 含义 | 判前时点 |
|---|---|---|---|
| `case_id` | str | 案件顺序代理编号（C0000–C2379） | —（代标识） |
| `year` | int | 判决年份（2022 / 2023） | 判前已知（诉讼年度） |
| `procedure_type` | str | 审理程序：一审 / 二审 / 再审 | 判前已知 |

### 3.2 判前司法认定特征（Pre-adjudication Judicial Findings）

| 字段 | 类型 | 含义 | 取值与时点说明 |
|---|---|---|---|
| `liability_ratio` | float | 交通事故责任比例（0–1） | 源自公安机关交管部门诉前出具的《道路交通事故认定书》（《道路交通安全法》第 73 条），判前客观已生成。**注意**：18.24%（434/2,380）为有效正值，其余 81.76% 编码为 0，属文书非结构化叙述导致的"信息抽取缺失"而非被告无责（零值组案件级支持率均值 0.6689，与"被告无责→支持率趋近 0"矛盾）。特征工程中拆解为 `liability_ratio_clean`（0→NaN）+ `liability_missing` 缺失标记，填补统计量仅在训练折内拟合；两种编码策略案件级 MAE 差异仅 0.0001（论文第 3.2 / 5.5 节）。详见 `configs/liability_extraction_decision.md` 第 7 节。 |
| `disability_level` | float | 伤残等级（0 = 无伤残；1–10 级，1 级最重、10 级最轻；25 = 异常值） | 由司法鉴定机构出具《司法鉴定意见书》确定，属判前/质证环节可得信息。特征工程中 25 → NaN + 异常标记（仅 1 例），缺失不删样本，双轨编码（有序数值 / 独热）。详见 `configs/disability_strategy.json`。 |

### 3.3 16 项法定赔偿项目的请求与判赔金额

- `*_claim`（16 项，判前入模特征）：原告在起诉状/庭审中主张的赔偿金额。
- `*_award`（16 项，**判后标签**）：法院对各分项的判定金额，严格列入特征
  黑名单（`configs/feature_blacklist.json`），仅用于构造标签与评价，
  任何情况下不得进入特征矩阵（Stage 1 起自动校验，4/4 PASS）。

| 分项（suffix） | 中文名称 |
|---|---|
| `medical` | 医疗费 |
| `future_medical` | 后续治疗费 |
| `hospital_meal` | 住院伙食补助费 |
| `nutrition` | 营养费 |
| `nursing` | 护理费 |
| `lost_income` | 误工费 |
| `transport` | 交通费 |
| `accommodation` | 住宿费 |
| `disability_comp` | 残疾赔偿金 |
| `assistive_device` | 残疾辅助器具费 |
| `death_comp` | 死亡赔偿金 |
| `funeral` | 丧葬费 |
| `mental_solatium` | 精神损害抚慰金 |
| `property_loss` | 财产损失 |
| `appraisal` | 鉴定费 |
| `other` | 其他费用 |

各分项有效主张样本量（claim > 0）与支持率中位数见
`configs/project_strategy_config.json` 的 `item_inventory`。

### 3.4 汇总指标（判后标签与评价）

| 字段 | 类型 | 含义 | 角色 |
|---|---|---|---|
| `claim_total` | float | 请求总额 = Σ `*_claim` | 判前特征 |
| `award_total` | float | 判赔总额 = Σ `*_award` | **判后**，仅用于审计/标签 |
| `support_rate` | float | 案件级支持率 = `award_total / claim_total` ∈ [0,1] | **最终评价标签**（黑名单） |
| `support_rate_pct` | float | 同上 ×100 | **最终评价标签**（黑名单） |

### 3.5 防泄漏硬校验

`scripts/01_feature_engineering.py` 对上述黑名单字段执行自动校验，
结果固化于 `results/stage1/stage1_leakage_check.json`（4/4 PASS）：
任何落入黑名单的字段（`*_award`、`award_total`、`support_rate`、
`support_rate_pct`、`n_nonzero_items`）不得进入特征矩阵。

## 4. 判前时点界定与程序法理说明（Pre-adjudication Timing Justification）

> 本节与 `configs/liability_extraction_decision.md` 第 7 节同步维护。

### 4.1 预测时点定义

本研究定位于**判决作出前（Pre-judgment / Pre-adjudication decision support）**
的司法辅助场景：庭审举证、质证与辩论程序完毕、法官落槌确定各分项判赔金额
**之前**。它不是"原告起诉立案时点（Filing time）"——因此在起诉后、判决前
进入诉讼的证据与司法认定（如鉴定意见），在本时点定义下属于合法输入。

### 4.2 程序法依据

1. **责任比例**：《道路交通事故认定书》由公安机关交通管理部门在诉前出具
   （《道路交通安全法》第 73 条），起诉前即已生成，属客观已存在的判前信息。
2. **伤残等级**：《司法鉴定意见书》由司法鉴定机构在诉前或庭审质证前出具
   （《民事诉讼法》证据种类之"鉴定意见"），在判决作出前可得。

### 4.3 缺失治理与稳健性

`liability_ratio` 的大面积编码缺失（81.76%）属"实质性缺失
（Substantive Missingness）"，其证据链与处理方案见第 3.2 节与
`configs/liability_extraction_decision.md`；主策略与敏感性策略的
案件级 MAE 差异仅 0.0001，研究结论对编码选择稳健。

## 5. 复现入口

```bash
python scripts/01_feature_engineering.py   # 由本表构建案件级/分项级数据集 + 泄漏检查
python tests/test_reproducibility.py       # 快速核验（数据规模 / 泄漏 4/4 PASS / OOF 指标）
```
