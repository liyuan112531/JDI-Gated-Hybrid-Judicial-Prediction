# TabPFN 处理策略（TabPFN Policy）

- 版本：1.0
- 生成阶段：Stage 0 决策修正
- 当前状态：`AVAILABLE_IMPORT_OK__NO_LOCAL_WEIGHT_CACHE__DOWNLOAD_REQUIRED_BUT_BLOCKED`

---

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| 包安装 | `tabpfn 2.1.4`（`d:/anaconda3`，import 正常） |
| 本地权重缓存 | **无**（未发现 `~/.cache/huggingface/hub` 下的 TabPFN 权重目录） |
| 权重下载 | **未执行**（Stage 0 脚本强制 `HF_HUB_OFFLINE=1`） |
| 实例化模型 | **未执行** |

依据：`results/environment/environment_snapshot.txt`

## 2. 当前阶段不下载权重的决定与理由

**决定：Stage 0 – Stage 3 期间不下载 TabPFN 权重。**

1. **TabPFN 只是候选模型之一**，实验原则明确禁止提前假设 TabPFN 一定最好，因此不应为单一候选模型提前投入下载与环境调试成本。
2. **避免环境风险**：权重下载与依赖变更可能引入新的依赖冲突，威胁 base 环境（`numpy 1.26.4` 必须保持不变）。
3. **研究顺序要求**：研究方案要求先完成数据构造（Stage 1）、JDI 分析（Stage 2）、规则引擎与基线（Stage 3），再进入模型横评（Stage 4）；在尚未确定任务形式（金额预测 vs 支持率预测）与分项归属之前，TabPFN 的接入方式也无法确定。

## 3. Stage 4 模型横评的前置条件

在申请启用 TabPFN 之前，必须先完成：

1. Stage 1：判前特征与标签构造（含责任比例、伤残等级的三字段方案）
2. Stage 2：JDI 分析与 A/B/C 三级项目划分
3. Stage 3：规则引擎与强基线体系
4. Stage 4 第一阶段：传统模型（Linear / Ridge / RF / XGBoost / LightGBM / CatBoost 等）的横向比较，形成可对照的基线成绩

## 4. 启用条件与申请流程

当且仅当满足以下条件时，向用户申请联网下载：

1. 传统模型横评已完成，存在可对照的性能基线；
2. 已确定任务形式（任务 A 金额预测 / 任务 B 支持率预测，或两者并行）；
3. 已确定参与建模的项目集合及其样本量（TabPFN 对样本量与特征数存在上下文长度限制）；
4. 已评估运行时间预算（TabPFN 在 CPU 上推理成本较高，需评估 5 折 × 3 次重复的可行性）。

申请需明确：下载内容（模型权重仓库与版本）、缓存位置、预计体积、是否写入 `environment_snapshot.txt`。

## 5. 不可用时的处理

若在 Stage 4 结束时仍未获得授权或权重不可用：

- 在模型横评表中将 TabPFN 标注为 `unavailable (weights not downloaded)`，**不得留白或删除该行**；
- 在论文实验章节说明该模型因权重未获取而未参与比较，并说明这不影响"不预先指定模型"的原则——其余候选模型的比较仍然完整；
- 在局限性章节说明结论未覆盖 TabPFN。

## 6. 禁止事项

- 禁止在未获得用户授权的情况下触发任何权重下载
- 禁止为适配 TabPFN 修改 `numpy` 版本或破坏 base 环境
- 禁止在横评前宣称或暗示 TabPFN 为预期最优模型
- 禁止用 TabPFN 的缺失作为推迟其他模型实验的理由

---

关联配置：`configs/experiment_policy.yaml`（`tabpfn_policy` 段）
