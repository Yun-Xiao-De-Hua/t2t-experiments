# T2T 文本通信基线实验综述

## 1. 实验概况

最终冻结模板在 M1（0.6B→1.7B）、M2（0.6B→4B）、M3（4B→0.6B）间共享，模型均为 Qwen3。GSM8K 与 HumanEval 使用 P1/P4，HotpotQA 使用 P1–P4，共 24 条件。所有条件调用 Sender；P4 由 Sender 在通信中逐字复述任务，Receiver 不从程序另获原问题。

单张 RTX 3090、双模型常驻、BF16/SDPA、batch 1、seed 42、关闭 thinking、贪心解码。数学、问答与代码分别使用 accuracy、EM/F1、pass@1。

## 2. 覆盖与失败情况

| 方向 | 总样本 | 正常生成 | OOM | 容量失败 |
|---|---:|---:|---:|---:|
| M1 | 32,526 | 32,512 | 0 | 14 |
| M2 | 32,526 | 32,482 | 30 | 14 |
| M3 | 32,526 | 32,430 | 75 | 21 |
| Total | 97,578 | 97,424 | 105 | 49 |

覆盖完整不等于所有样本都正常生成。OOM/容量失败按既定规则计入固定分母并赋零分。表1展示原始与去格式后质量评分，单位为百分比；表2、表3展示通信量与时延，单位见列名。HotpotQA 使用共同 7,345 题以支持跨协议比较。

## 3. 测试结果

### 表 1. 原始评分与去格式后评分

![原始评分与去格式后评分](../../docs/assets/tables/table_1_zh-CN.png)

### 表 2. 通信量与分阶段计算时延（P50 / P95）

![通信量与分阶段计算时延（P50 / P95）](../../docs/assets/tables/table_2_zh-CN.png)

### 表 3. 通信量与分阶段计算时延（算术均值）

![通信量与分阶段计算时延（算术均值）](../../docs/assets/tables/table_3_zh-CN.png)

这些结果只支持当前模型对、模板与评测集合下的描述性比较，未进行新的显著性检验。

[原始汇总及置信区间](../../results/summary.csv)

## 4. Receiver-only 单模型对照

独立单模型对照另外评测 Qwen3-0.6B、1.7B、4B，每模型覆盖三个任务，共 9 条件。数学输入 Q，问答输入 Q+C，代码输入原始函数前缀 Q；一次完整 Prefill，无 Sender，通信字节量为 0。同一任务共享 `receiver_only_v1` 提示词。单模型对照与上述 24 条通信条件分别配置和运行。

26,664 题次全部完成，无 OOM 或容量失败。以下三表采用 HotpotQA 共同 7,345 题；全量 7,405 题统计同时保留在[独立汇总](../../results/receiver_only/summary.csv)。

### 表 R1. 原始评分与去格式后评分

![Receiver-only — 原始评分与去格式后评分](../../docs/assets/tables/receiver_only_1_zh-CN.png)

### 表 R2. 计算时延（P50 / P95）

![Receiver-only — 计算时延（P50 / P95）](../../docs/assets/tables/receiver_only_2_zh-CN.png)

### 表 R3. 计算时延（算术均值）

![Receiver-only — 计算时延（算术均值）](../../docs/assets/tables/receiver_only_3_zh-CN.png)

[复现、评分与比较口径](../../docs/zh-CN/receiver_only.md)。M1/M2/M3 分别对应 R17/R40/R06；完整 Prefill 与分段 Prefill、单模型与双模型显存范围不同。
