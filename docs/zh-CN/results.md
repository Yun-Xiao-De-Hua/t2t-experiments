# 结果字段与统计口径

`results/M{1,2,3}/{dataset}/P{n}.jsonl.gz` 每行一条最终有效样本，包含 `sample_id`、`condition_id`、`schema_version` 和以下七个对象。

| 记录类别 | 内容 |
|---|---|
| 01_experiment_configuration | 共享配置引用、token 数和上下文容量 |
| 02_answer_quality | 既有得分、生成、格式、复述与终止状态 |
| 03_module_parameters | 共享模型和通信模块参数 |
| 04_communication_volume | Generated/envelope/metadata/serialized bytes, KV capacity and receive verification |
| 05_sender_compute_cost | Tokenization, H2D, prefill/decode, packing and memory |
| 06_receiver_compute_cost | Local/new prefill, decode, unpacking and memory |
| 07_device_environment | Shared hardware/software reference, pipeline time and available partial failure metrics |

时间单位秒；显存/通信单位字节；token 字段为数量，得分为 0–1。共享引用指向 `metadata.json` 的 configurations、module_parameters、environments。原实验提示词 SHA 与整理后 prompts.json 文件 SHA 分开记录。未知、未完成或不适用的数值为 null，并尽可能保留状态，不用零冒充测量。

逐题数据不含问题、证据、参考答案、展开提示词、模型生成文本、token ID 序列或完整堆栈。真实输入输出仅存在于原始完整记录。新复现实验则在外部运行目录保存评分所需输出。

每方向 GSM8K P1/P4 各 1,319；HotpotQA P1/P3/P4 各 7,405，P2 为 7,345；HumanEval P1/P4 各 164。每方向 32,526 条，总计 97,578 条，正常生成 97,424、OOM 105、容量失败 49。失败仍计入固定分母，质量分数为零。

summary.csv 的 33 行是 24 条件的统计视图，不代表新增条件：HotpotQA 另外报告共同 7,345 题视图。跨协议比较使用共同子集；完整视图不能与 P2 直接当作同题比较。GSM8K 用最终数字 accuracy，HotpotQA 用 answer_em/answer_f1，HumanEval 用隔离执行 pass@1，P4 另有逐字复述合规率及合规且正确率。

汇总点估计与 95% bootstrap 区间直接复用既有统计（10,000 次重采样、seed 42），本轮没有重算；去除了无意义的自身比较列。耗时/显存来自共享 GPU 上的实际测量，不应据此作不受其他负载影响的因果性能比较。流水线时间不包括模型加载、预热、写盘、评分及调度；CUDA 时间与同步墙钟含义不同。
