# Receiver-only 单模型对照复现

Qwen3-0.6B（R06）、1.7B（R17）、4B（R40）各自独立求解 GSM8K、HotpotQA、HumanEval，共 9 条件、26,664 题次。R06/R17/R40 表示独立模型，不表示 M1–M3 通信方向。没有新增权重训练。

## 实验设置

| 任务 | 模型实际输入 | 每模型正式题数 | 指标 |
|---|---|---:|---|
| GSM8K | 原题 Q | 1,319 | accuracy |
| HotpotQA | 原题 Q 与完整证据 C | 7,405 | EM / F1 |
| HumanEval | 原始 Python 函数前缀 Q | 164 | pass@1 |

提示词见 [receiver_only.prompts.json](../../configs/receiver_only/prompts.json)。任务输出指令沿用通信基线；System 移除上游协助措辞。同一任务跨三个模型共享。输入仅为 System + User；User 由 `<TASK>`、可选 `<EVIDENCE>` 和任务输出指令构成。不输入参考答案或测试代码。

单张 GPU、一次常驻一个模型、BF16/SDPA、batch 1、seed 42、2 CPU 线程、关闭 thinking、原手工贪心解码器、生成至 EOS 或真实剩余上下文容量，不截断输入。输入一次完整 Prefill，不执行通信实验的分段 Prefill。
## 准备、运行与评分

先按[使用说明](usage.md)安装依赖并配置 `configs/paths.local.json` 的模型、datasets、prepared 路径。Receiver-only 不读取 `warmup_datasets`。模型和数据 revision 沿用已有冻结设置。prepare 只在 prepared 不存在时执行；已有同版本 prepared 可复用，勿覆盖。

```bash
t2t-experiments --config configs/receiver_only/experiment.json prepare
CUDA_VISIBLE_DEVICES=3 t2t-experiments --config configs/receiver_only/experiment.json run \
  --wait-memory --tmux receiver_only --output /srv/t2t/runs/receiver_only
```

GPU 由外部环境选择，程序使用第一张可见卡 `cuda:0`，不绑定编号或 UUID。示例中的 3 可替换为分配给你的 GPU。可重复提供 `--model Qwen/Qwen3-0.6B` 和 `--dataset gsm8k` 筛选；不使用 `--direction` 或 `--protocol`。显存等待按所选模型的 BF16 权重估算加 2 GiB 工作空间，15 秒轮询，连续严格超过 180 秒并复查后加载。

```bash
# Resume with the same config, filters, GPU environment and output; add --resume.
CUDA_VISIBLE_DEVICES=3 t2t-experiments --config configs/receiver_only/experiment.json run \
  --wait-memory --resume --output /srv/t2t/runs/receiver_only
t2t-experiments --config configs/receiver_only/experiment.json status --run-dir /srv/t2t/runs/receiver_only
t2t-experiments --config configs/receiver_only/experiment.json score \
  --run-dir /srv/t2t/runs/receiver_only --output /srv/t2t/runs/receiver_only_scores
python -B -m t2t_experiments.receiver_only.verify
```

run 保存原始输出及七类提交记录，不在推理热路径评分；score 在 CPU 上生成原始/去格式逐题评分及 12 行汇总。HumanEval 评分通过 bubblewrap 隔离，失败则停止，不计为模型错误。OOM/容量失败保留记录并计零分，不原地重试；连续十次 OOM 冷却 60 秒。断点签名变化拒绝续跑。整理后的运行目录格式不用于原地续跑历史归档。

## 结果与比较口径

全部 26,664 题次已完成，无 OOM 或容量失败。[汇总结果](../../results/receiver_only/summary.csv)包含 9 个全量视图及 3 个 HotpotQA 共同 7,345 题视图；三线表采用共同子集，复用全量输出。

原始与去格式得分使用相同固定分母。去格式仅清理答案标签、外层标记等格式，不修改提示词或求解内容；两种得分均保留。

时延仅统计 CUDA event 测得的完整 Prefill 与 Decode，不含加载、分词、传输、评分、写盘或等待；总时延逐题相加后计算均值/P50/P95。通信量为 0。

按 Receiver 对齐：M1→R17、M2→R40、M3→R06。P1 最接近完整信息对照；P2/P3 的本地证据不同，差异不能只归因于通信。共享 GPU 下的时延仅作描述性比较，单模型与双模型常驻的显存口径不同。
