# 使用与复现

使用独立 Linux Python 3.11 环境，按已记录依赖安装；需要兼容 CUDA 12.8 的驱动及 bubblewrap。以下步骤是供后续复现使用的命令。

```bash
cd /path/to/t2t_experiments
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
cp configs/paths.example.json configs/paths.local.json
```

编辑 `paths.local.json`，配置外部模型、原始数据、非测试预热题、prepared 和 runs 路径。相对路径以声明它的配置文件为基准。模型须使用固定 revision 的完整本地快照，包括 tokenizer 和 safetensors；本代码不下载模型。

原始数据采用 UTF-8 JSONL，可 gzip：GSM8K 每行 `question, answer`；HotpotQA 每行 `_id, question, answer, context, supporting_facts`，context 为 `[title, sentences]`；HumanEval 每行 `task_id, prompt, canonical_solution, test, entry_point`。数据 revision 和数量见 experiment.json。预热使用 GSM8K train、HotpotQA train 与人工合成代码题，各至少三条；请勿使用正式测试题，HotpotQA 预热题须满足 P2 分配条件。

```bash
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json prepare
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json run \
  --direction M1 --wait-memory --tmux t2t_m1 \
  --output /srv/t2t/runs/m1
```

GPU 由外部 `CUDA_VISIBLE_DEVICES` 指定，程序使用第一张可见卡 `cuda:0`。省略方向/任务/协议筛选参数即可运行全部 24 条件；多次提供参数可选择多个值。P1–P3 优先于 P4。可选显存等待：M1 至少 6 GiB、M2/M3 至少 10.5 GiB，15 秒轮询、连续严格超过 180 秒后复查；实际资源仍可能发生变化，单题 OOM/容量失败记录后继续。

```bash
# Resume: repeat the same selection, GPU, configuration and output; add --resume.
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json run \
  --direction M1 --wait-memory --resume --output /srv/t2t/runs/m1
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json status \
  --run-dir /srv/t2t/runs/m1
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json score \
  --run-dir /srv/t2t/runs/m1 --output /srv/t2t/runs/m1_scores
```

首次 prepare/run/score 使用不存在的新目录。断点以七类记录提交为准，签名变化拒绝原地续跑；已提交 OOM 不自动重试，需另行设计补测批次。status 只读，无需 GPU。HumanEval 只在 bubblewrap 隔离环境中运行，探测失败不在宿主执行，得分保持待处理。通过 `T2T_BWRAP` 指定可用的 bubblewrap 路径；隔离需要内核支持。

代码评分会产生外部逐题评分表。`python -m pip install --no-deps -e .` 将当前源码目录以可编辑模式注册为 Python 包 `t2t_experiments`，并创建 `t2t-experiments` 命令；`--no-deps` 表示依赖由前一步锁定清单安装。这里安装的是项目代码，不包含模型权重或数据集，也不需要另找一个独立安装程序。`configs/`、`docs/`、`third_party/` 随完整仓库提供，不作为 Python 包数据内嵌；安装后仍需保留仓库并显式传入配置路径。

## GPU 选择

统一通过外部 `CUDA_VISIBLE_DEVICES` 指定 GPU，支持设备编号或 GPU UUID；程序不覆盖该变量，始终使用第一张可见卡 `cuda:0`。容器或调度器已分配设备时可直接运行。未设置该变量时使用系统第一张 CUDA 卡；值为空或没有可用 CUDA 设备时程序报错退出。双模型推理与显存等待使用同一卡，tmux 沿用启动时的设备环境。

```bash
CUDA_VISIBLE_DEVICES=3 t2t-experiments --config configs/experiment.json run --output /srv/t2t/runs/m1
```
