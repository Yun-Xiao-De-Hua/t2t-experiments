# Receiver-only single-model reproduction

Qwen3-0.6B (R06), 1.7B (R17), and 4B (R40) independently solve GSM8K, HotpotQA, and HumanEval: nine conditions and 26,664 outcomes. These identifiers denote models, not M1–M3 communication directions. No additional weights are trained.

## Setup

| Task | Actual model input | Formal tasks per model | Metric |
|---|---|---:|---|
| GSM8K | Original question Q | 1,319 | accuracy |
| HotpotQA | Q and complete evidence C | 7,405 | EM / F1 |
| HumanEval | Original Python function prefix Q | 164 | pass@1 |

The exact frozen [prompts](../../configs/receiver_only/prompts.json) use `receiver_only_v1`, SHA-256 `1c64b20eca29d69bc73faf82eb817247b9d0d81f719061063f4583c345ff8e06`. Task output instructions follow the communication baseline; system instructions remove upstream-specific wording. Each task shares one template across all models. System + User messages include `<TASK>`, optional `<EVIDENCE>`, and the output instruction. Reference answers and evaluation code never enter model inputs.

One GPU and one resident model at a time, BF16/SDPA, batch 1, seed 42, two CPU threads, thinking disabled, the original manual greedy decoder, and EOS or actual remaining context capacity termination. Inputs are not truncated. A single full prefill replaces the communication runner's split prefill. The historical formal campaign explicitly waived non-test acceptance: zero acceptance cases and no warmup inference. The reproduction config retains this behavior and does not claim acceptance passed.

## Prepare, run, and score

Install dependencies as described in [usage](usage.md), then configure models, datasets, and prepared paths in `configs/paths.local.json`. Receiver-only does not read warmup_datasets. Model/data revisions remain frozen. Run prepare only for a new prepared directory; existing data prepared under the same revisions may be reused without overwriting it.

```bash
t2t-experiments --config configs/receiver_only/experiment.json prepare
CUDA_VISIBLE_DEVICES=3 t2t-experiments --config configs/receiver_only/experiment.json run \
  --wait-memory --tmux receiver_only --output /srv/t2t/runs/receiver_only
```

External CUDA_VISIBLE_DEVICES selects the GPU; the program uses the first visible device, cuda:0, without binding a physical index or UUID. Replace 3 with your assigned device. Repeat `--model Qwen/Qwen3-0.6B` and `--dataset gsm8k` to filter conditions. Direction/protocol filters do not apply. Optional memory waiting uses estimated BF16 weights plus 2 GiB workspace, polls every 15 seconds, and rechecks after strictly more than 180 stable seconds.

```bash
# Repeat the same config, filters, GPU environment and output; add --resume.
CUDA_VISIBLE_DEVICES=3 t2t-experiments --config configs/receiver_only/experiment.json run \
  --wait-memory --resume --output /srv/t2t/runs/receiver_only
t2t-experiments --config configs/receiver_only/experiment.json status --run-dir /srv/t2t/runs/receiver_only
t2t-experiments --config configs/receiver_only/experiment.json score \
  --run-dir /srv/t2t/runs/receiver_only --output /srv/t2t/runs/receiver_only_scores
python -B -m t2t_experiments.receiver_only.verify
```

Run saves raw outputs and seven committed categories without scoring in the inference pipeline. Score produces original/format-normalized per-task scores and 12 aggregate views on CPU. HumanEval runs under bubblewrap; infrastructure failures stop scoring instead of becoming incorrect answers. OOM/capacity failures are retained and score zero; committed failures are not retried in place. Ten consecutive OOMs trigger a 60-second cooldown. Changed resume signatures are rejected. The reorganized run format does not resume historical archives in place.

## Results and comparisons

All 26,664 outcomes completed without OOM or capacity failures. The [summary](../../results/receiver_only/summary.csv) includes nine full views and three HotpotQA views on the common 7,345 tasks. The tables use this common subset, reusing full-run outputs.

Original and format-normalized scores use the same fixed denominators. Normalization only removes formatting such as answer labels and outer markup; it does not change prompts or solve tasks. Both score modes are retained.

CUDA-event latency covers only full prefill and decode, excluding loading, tokenization, transfers, scoring, writes, and waits. Total latency is summed per task before calculating the mean/P50/P95. Communication bytes are zero.

Receiver-matched controls are M1→R17, M2→R40, and M3→R06. P1 is the closest full-information control; P2/P3 use different local evidence, so differences cannot be attributed solely to communication. Shared-GPU timings are descriptive, and single-model and dual-resident memory measurements have different scopes.
