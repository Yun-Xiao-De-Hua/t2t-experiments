# Usage and reproduction

Use a separate Linux Python 3.11 environment, recorded dependency versions, a CUDA 12.8 compatible driver and bubblewrap. The following commands are instructions for reproduction.

```bash
cd /path/to/t2t_experiments
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
cp configs/paths.example.json configs/paths.local.json
```

Edit `paths.local.json` with external model snapshots, raw datasets, non-test warmup tasks, prepared inputs and runs. Relative paths resolve against the declaring config. Use complete local model snapshots at the fixed revisions, including tokenizers and safetensors. No model downloads are performed.

Raw datasets are UTF-8 JSONL, optionally gzip: GSM8K rows contain `question, answer`; HotpotQA rows contain `_id, question, answer, context, supporting_facts`, with context as `[title, sentences]`; HumanEval rows contain `task_id, prompt, canonical_solution, test, entry_point`. Data revisions and counts are in experiment.json. Supply at least three warmup tasks per dataset from GSM8K train, HotpotQA train and synthetic code tasks. Do not use evaluation tasks for warmup; HotpotQA warmup must satisfy P2 eligibility.

```bash
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json prepare
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json run \
  --direction M1 --wait-memory --tmux t2t_m1 \
  --output /srv/t2t/runs/m1
```

GPU selection is controlled externally through `CUDA_VISIBLE_DEVICES`; the program uses the first visible card, `cuda:0`. Omit direction/dataset/protocol filters for all 24 conditions; repeat filters to select multiple values. P1–P3 precede P4. Optional memory waiting requires M1 ≥6 GiB and M2/M3 ≥10.5 GiB, polling every 15 seconds and rechecking after strictly more than 180 continuous seconds. Occupancy can change; per-sample OOM/capacity failures are committed and execution continues.

```bash
# Resume: repeat the same selection, GPU, configuration and output; add --resume.
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json run \
  --direction M1 --wait-memory --resume --output /srv/t2t/runs/m1
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json status \
  --run-dir /srv/t2t/runs/m1
t2t-experiments --config /path/to/t2t_experiments/configs/experiment.json score \
  --run-dir /srv/t2t/runs/m1 --output /srv/t2t/runs/m1_scores
```

Use new directories for initial prepare/run/score. Resume uses committed seven-category transactions and rejects changed signatures. Committed OOMs are not retried automatically; use a separately designed campaign for backfill. Status is read-only and needs no GPU. HumanEval executes only inside bubblewrap: a failed isolation probe never falls back to host execution and leaves scoring pending. Set `T2T_BWRAP` to an available bubblewrap executable; kernel support is required.

Scoring creates external per-sample score files. `python -m pip install --no-deps -e .` registers this source checkout as the editable Python package `t2t_experiments` and creates the `t2t-experiments` command. `--no-deps` leaves dependency installation to the preceding locked-requirements step. This installs project code; it includes neither model weights nor datasets and requires no separate installer. `configs/`, `docs/`, and `third_party/` are supplied through the full repository rather than embedded as Python package data. Retain the checkout and pass an explicit configuration path.

## GPU selection

Select GPUs externally with `CUDA_VISIBLE_DEVICES`, using a device index or GPU UUID. The program never overrides it and always uses the first visible card, `cuda:0`. Run directly when a container or scheduler has already assigned devices. If the variable is unset, the first system CUDA card is used; if it is empty or no CUDA device is available, execution fails. Both models and memory waiting use the same card, and tmux inherits the launching CUDA environment.

```bash
CUDA_VISIBLE_DEVICES=3 t2t-experiments --config configs/experiment.json run --output /srv/t2t/runs/m1
```

## Receiver-only single-model controls

See [receiver-only reproduction](receiver_only.md) for the separate nine-condition config, inputs, commands, scoring, and results. The original 24 communication conditions are unchanged.
