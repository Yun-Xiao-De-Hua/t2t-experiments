# Result fields and reporting

Each line of `results/M{1,2,3}/{dataset}/P{n}.jsonl.gz` is one final valid sample, with `sample_id`, `condition_id`, `schema_version` and seven objects.

| Object | Contents |
|---|---|
| 01_experiment_configuration | Shared configuration reference, token counts and context limits |
| 02_answer_quality | Existing scores, generation/format/restatement/termination status |
| 03_module_parameters | Shared model and communication module parameters |
| 04_communication_volume | Generated/envelope/metadata/serialized bytes, KV capacity and receive verification |
| 05_sender_compute_cost | Tokenization, H2D, prefill/decode, packing and memory |
| 06_receiver_compute_cost | Local/new prefill, decode, unpacking and memory |
| 07_device_environment | Shared hardware/software reference, pipeline time and available partial failure metrics |

Times are seconds, memory/communication measures bytes, token fields counts, and scores fractions in [0,1]. Shared references resolve to configurations, module_parameters and environments in metadata.json. Original experiment prompt SHA and packaged prompts.json SHA are distinct. Unknown, incomplete or inapplicable measurements are null, with available state preserved; zero is never substituted for an unknown measurement.

Published per-sample data excludes questions, evidence, references, rendered prompts, generated text, token ID sequences and full tracebacks. Actual inputs/outputs remain in the original complete records. Future reproductions save scoring inputs in external run directories.

Per direction: GSM8K P1/P4 each 1,319; HotpotQA P1/P3/P4 each 7,405 and P2 7,345; HumanEval P1/P4 each 164. This gives 32,526 outcomes per direction and 97,578 total: 97,424 normal generations, 105 OOMs and 49 capacity failures. Failures remain in fixed denominators with zero quality score.

The 33 rows in summary.csv are reporting views of 24 conditions, including extra HotpotQA common-7,345 views. Use the common subset for cross-protocol comparisons; full-set scores are not matched comparisons with P2. Metrics are GSM8K final-number accuracy, HotpotQA answer_em/answer_f1 and isolated HumanEval pass@1, plus P4 verbatim restatement compliance and joint compliance/correctness.

Point estimates and existing 95% bootstrap intervals (10,000 resamples, seed 42) are copied without recomputation; meaningless self-comparison columns are removed. Costs were measured on a shared GPU and do not establish load-independent causal speed comparisons. Hot pipeline timing excludes model load, warmup, writes, scoring and scheduling. CUDA events and synchronized wall times have different scopes.

## Receiver-only single-model controls

See [receiver-only reproduction](receiver_only.md) for the separate nine-condition config, inputs, commands, scoring, and results. The original 24 communication conditions are unchanged.
