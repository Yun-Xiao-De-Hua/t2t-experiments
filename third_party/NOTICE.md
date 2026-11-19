# Third-party notices

The HumanEval execution helpers are derived from OpenAI HumanEval (MIT):
https://github.com/openai/human-eval — revision 6d43fb980f9fee3c892a914eda09951f772ad10d.
The unchanged helper is retained as `humaneval_execution.py`; see `licenses/humaneval_LICENSE`.

GSM8K numerical extraction follows OpenAI grade-school-math (MIT):
https://github.com/openai/grade-school-math — revision 3101c7d5072418e28b9008a6636bde82a006892c.
See `licenses/gsm8k_LICENSE`.

HotpotQA answer normalization, exact match and token F1 follow the official
HotpotQA evaluator: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py.
Its source attribution and applicable dataset terms remain with their owners.

Qwen3 model weights/tokenizers are supplied separately; model revisions are in
`configs/models.json`. Model and dataset licenses are independent of this code.
Bubblewrap is an external system dependency, not vendored here.

Project-level license, author list and citation are pending the project team's decision.
No project-level open-source license grant is asserted by this delivery.
