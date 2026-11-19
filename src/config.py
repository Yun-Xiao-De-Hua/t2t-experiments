"""Shared immutable experiment contracts; no torch import or inference side effects."""
import hashlib
import json
from pathlib import Path

CATEGORIES = ('01_experiment_configuration', '02_answer_quality', '03_module_parameters',
              '04_communication_volume', '05_sender_compute_cost', '06_receiver_compute_cost',
              '07_device_environment')
DATASET_NAMES = {'gsm8k': 'gsm8k_main_test', 'hotpotqa': 'hotpotqa_distractor_dev',
                 'humaneval': 'humaneval_original_test'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def repository_root():
    """Locate shared assets independently of configuration nesting depth."""
    return Path(__file__).resolve().parent.parent


def source_hashes():
    """Include subpackages in resume signatures, using unambiguous relative keys."""
    root = Path(__file__).resolve().parent
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*.py'))}


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else canonical(value) + '\n')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def conditions():
    import os
    return json.loads(Path(os.environ['T2T_EXPERIMENT_FILE']).read_text())['conditions']


def na(reason='T2T does not deploy this operation'):
    return {'value': None, 'status': 'not_applicable', 'reason': reason}


def load(path):
    """Resolve every relative resource path against its declaring config file."""
    import os
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text())
    def resource(value, base=path.parent):
        value = Path(value).expanduser()
        return (base / value).resolve() if not value.is_absolute() else value.resolve()
    cfg['_config'] = str(path)
    for key in ('models_file', 'prompts_file', 'paths_file'):
        cfg[key] = resource(cfg[key])
    paths = json.loads(cfg['paths_file'].read_text())
    cfg['paths'] = {key: ({k: str(resource(v, cfg['paths_file'].parent)) for k, v in val.items()}
                         if isinstance(val, dict) else str(resource(val, cfg['paths_file'].parent)))
                    for key, val in paths.items()}
    cfg['models'] = json.loads(cfg['models_file'].read_text())
    os.environ['T2T_EXPERIMENT_FILE'] = str(path)
    os.environ['T2T_PROMPT_FILE'] = str(cfg['prompts_file'])
    os.environ['T2T_HUMANEVAL_HELPER'] = str(repository_root() / 'third_party/humaneval_execution.py')
    return cfg
