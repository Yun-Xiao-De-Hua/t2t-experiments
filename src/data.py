"""Deterministic conversion of user-supplied JSONL datasets; no downloads."""
import gzip
import json
from pathlib import Path
from .config import canonical, file_hash, save_new
from .protocols import adapt


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(cfg):
    root = Path(cfg['paths']['prepared'])
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'sources': {}, 'conditions': cfg['conditions']}
    for split, key in [('evaluation', 'datasets'), ('warmup', 'warmup_datasets')]:
        out = root / split
        out.mkdir()
        for dataset, path in cfg['paths'][key].items():
            rows = read_rows(path)
            if split == 'evaluation' and len(rows) != cfg['datasets'][dataset]['records']:
                raise ValueError(f'{dataset}: unexpected evaluation sample count')
            public, references = [], []
            ids = set()
            for i, row in enumerate(rows):
                sample, ref = adapt(dataset, row, i)
                if split == 'warmup' and dataset == 'gsm8k':
                    sample['sample_id'] = f'gsm8k/main/train/{i:04d}'
                if sample['sample_id'] in ids:
                    raise ValueError('Duplicate sample identifier')
                ids.add(sample['sample_id'])
                public.append(sample)
                references.append({'sample_id': sample['sample_id'], **ref})
            save_new(out / f'{dataset}.public.jsonl', ''.join(canonical(s)+'\n' for s in public))
            save_new(out / f'{dataset}.references.jsonl', ''.join(canonical(s)+'\n' for s in references))
            manifest['sources'][split+':'+dataset] = {'sha256':file_hash(path), 'samples':len(rows)}
    # Warmup must contain non-test samples. Users provide synthetic code tasks.
    for dataset in cfg['datasets']:
        evaluation = read_rows(root / 'evaluation' / f'{dataset}.public.jsonl')
        warmup = read_rows(root / 'warmup' / f'{dataset}.public.jsonl')
        if {s['q'] for s in evaluation} & {s['q'] for s in warmup}:
            raise ValueError('Warmup overlaps evaluation questions')
        eligible = [s for s in warmup if dataset != 'hotpotqa' or s['common_7345']]
        if len(eligible) < 3:
            raise ValueError('At least three eligible non-test warmup samples are required')
    save_new(root / 'manifest.json', manifest)
    print(root)
