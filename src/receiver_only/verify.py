"""Recompute the published receiver-only summaries on CPU; no raw outputs needed."""
import argparse
import gzip
import json
from pathlib import Path
from ..config import CATEGORIES, file_hash, repository_root
from .scoring import cost, summary


def verify(root):
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    expected = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    total = 0
    for rel, entry in metadata['files'].items():
        p = root/rel
        if file_hash(p) != entry['sha256']:
            raise ValueError('Checksum mismatch: '+rel)
        records = []
        seen = set()
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if row['sample_id'] in seen or any(k not in row for k in CATEGORIES):
                    raise ValueError('Duplicate sample or missing category')
                seen.add(row['sample_id'])
                q = row[CATEGORIES[1]]
                records.append({'generation_status': q['status'], 'common_7345': q['common_7345'],
                                'raw': q['raw'], 'format_normalized': q['format_normalized'],
                                'output_changed': q['output_changed'],
                                'cost': cost({i: row[name] for i, name in enumerate(CATEGORIES, 1)})})
                c = metadata['configurations'][row[CATEGORIES[0]]['configuration_ref']]['condition']
        if len(records) != entry['samples']:
            raise ValueError('Sample count mismatch: '+rel)
        for subset in (('full', 'common_7345') if c['dataset'] == 'hotpotqa' else ('full',)):
            actual = summary(c, records, subset)
            target = next(s for s in expected if s['condition_id'] == c['condition_id'] and s['subset'] == subset)
            if actual != target:
                raise ValueError('Summary mismatch: '+rel+' '+subset)
        total += len(records)
    if total != 26664 or len(expected) != 12:
        raise ValueError('Unexpected complete receiver-only scope')
    print(f'Verified {total} outcomes, 9 conditions, 12 summary views; CPU only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=repository_root()/'results/receiver_only')
    verify(parser.parse_args().results)
