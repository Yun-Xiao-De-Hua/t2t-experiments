"""CPU-only receiver scoring and full/common-subset report generation."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from ..config import canonical, file_hash, read_jsonl, save_new, repository_root
from .formatting import normalize_code, normalize_gsm, normalize_hotpot
from ..scoring import extract_gsm, score

METRICS = {'gsm8k': ('accuracy',), 'hotpotqa': ('answer_em', 'answer_f1'), 'humaneval': ('pass@1',)}
COSTS = ('communication_bytes', 'prefill_cuda_ms', 'decode_cuda_ms', 'total_cuda_ms')


def normalize(dataset, text, prefix=''):
    if dataset == 'gsm8k':
        return normalize_gsm(text, extract_gsm)
    if dataset == 'hotpotqa':
        return normalize_hotpot(text)
    return normalize_code(text, prefix)


def percentile(values, p):
    values = sorted(values)
    rank = (len(values)-1)*p
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo]+(rank-lo)*(values[hi]-values[lo])


def summary(condition, records, subset='full'):
    selected = [r for r in records if subset == 'full' or r['common_7345']]
    if not selected:
        raise ValueError('Empty scoring subset')
    keys = METRICS[condition['dataset']]
    result = {**condition, 'subset': subset, 'samples': len(selected),
              'status_counts': dict(Counter(r['generation_status'] for r in selected)),
              'normalized_outputs': sum(r['output_changed'] for r in selected)}
    for mode in ('raw', 'format_normalized'):
        for r in selected:
            if any(type(r[mode][k]) not in (float, int) or not math.isfinite(r[mode][k])
                   or not 0 <= r[mode][k] <= 1 for k in keys):
                raise ValueError('Missing or invalid score; infrastructure failures are not zero scores')
        result[mode] = {k: math.fsum(r[mode][k] for r in selected)/len(selected) for k in keys}
    valid = [r['cost'] for r in selected if r.get('cost') is not None]
    result['cost_n'] = len(valid)
    for suffix, fn in (('mean', lambda v: math.fsum(v)/len(v)),
                       ('p50', lambda v: percentile(v, .5)), ('p95', lambda v: percentile(v, .95))):
        result['cost_'+suffix] = {k: fn([v[k] for v in valid]) if valid else None for k in COSTS}
    return result


def write_summary(destination, summaries):
    save_new(destination/'summary.json', summaries)
    fields = ['condition_id', 'model', 'dataset', 'subset', 'samples', 'normalized_outputs',
              'oom_failures', 'capacity_failures', 'cost_n']
    fields += [mode+'_'+k for mode in ('raw', 'format_normalized')
               for k in ('accuracy', 'answer_em', 'answer_f1', 'pass@1')]
    fields += [k+'_'+s for k in COSTS for s in ('mean', 'p50', 'p95')]
    with (destination/'summary.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            row = {k: s[k] for k in fields[:6]+['cost_n']}
            row.update(oom_failures=s['status_counts'].get('oom_failure', 0),
                       capacity_failures=s['status_counts'].get('capacity_failure', 0))
            for mode in ('raw', 'format_normalized'):
                row.update({mode+'_'+k: v for k, v in s[mode].items()})
            for suffix in ('mean', 'p50', 'p95'):
                row.update({k+'_'+suffix: v for k, v in s['cost_'+suffix].items()})
            writer.writerow(row)


def cost(records):
    try:
        pre = records[6]['timings']['prefill']['cuda_event_s']*1000
        dec = records[6]['timings']['decode']['cuda_event_s']*1000
        comm = records[4]['generated_text_bytes']
        values = [comm, pre, dec, pre+dec]
        if not all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values):
            return None
        return dict(zip(COSTS, values))
    except KeyError:
        return None


def score_run(cfg, args):
    from ..cli import committed_records
    from ..isolation import evaluate, probe
    root, dest = args.run_dir.resolve(), args.output.resolve()
    repo = repository_root()
    if dest == repo or repo in dest.parents:
        raise ValueError('Scoring output must be outside the repository')
    matrix = json.loads((root/'conditions.json').read_text())
    signature = json.loads((root/'signature.json').read_text())
    if signature['prompt_sha256'] != file_hash(cfg['prompts_file']):
        raise ValueError('Scoring prompt differs from generation signature')
    inputs = Path(cfg['paths']['prepared'])/'evaluation'
    for name, sha in signature['input_hashes'].items():
        if file_hash(inputs/name) != sha:
            raise ValueError('Scoring inputs differ from generation signature')
    isolation = probe() if any(c['dataset'] == 'humaneval' for c in matrix) else None
    if isolation is not None and not isolation.get('passed'):
        raise RuntimeError('HumanEval isolation probe failed')
    dest.mkdir(parents=True, exist_ok=False)
    summaries = []
    for c in matrix:
        ds = c['dataset']
        public = read_jsonl(inputs/f'{ds}.public.jsonl')
        refs = {r['sample_id']: r for r in read_jsonl(inputs/f'{ds}.references.jsonl')}
        committed = committed_records(root/c['condition_id'])
        if set(committed) != {r['sample_id'] for r in public}:
            raise ValueError('Scoring requires complete committed coverage')
        rows = []
        for p in public:
            records = committed[p['sample_id']]
            q = records[2]
            text = q.get('output', '')
            cleaned = normalize(ds, text, p['q'])
            if q['status'] in ('oom_failure', 'capacity_failure'):
                raw = normalized = {k: 0.0 for k in METRICS[ds]}
            elif q['status'] == 'generated_pending_offline_scoring':
                def evaluate_text(t):
                    value = score(ds, t, refs[p['sample_id']])
                    if ds == 'humaneval':
                        value.update(evaluate({'prompt': p['q'], **refs[p['sample_id']]}, t, isolation))
                        if value.get('pass@1') not in (0.0, 1.0):
                            raise RuntimeError('HumanEval isolation evaluation failed')
                    return {k: value[k] for k in METRICS[ds]}
                raw = evaluate_text(text)
                normalized = raw if cleaned == text else evaluate_text(cleaned)
            else:
                raise ValueError('Unexpected generation status')
            rows.append({'sample_id': p['sample_id'], 'attempt_id': q['attempt_id'],
                         'common_7345': p['common_7345'], 'generation_status': q['status'],
                         'raw': raw, 'format_normalized': normalized,
                         'output_changed': cleaned != text, 'cost': cost(records)})
        save_new(dest/(c['condition_id']+'.scored.jsonl'), ''.join(canonical(r)+'\n' for r in rows))
        summaries.append(summary(c, rows))
        if ds == 'hotpotqa':
            summaries.append(summary(c, rows, 'common_7345'))
    write_summary(dest, summaries)
    save_new(dest/'provenance.json', {'generation_signature_sha256': file_hash(root/'signature.json'),
             'formatting_sha256': file_hash(Path(__file__).with_name('formatting.py')),
             'scoring_sha256': file_hash(Path(__file__).with_name('scoring.py'))})
