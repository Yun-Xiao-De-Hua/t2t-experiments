"""Explicit entry point; imports CUDA code only for the run command."""
import argparse
import json
import os
from pathlib import Path
import sys
from .config import load, CATEGORIES, canonical, digest, save_new, read_jsonl


def committed_records(root):
    """Read committed records for scoring user runs; no writes to source records."""
    path = root/'commits.jsonl'
    commits = {}
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        commits[row['sample_id']] = row
    out = {sid:{} for sid in commits}
    for i,name in enumerate(CATEGORIES,1):
        f = root/name/'records.jsonl'
        if not f.exists():
            continue
        with f.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                commit = commits.get(row.get('sample_id'))
                if commit and row.get('attempt_id')==commit['attempt_id'] and digest(canonical(row))==commit['record_hashes'].get(str(i)):
                    out[row['sample_id']][i] = row
    if any(len(row)!=7 for row in out.values()):
        raise ValueError('Incomplete committed records in user run')
    return out


def score_run(cfg,args):
    from .scoring import score
    from .isolation import probe, evaluate
    helper = Path(cfg['_config']).parent.parent/'third_party/humaneval_execution.py'
    os.environ['T2T_HUMANEVAL_HELPER'] = str(helper)
    matrix = json.loads((args.run_dir/'conditions.json').read_text())
    isolated = probe() if any(c['dataset']=='humaneval' for c in matrix) else None
    destination = args.output.resolve()
    repo = Path(cfg['_config']).parent.parent
    if destination == repo or repo in destination.parents:
        raise ValueError('Scoring output must be outside the repository')
    destination.mkdir(parents=True,exist_ok=False)
    for c in matrix:
        ds = c['dataset']
        data = Path(cfg['paths']['prepared'])/'evaluation'
        refs = {r['sample_id']:r for r in read_jsonl(data/f'{ds}.references.jsonl')}
        public = {r['sample_id']:r for r in read_jsonl(data/f'{ds}.public.jsonl')}
        expected_hashes = json.loads((args.run_dir/'signature.json').read_text())['input_hashes']
        from .config import file_hash
        for suffix in ('public','references'):
            name=f'{ds}.{suffix}.jsonl'
            if file_hash(data/name)!=expected_hashes[name]:
                raise ValueError('Scoring inputs differ from run signature')
        lines=[]
        for sid,records in committed_records(args.run_dir/c['condition_id']).items():
            quality=records[2]
            if quality['status'] in ('oom_failure','capacity_failure'):
                keys={'gsm8k':['accuracy'],'hotpotqa':['answer_em','answer_f1'],'humaneval':['pass@1']}[ds]
                metrics={k:0.0 for k in keys}
            else:
                metrics=score(ds,quality['output'],refs[sid])
                if ds=='humaneval':
                    metrics.update(evaluate({'prompt':public[sid]['q'],**refs[sid]},quality['output'],isolated))
            if c['protocol']=='P4':
                compliance=float(quality.get('restatement',{}).get('compliant',False))
                key={'gsm8k':'accuracy','hotpotqa':'answer_em','humaneval':'pass@1'}[ds]
                metrics['restatement_compliance']=compliance
                metrics['restatement_and_correct']=None if metrics.get(key) is None else compliance*metrics[key]
            lines.append(canonical({'sample_id':sid,'generation_status':quality['status'],'scores':metrics})+'\n')
        save_new(destination/(c['condition_id']+'.scored.jsonl'),''.join(lines))


def status(root):
    from collections import Counter
    result={'run':str(root),'conditions':[]}
    path=root/'status.json'
    if path.exists():
        result['state']=json.loads(path.read_text())
    for c in json.loads((root/'conditions.json').read_text()):
        rows=committed_records(root/c['condition_id'])
        result['conditions'].append({'condition_id':c['condition_id'],'committed':len(rows),
                                     'statuses':dict(Counter(r[2]['status'] for r in rows.values()))})
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare')
    run=commands.add_parser('run')
    run.add_argument('--output',required=True,type=Path)
    run.add_argument('--resume',action='store_true')
    run.add_argument('--direction',action='append',choices=['M1','M2','M3'])
    run.add_argument('--dataset',action='append',choices=['gsm8k','hotpotqa','humaneval'])
    run.add_argument('--protocol',action='append',choices=['P1','P2','P3','P4'])
    run.add_argument('--wait-memory',action='store_true')
    run.add_argument('--tmux')
    score=commands.add_parser('score')
    score.add_argument('--run-dir',type=Path,required=True)
    score.add_argument('--output',type=Path,required=True)
    st=commands.add_parser('status')
    st.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    cfg=load(args.config)
    if args.command=='prepare':
        from .data import prepare
        prepare(cfg)
    elif args.command=='status':
        status(args.run_dir)
    elif args.command=='score':
        score_run(cfg,args)
    else:
        from .scheduling import tmux
        if args.tmux:
            argv=sys.argv[1:]
            i=argv.index('--tmux')
            argv=argv[:i]+argv[i+2:]
            # Child uses the same current directory; explicit config/output paths persist.
            tmux(args.tmux,argv)
            return
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        from .runner import run
        run(cfg,args)


if __name__=='__main__':
    main()
