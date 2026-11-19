"""Single-model controls with immutable inputs and append-only outcomes.

Message assembly and seven-category generation records follow receiver_only_v1.
Device selection is delegated to CUDA_VISIBLE_DEVICES; this runner uses cuda:0.
"""
import gc
import json
import os
from pathlib import Path
import time
import traceback
import uuid
from ..config import canonical, file_hash, na, read_jsonl, save_new, repository_root, source_hashes
from ..protocols import evidence, view


def messages(sample, prompts):
    task = prompts['per_task'][sample['dataset']]
    content = view(sample['q'], evidence(sample, sample['order']))
    return [{'role': 'system', 'content': task['receiver_system']},
            {'role': 'user', 'content': content + '\n\n' + task['receiver_task']}]


def build_records(sample, condition, result, env, model, request, prompt_sha, failure=None):
    common = {'condition': condition, 'seed': 42, 'no_added_training': True,
              'models': model.specs, 'generation': model.effective_generation,
              'assembly_version': 'receiver_only_v1', 'prompt_sha256': prompt_sha,
              'receiver_request': request, 'sender_executed': False,
              'prefill_mode': 'single_full_input', 'transmission_location': na('no upstream model')}
    rows = {
        1: common,
        2: {'common_7345': sample['common_7345'], 'status': 'generated_pending_offline_scoring'},
        3: {'base_models': model.load_metrics, 'sender': na('not loaded'),
            'communication_modules': {k: {'parameters': 0, 'deployed': False}
                                      for k in ('pooling', 'projection', 'conversion', 'fusion')}},
        4: {'transport': 'none', 'generated_text_bytes': 0, 'raw_text_bytes': 0,
            'metadata_json_bytes': 0, 'serialized_payload_bytes': 0, 'kv_sent_bytes': 0,
            'sender_prefill_kv': na('no Sender'), 'receive_verified': na('no communication')},
        5: {'executed': False, 'timings': na('no Sender'), 'memory': na('no Sender')},
        6: {'status': 'not_completed', 'value': None},
        7: {'hardware_environment': env, 'scoring': 'separate offline phase',
            'excluded_from_hot_pipeline': ['model_load', 'warmup', 'result_writes', 'scoring', 'scheduling']}}
    if failure:
        rows[2].update(status=failure['status'], output='', error=failure['error'])
        rows[6].update(reason=failure['status'])
        rows[7].update(event=failure['status'], traceback=failure['traceback'],
                       runtime_partial=model.current, cuda_mem_get_info=model.torch.cuda.mem_get_info())
    else:
        r = result['receiver']
        r['memory']['scope'] = 'one resident Receiver model + its KV/workspace; reset before inference'
        r['timings']['local_prefill'] = na('single complete prefill; no local/communication split')
        rows[1].update(receiver_input_tokens=r['input_tokens'], receiver_output_tokens=r['generated_tokens'],
            receiver_context_limit=r['context_limit'], receiver_max_new_tokens=r['context_limit']-r['input_tokens'],
            receiver_render={k:r[k] for k in ('rendered_prompt', 'input_token_ids', 'boundary', 'local_token_ids', 'new_token_ids')})
        rows[2].update(output=r['text'], generated_token_ids=r['generated_token_ids'],
                       receiver_termination=r['termination_reason'])
        rows[6] = {k:r[k] for k in ('timings', 'tokenize_wall_s', 'text_decode_wall_s', 'h2d_wall_s', 'memory',
                                  'prefill_kv', 'prefill_kv_logical_bytes')}
        rows[7].update(hot_pipeline_wall_s=result['pipeline_wall_s'],
            gpu_inference_hot_cuda_event_sum_s=sum(v['cuda_event_s'] for v in r['timings'].values() if 'cuda_event_s' in v))
    return rows


def infer_and_commit(model, sample, condition, store, prompts, env, prompt_sha):
    from ..runtime import CapacityError
    request = messages(sample, prompts)
    began = time.perf_counter()
    failure = None
    result = None
    try:
        result = {'receiver': model.generate('receiver', request), 'pipeline_wall_s': time.perf_counter()-began}
    except (model.torch.OutOfMemoryError, CapacityError) as error:
        failure = {'status': 'oom_failure' if isinstance(error, model.torch.OutOfMemoryError) else 'capacity_failure',
                   'error': str(error), 'traceback': traceback.format_exc()}
    rows = build_records(sample, condition, result, env, model, request, prompt_sha, failure)
    store.commit(sample['sample_id'], uuid.uuid4().hex, rows)
    status = rows[2]['status']
    del rows, result
    model.current = {}
    gc.collect()
    model.torch.cuda.empty_cache()
    return status


def select(cfg, args):
    if args.direction or args.protocol:
        raise ValueError('Receiver-only uses --model/--dataset, not direction/protocol filters')
    selected = [c for c in cfg['conditions'] if (not args.model or c['model'] in args.model)
                and (not args.dataset or c['dataset'] in args.dataset)]
    if not selected:
        raise ValueError('No active receiver-only conditions selected')
    return selected


def run(cfg, args):
    """Reuse the measured decoder with exactly one resident model; no acceptance run."""
    import fcntl
    from ..runtime import ResidentPair, weight_budget
    from ..runner import environment
    from ..records import ResultStore
    from ..scheduling import wait_memory
    selected = select(cfg, args)
    root, prepared = args.output.resolve(), Path(cfg['paths']['prepared'])
    repo = repository_root()
    if root == repo or repo in root.parents:
        raise ValueError('Choose an output directory outside the repository')
    for c in selected:
        samples = read_jsonl(prepared/'evaluation'/f'{c["dataset"]}.public.jsonl')
        if len(samples) != cfg['datasets'][c['dataset']]['records']:
            raise ValueError('Evaluation sample count differs from configured dataset')
        if len({s['sample_id'] for s in samples}) != len(samples):
            raise ValueError('Duplicate evaluation sample ID')
        if any(any(k in s for k in ('answer', 'test', 'canonical_solution', 'entry_point')) for s in samples):
            raise ValueError('Held-out reference leaked into public input')
        if any(s['dataset'] != c['dataset'] or sorted(s['order']) != list(range(len(s['context']))) for s in samples):
            raise ValueError('Invalid dataset or evidence ordering')
        if c['dataset'] == 'hotpotqa' and sum(s['common_7345'] for s in samples) != 7345:
            raise ValueError('HotpotQA common subset differs from the frozen scope')
    signature = {
        'experiment_type': 'receiver_only', 'conditions': selected,
        'prompt_sha256': file_hash(cfg['prompts_file']), 'models': cfg['models'],
        'runtime': cfg['runtime'], 'scheduling': cfg['scheduling'],
        'humaneval_helper_sha256': file_hash(repo/'third_party/humaneval_execution.py'),
        'device': 'cuda:0', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'), 'model_paths': cfg['paths']['models'],
        'input_hashes': {p.name: file_hash(p) for p in sorted((prepared/'evaluation').glob('*.jsonl'))},
        'source_hashes': source_hashes()}
    if args.resume:
        if json.loads((root/'signature.json').read_text()) != signature:
            raise ValueError('Resume signature changed; use a new output directory')
    else:
        root.mkdir(parents=True, exist_ok=False)
        save_new(root/'signature.json', signature)
        save_new(root/'conditions.json', selected)
        save_new(root/'prompts.json', json.loads(cfg['prompts_file'].read_text()))
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'running', 'experiment_type': 'receiver_only', 'current_condition': None}
        model, current_model = None, None
        last_write = [0.0]

        def update(**values):
            from datetime import datetime, timezone
            state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            temp = root/'status.tmp'
            temp.write_text(canonical(state)+'\n', encoding='utf-8')
            temp.replace(root/'status.json')
            last_write[0] = time.monotonic()

        def progress():
            if time.monotonic()-last_write[0] > 30:
                partial = getattr(model, 'current', {}) if model else {}
                update(runtime_stage=partial.get('stage'),
                       generated_tokens=len(partial.get('generated_token_ids', [])))

        try:
            prompts = json.loads(cfg['prompts_file'].read_text())
            for c in selected:
                store = ResultStore(root/c['condition_id'])
                samples = read_jsonl(prepared/'evaluation'/f'{c["dataset"]}.public.jsonl')
                done = store.committed()
                if not set(done) <= {s['sample_id'] for s in samples}:
                    raise ValueError('Unexpected committed sample ID')
                pending = [s for s in samples if s['sample_id'] not in done]
                if not pending:
                    continue
                if current_model != c['model']:
                    if model is not None:
                        del model
                        model = None
                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
                    spec = {**cfg['models']['models'][c['model']], 'repo_id': c['model'],
                            'local_path': cfg['paths']['models'][c['model']]}
                    if args.wait_memory:
                        minimum = weight_budget(spec['local_path'])['bf16_weight_bytes_estimate']/1024**3
                        minimum += cfg['scheduling']['workspace_gib']
                        update(status='waiting_memory', current_condition=c['condition_id'], minimum_free_gib=minimum)
                        wait_memory('cuda:0', minimum, cfg['scheduling']['poll_seconds'], cfg['scheduling']['stable_seconds'])
                    model = ResidentPair({'receiver': spec}, reserve_bytes=cfg['runtime']['reserve_bytes'], progress=progress)
                    current_model = c['model']
                    env = {**environment(model.device), 'resident_model_count': 1,
                           'model_revision': spec['revision'], 'prefill': 'single complete input'}
                update(status='running', current_condition=c['condition_id'], model=c['model'],
                       condition_committed=len(done), condition_expected=len(samples))
                consecutive_oom = 0
                for sample in pending:
                    update(sample_id=sample['sample_id'], runtime_stage='inference', generated_tokens=0)
                    outcome = infer_and_commit(model, sample, c, store, prompts, env, signature['prompt_sha256'])
                    consecutive_oom = consecutive_oom+1 if outcome == 'oom_failure' else 0
                    update(condition_committed=state['condition_committed']+1, last_outcome=outcome,
                           runtime_stage='committed')
                    print(c['condition_id'], sample['sample_id'], outcome, flush=True)
                    if consecutive_oom >= 10:
                        update(status='oom_cooldown')
                        for _ in range(4):
                            time.sleep(15)
                            progress()
                        consecutive_oom = 0
                        update(status='running')
                store.checksums()
            if not (root/'completed.json').exists():
                save_new(root/'completed.json', {'conditions': len(selected),
                    'formal_cases': sum(cfg['datasets'][c['dataset']]['records'] for c in selected),
                    'prompt_sha256': signature['prompt_sha256'], 'acceptance_cases': 0})
            update(status='complete', current_condition=None, sample_id=None)
        except BaseException:
            update(status='needs_attention', error=traceback.format_exc())
            raise
