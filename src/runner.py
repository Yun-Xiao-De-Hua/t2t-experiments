"""Sequential, append-only execution using the measured experiment pipeline."""
import gc
import json
import os
from pathlib import Path
import time
import traceback
import uuid
from .config import CATEGORIES, canonical, digest, file_hash, save_new, read_jsonl, na
from .protocols import inputs, pack, receiver_messages, restatement, prompt_path
from .records import ResultStore
from .runtime import ResidentPair, CapacityError


def environment(device='cuda:0'):
    import importlib.metadata
    import torch
    return {'gpu_name': torch.cuda.get_device_name(device), 'cuda_runtime': torch.version.cuda,
            'device': str(device), 'precision':'bfloat16','backend':'sdpa','threads':2,'batch_size':1,
            'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')},
            'timing':'CUDA Event and synchronized wall; model load/warmup/writes/scoring excluded'}


def release(pair):
    pair.current = {}
    gc.collect()
    pair.torch.cuda.empty_cache()


def failure_records(sample, condition, pair, error, env):
    """Preserve raw diagnostics externally; never retry a committed sample in-place."""
    status = 'oom_failure' if isinstance(error,pair.torch.OutOfMemoryError) else 'capacity_failure'
    rows = {i:{'status':'not_completed','reason':status,'value':None} for i in range(1,8)}
    rows[1] = {'condition':condition,'models':pair.specs,'prompt_sha256':file_hash(prompt_path()),'seed':42}
    rows[2] = {'status':status,'output':'','common_7345':sample['common_7345'],'error':str(error)}
    partial = getattr(error,'partial_result',{})
    sender = partial.get('sender')
    if sender and condition['path']=='sender_restated_q':
        rows[2]['restatement'] = restatement(sender['text'],sample['q'])
    rows[7] = {'event':status,'hardware_environment':env,'runtime_partial':pair.current,
               'partial_result':partial,'traceback':traceback.format_exc(),
               'cuda_mem_get_info':pair.torch.cuda.mem_get_info()}
    return rows


def run(cfg, args):
    import fcntl
    from .scheduling import wait_memory
    prepared = Path(cfg['paths']['prepared'])
    selected = [c for c in cfg['conditions'] if (not args.direction or c['pair'] in args.direction)
                and (not args.dataset or c['dataset'] in args.dataset)
                and (not args.protocol or c['protocol'] in args.protocol)]
    if not selected:
        raise ValueError('No active conditions selected')
    selected.sort(key=lambda c:(c['protocol']=='P4',c['pair'],c['dataset'],c['protocol']))
    root = Path(args.output).resolve()
    # Even personal run outputs must not be placed in the source repository.
    repo = Path(cfg['_config']).parent.parent
    if root == repo or repo in root.parents:
        raise ValueError('Choose an output directory outside the repository')
    specs_all = cfg['models']
    signature = {'conditions':selected,'prompt_sha256':file_hash(prompt_path()),
                 'models':specs_all,'runtime':cfg['runtime'],
                 'device':'cuda:0','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'cuda_device_order':os.environ.get('CUDA_DEVICE_ORDER'),
                 'model_paths':cfg['paths']['models'],
                 'input_hashes':{p.name:file_hash(p) for p in sorted((prepared/'evaluation').glob('*.jsonl'))},
                 'warmup_hashes':{p.name:file_hash(p) for p in sorted((prepared/'warmup').glob('*.jsonl'))},
                 'source_hashes':{p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    if args.resume:
        if json.loads((root/'signature.json').read_text()) != signature:
            raise ValueError('Resume signature changed; use a new output directory')
    else:
        root.mkdir(parents=True,exist_ok=False)
        save_new(root/'signature.json',signature)
        save_new(root/'conditions.json',selected)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pair = None
        current_pair = None
        state = {'status':'running','current_condition':None}
        def write_state():
            from datetime import datetime, timezone
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            temp = root/'status.tmp'
            temp.write_text(canonical(state)+'\n')
            temp.replace(root/'status.json')
        try:
            for c in selected:
                store = ResultStore(root/c['condition_id'])
                samples = read_jsonl(prepared/'evaluation'/f'{c["dataset"]}.public.jsonl')
                samples = [s for s in samples if c['protocol']!='P2' or s['common_7345']]
                done = store.committed()
                if all(s['sample_id'] in done for s in samples):
                    continue
                if current_pair != c['pair']:
                    if pair is not None:
                        del pair
                        import torch
                        gc.collect(); torch.cuda.empty_cache()
                    if args.wait_memory:
                        wait_memory('cuda:0',cfg['scheduling']['minimum_free_gib'][c['pair']],
                                    cfg['scheduling']['poll_seconds'],cfg['scheduling']['stable_seconds'])
                    specs = {role:{**specs_all['models'][name], 'repo_id':name,
                                   'local_path':cfg['paths']['models'][name]}
                             for role,name in specs_all['model_pairs_from_experiment_plan'][c['pair']].items()}
                    pair = ResidentPair(specs,reserve_bytes=cfg['runtime']['reserve_bytes'])
                    current_pair = c['pair']
                env = environment(pair.device)
                warmup = read_jsonl(prepared/'warmup'/f'{c["dataset"]}.public.jsonl')
                warmup = [s for s in warmup if c['protocol']!='P2' or s['common_7345']]
                for s in warmup[-cfg['runtime']['warmup_per_condition']:]:
                    try:
                        execute(pair,s,c)
                    except (pair.torch.OutOfMemoryError,CapacityError) as e:
                        store.append(7,{'phase':'warmup','event':type(e).__name__,'error':str(e)})
                    release(pair)
                for s in samples:
                    if s['sample_id'] in done:
                        continue
                    state.update(current_condition=c['condition_id'],sample_id=s['sample_id'])
                    write_state()
                    try:
                        result = execute(pair,s,c)
                        rows = records(s,c,result,env,pair)
                    except (pair.torch.OutOfMemoryError,CapacityError) as e:
                        rows = failure_records(s,c,pair,e,env)
                    store.commit(s['sample_id'],uuid.uuid4().hex,rows)
                    del rows
                    if 'result' in locals():
                        del result
                    release(pair)
                    print(c['condition_id'],s['sample_id'],'committed',flush=True)
            state.update(status='complete',current_condition=None)
            write_state()
        except BaseException:
            state.update(status='needs_attention',error=traceback.format_exc())
            write_state()
            raise

def execute(pair, sample, condition):
    began = time.perf_counter()
    prepared = inputs(sample, condition)
    sender = pair.generate('sender', prepared['sender_messages']) if condition['path'] != 'raw_forward' else None
    pack_start = time.perf_counter()
    wire, communication = pack(condition, sample['sample_id'], sender['text'] if sender else '', prepared['raw_text'])
    pack_s = time.perf_counter() - pack_start
    unpack_start = time.perf_counter()
    messages = receiver_messages(prepared, wire)
    unpack_s = time.perf_counter() - unpack_start
    try:
        receiver = pair.generate('receiver', messages, prepared['receiver_prefix_messages'])
    except BaseException as error:
        error.partial_result = {'sender': sender, 'payload_utf8': wire.decode('utf-8'),
                                'communication': communication, 'receiver_messages': messages}
        raise
    return {'sender': sender, 'receiver': receiver, 'wire': wire.decode('utf-8'), 'communication': communication,
            'pack_wall_s': pack_s, 'unpack_wall_s': unpack_s, 'pipeline_wall_s': time.perf_counter() - began,
            'restatement': restatement(sender['text'], sample['q']) if condition['path'] == 'sender_restated_q' else na('not sender-restated P4')}


_execute_pipeline = execute


def execute(pair, sample, condition):
    if getattr(pair, '_consecutive_oom', 0) >= 10:
        for _ in range(4):
            pair.progress()
            time.sleep(15)
        pair._consecutive_oom = 0
    try:
        result = _execute_pipeline(pair, sample, condition)
    except pair.torch.OutOfMemoryError:
        pair._consecutive_oom = getattr(pair, '_consecutive_oom', 0) + 1
        raise
    else:
        pair._consecutive_oom = 0
        return result


def records(sample, condition, result, env, pair):
    sender, receiver = result['sender'], result['receiver']
    gpu_times = [v['cuda_event_s'] for stage in (sender, receiver) if stage for v in stage['timings'].values() if 'cuda_event_s' in v]
    config = {'condition': condition, 'sample_id': sample['sample_id'], 'seed': 42,
              'no_added_training': True, 'models': pair.specs, 'generation': pair.effective_generation,
              'assembly_version': 't2t_generated_only_v1',
              'prompt_sha256': file_hash(prompt_path()), 'prompt_file': str(prompt_path()),
              'sender_input_tokens': sender['input_tokens'] if sender else na('raw forwarding'),
              'sender_output_tokens': sender['generated_tokens'] if sender else na('raw forwarding'),
              'receiver_input_tokens': receiver['input_tokens'], 'receiver_output_tokens': receiver['generated_tokens'],
              'receiver_local_tokens': receiver['boundary'], 'receiver_new_tokens': len(receiver['new_token_ids']),
              'sender_context_limit': sender['context_limit'] if sender else na('raw forwarding'),
              'receiver_context_limit': receiver['context_limit'],
              'sender_max_new_tokens': sender['context_limit'] - sender['input_tokens'] if sender else na('raw forwarding'),
              'receiver_max_new_tokens': receiver['context_limit'] - receiver['input_tokens'],
              'K': na(), 'k': na(), 'd': na(), 'transmission_location': 'separate user communication after local input',
              'sender_request': inputs(sample, condition)['sender_messages'] if sender else na('raw forwarding'),
              'sender_render': {k: sender[k] for k in ('rendered_prompt', 'input_token_ids')} if sender else na('raw forwarding'),
              'receiver_render': {k: receiver[k] for k in ('rendered_prompt', 'input_token_ids', 'local_token_ids', 'new_token_ids', 'boundary')}}
    return {
        1: config,
        2: {'status': 'generated_pending_offline_scoring', 'common_7345': sample['common_7345'],
            'sender_text': sender['text'] if sender else None, 'output': receiver['text'],
            'generated_token_ids': receiver['generated_token_ids'], 'sender_generated_token_ids': sender['generated_token_ids'] if sender else None,
            'restatement': result['restatement'], 'sender_termination': sender['termination_reason'] if sender else na(),
            'receiver_termination': receiver['termination_reason']},
        3: {'communication_modules': {k: {'parameters': 0, 'deployed': False} for k in ('pooling', 'projection', 'conversion', 'fusion')},
            'base_models': pair.load_metrics},
        4: {**result['communication'], 'payload_utf8': result['wire'],
            'sender_prefill_kv': sender['prefill_kv'] if sender else na('raw forwarding; sender not executed'),
            'sender_prefill_kv_logical_bytes': sender['prefill_kv_logical_bytes'] if sender else na('raw forwarding'),
            'transport': 'in-process serialized UTF-8 bytes; not network latency'},
        5: {'executed': sender is not None, 'timings': sender['timings'] if sender else na('raw forwarding'),
            'tokenize_wall_s': sender['tokenize_wall_s'] if sender else na(), 'text_decode_wall_s': sender['text_decode_wall_s'] if sender else na(),
            'h2d_wall_s': sender['h2d_wall_s'] if sender else na(), 'pack_wall_s': result['pack_wall_s'],
            'memory': sender['memory'] if sender else na(), 'screening': na(), 'pooling': na(), 'projection': na(), 'conversion': na()},
        6: {'timings': receiver['timings'], 'tokenize_wall_s': receiver['tokenize_wall_s'],
            'text_decode_wall_s': receiver['text_decode_wall_s'], 'h2d_wall_s': receiver['h2d_wall_s'],
            'unpack_wall_s': result['unpack_wall_s'], 'memory': receiver['memory'],
            'recovery': na(), 'broadcast': na(), 'fusion': na(),
            'prefill_total_cuda_event_s': receiver['timings']['local_prefill']['cuda_event_s'] + receiver['timings']['new_prefill']['cuda_event_s']},
        7: {'gpu_inference_hot_cuda_event_sum_s': sum(gpu_times), 'hot_pipeline_wall_s': result['pipeline_wall_s'],
            'hardware_environment': env, 'operations': {'forward': str(pair.device), 'greedy_control': 'CPU/GPU',
            'tokenize_decode_pack_unpack': 'CPU'}, 'scoring': 'separate offline phase',
            'excluded_from_hot_pipeline': ['model_load', 'warmup', 'result_writes', 'scoring', 'scheduling']}}
