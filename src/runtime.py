"""Resident BF16 model pairs with token-exact incremental prefill and CUDA timings."""
import json
import time
from pathlib import Path
from .config import na


class CapacityError(RuntimeError):
    pass


def weight_budget(model_path):
    """Count physical safetensor entries; subtract a tied alias only if BOTH are stored."""
    import math
    import struct
    tensors = {}
    for shard in sorted(Path(model_path).glob('*.safetensors')):
        with shard.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if not 0 < size < 32 * 1024 * 1024:
                raise ValueError('Invalid safetensors header size')
            tensors.update({k: v for k, v in json.loads(f.read(size)).items() if k != '__metadata__'})
    config = json.loads((Path(model_path) / 'config.json').read_text())
    stored = sum(math.prod(t['shape']) for t in tensors.values())
    alias = 0
    if config.get('tie_word_embeddings') and 'lm_head.weight' in tensors and 'model.embed_tokens.weight' in tensors:
        alias = math.prod(tensors['lm_head.weight']['shape'])
    return {'stored_tensor_elements': stored, 'unique_parameter_estimate': stored - alias,
            'bf16_weight_bytes_estimate': (stored - alias) * 2,
            'checkpoint_tensor_payload_bytes': sum(t['data_offsets'][1] - t['data_offsets'][0] for t in tensors.values())}


def render(tokenizer, messages, prefix_messages=None):
    """Split the final token stream by the longest valid rendered local prefix."""
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    start = time.perf_counter()
    ids = tokenizer.encode(full, add_special_tokens=False)
    boundary = 0
    prefix = ''
    if prefix_messages:
        prefix = tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prefix):
            raise ValueError('Local template is not a character prefix of the final rendered request')
        candidate = tokenizer.encode(prefix, add_special_tokens=False)
        for a, b in zip(candidate, ids):
            if a != b:
                break
            boundary += 1
        if boundary == 0 or boundary >= len(ids):
            raise ValueError('Invalid local/incremental token boundary')
    return {'rendered_prompt': full, 'input_token_ids': ids, 'local_token_ids': ids[:boundary],
            'new_token_ids': ids[boundary:], 'local_prefix_rendered': prefix,
            'boundary': boundary, 'tokenize_wall_s': time.perf_counter() - start}


def cache_info(cache):
    if cache is None:
        return []
    layers = []
    for i in range(len(cache)):
        key, value = cache[i]
        layers.append({'layer': i, 'key_shape': list(key.shape), 'value_shape': list(value.shape),
                       'dtype': str(key.dtype), 'logical_bytes': key.numel() * key.element_size() + value.numel() * value.element_size()})
    return layers


class ResidentPair:
    def __init__(self, specs, device='cuda:0', reserve_bytes=512 * 1024**2, progress=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, torch.device(device)
        self.progress = progress or (lambda: None)
        self.current = {}
        torch.set_num_threads(2)
        torch.manual_seed(42)
        if not torch.cuda.is_available():
            raise RuntimeError('No visible CUDA device; check CUDA_VISIBLE_DEVICES and the CUDA runtime')
        if self.device.type != 'cuda' or self.device.index is None or self.device.index >= torch.cuda.device_count():
            raise ValueError('The requested internal CUDA device is not visible')
        torch.cuda.set_device(self.device)
        self.specs, self.models, self.tokenizers, self.load_metrics = specs, {}, {}, {}
        budgets = {role: weight_budget(spec['local_path']) for role, spec in specs.items()}
        free, total = torch.cuda.mem_get_info(self.device)
        need = sum(b['bf16_weight_bytes_estimate'] for b in budgets.values()) + reserve_bytes
        if free < need:
            raise MemoryError(f'Resident pair requires at least {need} bytes including reserve; free={free}; budgets={budgets}')
        # Cap THIS process, without altering other users or the system allocator.
        torch.cuda.set_per_process_memory_fraction((free - reserve_bytes) / total, self.device)
        for role, spec in specs.items():
            start = time.perf_counter()
            self.tokenizers[role] = AutoTokenizer.from_pretrained(spec['local_path'], local_files_only=True, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(spec['local_path'], torch_dtype=torch.bfloat16,
                device_map={'': str(self.device)}, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, attn_implementation='sdpa')
            model.eval().requires_grad_(False)
            torch.cuda.synchronize(self.device)
            self.models[role] = model
            self.load_metrics[role] = {'load_wall_s': time.perf_counter() - start, **budgets[role],
                'actual_unique_parameters': sum(p.numel() for p in model.parameters()),
                'allocated_after_load': torch.cuda.memory_allocated(self.device),
                'reserved_after_load': torch.cuda.memory_reserved(self.device)}
            self.progress()
        self.effective_generation = {role: {'do_sample': False, 'num_beams': 1, 'enable_thinking': False,
            'temperature': None, 'top_p': None, 'top_k': None, 'use_cache': True,
            'eos_token_id': model.generation_config.eos_token_id, 'batch_size': 1,
            'precision': 'bfloat16', 'backend': 'sdpa', 'seed': 42,
            'max_new_tokens': 'remaining_actual_context_capacity', 'decode': 'manual_greedy_argmax_no_logits_processors'}
            for role, model in self.models.items()}

    def generate(self, role, messages, prefix_messages=None):
        torch, device = self.torch, self.device
        model, tokenizer = self.models[role], self.tokenizers[role]
        request = render(tokenizer, messages, prefix_messages)
        self.current = {'role': role, 'request': request, 'stage': 'input_check'}
        ids, boundary = request['input_token_ids'], request['boundary']
        limit = model.config.max_position_embeddings
        if len(ids) >= limit:
            raise CapacityError(f'{role}: input={len(ids)} context={limit}; input not truncated')
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        transfer = time.perf_counter()
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        torch.cuda.synchronize(device)
        h2d_s = time.perf_counter() - transfer
        cache, kv, timings = None, [], {}

        def measured(name, fn):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            wall = time.perf_counter()
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            timings[name] = {'cuda_event_s': start.elapsed_time(end) / 1000,
                             'synchronized_wall_s': time.perf_counter() - wall}
            self.progress()
            return result

        with torch.inference_mode():
            if boundary:
                out = measured('local_prefill', lambda: model(input_ids=tensor[:, :boundary],
                    use_cache=True, logits_to_keep=1))
                cache = out.past_key_values
                del out
                out = measured('new_prefill', lambda: model(input_ids=tensor[:, boundary:],
                    past_key_values=cache, use_cache=True, logits_to_keep=1))
            else:
                timings['local_prefill'] = na('Sender has no split local prefill')
                out = measured('prefill', lambda: model(input_ids=tensor, use_cache=True, logits_to_keep=1))
            cache = out.past_key_values
            kv = cache_info(cache)
            self.current.update(stage='decode', prefill_kv=kv)
            eos = self.effective_generation[role]['eos_token_id']
            eos = set(eos if isinstance(eos, list) else [eos])
            generated = []
            self.current['generated_token_ids'] = generated
            logits = out.logits[:, -1, :]
            del out

            def decode():
                nonlocal cache, logits
                # Prefill computes the first logits. Decode includes token selection,
                # EOS/capacity control and subsequent cached forwards consistently.
                while len(generated) < limit - len(ids):
                    token = torch.argmax(logits, dim=-1)
                    token_id = token.item()
                    generated.append(token_id)
                    self.progress()
                    if token_id in eos or len(generated) == limit - len(ids):
                        break
                    step = model(input_ids=token.reshape(1, 1), past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
                    cache, logits = step.past_key_values, step.logits[:, -1, :]
                return generated

            measured('decode', decode)
        peak = {'baseline_allocated_bytes': baseline, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'scope': 'both resident model weights + current role KV/workspace; reset before role'}
        start = time.perf_counter()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        text_decode_s = time.perf_counter() - start
        del cache, logits, tensor
        return {**request, 'text': text, 'generated_token_ids': generated,
                'input_tokens': len(ids), 'generated_tokens': len(generated),
                'termination_reason': 'eos' if generated[-1] in eos else 'context_capacity',
                'context_limit': limit, 'prefill_kv': kv,
                'prefill_kv_logical_bytes': sum(k['logical_bytes'] for k in kv),
                'timings': timings, 'h2d_wall_s': h2d_s, 'text_decode_wall_s': text_decode_s, 'memory': peak}
