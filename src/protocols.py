"""Deterministic evidence assignment and strictly separated public inputs."""
import json
import os
import random
import re
from .config import canonical, digest


def prompt_path():
    return __import__('pathlib').Path(os.environ['T2T_PROMPT_FILE']).resolve()


def templates():
    return json.loads(prompt_path().read_text(encoding='utf-8'))


def rng(sample_id, phase):
    return random.Random(int(digest(f'42:{sample_id}:{phase}'), 16))


def adapt(dataset, row, index):
    sid = row.get('_id', row.get('task_id', f'gsm8k/main/test/{index:04d}'))
    q = row['prompt'] if dataset == 'humaneval' else row['question']
    context = row.get('context', [])
    order = list(range(len(context)))
    rng(sid, 'full_order').shuffle(order)
    support = {title for title, _ in row.get('supporting_facts', [])}
    support_indices = [i for i, (title, _) in enumerate(context) if title in support]
    eligible = (dataset == 'hotpotqa' and len(context) == 10
                and len({p[0] for p in context}) == 10 and len(support) == 2
                and len(support_indices) == 2)
    sender, receiver = [], []
    if eligible:
        noise = [i for i in range(10) if i not in support_indices]
        rng(sid, 'support').shuffle(support_indices)
        rng(sid, 'distractors').shuffle(noise)
        sender = [support_indices[0]] + noise[:4]
        receiver = [support_indices[1]] + noise[4:]
        # Preserve the common full ordering within each endpoint.
        sender = [i for i in order if i in sender]
        receiver = [i for i in order if i in receiver]
    public = {'sample_id': sid, 'dataset': dataset, 'q': q, 'context': context,
              'order': order, 'sender_indices': sender, 'receiver_indices': receiver,
              'common_7345': eligible, 'p2_exclusion': None if eligible else 'not_10_unique_paragraphs_with_two_supports'}
    reference = {k: row[k] for k in ('answer', 'canonical_solution', 'test', 'entry_point') if k in row}
    return public, reference


def evidence(sample, indices):
    return '\n\n'.join(f'[{sample["context"][i][0]}]\n' + ''.join(sample['context'][i][1]) for i in indices)


def view(q, c):
    return f'<TASK>\n{q}\n</TASK>' + (f'\n<EVIDENCE>\n{c}\n</EVIDENCE>' if c else '')


def inputs(sample, condition, prompt=None):
    from .config import conditions
    def resolve_task_prompt(prompt, dataset):
        task = prompt["per_task"][dataset]
        return {**task, "sender_task": {dataset: task["sender_task"]}, "receiver_task": {dataset: task["receiver_task"]}}
    p = resolve_task_prompt(prompt or templates(), sample["dataset"])
    if condition['condition_id'] not in {c['condition_id'] for c in conditions()}:
        raise ValueError('Condition is outside the active 24-condition scope')
    protocol = condition['protocol']
    full = evidence(sample, sample['order'])
    if protocol == 'P2':
        if not sample['common_7345']:
            raise ValueError('P2 ineligible sample')
        cs, cr = evidence(sample, sample['sender_indices']), evidence(sample, sample['receiver_indices'])
    else:
        cs, cr = full, full if protocol == 'P1' else ''
    q = sample['q']
    local = p['p0'] if protocol == 'P4' else view(q, cr)
    sender_instruction = p['sender_task'][sample['dataset']]
    if condition['path'] == 'sender_restated_q':
        sender_instruction += '\n' + p['restatement_instruction']
    sender = [{'role': 'system', 'content': p['sender_system']},
              {'role': 'user', 'content': view(q, cs) + '\n\n' + sender_instruction}]
    receiver_prefix = [{'role': 'system', 'content': p['receiver_system']},
                       {'role': 'user', 'content': local}]
    return {'sender_messages': sender, 'receiver_prefix_messages': receiver_prefix,
            'receiver_instruction': p['receiver_task'][sample['dataset']], 'raw_text': ''}


def pack(condition, sample_id, generated, raw=''):
    from .config import conditions
    if raw or condition['condition_id'] not in {c['condition_id'] for c in conditions()}:
        raise ValueError('Only generated-text active conditions may be serialized')
    metadata = {'schema_version': 1, 'sample_id': sample_id, 'condition_id': condition['condition_id']}
    obj = {'metadata': metadata, 'generated_text': generated}
    wire = canonical(obj).encode('utf-8')
    assert json.loads(wire) == obj
    g, m = len(generated.encode('utf-8')), len(canonical(metadata).encode('utf-8'))
    return wire, {'generated_text_bytes': g, 'raw_text_bytes': 0, 'metadata_json_bytes': m,
                  'json_escaping_and_envelope_bytes': len(wire)-g-m,
                  'serialized_payload_bytes': len(wire), 'payload_sha256': digest(wire),
                  'receive_sha256': digest(wire), 'kv_sent_bytes': 0, 'receive_verified': True}


def receiver_messages(prepared, wire):
    received = json.loads(wire)
    if set(received) != {'metadata', 'generated_text'}:
        raise ValueError('Unexpected fields in generated-only wire payload')
    communication = canonical({'generated_text': received['generated_text']})
    return prepared['receiver_prefix_messages'] + [
        {'role': 'user', 'content': '<COMMUNICATION>\n' + communication + '\n</COMMUNICATION>\n\n' + prepared['receiver_instruction']}]


def restatement(text, original):
    text, original = text.replace('\r\n', '\n').replace('\r', '\n'), original.replace('\r\n', '\n').replace('\r', '\n')
    matches = list(re.finditer(r'(?:\A|\n)<TASK>\n(.*?)\n</TASK>(?:\n|\Z)', text, re.DOTALL))
    analyses = list(re.finditer(r'(?:\A|\n)<ANALYSIS>\n(.*?)\n</ANALYSIS>(?:\n|\Z)', text, re.DOTALL))
    exact = len(matches) == 1 and matches[0].group(1) == original
    structure = len(analyses) == 1 and len(matches) == 1 and matches[0].end() <= analyses[0].start() + 1
    return {'task_exact_match': exact, 'blocks_valid': structure,
            'compliant': exact and structure, 'extracted_task': matches[0].group(1) if len(matches) == 1 else None}
