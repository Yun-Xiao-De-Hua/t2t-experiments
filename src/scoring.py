"""Answer-only adapters matching the pinned upstream scoring functions."""
from collections import Counter
import re
import string

ANS_RE = re.compile(r'#### (\-?[0-9\.\,]+)')


def extract_gsm(text):
    match = ANS_RE.search(text)
    return match.group(1).strip().replace(',', '') if match else '[invalid]'


def normalize_hotpot(text):
    text = ''.join(c for c in text.lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def hotpot_f1(prediction, answer):
    p, a = normalize_hotpot(prediction), normalize_hotpot(answer)
    if (p in ('yes', 'no', 'noanswer') or a in ('yes', 'no', 'noanswer')) and p != a:
        return 0.0
    ps, ans = p.split(), a.split()
    n = sum((Counter(ps) & Counter(ans)).values())
    if not n:
        return 0.0
    precision, recall = n / len(ps), n / len(ans)
    return 2 * precision * recall / (precision + recall)


def score(dataset, output, reference):
    if dataset == 'gsm8k':
        extracted, target = extract_gsm(output), extract_gsm(reference['answer'])
        if target == '[invalid]':
            raise ValueError('Invalid gold GSM8K answer')
        return {'extracted': extracted, 'accuracy': float(extracted == target),
                'format_valid': bool(re.search(r'(?:^|\n)#### -?[0-9][0-9,.]*\s*\Z', output))}
    if dataset == 'hotpotqa':
        return {'extracted': output, 'answer_em': float(normalize_hotpot(output) == normalize_hotpot(reference['answer'])),
                'answer_f1': hotpot_f1(output, reference['answer']),
                'format_valid': bool(output.strip()) and len(output.strip().splitlines()) == 1}
    return {'extracted': output, 'pass@1': None, 'status': 'pending_isolated_evaluation',
            'format_valid': '```' not in output}
