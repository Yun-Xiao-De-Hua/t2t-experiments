"""Frozen v8 output normalization; no model inference or answer repair.

Source SHA-256: e15f1fbce64a1c45bfa0e029cdd6761d76a9c6c1ff924c83f5e2df53bf7b0869
"""
import re


NUMBER = r'-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?'


NUMERIC_LINE = re.compile(r'(?:final answer:\s*|answer:\s*)?(' + NUMBER + r')', re.I)


ANSWER_LABEL = re.compile(r'(?:(?:final answer|answer)\s*:|\*\*(?:final answer|answer)\s*:\*\*|__(?:final answer|answer)\s*:__)\s*', re.I)


TICK = chr(96)


FENCE = TICK * 3


FENCE_OPEN = re.compile(re.escape(FENCE) + r'(?:python|py)?', re.I)


def outer_markup(text):
    for marker in ('**', '__', TICK):
        if text.startswith(marker) and text.endswith(marker) and len(text) > 2 * len(marker):
            body = text[len(marker):-len(marker)]
            if marker not in body and body.strip():
                return body.strip()
    return text


def normalize_gsm(text, extract_gsm):
    if extract_gsm(text) != '[invalid]':
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = NUMERIC_LINE.fullmatch(lines[-1]) if lines else None
    return '#### ' + match.group(1) if match else text


def normalize_hotpot(text):
    lines = [outer_markup(line.strip()) for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    matches = [ANSWER_LABEL.match(line) for line in lines]
    last = matches[-1]
    if last and sum(match is not None for match in matches) == 1:
        answer = outer_markup(lines[-1][last.end():].strip())
        if answer and not ANSWER_LABEL.match(answer):
            return answer
    if len(lines) == 1 and not last:
        return lines[0]
    return text


def normalize_code(text, prefix):
    # Keep all bytes inside a recognized wrapper, including indentation.
    lines = text.strip('\r\n').splitlines(keepends=True)
    if (len(lines) >= 3 and FENCE_OPEN.fullmatch(lines[0].strip())
            and lines[-1].strip() == FENCE
            and not any(line.lstrip().startswith(FENCE) for line in lines[1:-1])):
        text = ''.join(lines[1:-1])
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text
