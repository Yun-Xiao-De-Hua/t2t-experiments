"""Applicable regression cases retained from the experiment; not rerun on organization."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from t2t_experiments.config import conditions, canonical, digest
from t2t_experiments.protocols import adapt, inputs, pack, receiver_messages, restatement
from t2t_experiments.scoring import score, extract_gsm, hotpot_f1, normalize_hotpot
from t2t_experiments.records import ResultStore

class Contracts(unittest.TestCase):
    def setUp(self):
        config = Path(__file__).resolve().parents[1] / 'configs'
        self.env = patch.dict(os.environ, {'T2T_EXPERIMENT_FILE':str(config/'experiment.json'),
                                          'T2T_PROMPT_FILE':str(config/'prompts.json')})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_p2_exclusion(self):
        row = hotrow()
        row['context'].pop()
        sample, _ = adapt('hotpotqa', row, 0)
        self.assertFalse(sample['common_7345'])
        with self.assertRaises(ValueError):
            inputs(sample, next(c for c in conditions() if c['protocol']=='P2'))

    def test_restatement_preserves_whitespace(self):
        q = 'def f(x=1):\n    """doc"""\n'
        valid = '<TASK>\n' + q + '\n</TASK>\n<ANALYSIS>\nplan\n</ANALYSIS>'
        self.assertTrue(restatement(valid, q)['compliant'])
        self.assertTrue(restatement(valid.replace('\n', '\r\n'), q)['compliant'])
        self.assertFalse(restatement(valid.replace('    ', '  '), q)['compliant'])
        self.assertFalse(restatement(valid + '\n' + valid, q)['compliant'])

    def test_utf8_wire_accounting(self):
        c = conditions()[0]
        wire, m = pack(c, 'id', '中文\n"\\', '')
        self.assertEqual(m['serialized_payload_bytes'], len(wire))
        self.assertEqual(sum(m[k] for k in ('generated_text_bytes', 'raw_text_bytes', 'metadata_json_bytes', 'json_escaping_and_envelope_bytes')), len(wire))
        self.assertEqual(digest(wire), m['payload_sha256'])
        self.assertEqual(json.loads(wire)['generated_text'], '中文\n"\\')


def hotrow():
    return {'_id': 'fixture', 'question': 'QUESTION_SENTINEL', 'answer': 'GOLD_SENTINEL',
            'context': [[f't{i}', [f'paragraph_{i}_sentinel']] for i in range(10)],
            'supporting_facts': [['t0', 0], ['t1', 0]]}
