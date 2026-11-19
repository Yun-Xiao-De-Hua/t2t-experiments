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

    def test_append_only_recovery(self):
        # Keep the fixture for inspection; never recursively clean test artifacts.
        root = Path(tempfile.mkdtemp(prefix='t2t_recovery_'))
        store = ResultStore(root)
        store.commit('a', 'attempt1', {2: {'output': 'a'}})
        with (root / 'commits.jsonl').open('ab') as f:
            f.write(b'{"broken"')
        store.commit('b', 'attempt2', {2: {'output': 'b'}})
        self.assertEqual(set(store.committed()), {'a', 'b'})
        self.assertIn(b'{"broken"', (root / 'commits.jsonl').read_bytes())
