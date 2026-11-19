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

    def test_gsm_official_edge_behavior(self):
        self.assertEqual(extract_gsm('####18'), '[invalid]')
        self.assertEqual(extract_gsm('#### 1,000'), '1000')
        self.assertEqual(extract_gsm('#### 1 then #### 2'), '1')
        self.assertEqual(score('gsm8k', '#### 18.0', {'answer': '#### 18'})['accuracy'], 0)

    def test_hotpot_edge_behavior(self):
        self.assertEqual(hotpot_f1('yes indeed', 'yes'), 0)
        self.assertEqual(hotpot_f1('', ''), 0)
        self.assertEqual(score('hotpotqa', 'The U.S.', {'answer': 'US'})['answer_em'], 1)
