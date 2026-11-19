"""CPU regressions for receiver assembly, score normalization and report populations."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from t2t_experiments import cli
from t2t_experiments.receiver_only.runner import messages, select
from t2t_experiments.receiver_only.scoring import normalize, summary, cost


class ReceiverOnly(unittest.TestCase):
    def test_full_input_has_no_upstream_or_reference(self):
        p = Path(__file__).resolve().parents[1]/'configs/receiver_only/prompts.json'
        prompts = json.loads(p.read_text(encoding='utf-8'))
        sample = {'dataset': 'hotpotqa', 'q': 'Who?', 'context': [['B', ['B fact.']], ['A', ['A fact.']]],
                  'order': [1, 0], 'answer': 'HELD_OUT'}
        request = messages(sample, prompts)
        self.assertEqual([r['role'] for r in request], ['system', 'user'])
        self.assertIn('<TASK>\nWho?\n</TASK>', request[1]['content'])
        self.assertIn('[A]\nA fact.\n\n[B]\nB fact.', request[1]['content'])
        self.assertNotIn('HELD_OUT', str(request))
        for ds in ('gsm8k', 'humaneval'):
            request = messages({'dataset': ds, 'q': 'exact prefix\n    ', 'context': [], 'order': []}, prompts)
            self.assertIn('exact prefix\n    \n</TASK>', request[1]['content'])

    def test_frozen_normalization_preserves_ambiguity_and_indentation(self):
        self.assertEqual(normalize('gsm8k', 'Calculation\nAnswer: 12'), '#### 12')
        self.assertEqual(normalize('gsm8k', '#### 12.00'), '#### 12.00')
        self.assertEqual(normalize('gsm8k', 'maybe twelve'), 'maybe twelve')
        self.assertEqual(normalize('hotpotqa', '**Answer:** Paris'), 'Paris')
        ambiguous = 'Answer: Paris\nAnswer: Rome'
        self.assertEqual(normalize('hotpotqa', ambiguous), ambiguous)
        prefix = 'def f():\n'
        self.assertEqual(normalize('humaneval', '```python\n'+prefix+'    return 1\n```', prefix), '    return 1\n')

    def test_fixed_denominator_and_common_population(self):
        c = {'condition_id': 'r', 'model': 'qwen', 'dataset': 'hotpotqa'}
        rows = [{'generation_status': s, 'common_7345': common, 'raw': {'answer_em': v, 'answer_f1': v},
                 'format_normalized': {'answer_em': v, 'answer_f1': v}, 'output_changed': False, 'cost': None}
                for s, common, v in [('generated_pending_offline_scoring', True, 1.), ('oom_failure', False, 0.)]]
        self.assertEqual(summary(c, rows)['raw']['answer_em'], .5)
        self.assertEqual(summary(c, rows, 'common_7345')['samples'], 1)
        rows[1]['raw']['answer_em'] = None
        with self.assertRaises(ValueError):
            summary(c, rows)

    def test_total_latency_is_per_sample_sum(self):
        rows = []
        for p, d in [(1, 100), (100, 1)]:
            v = cost({4: {'generated_text_bytes': 0}, 6: {'timings': {
                'prefill': {'cuda_event_s': p/1000}, 'decode': {'cuda_event_s': d/1000}}}})
            rows.append({'generation_status': 'generated_pending_offline_scoring', 'common_7345': False,
                         'raw': {'accuracy': 1.}, 'format_normalized': {'accuracy': 1.},
                         'output_changed': False, 'cost': v})
        s = summary({'dataset': 'gsm8k'}, rows)
        self.assertEqual(s['cost_p95']['total_cuda_ms'], 101)
        self.assertNotEqual(s['cost_p95']['total_cuda_ms'],
                            s['cost_p95']['prefill_cuda_ms']+s['cost_p95']['decode_cuda_ms'])

    def test_configuration_dispatch_preserves_gpu_environment(self):
        run = Mock()
        argv = ['t2t-experiments', '--config', 'receiver.json', 'run', '--model', 'Qwen/Qwen3-4B', '--output', '/tmp/r']
        with patch.object(sys, 'argv', argv), patch.object(cli, 'load', return_value={'experiment_type': 'receiver_only'}), \
             patch.dict(sys.modules, {'t2t_experiments.receiver_only.runner': SimpleNamespace(run=run)}), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-external'}, clear=True):
            cli.main()
            run.assert_called_once()
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-external')

    def test_selection_is_independent_of_directions(self):
        cfg = {'conditions': [{'model': 'Qwen/Qwen3-4B', 'dataset': 'gsm8k'}]}
        args = SimpleNamespace(direction=None, protocol=None, model=['Qwen/Qwen3-4B'], dataset=['gsm8k'])
        self.assertEqual(select(cfg, args), cfg['conditions'])
        args.direction = ['M3']
        with self.assertRaises(ValueError):
            select(cfg, args)

    def test_nested_config_resolves_shared_assets_and_tracks_subpackage(self):
        from t2t_experiments.config import load, repository_root, source_hashes
        repo = Path(__file__).resolve().parents[1]
        fixture = Path(tempfile.mkdtemp(prefix='receiver_config_'))
        nested = fixture/'configs/receiver_only'
        nested.mkdir(parents=True)
        for name in ('experiment.json', 'prompts.json'):
            (nested/name).write_bytes((repo/'configs/receiver_only'/name).read_bytes())
        (nested.parent/'models.json').write_bytes((repo/'configs/models.json').read_bytes())
        (nested.parent/'paths.local.json').write_text(json.dumps({'prepared': '../data/prepared'}))
        with patch.dict(os.environ, {}, clear=True):
            cfg = load(nested/'experiment.json')
            self.assertEqual(cfg['prompts_file'], (nested/'prompts.json').resolve())
            self.assertEqual(cfg['models_file'], (nested.parent/'models.json').resolve())
            self.assertEqual(Path(cfg['paths']['prepared']), (fixture/'data/prepared').resolve())
            self.assertEqual(Path(os.environ['T2T_HUMANEVAL_HELPER']),
                             repository_root()/'third_party/humaneval_execution.py')
        self.assertIn('receiver_only/runner.py', source_hashes())
        self.assertIn('runtime.py', source_hashes())

    def test_capacity_failure_commits_resumes_and_scores_without_cuda(self):
        from t2t_experiments import runtime, runner
        from t2t_experiments.receiver_only import runner as receiver_only, scoring as receiver_scoring
        from t2t_experiments.cli import committed_records
        repo = Path(__file__).resolve().parents[1]
        fixture = Path(tempfile.mkdtemp(prefix='receiver_replay_'))
        prepared = fixture/'prepared/evaluation'
        prepared.mkdir(parents=True)
        public = [{'sample_id': f'gsm8k/{i}', 'dataset': 'gsm8k', 'q': 'Q', 'context': [],
                   'order': [], 'common_7345': False} for i in range(2)]
        (prepared/'gsm8k.public.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public))
        (prepared/'gsm8k.references.jsonl').write_text(''.join(
            json.dumps({'sample_id': r['sample_id'], 'answer': '#### 1'})+'\n' for r in public))
        cfg = json.loads((repo/'configs/receiver_only/experiment.json').read_text())
        cfg['_config'] = str(repo/'configs/receiver_only/experiment.json')
        cfg['prompts_file'] = repo/'configs/receiver_only/prompts.json'
        cfg['models'] = json.loads((repo/'configs/models.json').read_text())
        cfg['conditions'] = cfg['conditions'][:1]
        cfg['datasets']['gsm8k']['records'] = 2
        cfg['paths'] = {'prepared': str(prepared.parent), 'models': {'Qwen/Qwen3-0.6B': '/unused'}}
        pair = SimpleNamespace(specs={}, effective_generation={}, load_metrics={}, device='cuda:0', current={},
            generate=Mock(side_effect=runtime.CapacityError('synthetic context overflow')),
            torch=SimpleNamespace(OutOfMemoryError=MemoryError, cuda=SimpleNamespace(
                empty_cache=Mock(), mem_get_info=Mock(return_value=(1, 2)))))
        args = SimpleNamespace(direction=None, protocol=None, model=None, dataset=None,
                               output=fixture/'run', resume=False, wait_memory=False)
        fcntl = SimpleNamespace(flock=Mock(), LOCK_EX=1, LOCK_NB=2)
        with patch.dict(sys.modules, {'fcntl': fcntl}), patch.object(runtime, 'ResidentPair', return_value=pair), \
             patch.object(runner, 'environment', return_value={}):
            receiver_only.run(cfg, args)
            self.assertEqual(pair.generate.call_count, 2)
            args.resume = True
            receiver_only.run(cfg, args)
            self.assertEqual(pair.generate.call_count, 2)
        c = cfg['conditions'][0]
        records = committed_records(args.output/c['condition_id'])
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r[2]['status'] == 'capacity_failure' for r in records.values()))
        receiver_scoring.score_run(cfg, SimpleNamespace(run_dir=args.output, output=fixture/'scores'))
        aggregate = json.loads((fixture/'scores/summary.json').read_text())[0]
        self.assertEqual(aggregate['raw']['accuracy'], 0)
        self.assertEqual(aggregate['samples'], 2)
        self.assertEqual(aggregate['cost_n'], 0)
        cfg['runtime']['seed'] = 7
        with patch.dict(sys.modules, {'fcntl': fcntl}), self.assertRaisesRegex(ValueError, 'signature'):
            receiver_only.run(cfg, args)
