"""GPU routing regressions; mocks only, never initializes CUDA."""
import contextlib
import io
import os
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from t2t_experiments import cli, scheduling


class GPUSelection(unittest.TestCase):
    def test_cli_preserves_external_visibility(self):
        for visible in (None, '', '3', 'GPU-example', '4,2'):
            env = {} if visible is None else {'CUDA_VISIBLE_DEVICES': visible}
            run=Mock()
            argv=['t2t-experiments','--config','unused.json','run','--output','/tmp/test-run']
            with self.subTest(visible=visible), patch.dict(os.environ, env, clear=True), \
                 patch.object(sys,'argv',argv), patch.object(cli,'load',return_value={}), \
                 patch.dict(sys.modules,{'t2t_experiments.runner':SimpleNamespace(run=run)}):
                cli.main()
                run.assert_called_once()
                self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), visible)

    def test_removed_device_flags_are_rejected(self):
        for flag,value in (('--gpu','3'),('--device','cuda:1')):
            argv=['t2t-experiments','--config','unused.json','run','--output','/tmp/test-run',flag,value]
            with self.subTest(flag=flag), patch.object(sys,'argv',argv), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code,2)

    def test_cli_run_without_gpu(self):
        run = Mock()
        argv = ['t2t-experiments','--config','unused.json','run','--output','/tmp/test-run']
        with patch.object(sys,'argv',argv), patch.object(cli,'load',return_value={}), \
             patch.dict(sys.modules,{'t2t_experiments.runner':SimpleNamespace(run=run)}), \
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4,2'},clear=True):
            cli.main()
            self.assertFalse(hasattr(run.call_args.args[1],'gpu'))
            self.assertFalse(hasattr(run.call_args.args[1],'device'))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '4,2')

    def test_memory_wait_uses_logical_cuda_device(self):
        mem = Mock(return_value=(16*1024**3,24*1024**3))
        cuda = SimpleNamespace(is_available=lambda:True,mem_get_info=mem)
        with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=cuda)}), \
             patch.object(scheduling.time,'monotonic',side_effect=[0,181]), \
             patch.object(scheduling.time,'sleep'), patch.object(scheduling.subprocess,'run') as process:
            scheduling.wait_memory('cuda:0',6)
        self.assertEqual(mem.call_count,3)
        self.assertTrue(all(c.args==('cuda:0',) for c in mem.call_args_list))
        process.assert_not_called()

    def test_no_visible_cuda_fails_without_memory_query(self):
        mem=Mock()
        cuda=SimpleNamespace(is_available=lambda:False,mem_get_info=mem)
        with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=cuda)}), self.assertRaises(RuntimeError):
            scheduling.wait_memory('cuda:0',6)
        mem.assert_not_called()

    def test_tmux_preserves_callers_cuda_environment(self):
        for visible in (None, '', 'GPU-example'):
            env={} if visible is None else {'CUDA_VISIBLE_DEVICES':visible}
            with patch.dict(os.environ,env,clear=True), patch.object(scheduling.subprocess,'run') as run:
                scheduling.tmux('test',['--config','config.json','run'])
                command=shlex.split(run.call_args.args[0][-1])
                self.assertIn('-u',command)
                assignments=[v for v in command if v.startswith('CUDA_VISIBLE_DEVICES=')]
                self.assertEqual(assignments,[] if visible is None else ['CUDA_VISIBLE_DEVICES='+visible])
