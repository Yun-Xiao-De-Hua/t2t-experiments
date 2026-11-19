"""HumanEval execution in a minimal bwrap namespace; never runs task code on host."""
import json
from pathlib import Path
import subprocess
import sys
import os

RUNTIME = Path(sys.prefix).resolve()


def command(code):
    args = [os.environ.get('T2T_BWRAP', 'bwrap'), '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
            '--clearenv', '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
            '--ro-bind', str(RUNTIME), '/opt/python', '--tmpfs', '/tmp', '--tmpfs', '/run',
            '--tmpfs', '/work', '--proc', '/proc', '--dev', '/dev', '--chdir', '/work',
            '--setenv', 'PATH', '/opt/python/bin:/usr/bin', '--setenv', 'HOME', '/work']
    if Path('/lib64').exists():
        args += ['--ro-bind', '/lib64', '/lib64']
    args += ['--', '/opt/python/bin/python', '-I', '-S', '-c', code]
    return args


PROBE = '''import json, os, resource, socket
resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
assert not os.path.exists('/home')
assert not os.path.exists('/mnt')
assert not os.path.exists('/etc/passwd')
try:
    open('/usr/t2t_write_probe', 'w')
except OSError:
    pass
else:
    raise AssertionError('runtime writable')
assert os.listdir('/sys/class/net') == [] if os.path.exists('/sys/class/net') else True
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('network accessible')
assert resource.getrlimit(resource.RLIMIT_AS)[0] == 2*1024**3
print(json.dumps({'isolated': True, 'network_blocked': True, 'host_private_paths_hidden': True, 'runtime_readonly': True, 'memory_limit_bytes': 2*1024**3}))
'''


def probe():
    try:
        r = subprocess.run(command(PROBE), capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return {'passed': False, 'returncode': r.returncode, 'stderr': r.stderr}
        return {'passed': True, **json.loads(r.stdout)}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {'passed': False, 'error': repr(e)}


def evaluate(problem, completion, probe_result):
    if not probe_result.get('passed'):
        return {'pass@1': None, 'status': 'pending_isolation_acceptance', 'probe': probe_result}
    source = Path(os.environ['T2T_HUMANEVAL_HELPER']).read_text()
    # Import the unchanged upstream helpers. Do not call create_tempdir or
    # unsafe_execute: ephemeral namespace files disappear when bwrap exits.
    wrapper = '''import contextlib, io, json, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024*1024, 16*1024*1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
payload = json.load(sys.stdin)
upstream = {}
exec(payload['upstream'], upstream)
upstream['reliability_guard'](maximum_memory_bytes=2*1024**3)
try:
    with upstream['swallow_io']():
        with upstream['time_limit'](3.0):
            exec(payload['program'], {})
    result = {'pass@1': 1.0, 'status': 'passed'}
except upstream['TimeoutException']:
    result = {'pass@1': 0.0, 'status': 'timed_out'}
except BaseException as e:
    result = {'pass@1': 0.0, 'status': 'failed', 'exception_type': type(e).__name__}
print(json.dumps(result))
'''
    program = problem['prompt'] + completion + '\n' + problem['test'] + '\n' + f'check({problem["entry_point"]})'
    try:
        r = subprocess.run(command(wrapper), input=json.dumps({'upstream': source, 'program': program}),
                           text=True, capture_output=True, timeout=8)
    except subprocess.TimeoutExpired:
        return {'pass@1': 0.0, 'status': 'sandbox_process_timeout'}
    if r.returncode != 0:
        return {'pass@1': None, 'status': 'infrastructure_failure', 'returncode': r.returncode, 'stderr': r.stderr[-2000:]}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {'pass@1': None, 'status': 'infrastructure_failure', 'stderr': r.stderr[-2000:]}
