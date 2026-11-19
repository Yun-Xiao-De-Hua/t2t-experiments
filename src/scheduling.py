"""Memory waiting and tmux startup with externally assigned CUDA visibility."""
import os
import shlex
import subprocess
import sys
import time


def wait_memory(device, minimum_gib, poll_seconds=15, stable_seconds=180):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A visible CUDA device is required')

    def free_gib():
        return torch.cuda.mem_get_info(device)[0] / 1024**3

    since = None
    while True:
        free = free_gib()
        now = time.monotonic()
        if free >= minimum_gib:
            since = now if since is None else since
            if now - since > stable_seconds:
                # Requery before returning; no occupancy guarantee is assumed.
                if free_gib() >= minimum_gib:
                    return
                since = None
        else:
            since = None
        time.sleep(poll_seconds)


def tmux(name, argv):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Use letters, numbers, underscore or hyphen for the tmux session')
    # Existing tmux servers may retain an older environment. Pass current CUDA
    # visibility explicitly, including the distinction between unset and empty.
    env = ['env', '-u', 'CUDA_VISIBLE_DEVICES', '-u', 'CUDA_DEVICE_ORDER']
    env += [key+'='+os.environ[key] for key in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')
            if key in os.environ]
    subprocess.run(['tmux','new-session','-d','-s',name,shlex.join([*env,sys.executable,'-m',
                    't2t_experiments.cli',*argv])],check=True)
