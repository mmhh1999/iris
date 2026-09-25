"""Stop experiments before they fill the disk and freeze WSL.

Inside WSL, `df /` reports the ext4.vhdx's virtual size (1 TB), not the free
space of the Windows drive that actually holds the vhdx (D:\\WSL\\Ubuntu). When
that drive fills, the vhdx cannot grow and WSL hangs. So the usable headroom is
the minimum of the Linux filesystem's free space and the host drive's.

Environment:
  IRIS_VHDX_HOST     mount of the drive holding ext4.vhdx (default /mnt/d)
  IRIS_MIN_FREE_GB   refuse/kill below this many GB free (default 50)
"""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

VHDX_HOST = Path(os.environ.get('IRIS_VHDX_HOST', '/mnt/d'))
MIN_FREE_GB = float(os.environ.get('IRIS_MIN_FREE_GB', '50'))


class DiskLow(RuntimeError):
    pass


def free_gb(path='.'):
    path = Path(path)
    while not path.exists():
        path = path.parent
    paths = [path] + ([VHDX_HOST] if VHDX_HOST.exists() else [])
    return min(shutil.disk_usage(p).free for p in paths) / 1e9


def check(path='.', min_gb=None, need_gb=0.0):
    """Raise DiskLow unless `need_gb` more can be written and `min_gb` still remain."""
    min_gb = MIN_FREE_GB if min_gb is None else min_gb
    free = free_gb(path)
    if free - need_gb < min_gb:
        raise DiskLow(f'{free:.0f} GB free (incl. {VHDX_HOST}), need {need_gb:.0f} GB + {min_gb:.0f} GB reserve')
    return free


def _kill(child):
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run(cmd, path='.', min_gb=None, timeout=None, poll=5, **kw):
    """subprocess.run replacement that kills the child's process group if disk drops below min_gb."""
    check(path, min_gb)
    start = time.monotonic()
    child = subprocess.Popen(cmd, start_new_session=True, **kw)
    while child.poll() is None:
        if timeout is not None and time.monotonic() - start > timeout:
            _kill(child)
            raise subprocess.TimeoutExpired(cmd, timeout)
        try:
            check(path, min_gb)
        except DiskLow:
            _kill(child)
            raise
        time.sleep(poll)
    return subprocess.CompletedProcess(cmd, child.returncode)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Print usable free space; exit 1 below threshold.')
    p.add_argument('path', nargs='?', default='.')
    p.add_argument('--need-gb', type=float, default=0.0, help='planned output size, e.g. runs x GB per run')
    p.add_argument('--min-gb', type=float, default=None)
    a = p.parse_args()
    try:
        print(f'{check(a.path, a.min_gb, a.need_gb):.0f} GB free')
    except DiskLow as e:
        raise SystemExit(f'DISK LOW: {e}')
