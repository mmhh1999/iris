"""Build the pinned local tiny-cuda-nn checkout with the local CUDA toolchain."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
prefix = root / '.toolchain/compiler'
env = os.environ.copy()
env.update(CUDA_HOME=str(prefix), TCNN_CUDA_ARCHITECTURES='120', MAX_JOBS='2')
includes = list((root / '.venv/lib/python3.10/site-packages/nvidia').glob('*/include'))
env['CPATH'] = ':'.join(map(str, includes + [prefix / 'targets/x86_64-linux/include']))
env['LIBRARY_PATH'] = ':'.join(map(str, [prefix / 'lib', prefix / 'targets/x86_64-linux/lib/stubs']))
subprocess.run([
    str(root / '.toolchain/bootstrap/bin/micromamba'), 'run', '-p', str(prefix),
    str(Path.home() / '.local/bin/uv'), 'pip', 'install', '--python', sys.executable,
    '--no-build-isolation', str(root / '.toolchain/tiny-cuda-nn/bindings/torch'),
], env=env, cwd=root, check=True)
