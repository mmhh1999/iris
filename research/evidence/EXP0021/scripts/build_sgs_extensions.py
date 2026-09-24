#!/usr/bin/env python3
"""Build selected public SGS dependencies in the isolated package target."""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('packages',nargs='+')
p.add_argument('--log',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
tool=root/'.toolchain/compiler'
nvidia=root/'.venv/lib/python3.10/site-packages/nvidia'
env=os.environ.copy()
env.update(CUDA_HOME=str(tool),CC=str(tool/'bin/x86_64-conda-linux-gnu-gcc'),
           CXX=str(tool/'bin/x86_64-conda-linux-gnu-g++'),TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='4',
           LD_LIBRARY_PATH=str(tool/'lib'),
           CPATH=':'.join(map(str,[tool/'targets/x86_64-linux/include',*sorted(nvidia.glob('*/include'))])),
           LIBRARY_PATH=':'.join(map(str,[tool/'targets/x86_64-linux/lib',*sorted(nvidia.glob('*/lib'))])))
cmd=['uv','pip','install','--python',str(root/'.venv/bin/python'),'--target',str(root/'.baseline_deps/sgs'),
     '--no-build-isolation','--no-deps',*a.packages]
with open(a.log,'w') as log:
    run=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
raise SystemExit(run.returncode)
