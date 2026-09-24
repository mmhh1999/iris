#!/usr/bin/env python3
"""Bounded runner for the explicitly labelled public-upstream SGS adaptation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--timeout',type=int,default=120)
p.add_argument('sgs_args',nargs=argparse.REMAINDER)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
args=a.sgs_args[1:] if a.sgs_args[:1]==['--'] else a.sgs_args
if not args:args=['--help']
env=os.environ.copy()
env.update(PYTHONPATH=str(root/'.baseline_deps/sgs'),LD_LIBRARY_PATH=str(root/'.toolchain/compiler/lib'),
           HF_HOME=str(root/'.baseline_deps/hf_cache'),TORCH_HOME=str(root/'.baseline_deps/torch_cache'),
           MPLCONFIGDIR=str(out/'matplotlib'),PYTHONUNBUFFERED='1')
cmd=[str(root/'.venv/bin/python'),'train.py',*args]
start=time.monotonic();timed_out=False
with (out/'run.log').open('w') as log:
    try:
        result=subprocess.run(cmd,cwd=root/'third_party/sgs_adapted',env=env,
                              stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
        rc=result.returncode
    except subprocess.TimeoutExpired:
        rc=None;timed_out=True
record={'label':'SGS-public-upstream-adaptation','command':cmd,'returncode':rc,'timeout':timed_out,
        'seconds':time.monotonic()-start,'entrypoint_help_passed':args==['--help'] and rc==0,
        'scientific_metrics_validated':False,'official_reproduction':False}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(rc if rc is not None else 124)
