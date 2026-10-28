"""Persistent local postcompute pipeline; failed checks never mark a release ready."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from nucleus4d_baked_relight import ROOT,atomic_json
from nucleus4d_transport_geometry import OUT


def status(stage,**values):
    value=dict(stage=stage,updated=time.time(),**values);atomic_json(OUT/'finish_status.json',value);print('FINISH',json.dumps(value),flush=True)


def run(script,arguments=(),env=None):
    status(script)
    subprocess.run([sys.executable,str(ROOT/'experiments'/script),*arguments],cwd=ROOT,env=env,check=True)


def main(check_only=False):
    if check_only:
        run('nucleus4d_transport_viewer.py')
        run('nucleus4d_transport_check.py')
        env=os.environ.copy();env['DRJIT_LIBCUDA_PATH']='/dev/null'
        run('nucleus4d_transport_shadow_check.py',env=env)
        status('numerical_and_browser_checks_passed',visual_review_pending=True,physical_room_ground_truth_verified=False)
        return
    status('waiting_for_light_bank')
    while not (OUT/'bank_manifest.json').exists():
        status('waiting_for_light_bank',fine_lamps=len(list(OUT.glob('fine_lamp_*.npy'))),total_lamps=25,
               solar_indirect_samples=len(list((OUT/'sun_gi').glob('coarse_*.npy'))))
        if subprocess.run(['systemctl','--user','is-active','--quiet','iris-transport-v3.service']).returncode:
            raise RuntimeError('Light bake stopped before completing its bank; see compute.log')
        time.sleep(15)
    run('nucleus4d_transport_audit.py')
    env=os.environ.copy();args=[]
    smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    free=int(subprocess.check_output([smi,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<6500:args=['--cpu'];env['DRJIT_LIBCUDA_PATH']='/dev/null'
    run('nucleus4d_transport_review.py',args,env)
    run('nucleus4d_transport_audit.py')
    run('nucleus4d_transport_viewer.py')
    run('nucleus4d_transport_check.py')
    env=os.environ.copy();env['DRJIT_LIBCUDA_PATH']='/dev/null'
    run('nucleus4d_transport_shadow_check.py',env=env)
    status('numerical_and_browser_checks_passed',visual_review_pending=True,physical_room_ground_truth_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    try:main(args.check_only)
    except Exception as error:
        status('failed',error=str(error));raise
