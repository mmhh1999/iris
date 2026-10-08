"""Durable, restartable full PTIR job with independent appearance quality gates."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from nucleus4d_ptir_full_data import ROOT,LOCAL,STORAGE,prior_path


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-stage',choices=['geometry','inverse'])
    parser.add_argument('--adopt-pid',type=int)
    parser.add_argument('--revalidate-stage',choices=['geometry','inverse'],help='Re-evaluate a final checkpoint without further training')
    parser.add_argument('--static-exports',action='store_true',help='Also render the earlier fixed-camera comparisons')
    options=parser.parse_args()
    if bool(options.adopt_stage)!=bool(options.adopt_pid):parser.error('Both adoption arguments are required')
    LOCAL.mkdir(exist_ok=True,parents=True);STORAGE.mkdir(exist_ok=True,parents=True)
    lock=(LOCAL/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=LOCAL/'pipeline_status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'completed_stages':[],'started':time.time()}
    if options.revalidate_stage:
        final=json.loads((STORAGE/options.revalidate_stage/'checkpoint.json').read_text())
        expected=30000 if options.revalidate_stage=='geometry' else 16000
        if final['step']!=expected:raise RuntimeError('Revalidation requires a complete final checkpoint')
        state['completed_stages']=[name for name in state['completed_stages'] if name!=options.revalidate_stage]
    adopted_identity=None
    def identity(pid):
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
            return fields[19] if fields[0]!='Z' else None
        except (OSError,IndexError):return None
    if options.adopt_pid:
        if state.get('stage')!=options.adopt_stage or state.get('child_pid')!=options.adopt_pid:
            raise RuntimeError('Adopted worker must match the existing pipeline status')
        command=Path(f'/proc/{options.adopt_pid}/cmdline').read_bytes().split(b'\0')
        if b'experiments/nucleus4d_ptir_full_train.py' not in command or options.adopt_stage.encode() not in command:
            raise RuntimeError('Adopted process is not the expected training worker')
        adopted_identity=identity(options.adopt_pid)
        if adopted_identity is None:raise RuntimeError('Adopted worker has already exited')
    if 'error' in state:
        state.setdefault('previous_errors',[]).append(dict(error=state.pop('error'),updated=state.get('updated')))
    state.update(pid=os.getpid(),status='running',storage=str(STORAGE),deliverable='arbitrary-camera daylight viewer',
                 viewer_url='http://localhost:8770/',
                 settings={'geometry_steps':30000,'inverse_steps':16000,'training_crop':512,
                           'training_spp':64,'training_bounces':4,'prior_long_edge':1024,'prior_steps':50,
                           'export_width':3840,'export_spp':1024,'export_bounces':8,
                           'float_precision':'float32 training, EXR and browser preview; float16 weak priors',
                           'initial_gaussians':6100978,'densification_cap':8000000,
                           'max_growth_per_update':100000,'training_views':7446,'held_out_views':504,
                           'checkpoint_every_steps':1000,
                           'cuda_allocator':os.environ.get('PYTORCH_ALLOC_CONF','native'),
                           'validation_scope':'24 fixed held-out native 256-pixel crops per evaluation'} )
    write(state_path,state)
    def stage(name,arguments):
        if name in state['completed_stages']:return
        if shutil.disk_usage(STORAGE).free < 20*2**30:raise RuntimeError('E: free space below the 20 GiB reserve')
        log=LOCAL/(name+'.log');state.update(stage=name,log=log.name,updated=time.time());write(state_path,state)
        with log.open('a') as f:
            f.write(f'\nSTART {time.time()} {arguments!r}\n');f.flush()
            if options.adopt_stage==name:
                f.write(f'ADOPT running worker {options.adopt_pid}; no training restart\n');f.flush()
                state['child_pid']=options.adopt_pid;write(state_path,state)
                while identity(options.adopt_pid)==adopted_identity:
                    state['updated']=time.time();write(state_path,state);time.sleep(15)
                checkpoint=json.loads((STORAGE/name/'checkpoint.json').read_text())
                after=json.loads((STORAGE/name/'after.json').read_text())
                expected=30000 if name=='geometry' else 16000
                if checkpoint['step']<expected or after['step']<expected or len(after.get('views',[]))!=24:
                    raise RuntimeError(f'Adopted {name} worker exited before complete training and evaluation; see {log.name}')
            else:
                process=subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                state['child_pid']=process.pid;write(state_path,state)
                while process.poll() is None:
                    state['updated']=time.time();write(state_path,state);time.sleep(15)
                if process.returncode:raise RuntimeError(f'{name} exited {process.returncode}; see {log.name}')
        state['completed_stages'].append(name);state['child_pid']=None;write(state_path,state)
    def quality_gate(stage_name,reference,actual,psnr_drop,ssim_drop):
        before=json.loads(reference.read_text());after=json.loads(actual.read_text())
        passed=after['psnr']>=before['psnr']-psnr_drop and after['ssim']>=before['ssim']-ssim_drop
        report=dict(passed=passed,reference=before,actual=after,allowed_psnr_drop=psnr_drop,
                    allowed_ssim_drop=ssim_drop,claim='Original-lighting appearance check only; not relighting ground truth')
        write(LOCAL/(stage_name+'.json'),report)
        if not passed:
            raise RuntimeError(f'{stage_name}: original-lighting reconstruction failed the quality gate; daylight publication stopped')
    try:
        for key in ('geometry_gradient_check','inverse_gradient_check','replay_validation'):
            result=json.loads((LOCAL/(key+'.json')).read_text())
            if not result.get('passed',result.get('finite',False)):raise RuntimeError(f'Preflight failed: {key}')
        if shutil.disk_usage(STORAGE).free < 80*2**30:raise RuntimeError('Full job needs 80 GiB free including temporary checkpoints')
        stage('priors',['experiments/nucleus4d_ptir_full_priors.py'])
        frames=json.loads((LOCAL/'dataset.json').read_text())['frames']
        missing=[r['name'] for r in frames if r['split']=='train' and not prior_path(r).exists()]
        if missing:raise RuntimeError(f'{len(missing)} training priors missing')
        for name in ('geometry','inverse'):
            args=['experiments/nucleus4d_ptir_full_train.py','--stage',name,'--crop','512','--spp','64']
            checkpoint=STORAGE/name/'resume.pt'
            if checkpoint.exists() and name not in state['completed_stages']:args+=['--resume',str(checkpoint)]
            stage(name,args)
            if name=='geometry':
                quality_gate('geometry_quality',STORAGE/'geometry/before.json',STORAGE/'geometry/after.json',.5,.02)
            else:
                quality_gate('inverse_quality',STORAGE/'geometry/after.json',STORAGE/'inverse/after.json',2.0,.05)
        if options.static_exports:
            stage('reference_4k',['experiments/nucleus4d_ptir_full_render.py','--reference'])
            stage('daylight_4k',['experiments/nucleus4d_ptir_full_render.py'])
        stage('viewer',['experiments/nucleus4d_ptir_full_ui.py','--results'])
        try:
            with urllib.request.urlopen('http://127.0.0.1:8770/api/status',timeout=3) as response:
                running=json.load(response)
            if running.get('probe') or running.get('quality',{}).get('spp')!=1024:
                raise RuntimeError('Port 8770 is occupied by a different rendering service')
        except OSError:
            with (LOCAL/'freeview.log').open('ab',buffering=0) as log:
                viewer=subprocess.Popen([sys.executable,'-u','experiments/nucleus4d_ptir_freeview.py'],cwd=ROOT,
                                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['viewer_pid']=viewer.pid
            for attempt in range(30):
                if viewer.poll() is not None:raise RuntimeError('Free-view service failed to start; see freeview.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8770/api/status',timeout=1) as response:json.load(response)
                    break
                except OSError:time.sleep(1)
            else:raise RuntimeError('Free-view service startup timed out; see freeview.log')
        state.update(status='completed',stage='completed',finished=time.time())
    except Exception as exc:
        state.update(status='needs_attention',error=str(exc),updated=time.time())
        write(state_path,state);raise
    finally:
        state['updated']=time.time();write(state_path,state)


if __name__=='__main__':main()
