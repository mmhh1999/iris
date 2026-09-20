"""One-command local evidence cycle. Completion is not a scientific success claim.

Run IDs are unique; failures and subprocess logs are retained. No downloads,
training, publications or data changes are performed by this validation cycle.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=Path(args.out_root)/run_id;out.mkdir(parents=True,exist_ok=False)
    state=dict(run_id=run_id,status='running',stages=[])
    commands=[('tests',[sys.executable,'-m','unittest','discover','-s','tests','-v']),
              ('real',[sys.executable,'experiments/real_sun_visibility.py','--out',str(out/'real')]),
              ('portal_ablation',[sys.executable,'experiments/portal_ablation.py','--out',str(out/'portal_ablation')]),
              ('synthetic',[sys.executable,'experiments/heldout_sun_control.py','--out',str(out/'synthetic')])]
    if args.screen:
        root=ROOT/'data_download/scannetpp/data'
        scenes=[p.name for p in sorted(root.iterdir()) if (p/'scans/mesh_aligned_0.05.ply').exists() and (p/'dslr/nerfstudio/transforms_undistorted.json').exists() and (p/'dslr/resized_undistorted_images').is_dir()]
        commands.append(('screen',[sys.executable,'experiments/scannetpp_geometry_audit.py','--images','8','--out',str(out/'screen'),'--scenes',*scenes]))
    try:
        for stage,command in commands:
            print('Running',stage,'->',out,flush=True);start=time.perf_counter()
            with (out/(stage+'.log')).open('w') as log:
                proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['stages'].append(dict(stage=stage,command=command,exit_code=proc.returncode,seconds=time.perf_counter()-start))
            (out/'run.json').write_text(json.dumps(state,indent=2))
            if proc.returncode:raise RuntimeError(f'{stage} failed; see {out/stage}.log')
            manifest=out/stage/'manifest.json'
            if manifest.exists():
                m=json.loads(manifest.read_text())
                for name in m.get('sources',m.get('source_hashes',{})):
                    dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        r=json.loads((out/'real/results.json').read_text());s=json.loads((out/'synthetic/results.json').read_text())
        ablation=json.loads((out/'portal_ablation/results.json').read_text())
        rows=[x for x in r['images'] if x['split']=='heldout_view']
        mean=lambda key:sum(x['metrics'][key]['iou'] for x in rows)/len(rows)
        state['findings']=dict(real_physical_proxy_iou=mean('fitted'),real_memory_proxy_iou=mean('memorized_floor_1nn'),
                              real_physical_beats_memory=mean('fitted')>mean('memorized_floor_1nn'),
                              implicit_mesh_openings_proxy_iou=ablation['heldout_mean_iou'],
                              synthetic=s['summary'],claim_limit='Component validation only; no full IRIS comparison or real heldout-illumination validation.')
        state['status']='completed'
    except Exception as exc:
        state['status']='failed';state['error']=str(exc);raise
    finally:
        (out/'run.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state['findings'],indent=2));print('Evidence:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out-root',default=str(ROOT/'experiments/out/validation_cycles'))
    p.add_argument('--screen',action='store_true',help='Also repeat all locally ready scenes (roughly 1-2 minutes CPU).')
    run(p.parse_args())
