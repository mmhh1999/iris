"""Bounded paired SGS material-stage run, followed by audits and heldout scoring."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];base=root/'experiments/out/EXP0022_sgs_sunpatch';scripts=root/'experiments/baselines';py=str(root/'.venv/bin/python')
inputs=base/'input_v2'
assert 'input_color_conversion' in json.loads((inputs/'prior_manifest.json').read_text())
for condition in ['sun_a','sky']:
    run=base/f'{condition}_v2_1000_600'
    cmd=[py,str(scripts/'launch_sgs_p1.py'),'--input',str(inputs),'--condition',condition,'--out',str(run),'--warmup','1000','--steps','600']
    with (base/f'{condition}_v2_1000_600.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200,cwd=root)
    subprocess.run([py,str(scripts/'audit_sgs_p1_run.py'),'--run',str(run)],check=True,timeout=120,cwd=root)
subprocess.run([py,str(scripts/'evaluate_sgs_sunpatch.py'),'--base',str(base),'--truth',str(root/'experiments/out/EXP0019_sunpatch/data_03'),'--input',str(inputs),'--suffix','v2_1000_600','--out',str(base/'evaluation_v2')],check=True,timeout=120,cwd=root)
(base/'pair_completed.json').write_text(json.dumps(dict(completed=True,scientific_convergence_established=False),indent=2)+'\n')
