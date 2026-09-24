"""Check step counts, frozen geometry and executed material constraints after training."""
import argparse,json,hashlib
from pathlib import Path
import torch,numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.run/'config.json').read_text());warm=cfg['warmup_steps'];last=warm+cfg['material_steps']
wc=a.run/'warmup'/f'chkpnt{warm}.pth';mc=a.run/'material'/f'chkpnt{last}.pth'
w,wi=torch.load(wc,map_location='cpu',weights_only=False);m,mi=torch.load(mc,map_location='cpu',weights_only=False)
assert wi==warm and mi==last
# capture() order is pinned SGS: degree, xyz, normal, SH dc/rest, scale, rotation, opacity.
frozen={name:torch.equal(w[i],m[i]) for i,name in [(1,'positions'),(2,'normals'),(3,'SH_dc'),(4,'SH_rest'),(5,'scales'),(6,'rotations'),(7,'opacity')]}
assert all(frozen.values()),frozen
h=[json.loads(s) for s in (a.run/'history.jsonl').read_text().splitlines()];mat=[r for r in h if r['phase']=='material'];pre=[r for r in h if r['phase']=='warmup']
assert len(pre)==warm and len(mat)==cfg['material_steps']
assert [r['iteration'] for r in mat]==list(range(warm+1,last+1))
assert all(np.isfinite(r['loss_total_with_invariance']) for r in h)
cross=sum(int(r['cross_view_active']) for r in mat);self_inv=sum(int(r['self_invariance_active']) for r in mat)
assert cross>0 and self_inv>0
material_updates={}
for i,name in [(16,'base_color'),(17,'roughness')]:
    assert torch.isfinite(m[i]).all()
    fraction=float((m[i]!=0).float().mean());assert fraction>0
    material_updates[name]=dict(finite=True,changed_from_zero_initialization_fraction=fraction)
light_steps={}
for f in ['point_light','env_light','visibility_enc']:
    assert (a.run/'material'/f'{f}_chkpnt{last}.pth').is_file(),f
    if f!='visibility_enc':
        state,opt,iteration=torch.load(a.run/'material'/f'{f}_chkpnt{last}.pth',map_location='cpu',weights_only=False)
        assert iteration==last and all(torch.isfinite(x).all() for x in state.values())
        steps=sorted(set(float(s['step']) for s in opt['state'].values() if 'step' in s))
        assert steps==[float(cfg['material_steps'])],(f,steps)
        light_steps[f]=steps
def digest(p):
    sha=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
    return sha.hexdigest()
report=dict(passed=True,warmup_checkpoint_step=wi,material_checkpoint_step=mi,frozen_parameters_exactly_equal=frozen,cross_view_steps=cross,self_invariance_steps=self_inv,finite_losses=True,material_updates=material_updates,light_optimizer_steps=light_steps,checkpoint_sha256={str(p.relative_to(a.run)):digest(p) for p in [wc,*sorted((a.run/'material').glob('*.pth'))]},scientific_convergence_established=False)
(a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
