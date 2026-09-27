"""Matched-ray, fixed-material, unsaturated-floor and sampling checks."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_sunpatch_benchmark import mi,exr,cv2
b=Path('experiments/out/EXP0019_sunpatch/data_03');out=b.parent/'quality';out.mkdir(exist_ok=False)
m=json.loads((b/'manifest.json').read_text());rows=[]
for condition in ['sun_a','sun_b','sky']:
    c=next(x for x in m['conditions'] if x['name']==condition)
    for view in [3,9]:
        row=next(v for v in c['views'] if v['view']==view);g=np.load(b/'generator_truth'/f'{condition}_view{view:02d}.npz');g0=np.load(b/'generator_truth'/f'sky_view{view:02d}.npz')
        assert np.array_equal(g['floor'],g0['floor']);mask=g['floor']
        np.testing.assert_allclose(g['position'][mask],g0['position'][mask],atol=1e-6)
        f=b/condition/'val/Image'/f"{row['index']:03d}_0001.exr";im=cv2.imread(str(f),-1)[...,::-1]
        second=np.array(mi.render(mi.load_file(str((b/'generator_truth'/f'{condition}_view{view:02d}.xml').resolve())),spp=1024,seed=202))
        exr(out/f'{condition}_view{view}_seed202.exr',second)
        rmse=float(np.sqrt(((im[mask]-second[mask])**2).mean()));r={'condition':condition,'view':view,'floor_seed_rmse':rmse,'floor_seed_rmse_div_mean':rmse/float(im[mask].mean()),'floor_saturated_fraction':float((im[mask]>=1).any(-1).mean()),'floor_pixels':int(mask.sum())};rows.append(r);print(json.dumps(r),flush=True)
(out/'report.json').write_text(json.dumps({'matching_floor_positions_across_conditions':True,'spp':1024,'seeds':[101,202],'views':rows},indent=2))
