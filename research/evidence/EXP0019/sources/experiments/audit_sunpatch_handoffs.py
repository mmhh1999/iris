"""Verify final step counts, actual IRIS rays, and floor emitter classification."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.dataset.synthetic_ldr import get_ray_directions,get_rays
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
b=Path('experiments/out/EXP0019_sunpatch');rows=[]
for condition in ['sun_a','sun_b','sky']:
    root=b/('iris_'+condition+'_v2');folder=root/'checkpoints'/('sunpatch_'+root.name)
    checks=[]
    for name in ['init','last_0','last_1']:
        p=folder/(name+'.ckpt');c=torch.load(p,map_location='cpu',weights_only=False)
        step=c['global_step'];assert step==200,(p,step)
        checks.append({'name':name,'global_step':step,'bytes':p.stat().st_size})
        del c
    data=b/'data_03'/condition;scene=mi.load_dict({'type':'scene','shape_id':{'type':'obj','filename':str((data/'scene.obj').resolve())}})
    flags=torch.load(root/'bake/emitter.pth',map_location='cpu',weights_only=False)['is_emitter'].numpy()
    metadata=json.loads((data/'val/transforms.json').read_text());geometry=[]
    for obs in metadata['frames']:
        v=obs['view'];g=np.load(b/'data_03/generator_truth'/f'{condition}_view{v:02d}.npz');mask=g['floor']
        dirs=get_ray_directions(96,128,64/np.tan(metadata['camera_angle_x']/2))
        origin,direction=get_rays(dirs,torch.tensor(obs['transform_matrix'],dtype=torch.float32)[:3,:4])
        o=origin.numpy();d=direction.numpy();ray=mi.Ray3f(mi.Point3f(mi.Float(o[:,0]),mi.Float(o[:,1]),mi.Float(o[:,2])),mi.Vector3f(mi.Float(d[:,0]),mi.Float(d[:,1]),mi.Float(d[:,2])))
        si=scene.ray_intersect(ray);p=np.array(si.p).T.reshape(96,128,3)
        error=np.linalg.norm(p[mask]-g['position'][mask],axis=-1);assert error.max()<1e-4,error.max()
        idx=np.array(si.prim_index).reshape(96,128)[mask]
        geometry.append({'view':v,'max_floor_position_error':float(error.max()),'floor_pixels_classified_emitter':int(flags[idx].sum()),'floor_pixels':int(mask.sum())})
    rows.append({'condition':condition,'checkpoints':checks,'emitter_triangle_count':int(flags.sum()),'geometry':geometry})
(b/'handoff_geometry_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
