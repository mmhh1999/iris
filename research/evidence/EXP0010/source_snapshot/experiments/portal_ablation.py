"""EXP0010: explicit annotated apertures vs implicit openings in the same mesh.

Both models have full receiver and occluder geometry. This does NOT remove
windows from the mesh and must not be described as a no-window-geometry model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mitsuba as mi
from experiments.real_sun_visibility import observation,metrics
from utils.multiview_geometry import intersect_rays
from utils.solar_geometry import sun_vector_world
from utils.geometry_screening import validate_scannetpp_bounds


def sky_visibility(points,directions,scene):
    p=np.asarray(points);directions=np.asarray(directions);directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    result=[]
    for begin in range(0,len(directions),32):
        ds=directions[begin:begin+32];rays=np.broadcast_to(ds[:,None,:],(len(ds),len(p),3)).copy()
        origins=np.broadcast_to(p,rays.shape)+.002*rays
        _,valid,_=intersect_rays(scene,origins.reshape(-1,3),rays.reshape(-1,3))
        result.append((~valid).reshape(len(ds),len(p)))
    return np.concatenate(result)


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();mi.set_variant('llvm_ad_rgb')
    cfg=json.loads(Path(args.config).read_text());folder=ROOT/'data_download/scannetpp/data'/cfg['scene']
    tf=folder/'dslr/nerfstudio/transforms_undistorted.json';mesh=folder/'scans/mesh_aligned_0.05.ply';d=json.loads(tf.read_text())
    scene=mi.load_dict({'type':'scene','mesh':{'type':'ply','filename':str(mesh)}})
    validate_scannetpp_bounds(d['aabb_range'],[np.asarray(scene.bbox().min),np.asarray(scene.bbox().max)])
    train=observation(cfg['train_image'],d,scene,folder,cfg)
    def fit(angles):
        pred=sky_visibility(train['points'],[sun_vector_world(*v) for v in angles],scene)
        scores=np.array([metrics(p,train['labels'])['iou'] for p in pred]);i=int(scores.argmax())
        return angles[i],float(scores[i])
    (a,e),_=fit([(a,e) for a in range(0,360,10) for e in range(5,90,5)])
    (a,e),score=fit([(float(a0%360),float(e0)) for a0 in np.arange(a-10,a+10.1) for e0 in np.arange(max(1,e-5),min(89,e+5)+.1)])
    rows=[]
    for split,name in [('train',cfg['train_image'])]+[('heldout_view',n) for n in cfg['heldout_images']]:
        obs=train if split=='train' else observation(name,d,scene,folder,cfg)
        pred=sky_visibility(obs['points'],[sun_vector_world(a,e)],scene)[0]
        rows.append(dict(split=split,**obs['info'],metrics=metrics(pred,obs['labels'])))
    result=dict(experiment_id='EXP0010',recovered_az=a,recovered_el=e,train_iou=score,images=rows,
                heldout_mean_iou=float(np.mean([r['metrics']['iou'] for r in rows[1:]])),elapsed_seconds=time.perf_counter()-start,
                disclosure='Same geometry, implicit mesh openings, no explicit aperture labels. Geometry fixed; brightness labels are proxies. Not IRIS.')
    sources=['experiments/portal_ablation.py','experiments/real_sun_visibility.py','utils/multiview_geometry.py','utils/geometry_screening.py','utils/solar_geometry.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    manifest=dict(config=cfg,source_hashes=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        data_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [tf,mesh]},mitsuba=mi.__version__,numpy=np.__version__)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/daylight/exp0008_real_visibility.json'));p.add_argument('--out',default=str(ROOT/'experiments/out/EXP0010'));run(p.parse_args())
