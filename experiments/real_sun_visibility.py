"""EXP0008: falsifiable one-view fit / disjoint-view daylight-visibility proxy.

Does not optimize on heldout views. Approximate apertures and brightness masks
are hypotheses, explicitly not ground truth. Original IRIS is not a comparator.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import platform
import resource
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import mitsuba as mi
from PIL import Image
from scipy.spatial import cKDTree
from utils.geometry_screening import pinhole_rays,scannetpp_pose_in_mesh_frame,validate_scannetpp_bounds
from utils.multiview_geometry import intersect_rays,sunlight_visibility
from utils.solar_geometry import sun_vector_world


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(pred,obs):
    pred=np.asarray(pred,bool);obs=np.asarray(obs,bool)
    tp=int(np.sum(pred&obs));fp=int(np.sum(pred&~obs));fn=int(np.sum(~pred&obs));tn=int(np.sum(~pred&~obs))
    return dict(iou=tp/max(1,tp+fp+fn),balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),
                tp=tp,fp=fp,fn=fn,tn=tn,n=len(obs))


def observation(name,d,scene,folder,cfg):
    f=next(f for f in d['frames'] if f['file_path']==name)
    if f.get('is_bad',False):raise ValueError('bad image')
    path=folder/'dslr/resized_undistorted_images'/name
    image=cv2.imread(str(path));h,w=image.shape[:2]
    if (h,w)!=(d['h'],d['w']):raise ValueError('image size mismatch')
    mask_path=folder/'dslr/resized_undistorted_masks'/Path(name).with_suffix('.png')
    valid_mask=cv2.imread(str(mask_path),0)
    if valid_mask is None or valid_mask.shape!=(h,w):raise ValueError('missing validity mask')
    K=np.array([[d['fl_x'],0,d['cx']],[0,d['fl_y'],d['cy']],[0,0,1.]])
    pose=scannetpp_pose_in_mesh_frame(f['transform_matrix'])
    yy,xx=np.mgrid[0:h:cfg['pixel_stride'],0:w:cfg['pixel_stride']]
    px=np.column_stack((xx.ravel(),yy.ravel()))
    origins,dirs=pinhole_rays(px,K,pose);points,valid,_=intersect_rays(scene,origins,dirs)
    floor=valid&(np.abs(points[:,2]-cfg['floor_z'])<cfg['floor_tolerance'])&(valid_mask[px[:,1],px[:,0]]==255)
    ids=np.flatnonzero(floor)
    if len(ids)<200:raise ValueError('insufficient floor support')
    ids=np.random.default_rng(cfg['seed']).choice(ids,min(cfg['max_points'],len(ids)),replace=False)
    px=px[ids];points=points[ids]
    lum=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)[px[:,1],px[:,0]]
    threshold,_=cv2.threshold(lum,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labels=lum>threshold
    exif=Image.open(path).getexif()
    info=dict(image=name,n_floor_samples=len(ids),proxy_threshold=float(threshold),bright_fraction=float(labels.mean()),
              image_sha256=digest(path),mask_sha256=digest(mask_path),datetime_exif=exif.get(306),
              floor_z_median=float(np.median(points[:,2])))
    return dict(points=points,labels=labels,pixels=px,image=image,info=info)


def main(args):
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    mi.set_variant('llvm_ad_rgb');start=time.perf_counter()
    if cfg['train_image'] in cfg['heldout_images']:raise ValueError('split leakage')
    folder=Path(args.data_root)/cfg['scene'];tf=folder/'dslr/nerfstudio/transforms_undistorted.json';mesh=folder/'scans/mesh_aligned_0.05.ply'
    d=json.loads(tf.read_text())
    if d['camera_model']!='PINHOLE':raise ValueError('requires pinhole')
    scene=mi.load_dict({'type':'scene','mesh':{'type':'ply','filename':str(mesh.resolve())}})
    validate_scannetpp_bounds(d['aabb_range'],[np.asarray(scene.bbox().min),np.asarray(scene.bbox().max)])
    sources=['experiments/real_sun_visibility.py','utils/multiview_geometry.py','utils/geometry_screening.py','utils/solar_geometry.py']
    manifest=dict(experiment_id=cfg['experiment_id'],config=cfg,config_sha256=digest(args.config),
        sources={s:digest(ROOT/s) for s in sources},mesh_sha256=digest(mesh),transforms_sha256=digest(tf),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True,cwd=ROOT),
        versions=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,mitsuba=mi.__version__),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    train=observation(cfg['train_image'],d,scene,folder,cfg)
    def evaluate_angles(angles):
        directions=np.array([sun_vector_world(az,el) for az,el in angles])
        pred=sunlight_visibility(train['points'],directions,scene,cfg['apertures'],cfg['shadow_tolerance'])
        scores=np.array([metrics(p,train['labels'])['iou'] for p in pred])
        return scores
    angles=[(a,e) for a in np.arange(0,360,cfg['coarse_az_step']) for e in np.arange(5,90,cfg['coarse_el_step'])]
    scores=evaluate_angles(angles);best=int(scores.argmax());a,e=angles[best]
    coarse_top=[dict(az=float(angles[i][0]),el=float(angles[i][1]),iou=float(scores[i])) for i in np.argsort(-scores)[:10]]
    fine=[(az%360,el) for az in np.arange(a-10,a+10.1,1) for el in np.arange(max(1,e-5),min(89,e+5)+.1,1)]
    fs=evaluate_angles(fine);bi=int(fs.argmax());az,el=map(float,fine[bi]);direction=sun_vector_world(az,el)
    print('Train-only fitted direction',az,el,'proxy IoU',float(fs[bi]),flush=True)
    (out/'train_fit.json').write_text(json.dumps(dict(azimuth_scene_deg=az,elevation_deg=el,coarse_top=coarse_top),indent=2))
    variants=[('fitted',direction,cfg['apertures'],cfg['shadow_tolerance']),
              ('wrong_direction_plus45',sun_vector_world((az+45)%360,el),cfg['apertures'],cfg['shadow_tolerance'])]
    for ap in cfg['apertures']:variants.append((ap['name']+'_only',direction,[ap],cfg['shadow_tolerance']))
    for shift in [-.1,.1,1.]:
        apertures=copy.deepcopy(cfg['apertures'])
        for ap in apertures:ap['endpoints_xy']=(np.array(ap['endpoints_xy'])+[shift,0]).tolist()
        variants.append((f'window_shift_x_{shift:g}m',direction,apertures,cfg['shadow_tolerance']))
    for tol in [.02,.08]:variants.append((f'visibility_tolerance_{tol:g}m',direction,cfg['apertures'],tol))
    rows=[]
    memory=cKDTree(train['points'])
    # Heldout observations are first loaded after the direction has been frozen.
    for split,name in [('train',cfg['train_image'])]+[('heldout_view',n) for n in cfg['heldout_images']]:
        obs=train if split=='train' else observation(name,d,scene,folder,cfg)
        predictions={tag:sunlight_visibility(obs['points'],[vec],scene,aps,tol)[0] for tag,vec,aps,tol in variants}
        predictions['all_bright']=np.ones(len(obs['labels']),bool)
        predictions['all_dark']=np.zeros(len(obs['labels']),bool)
        distance,nearest=memory.query(obs['points'])
        predictions['memorized_floor_1nn']=train['labels'][nearest]
        row=dict(split=split,**obs['info'],memory_coverage_within_10cm=float(np.mean(distance<=.1)),metrics={tag:metrics(p,obs['labels']) for tag,p in predictions.items()})
        rows.append(row)
        vis=obs['image'].copy();px=obs['pixels'];pr=predictions['fitted'];truth=obs['labels']
        colors=np.where((pr==truth)[:,None],np.array([0,200,0]),np.array([0,0,255]))
        for xy,c in zip(px,colors):cv2.circle(vis,tuple(xy),2,tuple(map(int,c)),-1)
        cv2.imwrite(str(out/(Path(name).stem+'_agreement.jpg')),cv2.resize(vis,(876,584)))
        np.savez_compressed(out/(Path(name).stem+'_samples.npz'),points=obs['points'],pixels=px,proxy_labels=truth,predicted=pr)
        print(split,name,row['metrics']['fitted'],flush=True)
    result=dict(fit=dict(azimuth_scene_deg=az,elevation_deg=el,train_proxy_iou=float(fs[bi]),coarse_top=coarse_top),
                images=rows,elapsed_seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                limitations=cfg['evaluation'],aperture_provenance=cfg['annotation_provenance'])
    (out/'results.json').write_text(json.dumps(result,indent=2))
    lines=['| Model | Train proxy IoU | Mean heldout-view proxy IoU |','|---|---:|---:|']
    for tag in rows[0]['metrics']:
        lines.append(f"| {tag} | {rows[0]['metrics'][tag]['iou']:.4f} | {np.mean([r['metrics'][tag]['iou'] for r in rows[1:]]):.4f} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/daylight/exp0008_real_visibility.json'))
    p.add_argument('--data-root',default=str(ROOT/'data_download/scannetpp/data'));p.add_argument('--out',default=str(ROOT/'experiments/out/EXP0008'))
    main(p.parse_args())
