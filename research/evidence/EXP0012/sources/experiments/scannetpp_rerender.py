"""Path-traced relighting on an unchanged real ScanNet++ room mesh.

Generate a frozen cross-light split, float radiance and geometric truth, then
fit a sunlight direction on one training observation. Test labels come from a
separate direct-light render, not the visibility predictor's output.
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iris-mpl')
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import mitsuba as mi
from utils.geometry_screening import scannetpp_pose_in_mesh_frame, pinhole_rays
from utils.multiview_geometry import intersect_rays, aperture_distances
from utils.solar_geometry import sun_vector_world, angular_error_deg

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8*1024*1024): h.update(b)
    return h.hexdigest()

def score(pred, label):
    tp = int(np.sum(pred & label)); fp = int(np.sum(pred & ~label)); fn = int(np.sum(~pred & label))
    return dict(iou=tp/max(1,tp+fp+fn),tp=tp,fp=fp,fn=fn,n=len(label))

def visible(scene, points, direction, offset):
    d = np.broadcast_to(np.asarray(direction), points.shape)
    _, hit, _ = intersect_rays(scene, points + offset*d, d)
    return ~hit

def save_image(path, img):
    img = np.asarray(img, dtype=np.float32)
    if not np.isfinite(img).all(): raise ValueError(f'Nonfinite render {path}')
    if not cv2.imwrite(str(path.with_suffix('.exr')), img[...,::-1]): raise IOError(path)
    preview = np.clip(img,0,1)**(1/2.2)
    cv2.imwrite(str(path.with_suffix('.png')), np.round(preview[...,::-1]*255).astype(np.uint8))

def run(args):
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    mi.set_variant(args.variant)
    start = time.perf_counter()
    folder = Path(args.data_root)/cfg['scene']
    mesh_path = folder/'scans/mesh_aligned_0.05.ply'
    transform_path = folder/'dslr/nerfstudio/transforms_undistorted.json'
    meta = json.loads(transform_path.read_text())
    w,h = cfg['resolution']
    fx = meta['fl_x']*w/meta['w']; fy_original = meta['fl_y']*h/meta['h']
    K = np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1.]])
    if abs(meta['cx']/meta['w']-.5)>1e-8 or abs(meta['cy']/meta['h']-.5)>1e-8:
        raise ValueError('This calibrated sensor path requires centered principal point')
    cameras=[]; sensors=[]
    for index,name in enumerate(cfg['views']):
        frame=next(f for f in meta['frames'] if f['file_path']==name)
        if frame.get('is_bad',False): raise ValueError('Bad camera frame')
        pose=scannetpp_pose_in_mesh_frame(frame['transform_matrix'])
        sensor=mi.load_dict({'type':'perspective','fov_axis':'x',
            'fov':float(np.degrees(2*np.arctan(w/(2*fx)))),
            'to_world':mi.ScalarTransform4f(pose @ np.diag([-1.,-1.,1.,1.])),
            'near_clip':.001,'far_clip':1000,
            'sampler':{'type':'independent','sample_count':cfg['spp']},
            'film':{'type':'hdrfilm','width':w,'height':h,'pixel_format':'rgb',
                    'rfilter':{'type':'box'}}})
        sensors.append(sensor);cameras.append(dict(name=name,K=K.tolist(),c2w=pose.tolist()))
        photo=cv2.imread(str(folder/'dslr/resized_undistorted_images'/name))
        cv2.imwrite(str(out/f'view_{index}_real_reference.jpg'),cv2.resize(photo,(w,h)))
    shape=mi.load_dict({'type':'ply','filename':str(mesh_path.resolve()),
        'bsdf':{'type':'twosided','bsdf':{'type':'diffuse','reflectance':cfg['diffuse_reflectance']}}})
    geometry=mi.load_dict({'type':'scene','mesh':shape})
    manifest=dict(config=cfg,variant=args.variant,mitsuba=mi.__version__,numpy=np.__version__,
        mesh_sha256=sha(mesh_path),transforms_sha256=sha(transform_path),
        triangle_count=shape.primitive_count(),cameras=cameras,
        original_render_scaled_fy=fy_original,render_fy=fx,
        fy_relative_error=abs(fx/fy_original-1),
        source_sha256=sha(__file__),config_sha256=sha(args.config),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(__file__,out/'source_snapshot.py')
    geometry_maps=[]; yy,xx=np.mgrid[:h,:w]
    for index,camera in enumerate(cameras):
        o,d=pinhole_rays(np.c_[xx.ravel(),yy.ravel()],K,camera['c2w'])
        ray,_=sensors[index].sample_ray(time=0.,sample1=0.,
            sample2=mi.Point2f(mi.Float((xx.ravel()+.5)/w),mi.Float((yy.ravel()+.5)/h)),sample3=mi.Point2f(.5,.5))
        sensor_d=np.column_stack([np.asarray(ray.d[i]) for i in range(3)])
        error=float(np.max(np.abs(sensor_d-d)))
        if error>2e-5:raise ValueError(f'Sensor coordinate mismatch {error}')
        si=geometry.ray_intersect(ray)
        pos=np.column_stack([np.asarray(si.p[i]) for i in range(3)])
        normal=np.column_stack([np.asarray(si.sh_frame.n[i]) for i in range(3)])
        normal=np.where((normal*d).sum(-1,keepdims=True)>0,-normal,normal)
        valid=np.asarray(si.is_valid()); depth=np.asarray(si.t)
        floor=valid&(np.abs(pos[:,2]-cfg['floor_z'])<cfg['floor_tolerance'])&(normal[:,2]>.8)
        # Official masks exclude undistortion padding and anonymized areas.
        mask=cv2.imread(str(folder/'dslr/resized_undistorted_masks'/Path(camera['name']).with_suffix('.png')),0)
        if mask is None:raise FileNotFoundError('Official validity mask required')
        usable=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST).ravel()==255
        floor &= usable
        np.savez_compressed(out/f'view_{index}_geometry.npz',position=pos.reshape(h,w,3),
            normal=normal.reshape(h,w,3),depth=depth.reshape(h,w),valid=valid.reshape(h,w),
            floor=floor.reshape(h,w),photo_usable=usable.reshape(h,w))
        geometry_maps.append(dict(pos=pos,normal=normal,floor=floor))
        print('CAMERA',index,'floor pixels',int(floor.sum()),'ray error',error,flush=True)
    observations=[]
    for condition in cfg['conditions']:
        sun=sun_vector_world(condition['azimuth'],condition['elevation'])
        common={'type':'scene','mesh':shape,'sun':{'type':'directional','direction':(-sun).tolist(),
                    'irradiance':{'type':'rgb','value':cfg['sun_irradiance']}}}
        for i,sensor in enumerate(sensors):common[f'camera_{i}']=sensor
        full=mi.load_dict(dict(common,integrator={'type':'path','max_depth':cfg['max_depth']},
            sky={'type':'constant','radiance':{'type':'rgb','value':cfg['sky_radiance']}}))
        direct=mi.load_dict(dict(common,integrator={'type':'direct','emitter_samples':1,'bsdf_samples':0}))
        for view in range(len(sensors)):
            g=geometry_maps[view];floor=g['floor'];n=g['normal']
            expected=cfg['diffuse_reflectance']/np.pi * np.mean(cfg['sun_irradiance']) * np.maximum(0,n@sun)
            for seed in cfg['seeds']:
                case=out/condition['name']/f'view_{view}_seed_{seed}';case.mkdir(parents=True)
                rgb=np.array(mi.render(full,sensor=view,spp=cfg['spp'],seed=seed))
                direct_rgb=np.array(mi.render(direct,sensor=view,spp=cfg['spp'],seed=seed+10000))
                save_image(case/'rgb',rgb);save_image(case/'direct_sun',direct_rgb)
                # Half the known fully illuminated direct radiance; edge pixels retain MC uncertainty.
                labels=(direct_rgb.mean(-1).ravel()>.5*expected)&(expected>1e-5)&floor
                cv2.imwrite(str(case/'direct_sun_proxy.png'),labels.reshape(h,w).astype(np.uint8)*255)
                observations.append(dict(condition=condition['name'],view=view,seed=seed,path=str(case.relative_to(out))))
                print('RENDER',condition['name'],view,seed,'sunlit floor',int(labels.sum()),flush=True)
        del full,direct
    (out/'observations.json').write_text(json.dumps(observations,indent=2))
    print('Rendering completed in', time.perf_counter()-start, 'seconds', flush=True)
    if not args.render_only:
        from experiments.evaluate_scannetpp_rerender import run as evaluate
        evaluate(out, scene=geometry)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/daylight/exp0012_scannetpp_rerender.json'))
    p.add_argument('--data-root',default=str(ROOT/'data_download/scannetpp/data'))
    p.add_argument('--out',required=True);p.add_argument('--variant',default='cuda_ad_rgb',choices=['cuda_ad_rgb','llvm_ad_rgb'])
    p.add_argument('--render-only', action='store_true')
    run(p.parse_args())
