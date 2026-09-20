"""Reproject the captured appearance texture to audit camera/mesh alignment.

This is deliberately NOT a relighting render or a recovered BRDF. Texture values
contain captured illumination. Missing mesh pixels are magenta in diagnostics.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import mitsuba as mi


def run(root, scene_name, variant):
    mi.set_variant(variant)
    out=root/scene_name
    data=json.loads((out/'manifest.json').read_text())
    mesh=Path(data['mesh'])
    scene=mi.load_dict({'type':'scene','mesh':{'type':'obj','filename':str(mesh),
        'bsdf':{'type':'diffuse'}}})
    texture=cv2.imread(str(mesh.with_suffix('.jpg')))
    if texture is None:
        raise FileNotFoundError(mesh.with_suffix('.jpg'))
    th,tw=texture.shape[:2]
    rows=[];stats=[]
    for c in data['cameras']:
        h,w=256,171
        K=np.array(c['K']);K[0]*=w/c['width'];K[1]*=h/c['height']
        pose=np.array(c['c2w_opencv'])
        yy,xx=np.mgrid[:h,:w]
        # OpenCV calibration/remap uses integer pixel centers.
        dirs=np.c_[xx.ravel(),yy.ravel(),np.ones(h*w)]@np.linalg.inv(K).T
        dirs=dirs@pose[:3,:3].T;dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        ray=mi.Ray3f(mi.Point3f(*pose[:3,3].tolist()),mi.Vector3f(*dirs.astype('float32').T))
        si=scene.ray_intersect(ray)
        valid=np.array(si.is_valid()).reshape(h,w)
        uv=np.stack([np.asarray(si.uv.x),np.asarray(si.uv.y)],-1).reshape(h,w,2)
        # Mitsuba's OBJ loader flips OBJ V coordinates during loading.
        tx=(uv[...,0]*tw-.5).astype('float32')
        ty=(uv[...,1]*th-.5).astype('float32')
        projected=cv2.remap(texture,tx,ty,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        projected[~valid]=[255,0,255]
        normal=np.stack([np.asarray(si.sh_frame.n[i]) for i in range(3)],-1).reshape(h,w,3)
        normal=np.uint8(np.clip((normal+1)*127.5,0,255))[...,::-1]
        normal[~valid]=0
        stem=Path(c['file']).stem
        ref=cv2.resize(cv2.imread(str(out/(stem+'.jpg'))),(w,h))
        row=np.hstack([ref,projected,normal])
        cv2.putText(row,c['camera_id'],(4,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
        cv2.imwrite(str(out/(stem+'_geometry.jpg')),row)
        # Retain geometry for diagnostics only; not photo-only estimator inputs.
        np.savez_compressed(out/(stem+'_geometry.npz'),valid=valid,
                            depth=np.asarray(si.t).reshape(h,w),uv=uv,
                            K=K,c2w=pose)
        if len(rows)<8 and c['camera_id'].startswith(('19/','20/')):
            rows.append(row)
        mask=cv2.resize(cv2.imread(str(out/c['mask']),0),(w,h),interpolation=cv2.INTER_NEAREST)>0
        stats.append(dict(camera_id=c['camera_id'],mesh_hit_fraction=float(valid[mask].mean())))
    cv2.imwrite(str(out/'geometry_comparison.jpg'),np.vstack(rows))
    report=dict(scene=scene_name,variant=variant,mitsuba=mi.__version__,
        triangles=sum(s.primitive_count() for s in scene.shapes()),views=stats,
        diagnostic='captured texture reprojection, NOT relighting or material estimation',
        columns=['real HDR preview','captured mesh texture','geometry normals'])
    (out/'geometry_report.json').write_text(json.dumps(report,indent=2))
    print(scene_name,report['triangles'],'triangles; projections complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('experiments/out/EXP0014_eyeful'))
    p.add_argument('--scenes',nargs='+',default=['riverview','apartment'])
    p.add_argument('--variant',default='cuda_ad_rgb')
    a=p.parse_args()
    for s in a.scenes:run(a.root,s,a.variant)
