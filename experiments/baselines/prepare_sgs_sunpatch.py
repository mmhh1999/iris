"""P1 dataset: known mesh/cameras, photo-only appearance, fixed 12/2 split.
No BRDF or solar ground truth is read or exported into training inputs.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np,mitsuba as mi
mi.set_variant('llvm_ad_rgb')
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
source=a.source.resolve();scene=mi.load_dict({'type':'scene','mesh':{'type':'obj','filename':str(source/'scene.obj'),'face_normals':True}})
rows=[];ps=[];ns=[];camera_checks=[]
for split in ['train','val']:
 t=json.loads((source/'sun_a'/split/'transforms.json').read_text());fov=t['camera_angle_x']
 for row in t['frames']:
  pose=np.array(row['transform_matrix']);h,w=96,128
  sensor=mi.load_dict({'type':'perspective','to_world':mi.ScalarTransform4f(pose),'fov':float(np.rad2deg(fov)),'fov_axis':'x','film':{'type':'hdrfilm','width':w,'height':h,'sample_border':False}})
  yy,xx=np.mgrid[:h,:w]
  ray,_=sensor.sample_ray(0,0,mi.Point2f(mi.Float(((xx+.5)/w).ravel()),mi.Float(((yy+.5)/h).ravel())),mi.Point2f(.5))
  si=scene.ray_intersect(ray);pos=np.array(si.p).T.reshape(h,w,3);normal=np.array(si.n).T.reshape(h,w,3);valid=np.array(si.is_valid()).reshape(h,w)
  c2w=pose@np.diag([-1.,-1.,1.,1.]);w2c=np.linalg.inv(c2w)
  pc=pos@w2c[:3,:3].T+w2c[:3,3];f=w/(2*np.tan(fov/2));uv=pc[...,:2]/np.maximum(pc[...,2:],1e-9)*f+[w/2,h/2]
  err=float(np.max(np.abs(uv[valid]-np.stack([xx+.5,yy+.5],-1)[valid])));assert err<.002,err
  camera_checks.append(err)
  if split=='train':ps.append(pos[valid]);ns.append(normal[valid])
  rows.append(dict(view=row['view'],split=split,index=row['index'],c2w=c2w.tolist()))
points=np.concatenate(ps);normals=np.concatenate(ns)
# Deduplicate geometry only. Shared across lighting conditions; never fit heldout RGB.
_,idx=np.unique(np.round(points/.015).astype(np.int64),axis=0,return_index=True)
points,normals=points[idx],normals[idx]
center=(points.max(0)+points.min(0))/2;scale=float((points.max(0)-points.min(0)).max()/2)
np.savez_compressed(a.out/'geometry.npz',positions=(points-center)/scale,normals=normals,center=center,scale=scale)
for r in rows:
 c=np.array(r['c2w']);c[:3,3]=(c[:3,3]-center)/scale;r['c2w']=c.tolist()
for condition in ['sun_a','sky']:
 d=a.out/condition;d.mkdir()
 for r in rows:
  dest=d/r['split'];dest.mkdir(exist_ok=True)
  f=source/condition/r['split']/'Image'/f"{r['index']:03d}_0001.exr"
  im=cv2.imread(str(f),-1)[...,::-1];linear=np.clip(im,0,1)
  srgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
  cv2.imwrite(str(dest/f"view{r['view']:02d}.png"),np.uint8(np.clip(srgb,0,1)*255+.5)[...,::-1])
manifest=dict(protocol='P1 known geometry material-stage diagnostic; not full SGS reconstruction',source=str(source),conditions=['sun_a','sky'],width=128,height=96,fov_x=float(fov),frames=rows,point_count=len(points),camera_projection_max_error=max(camera_checks),mesh_sha256=hashlib.sha256((source/'scene.obj').read_bytes()).hexdigest(),image_mapping='fixed exposure=1, clip HDR to [0,1], standard sRGB; differs from EXP0019 Emor images',material_gt_in_training=False,sun_gt_in_training=False)
(a.out/'input.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='frames'},indent=2))
