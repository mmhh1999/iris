"""Bind original Gaussians to full-house receivers without editing the model."""
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from nucleus4d_baked_relight import SOURCE,PHOTOS,atomic_json
from nucleus4d_transport_geometry import OUT,BOUNDS


def main():
    with np.load(OUT/'geometry.npz') as z:
        points=z['fine_points'];normals=z['fine_normals']
    tree=cKDTree(points)
    frames=json.loads((PHOTOS/'dataset.json').read_text())['frames']
    cameras=np.array([f['pose'] for f in frames],np.float32)[:,:3,3];camera_tree=cKDTree(cameras)
    rows=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    a=np.memmap(OUT/'gaussian_binding.u32',mode='w+',dtype='<u4',shape=(len(rows),4));a[:]=0
    b=np.memmap(OUT/'gaussian_normal_weights.f32',mode='w+',dtype='<f4',shape=(len(rows),4));b[:]=0
    active=np.zeros(len(points)*2,bool);counts=dict(interior=0,supported=0);distance=[]
    for start in range(0,len(rows),65536):
        r=rows[start:start+65536];p=r[:,:12].copy().view('<f4').reshape(-1,3)
        indoor=((p>=BOUNDS[0])&(p<=BOUNDS[1])).all(1);counts['interior']+=int(indoor.sum())
        d,ids=tree.query(p,k=12,workers=8);_,cam=camera_tree.query(p,workers=8)
        toward=cameras[cam]-p
        n=normals[ids];side=np.sum(n*toward[:,None,:],axis=2)<0
        n=n*np.where(side,-1,1)[...,None]
        # Keep corner normals coherent rather than mixing opposite wall sides.
        best=n[:,0];compatible=(n*best[:,None,:]).sum(2)>.75
        rank=np.where(compatible,d,np.inf);order=np.argsort(rank,axis=1)[:,:3]
        chosen=np.take_along_axis(ids,order,axis=1);chosen_side=np.take_along_axis(side,order,axis=1)
        dd=np.take_along_axis(rank,order,axis=1)
        valid=indoor&(d[:,0]<.16)
        weights=1/np.maximum(dd,.004)**2;weights[~np.isfinite(weights)]=0;weights/=np.maximum(weights.sum(1,keepdims=True),1e-20)
        selected_n=np.take_along_axis(n,order[:,:,None],axis=1)
        weighted_n=(selected_n*weights[:,:,None]).sum(1);weighted_n/=np.maximum(np.linalg.norm(weighted_n,axis=1,keepdims=True),1e-10)
        node=(2*chosen+chosen_side).astype('<u4');node[~valid]=0xffffffff
        a[start:start+len(r),:3]=node;a[start:start+len(r),3]=weights[:,0].astype('<f4').view('<u4')
        b[start:start+len(r),0]=weights[:,1];b[start:start+len(r),1:]=weighted_n
        active[node[valid].ravel()]=True;counts['supported']+=int(valid.sum());distance.extend(d[indoor,0][::16].tolist())
        status=dict(stage='binding_original_gaussians',completed=start+len(r),total=len(rows),**counts,updated=time.time())
        atomic_json(OUT/'mapping_status.json',status)
    a.flush();b.flush();np.save(OUT/'active_nodes.npy',active)
    report=dict(gaussians=len(rows),**counts,coverage=counts['supported']/max(counts['interior'],1),
        active_receiver_nodes=int(active.sum()),total_receiver_nodes=len(active),max_supported_distance_m=.16,
        distance_quantiles=np.quantile(distance,[.5,.9,.99]).tolist(),source_geometry_unchanged=True,
        unsupported_regions='Explicitly retain captured appearance; excluded from claims of verified relighting')
    atomic_json(OUT/'mapping_report.json',report);print('MAPPING',json.dumps(report),flush=True)


if __name__=='__main__':main()
