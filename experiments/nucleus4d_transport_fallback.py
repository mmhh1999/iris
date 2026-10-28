"""Explicit low-confidence volume bindings for unsupported interior splats."""
import json
import shutil

import numpy as np
from scipy.spatial import cKDTree

from nucleus4d_baked_relight import SOURCE,PHOTOS,atomic_json
from nucleus4d_transport_geometry import OUT,BOUNDS


def main():
    data=np.load(OUT/'geometry.npz');active=np.flatnonzero(np.load(OUT/'active_nodes.npy'))
    pos=data['fine_points'][active//2];normal=data['fine_normals'][active//2]*np.where(active%2,-1,1)[:,None]
    tree=cKDTree(pos)
    for src,dst in [('gaussian_binding.u32','binding_full.u32'),('gaussian_normal_weights.f32','normal_weights_full.f32')]:shutil.copyfile(OUT/src,OUT/dst)
    rows=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    a=np.memmap(OUT/'binding_full.u32',mode='r+',dtype='<u4').reshape(-1,4)
    b=np.memmap(OUT/'normal_weights_full.f32',mode='r+',dtype='<f4').reshape(-1,4)
    cameras=np.array([f['pose'] for f in json.loads((PHOTOS/'dataset.json').read_text())['frames']])[:,:3,3];ct=cKDTree(cameras)
    count=0;distances=[]
    for start in range(0,len(rows),65536):
        r=rows[start:start+65536];p=r[:,:12].copy().view('<f4').reshape(-1,3)
        inside=((p>=BOUNDS[0])&(p<=BOUNDS[1])).all(1)
        missing=inside&(a[start:start+len(r),0]==0xffffffff)
        local=np.flatnonzero(missing)
        if not len(local):continue
        pp=p[local];d,ii=tree.query(pp,k=3,workers=8);w=1/np.maximum(d,.005)**2;w/=w.sum(1,keepdims=True)
        # Estimate a thin Gaussian's surface axis without altering its covariance.
        q=(r[local,28:32].astype(np.float32)-128)/128;q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
        scales=r[local,12:24].copy().view('<f4').reshape(-1,3);axis=np.eye(3,dtype=np.float32)[scales.argmin(1)]
        cross=np.cross(q[:,1:],axis);n=axis+2*q[:,:1]*cross+2*np.cross(q[:,1:],cross)
        _,cam=ct.query(pp,workers=8);n*=np.where((n*(cameras[cam]-pp)).sum(1)<0,-1,1)[:,None]
        isotropic=scales.min(1)/np.maximum(scales.max(1),1e-10)>.5;n[isotropic]=normal[ii[isotropic,0]]
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        target=start+local;a[target,:3]=active[ii];a[target,0]|=np.uint32(0x80000000)
        a[target,3]=w[:,0].astype('<f4').view('<u4');b[target,0]=w[:,1];b[target,1:]=n
        count+=len(local);distances.extend(d[:,0].tolist())
    a.flush();b.flush()
    report=dict(fallback_gaussians=count,method='nearest local incident-light field with covariance-axis normals; source positions retained for solar shadows',
        confidence='unverified volume estimate, not recovered surface geometry',
        distance_quantiles=np.quantile(distances,[.5,.9,.99]).tolist(),
        surface_verified_fraction=json.loads((OUT/'mapping_report.json').read_text())['coverage'],
        indoor_lighting_participation=1.,outside_scope='exterior appearance uses an approximate daylight scale')
    atomic_json(OUT/'fallback_report.json',report);print('FALLBACK',json.dumps(report),flush=True)


if __name__=='__main__':main()
