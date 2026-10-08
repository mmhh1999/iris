"""Surface-normal initialization from the full Gaussian scene, no proxy mesh.

Voxel representatives are used ONLY for estimating smooth shading normals.
All 6,100,978 source Gaussians still render with unchanged geometry/opacity.
"""
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from nucleus4d_ptir import ROOT, OUT, source_arrays


def main():
    t=time.time();data,_=source_arrays();xyz=data[:,:3]
    with np.load(OUT/'materials.npz') as f: m={k:f[k].copy() for k in f.files}
    # Reliable, relatively thin Gaussians provide a surface neighborhood.
    logscale=data[:,10:13]
    good=(data[:,9]>0)&((logscale.max(1)-logscale.min(1))>1.)
    ids=np.flatnonzero(good)
    voxel=np.floor(xyz[ids]/.025).astype('int32')
    _,first=np.unique(voxel,axis=0,return_index=True)
    representative=xyz[ids[first]].copy()
    tree=cKDTree(representative)
    rn=np.zeros_like(representative)
    for start in range(0,len(rn),20000):
        stop=min(start+20000,len(rn));dist,near=tree.query(representative[start:stop],k=24,workers=8)
        p=representative[near];weight=np.exp(-(dist/.1)**2)
        center=(p*weight[:,:,None]).sum(1)/weight.sum(1)[:,None]
        delta=p-center[:,None,:]
        cov=np.einsum('nki,nkj,nk->nij',delta,delta,weight)
        _,vec=np.linalg.eigh(cov);rn[start:stop]=vec[:,:,0]
    cameras=np.array([v['position'] for v in json.loads((ROOT/'experiments/out/nucleus4d_material/raw_images.json').read_text())])
    _,camera=cKDTree(cameras).query(representative,workers=8)
    rn*=np.where(np.sum(rn*(cameras[camera]-representative),axis=1)<0,-1,1)[:,None]
    for start in range(0,len(xyz),200000):
        stop=min(start+200000,len(xyz));dist,near=tree.query(xyz[start:stop],k=4,workers=8)
        nn=rn[near];nn*=np.where(np.sum(nn*nn[:,:1],axis=-1)<0,-1,1)[:,:,None]
        weights=np.exp(-(dist/.07)**2);normal=(nn*weights[:,:,None]).sum(1)
        norm=np.linalg.norm(normal,axis=1);ok=(dist[:,0]<.2)&(norm>1e-8)
        normal/=np.maximum(norm[:,None],1e-8)
        m['normals'][start:stop][ok]=normal[ok].astype('float16')
    np.savez_compressed(OUT/'materials.npz',**m)
    report={'method':'local PCA of high-opacity thin Gaussian centers, spatial interpolation of shading normals only',
            'representatives':len(representative),'source_render_count':len(xyz),'geometry_unchanged':True,'seconds':time.time()-t}
    (OUT/'normal_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__': main()
