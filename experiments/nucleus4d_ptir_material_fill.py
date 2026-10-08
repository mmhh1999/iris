"""Smooth incomplete photo delighting ratios over compatible Gaussian surfaces.

Keep each Gaussian's own high-frequency source color; never replace it with a
mesh vertex color. Filled material values remain uncertain estimates.
"""
import json
import numpy as np
from scipy.spatial import cKDTree
from nucleus4d_ptir import OUT, source_arrays

data,_=source_arrays();xyz=data[:,:3]
with np.load(OUT/'materials.npz') as f:m={k:f[k].copy() for k in f.files}
rgb=np.clip(.5+.28209479177387814*data[:,6:9],0,1)
base=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
ids=np.flatnonzero(m['support']>.1)
ratios=np.log(np.clip(m['albedo'][ids].astype('float32')/np.maximum(base[ids],.01),.5,2.5))
tree=cKDTree(xyz[ids]);filled=0
normals=m['normals'].astype('float32')
normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
for start in range(0,len(xyz),100000):
    stop=min(start+100000,len(xyz))
    dist,near=tree.query(xyz[start:stop],k=12,workers=8)
    similarity=np.abs(np.sum(normals[ids[near]]*normals[start:stop,None,:],axis=-1))
    weight=np.exp(-(dist/.18)**2)*(similarity>.85)
    denom=weight.sum(1)
    mean=(ratios[near]*weight[:,:,None]).sum(1)/np.maximum(denom[:,None],1e-8)
    confidence=np.minimum(denom,1.)
    result=base[start:stop]*np.exp(mean*confidence[:,None])
    m['albedo'][start:stop]=np.clip(result,.002,.98).astype('float16')
    filled+=int((confidence>.1).sum())
np.savez_compressed(OUT/'materials.npz',**m)
report={'photo_supported_gaussians':len(ids),'supported_or_spatially_filled_gaussians':filled,
        'method':'normal-compatible local interpolation of log delighting ratio; original per-Gaussian texture retained',
        'geometry_changed':False,'measured_material':False,'joint_optimization':False}
(OUT/'material_fill_report.json').write_text(json.dumps(report,indent=2));print(report)
