"""Lift photo material priors directly to ORIGINAL Gaussians, with GS visibility.

Preserve SH0 fine texture using a low-frequency delighting ratio. This is a
prior-based initialization, not calibrated or jointly optimized material truth.
No proxy mesh participates in this stage.
"""
import json
import time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from nucleus4d_ptir import ROOT, OUT, source_arrays, load_integrators, shape, sensor, mi, dr


def linear(a):
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055)**2.4)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    mi.set_variant('cuda_ad_rgb')
    load_integrators()
    data, _ = source_arrays()
    xyz = data[:, :3]
    desc = shape(data, material=True)
    original_normals = np.array(desc['normals'])
    scene = mi.load_dict({'type':'scene', 'shape':desc,
                         'integrator':{'type':'gsprim_prb', 'max_depth':1, 'gaussian_max_depth':256, 'separate_direct_indirect':True},
                         'sky':{'type':'constant', 'radiance':0.0}})
    n = len(data)
    sums = np.zeros((n, 9), 'float32') # log ratio RGB, normal XYZ, roughness, metallic, weight
    records=[]
    base = ROOT/'experiments/out/nucleus4d_material'
    views = json.loads((base/'views.json').read_text())
    for view in views:
        if view['split'] != 'train':
            continue
        t = time.time()
        i=view['index']; cam=sensor(512, i)
        buffer=np.array(scene.integrator().render(scene, sensor=cam, spp=2, seed=17))
        depth=buffer[:,:,18]
        np.save(OUT/f'depth_{i:02}.npy',depth.astype('float16'))
        r=np.array(view['rotation'],dtype='float32'); eye=np.array(view['position'],dtype='float32')
        yy,xx=np.mgrid[:512,:512]
        directions=np.stack([(xx+.5-256)/350,(yy+.5-256)/350,np.ones_like(xx)],axis=-1)@r.T
        directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
        smooth=gaussian_filter(depth,1.)
        points=eye+directions*smooth[:,:,None]
        dx=np.gradient(points,axis=1); dy=np.gradient(points,axis=0)
        normals=np.cross(dx,dy); normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-8)
        normals*=np.where(np.sum(normals*directions,axis=-1)>0,-1,1)[:,:,None]
        edges=np.maximum(abs(np.gradient(smooth,axis=0)),abs(np.gradient(smooth,axis=1)))
        photo=linear(np.asarray(Image.open(base/view['photo']).convert('RGB'),dtype='float32')/255)
        with np.load(base/'priors'/f'{i:02}.npz') as p:
            prior=np.clip(p['albedo'],.005,1.)
            rough=p['roughness'].squeeze(); metal=p['metallic'].squeeze()
        if rough.ndim==3: rough=rough.mean(-1)
        if metal.ndim==3: metal=metal.mean(-1)
        ratio=np.log(np.clip(gaussian_filter(prior,(3,3,0))/np.maximum(gaussian_filter(photo,(3,3,0)),.015),.25,4.))
        count=0
        for start in range(0,n,200000):
            end=min(start+200000,n); delta=xyz[start:end]-eye; cp=delta@r
            distance=np.linalg.norm(delta,axis=1)
            uv=cp[:,:2]/np.maximum(cp[:,2:],1e-6)*350+256
            good=(cp[:,2]>.15)&(uv[:,0]>3)&(uv[:,0]<508)&(uv[:,1]>3)&(uv[:,1]<508)
            ids=np.flatnonzero(good); uv=uv[ids]; coords=[uv[:,1],uv[:,0]]
            observed=map_coordinates(depth,coords,order=1,mode='nearest')
            err=abs(observed-distance[ids])
            ok=(err<.08)&(map_coordinates(edges,coords,order=1)<.035)
            ids=ids[ok];coords=[uv[ok,1],uv[ok,0]]
            normal=np.stack([map_coordinates(normals[:,:,j],coords,order=1) for j in range(3)],-1)
            facing=np.maximum(-np.sum(normal*delta[ids]/distance[ids,None],axis=1),0)
            weight=facing**2*np.exp(-(err[ok]/.04)**2)
            vals=np.stack([map_coordinates(ratio[:,:,j],coords,order=1) for j in range(3)]+[normal[:,j] for j in range(3)]+[map_coordinates(rough,coords,order=1),map_coordinates(metal,coords,order=1),np.ones(len(ids))],-1)
            sums[ids+start]+=vals*weight[:,None];count+=len(ids)
        records.append({'view':i,'visible_gaussians':count,'seconds':round(time.time()-t,2)})
        print(records[-1],flush=True)
    support=sums[:,-1];valid=support>.1
    means=sums/np.maximum(support[:,None],1e-8)
    rgb=linear(np.clip(.5+.28209479177387814*data[:,6:9],0,1))
    rgb[valid]*=np.exp(means[valid,:3]);rgb=np.clip(rgb,.005,.98)
    normal=original_normals
    normal[valid]=means[valid,3:6]
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
    rough=np.full(n,.65,'float32');rough[valid]=np.clip(means[valid,6],.15,.95)
    # Learned metal classification is uncertain; retain it only at strong support.
    metal=np.zeros(n,'float32');metal[valid]=np.clip(means[valid,7],0,1)
    np.savez_compressed(OUT/'materials.npz',albedo=rgb.astype('float16'),normals=normal.astype('float16'),
                        roughness=rough.astype('float16'),metallic=metal.astype('float16'),support=support.astype('float16'))
    report={'gaussians':n,'direct_photo_supported':int(valid.sum()),'views':records,
            'method':'visibility from original full GS; photo RGB-X priors; low-frequency ratio preserves original DC detail; depth-derived normals',
            'no_mesh':True,'joint_inverse_optimization':False,'unobserved':'original baked DC as initialization; covariance-axis normal',
            'material_status':'estimated initialization, not measured reflectance'}
    (OUT/'material_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__': main()
