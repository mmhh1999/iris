"""Unbiased cosine/window MIS sky candidate, preserving all source textures.

The two reviewed windows are sampling portals, not extra emitters. The cosine
branch retains coverage of every other opening in the complete supplied mesh.
"""
import json
import shutil
import subprocess
import time

import numpy as np
from scipy.stats import qmc

from nucleus4d_baked_relight import atomic_json
from nucleus4d_transport_geometry import OUT,EPS,make_scene
from nucleus4d_transport_bake import cosine_rays,environment_emitters

CANDIDATE=OUT/'sky_mis_bases'
RECTS=np.array([[-2.10,-1.38,-.55,.07],[-2.10,-1.38,.11,.60],[.78,1.50,-.55,.07],[.78,1.50,.11,.60]])
AREAS=(RECTS[:,1]-RECTS[:,0])*(RECTS[:,3]-RECTS[:,2]);AREA=float(AREAS.sum());WINDOW_X=6.


def portal_pdf(points,direction):
    dx=direction[...,0];t=(WINDOW_X-points[...,0])/np.where(abs(dx)>1e-12,dx,1e-12)
    y=points[...,1]+t*direction[...,1];z=points[...,2]+t*direction[...,2];inside=np.zeros(t.shape,bool)
    for a,b,c,d in RECTS:inside|=(y>=a)&(y<=b)&(z>=c)&(z<=d)
    return np.where((t>0)&inside,t*t/(AREA*np.maximum(abs(dx),1e-12)),0)


def portal_rays(points,normals,spp,start):
    uv=qmc.Sobol(3,scramble=True,seed=8873).random_base2(int(np.log2(spp)))
    shifts=np.mod((np.arange(len(points))+start+1)[:,None]*np.array([.7548776662,.569840291,.438579]),1)
    v=np.mod(uv[None]+shifts[:,None],1);which=np.searchsorted(np.cumsum(AREAS)/AREA,v[...,0]);rect=RECTS[which]
    q=np.empty((len(points),spp,3),np.float32);q[...,0]=WINDOW_X;q[...,1]=rect[...,0]+v[...,1]*(rect[...,1]-rect[...,0]);q[...,2]=rect[...,2]+v[...,2]*(rect[...,3]-rect[...,2])
    delta=q-points[:,None];d=delta/np.maximum(np.linalg.norm(delta,axis=2,keepdims=True),1e-12)
    return np.repeat(points+normals*EPS,spp,axis=0),d.reshape(-1,3)


def evaluate(sc,environments,points,normals,spp=1024,start=0):
    import mitsuba as mi
    import drjit as dr
    result=np.zeros((len(points),2,3),np.float64)
    portal_weight=np.where(abs(WINDOW_X-points[:,0])>.01,.5,0.)[:,None]
    for branch,sampler in enumerate([cosine_rays,portal_rays]):
        origins,d=sampler(points,normals,spp,start);direction=d.reshape(len(points),spp,3)
        cosine=np.maximum((normals[:,None]*direction).sum(2),0)/np.pi
        pdf=(1-portal_weight)*cosine+portal_weight*portal_pdf(points[:,None],direction);factor=cosine/np.maximum(pdf,1e-20)
        if factor.max()>2.00001:raise RuntimeError('Invalid MIS weight')
        escaped=~sc.ray_test(mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(d.T)));dr.eval(escaped)
        si=dr.zeros(mi.SurfaceInteraction3f,len(d));si.wi=mi.Vector3f(-d.T)
        for k,environment in enumerate(environments):
            rgb=np.broadcast_to(np.asarray(environment.eval(si,escaped)).T,(len(d),3)).copy();rgb[~np.asarray(escaped)]=0
            branch_weight=portal_weight if branch else 1-portal_weight
            result[:,k]+=(rgb.reshape(len(points),spp,3)*factor[...,None]*branch_weight[...,None]).mean(1,dtype=np.float64)
    return result.astype(np.float32)


def main():
    smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    def native_review_running():
        path=OUT/'finish_status.json'
        return path.exists() and json.loads(path.read_text()).get('stage')=='nucleus4d_transport_review.py'
    while native_review_running() or int(subprocess.check_output([smi,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6500:
        print('SKY_MIS waiting for native full-GS render to release GPU memory',flush=True);time.sleep(15)
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('cuda_ad_rgb');dr.set_thread_count(6)
    CANDIDATE.mkdir(exist_ok=True);chunks=CANDIDATE/'chunks';chunks.mkdir(exist_ok=True)
    with np.load(OUT/'geometry.npz') as z:data={k:z[k] for k in z.files}
    ids=np.load(OUT/'compact_node_ids.npy');p=np.repeat(data['fine_points'],2,axis=0)[ids];n=np.repeat(data['fine_normals'],2,axis=0);n[1::2]*=-1;n=n[ids]
    sc=make_scene(data);environments=environment_emitters();result=np.empty((len(p),2,3),np.float32)
    for start in range(0,len(p),128):
        path=chunks/f'{start:07}.npy';end=min(start+128,len(p))
        if path.exists():value=np.load(path)
        else:
            value=evaluate(sc,environments,p[start:end],n[start:end],1024,start)
            temporary=path.with_suffix('.partial')
            with temporary.open('wb') as stream:np.save(stream,value)
            temporary.replace(path)
        result[start:end]=value
        if end%8192<128:print('SKY_MIS',json.dumps(dict(completed=end,total=len(p),rays_per_receiver=2048)),flush=True)
    original_sky=np.load(OUT/'fine_sky_total.npy');original_direct=np.load(OUT/'fine_sky_direct.npy')
    total=result+np.maximum(original_sky-original_direct,0)
    np.save(CANDIDATE/'fine_sky_direct.npy',result);np.save(CANDIDATE/'fine_sky_total.npy',total);reference=total[:,1].copy()
    for i in range(25):reference+=np.load(OUT/f'fine_lamp_{i:02}.npy')
    reference.tofile(CANDIDATE/'reference_irradiance.f32')
    # Lamp bases are immutable shared inputs. Links preserve the raw candidate
    # and avoid copying hundreds of megabytes for this sky-only comparison.
    for i in range(25):
        link=CANDIDATE/f'fine_lamp_{i:02}.npy'
        if not link.exists():link.symlink_to(f'../fine_lamp_{i:02}.npy')
    atomic_json(CANDIDATE/'sampling_report.json',dict(completed=time.time(),candidate_only=True,rays_per_receiver=2048,
        method='equal mixture of cosine hemisphere and area-proportional reviewed-window sampling with exact mixture PDF',
        all_other_openings_covered=True,source_textures_and_geometry_unchanged=True,no_spatial_smoothing=True,manual_visual_review_pending=True))
    print('SKY_MIS_COMPLETE',flush=True)


if __name__=='__main__':main()
