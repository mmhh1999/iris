"""Second attempt: preserve captured GS appearance and bake additional window light.

This is differential diffuse lighting, not recovered/replaced original lighting.
No estimated per-Gaussian albedo replaces captured texture. Neutral additional
light changes brightness only; original geometry, opacity and color ratios stay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.ndimage import map_coordinates

from nucleus4d_baked_relight import (ROOT,SOURCE,PHOTOS,SH_C0,atomic_json,
    file_identity,geometry_digest,read_source_ply,srgb_to_linear,linear_to_srgb)
from nucleus4d_mesh_bake import setup,sample_surface_radiance

OUT=ROOT/'experiments/out/nucleus4d_sun_increment_v2'
LIGHT=np.array([.69,.28,.667],dtype=np.float32);LIGHT/=np.linalg.norm(LIGHT)
WINDOWS=[(-2.10,-1.38),(.78,1.50)]
# axis, physical plane, Gaussian-center layer, inward normal, other two axes,
# plane-coordinate ranges, explicit neutral diffuse reflectance.
PLANES=[
 dict(name='floor',axis=2,bound=-1.18,layer=-1.13,n=[0,0,1],uv=[0,1],lo=[-1.8,-2.5],hi=[5.98,2.45],rho=.35),
 dict(name='ceiling',axis=2,bound=1.38,layer=1.33,n=[0,0,-1],uv=[0,1],lo=[-1.8,-2.5],hi=[5.98,2.45],rho=.7),
 dict(name='west',axis=0,bound=-1.8,layer=-1.805,n=[1,0,0],uv=[1,2],lo=[-2.5,-1.18],hi=[2.45,1.38],rho=.7),
 dict(name='east',axis=0,bound=5.98,layer=6.055,n=[-1,0,0],uv=[1,2],lo=[-2.5,-1.18],hi=[2.45,1.38],rho=.7),
 dict(name='south',axis=1,bound=-2.5,layer=-2.525,n=[0,1,0],uv=[0,2],lo=[-1.8,-1.18],hi=[5.98,1.38],rho=.7),
 dict(name='north',axis=1,bound=2.45,layer=2.4625,n=[0,-1,0],uv=[0,2],lo=[-1.8,-1.18],hi=[5.98,1.38],rho=.7)]


def classify(points):
    p=np.asarray(points);distance=np.full((len(p),len(PLANES)),np.inf,dtype=np.float32)
    for j,plane in enumerate(PLANES):
        uv=p[:,plane['uv']]
        valid=((uv>=np.array(plane['lo'])+.025)&(uv<=np.array(plane['hi'])-.025)).all(1)
        d=np.abs(p[:,plane['axis']]-plane['layer'])
        distance[:,j]=np.where(valid,d,np.inf)
    index=distance.argmin(1);d=distance[np.arange(len(p)),index]
    index[d>=.12]=-1
    # A continuous boundary, rather than abrupt material/confidence patches.
    weight=np.clip((.12-d)/.04,0,1).astype(np.float32)
    return index,weight


def sun_directions(spp):
    # Deterministic equal-area samples of the 0.53-degree solar disc.
    u=(np.arange(spp)+.5)/spp;phi=np.arange(spp)*np.pi*(3-np.sqrt(5))
    radius=np.sqrt(u)*np.tan(np.deg2rad(.265))
    tangent=np.cross(LIGHT,[0,0,1]);tangent/=np.linalg.norm(tangent)
    bitangent=np.cross(LIGHT,tangent)
    d=LIGHT+radius[:,None]*(np.cos(phi)[:,None]*tangent+np.sin(phi)[:,None]*bitangent)
    return (d/np.linalg.norm(d,axis=1,keepdims=True)).astype(np.float32)


def project(points,plane):
    p=np.array(points,dtype=np.float32,copy=True);p[:,plane['axis']]=plane['bound']
    return p


def scene(blocked=False):
    import mitsuba as mi
    import trimesh
    scan=trimesh.load(ROOT/'experiments/out/nucleus4d_daylight/living_scan.obj',process=False)
    triangles=scan.triangles;centers=scan.triangles_center;normals=scan.face_normals
    keep=np.ones(len(centers),bool)
    # Remove the scan's enclosing surfaces before constructing an aperture-aware
    # shell. The old experiment traced through overlapping wall/glazing meshes.
    for p in PLANES:
        near=np.max(np.abs(triangles[:,:,p['axis']]-p['bound']),axis=1)<.25
        keep&=~(near&(np.abs(normals[:,p['axis']])>.7))
    scan.update_faces(keep);scan.remove_unreferenced_vertices()
    props=mi.Properties();props['bsdf']=mi.load_dict({'type':'twosided','bsdf':{'type':'diffuse','reflectance':.5}})
    furniture=mi.Mesh('furniture',len(scan.vertices),len(scan.faces),props)
    params=mi.traverse(furniture);params['vertex_positions']=np.asarray(scan.vertices,dtype=np.float32).ravel()
    params['faces']=np.asarray(scan.faces,dtype=np.uint32).ravel();params.update()
    desc={'type':'scene','integrator':{'type':'path','max_depth':9,'rr_depth':5},'furniture':furniture,
          'sun':{'type':'directional','direction':(-LIGHT).tolist(),'irradiance':1}}
    def box(name,lo,hi,rho):
        lo=np.asarray(lo);hi=np.asarray(hi)
        desc[name]={'type':'cube','to_world':mi.ScalarTransform4f.translate((lo+hi)/2).scale((hi-lo)/2),
            'bsdf':{'type':'twosided','bsdf':{'type':'diffuse','reflectance':rho}}}
    box('floor',[-1.86,-2.56,-1.24],[6.06,2.51,-1.18],.35)
    box('ceiling',[-1.86,-2.56,1.38],[6.06,2.51,1.44],.7)
    box('west',[-1.86,-2.56,-1.18],[-1.8,2.51,1.38],.7)
    box('south',[-1.8,-2.56,-1.18],[5.98,-2.5,1.38],.7)
    box('north',[-1.8,2.45,-1.18],[5.98,2.51,1.38],.7)
    box('east_bottom',[5.98,-2.5,-1.18],[6.06,2.45,-.55],.7)
    box('east_top',[5.98,-2.5,.60],[6.06,2.45,1.38],.7)
    for j,(a,b) in enumerate([(-2.5,-2.10),(-1.38,.78),(1.50,2.45)]):
        box(f'east_pier_{j}',[5.98,a,-.55],[6.06,b,.60],.7)
    for j,(a,b) in enumerate(WINDOWS):box(f'crossbar_{j}',[5.96,a,.07],[6.10,b,.11],.7)
    if blocked:box('blackout',[5.94,-2.5,-1.18],[5.96,2.45,1.38],0)
    return mi.load_dict(desc)


def direct_visibility(sc,points,normal,directions):
    import mitsuba as mi
    import drjit as dr
    n=np.asarray(normal,dtype=np.float32);p=np.asarray(points,dtype=np.float32)
    cosine=np.maximum(directions@n,0)
    origins=np.repeat(p+n*.00002,len(directions),axis=0)
    ds=np.tile(directions,(len(p),1))
    ray=mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(ds.T));ray.maxt=100
    visible=~sc.ray_test(ray);dr.eval(visible)
    return (np.asarray(visible).reshape(len(p),len(directions))*cosine).mean(1).astype(np.float32)/np.pi


def native_direct(sc,points,normals):
    """Use the exact first interaction, BSDF and visibility of the path tracer."""
    import mitsuba as mi
    import drjit as dr
    p=np.asarray(points,dtype=np.float32);n=np.asarray(normals,dtype=np.float32)
    ray=mi.Ray3f(mi.Point3f((p+n*.005).T),mi.Vector3f((-n).T))
    si=sc.ray_intersect(ray)
    ds,value=sc.sample_emitter_direction(si,mi.Point2f(.5,.5),test_visibility=True)
    direct=value*si.bsdf().eval(mi.BSDFContext(),si,si.to_local(ds.d))
    rho=si.bsdf().eval_diffuse_reflectance(si)
    dr.eval(direct,rho)
    return np.asarray(direct).mean(0),np.asarray(rho).mean(0)


def indirect_fields(output,sc,spp,progress):
    fields=[];spacing=.04
    for k,plane in enumerate(PLANES):
        path=output/f'indirect_{plane["name"]}.npy'
        u=np.arange(plane['lo'][0]+.025,plane['hi'][0]-.025,spacing,dtype=np.float32)
        v=np.arange(plane['lo'][1]+.025,plane['hi'][1]-.025,spacing,dtype=np.float32)
        xx,yy=np.meshgrid(u,v);p=np.zeros((xx.size,3),dtype=np.float32)
        p[:,plane['axis']]=plane['bound'];p[:,plane['uv'][0]]=xx.ravel();p[:,plane['uv'][1]]=yy.ravel()
        if path.exists():
            field=np.load(path)
            if field.shape!=(len(v),len(u)):raise RuntimeError('Cached light map shape changed')
        else:
            done_path=output/f'indirect_{plane["name"]}_done.npy'
            partial=output/f'indirect_{plane["name"]}_partial.npy'
            if partial.exists() and done_path.exists():
                values=np.lib.format.open_memmap(partial,mode='r+');done=np.lib.format.open_memmap(done_path,mode='r+')
            else:
                values=np.lib.format.open_memmap(partial,mode='w+',dtype='<f4',shape=(len(p),));values[:]=0;values.flush()
                done=np.lib.format.open_memmap(done_path,mode='w+',dtype='u1',shape=(len(p),));done[:]=0;done.flush()
            n=np.tile(plane['n'],(128,1))
            for start in range(0,len(p),128):
                end=min(start+128,len(p))
                if done[start:end].all():continue
                total=sample_surface_radiance(sc,p[start:end],n[:end-start],spp,seed=80000+k*100000+start).mean(1)
                direct,rho=native_direct(sc,p[start:end],n[:end-start])
                # The directional direct term is deterministic. Only a tiny
                # negative roundoff is allowed; a model disagreement is a failure.
                valid=np.isfinite(total)
                difference=total-direct
                if (difference[valid]<-.005).any():
                    raise RuntimeError(f'Direct/total disagreement on {plane["name"]}: {difference[valid].min()}')
                values[start:end]=np.where(valid,np.maximum(difference,0)/np.maximum(rho,1e-6),0)
                values.flush();done[start:end]=1;done.flush()
                progress('indirect_transport',plane=plane['name'],completed=int(done.sum()),total=len(p),spp=spp)
            field=np.asarray(values).reshape(len(v),len(u)).copy();np.save(path,field)
        fields.append(dict(data=field,u0=float(u[0]),v0=float(v[0]),spacing=spacing))
    return fields


def incident_increment(points,sc,fields,spp):
    index,weight=classify(points);result=np.zeros(len(points),dtype=np.float32)
    directions=sun_directions(spp)
    for k,plane in enumerate(PLANES):
        ids=np.flatnonzero(index==k)
        if not len(ids):continue
        p=project(points[ids],plane);f=fields[k]
        uv=p[:,plane['uv']]
        grid=np.stack([(uv[:,1]-f['v0'])/f['spacing'],(uv[:,0]-f['u0'])/f['spacing']])
        result[ids]=map_coordinates(f['data'],grid,order=1,mode='nearest')
        if np.dot(plane['n'],LIGHT)<=0:continue
        t=(5.98-p[:,0])/LIGHT[0];hit=p+LIGHT*t[:,None]
        window=np.zeros(len(ids),bool)
        for a,b in WINDOWS:window|=(hit[:,1]>a-.06)&(hit[:,1]<b+.06)
        possible=(t>0)&window&(hit[:,2]>-.61)&(hit[:,2]<.66)
        lit=np.flatnonzero(possible)
        for start in range(0,len(lit),128):
            j=lit[start:start+128]
            result[ids[j]]+=direct_visibility(sc,p[j],plane['n'],directions)
    return result*weight,index


def write_outputs(output,sc,fields,reference,spp,strength,progress):
    original=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    partial=output/'added_sun.partial.splat';shutil.copyfile(SOURCE/'full.splat',partial)
    baked=np.memmap(partial,mode='r+',dtype='u1').reshape(-1,32)
    gain_path=output/'gain.npy'
    gain=np.lib.format.open_memmap(gain_path,mode='w+',dtype='<f4',shape=(len(original),))
    direct_count=0;clipped=0
    for start in range(0,len(original),100000):
        end=min(start+100000,len(original));r=original[start:end]
        p=r[:,:12].copy().view('<f4').reshape(-1,3)
        light,index=incident_increment(p,sc,fields,spp)
        g=1+strength*light/np.asarray(reference)[np.maximum(index,0)]
        gain[start:end]=g
        mask=g>1+1e-7;direct_count+=int(mask.sum())
        base=srgb_to_linear(r[mask,24:27].astype(np.float32)/255)
        encoded=linear_to_srgb(base*g[mask,None])
        clipped+=int((encoded>1).any(1).sum())
        baked[start+np.flatnonzero(mask),24:27]=np.rint(encoded.clip(0,1)*255).astype('u1')
        progress('writing_gaussians',completed=end,total=len(original),affected=direct_count)
    baked.flush();gain.flush();del baked
    partial.replace(output/'added_sun.splat')
    header,source=read_source_ply(PHOTOS/'source.cache.ply')
    master=output/'added_sun.partial.ply'
    with master.open('wb') as stream:stream.write(header);stream.truncate(len(header)+source.nbytes)
    target=np.memmap(master,mode='r+',dtype='<f4',offset=len(header),shape=source.shape)
    for start in range(0,len(source),100000):
        end=min(start+100000,len(source));target[start:end]=source[start:end]
        light,index=incident_increment(np.asarray(source[start:end,:3]),sc,fields,spp)
        g=1+strength*light/np.asarray(reference)[np.maximum(index,0)]
        mask=g>1+1e-7
        encoded=np.maximum(.5+source[start:end,6:9][mask]*SH_C0,0)
        radiance=srgb_to_linear(encoded)*g[mask,None]
        target[start+np.flatnonzero(mask),6:9]=(linear_to_srgb(radiance)-.5)/SH_C0
    target.flush();del target
    master.replace(output/'added_sun.ply')
    g=np.asarray(gain)
    report=dict(status='awaiting_visual_review',gaussians=len(original),affected_gaussians=direct_count,
       gain_min=float(g.min()),gain_p50_p95_max=np.quantile(g[g>1],[.5,.95,1]).tolist(),
       darkened_gaussians=0,geometry_bitwise_unchanged=geometry_digest(original)==geometry_digest(np.memmap(output/'added_sun.splat',mode='r',dtype='u1').reshape(-1,32)),
       clipped_point_count=clipped,scope='additional neutral window light on room enclosure; captured lighting and furnishings retained',
       original_lighting_replaced=False,full_relighting=False,appearance_passed=False,
       physical_site_calibrated=False,completed_at=time.time())
    atomic_json(output/'result.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--strength',type=float,default=2.5)
    parser.add_argument('--spp',type=int,default=1024)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.spp<1024:raise ValueError('Do not reduce final sampling')
    if not 0<args.strength<=10:raise ValueError('Use a bounded positive light strength')
    def progress(stage,**values):
        record=dict(status='running',stage=stage,pid=os.getpid(),updated=time.time())
        record.update(values)
        atomic_json(output/'status.json',record);print('SUN_INCREMENT',json.dumps(record),flush=True)
    inputs=[file_identity(p) for p in [SOURCE/'full.splat',PHOTOS/'source.cache.ply',
        ROOT/'experiments/out/nucleus4d_daylight/living_scan.obj',Path(__file__)]]
    plan=dict(inputs=inputs,spp=args.spp,strength=args.strength,light_direction=LIGHT.tolist(),
        angular_diameter_degrees=.53,geometry='repaired aperture shell + scan furniture; approximate',
        materials='neutral diffuse proxy only; captured GS appearance is preserved',
        original_lighting_replaced=False,planes=PLANES)
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan:
        raise RuntimeError('Inputs/settings changed; select a fresh output directory')
    atomic_json(output/'plan.json',plan)
    setup(4);sc=scene()
    original=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    samples=[[] for _ in PLANES]
    for start in range(0,len(original),100000):
        r=original[start:start+100000:20];p=r[:,:12].copy().view('<f4').reshape(-1,3)
        k,_=classify(p);l=srgb_to_linear(r[:,24:27].astype(np.float32)/255).mean(1)
        for j in range(len(PLANES)):samples[j].extend(l[(k==j)&(l>.03)&(l<.9)].tolist())
    reference=[float(np.median(x))/p['rho'] if x else 1 for x,p in zip(samples,PLANES)]
    atomic_json(output/'reference_scale.json',dict(incident_over_pi=reference,
        method='one neutral photometric reference per enclosing surface; no per-Gaussian albedo replacement'))
    fields=indirect_fields(output,sc,args.spp,progress)
    report=write_outputs(output,sc,fields,reference,args.spp,args.strength,progress)
    progress('awaiting_visual_review',**report)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        output=Path(sys.argv[sys.argv.index('--output')+1]).resolve() if '--output' in sys.argv else OUT
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/'status.json',dict(status='needs_attention',error=f'{type(exc).__name__}: {exc}',updated=time.time()))
        raise
