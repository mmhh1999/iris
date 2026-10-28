"""Independent full-house diffuse source bases, with geometric visibility.

Geometry-only ray queries use bounded GPU batches; no long CUDA path kernels.
The reference appearance remains the original GS. Secondary diffuse transport
uses a separately audited two-sided radiosity operator and eight reflections.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.stats import qmc

from nucleus4d_baked_relight import atomic_json,ROOT
from nucleus4d_transport_geometry import OUT,EPS,make_scene


def progress(stage,**values):
    r=dict(stage=stage,updated=time.time(),**values);atomic_json(OUT/'bake_status.json',r)
    print('TRANSPORT',json.dumps(r),flush=True)


def cosine_rays(points,normals,spp,offset=0):
    samples=qmc.Sobol(2,scramble=True,seed=730).random_base2(int(np.log2(spp)))
    shifts=np.mod((np.arange(len(points))+offset)[:,None]*np.array([.7548776662466927,.5698402909980532]),1)
    uv=np.mod(samples[None]+shifts[:,None],1)
    r=np.sqrt(uv[...,0]);phi=2*np.pi*uv[...,1]
    n=np.asarray(normals,np.float32);axis=np.tile([0,0,1.],(len(n),1));axis[np.abs(n[:,2])>.8]=[0,1,0]
    tangent=np.cross(axis,n);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True);bitangent=np.cross(n,tangent)
    d=(tangent[:,None]*((r*np.cos(phi))[...,None])+bitangent[:,None]*((r*np.sin(phi))[...,None])+
       n[:,None]*np.sqrt(1-r*r)[...,None])
    origins=np.repeat(np.asarray(points,np.float32)+n*EPS,spp,axis=0)
    return origins,d.reshape(-1,3).astype(np.float32)


def environment_emitters():
    import mitsuba as mi
    result=[]
    for name in ['direct','overcast']:
        folder=ROOT/'experiments/out/nucleus4d_weather_presets'/name;info=json.loads((folder/'lighting.json').read_text())
        result.append(mi.load_dict({'type':'envmap','filename':str(folder/'sky_linear_8k.exr'),
            'to_world':mi.ScalarTransform4f(np.array(info['environment_rotation']))}))
    return result


def build_operator(data,sc,environments,spp=4096):
    import mitsuba as mi
    import drjit as dr
    points=np.repeat(data['coarse_points'],2,axis=0);normals=np.repeat(data['coarse_normals'],2,axis=0);normals[1::2]*=-1
    count=len(points);folder=OUT/'operator_chunks';folder.mkdir(exist_ok=True)
    sky_path=OUT/'coarse_sky_direct.npy';escape_path=OUT/'coarse_escape.npy'
    sky=np.lib.format.open_memmap(sky_path,mode='r+' if sky_path.exists() else 'w+',dtype='<f4',shape=(count,2,3))
    escape=np.lib.format.open_memmap(escape_path,mode='r+' if escape_path.exists() else 'w+',dtype='<f4',shape=(count,))
    pieces=[]
    for start in range(0,count,128):
        end=min(start+128,count);path=folder/f'{start:07}.npz'
        if path.exists():pieces.append(sparse.load_npz(path));continue
        origins,d=cosine_rays(points[start:end],normals[start:end],spp,start)
        si=sc.ray_intersect(mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(d.T)))
        dr.eval(si.prim_index,si.t)
        primitive=np.asarray(si.prim_index);valid=np.asarray(si.is_valid());target=np.full(len(d),-1,np.int32)
        hit=np.flatnonzero(valid);f=primitive[hit]
        local=data['face_to_node'][f];side=(data['face_normals'][f]*d[hit]).sum(1)>0
        good=local>=0;target[hit[good]]=local[good]*2+side[good]
        values=[];indices=[];indptr=[0]
        for row in target.reshape(end-start,spp):
            ids,counts=np.unique(row[row>=0],return_counts=True);indices.extend(ids);values.extend(counts.astype(np.float32)/spp);indptr.append(len(indices))
        matrix=sparse.csr_matrix((np.asarray(values,np.float32),np.asarray(indices,np.int32),np.asarray(indptr,np.int32)),shape=(end-start,count))
        miss=~valid;escape[start:end]=miss.reshape(end-start,spp).mean(1)
        interaction=dr.zeros(mi.SurfaceInteraction3f,len(d));interaction.wi=mi.Vector3f(-d.T)
        for k,emitter in enumerate(environments):
            light=emitter.eval(interaction,mi.Bool(miss));dr.eval(light)
            rgb=np.asarray(light).T;rgb[~miss]=0
            sky[start:end,k]=rgb.reshape(end-start,spp,3).mean(1,dtype=np.float64)
        sky.flush();escape.flush();sparse.save_npz(path,matrix,compressed=False);pieces.append(matrix)
        if end%2048<128:progress('geometry_transfer',completed=end,total=count,spp=spp)
    matrix=sparse.vstack(pieces,format='csr');sparse.save_npz(OUT/'form_factors.npz',matrix,compressed=False)
    row_sum=np.asarray(matrix.sum(1)).ravel()
    if (row_sum+np.asarray(escape)>1.00001).any():raise RuntimeError('Nonconservative geometry transfer')
    atomic_json(OUT/'operator_validation.json',dict(rows=count,entries=matrix.nnz,spp=spp,
        maximum_row_sum=float(row_sum.max()),maximum_total_probability=float((row_sum+escape).max()),
        nonnegative=bool(matrix.data.min(initial=0)>=0),two_sided=True,finite_volume_reflections=8))
    progress('geometry_transfer_complete',rows=count,entries=matrix.nnz)
    return matrix,np.asarray(sky)


def emitter_samples(light,spp):
    uv=qmc.Sobol(2,scramble=True,seed=820+light['id']).random_base2(int(np.log2(spp)))
    c=np.array(light['center'],np.float32);radius=light['radius']
    if light['type']=='disk':
        r=radius*np.sqrt(uv[:,0]);phi=2*np.pi*uv[:,1]
        p=c+np.stack([r*np.cos(phi),r*np.sin(phi),np.zeros(spp)],axis=1)
        n=np.tile([0,0,-1.],(spp,1));area=np.pi*radius*radius
    else:
        phi=2*np.pi*uv[:,0];n=np.stack([np.cos(phi),np.sin(phi),np.zeros(spp)],axis=1)
        p=c+n*radius;p[:,2]+=(uv[:,1]-.5)*light['height'];area=2*np.pi*radius*light['height']
    return p.astype(np.float32),n.astype(np.float32),area


def occluded_cones(sc,points,normals,light,triangles):
    """Conservatively prove a finite emitter hidden by one opaque triangle."""
    import mitsuba as mi
    import drjit as dr
    origins=points+normals*EPS;delta=np.array(light['center'])-origins;distance=np.linalg.norm(delta,axis=1)
    direction=delta/np.maximum(distance[:,None],1e-12)
    si=sc.ray_intersect(mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(direction.T)));dr.eval(si.t,si.prim_index)
    valid=np.asarray(si.is_valid());t=np.asarray(si.t);primitive=np.asarray(si.prim_index)
    radius=np.sqrt(light['radius']**2+(.5*light.get('height',0))**2)+2*EPS
    ids=np.flatnonzero(valid&(t<distance-radius-.003));hidden=np.zeros(len(points),bool)
    if not len(ids):return hidden
    tri=triangles[primitive[ids]];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-20)
    hit=origins[ids]+direction[ids]*t[ids,None];denom=np.abs((normal*delta[ids]).sum(1))
    bound=(t[ids]/distance[ids])*radius*(denom+distance[ids])/np.maximum(denom-radius,1e-20)
    margin=np.full(len(ids),np.inf)
    for j in range(3):
        edge=tri[:,(j+1)%3]-tri[:,j]
        perp=np.linalg.norm(np.cross(edge,hit-tri[:,j]),axis=1)/np.maximum(np.linalg.norm(edge,axis=1),1e-20)
        margin=np.minimum(margin,perp)
    hidden[ids]=(denom>radius)&(margin>bound+.003)
    return hidden


def near_emitter_direct(sc,points,normals,light,spp=16384):
    """Cosine-angle sampling is bounded even very near an emitting surface."""
    import mitsuba as mi
    import drjit as dr
    result=np.zeros((len(points),3),np.float32);c=np.array(light['center']);radius=light['radius']
    for start in range(0,len(points),32):
        end=min(start+32,len(points));origin,d=cosine_rays(points[start:end],normals[start:end],spp,start+902001)
        physical=origin-np.repeat(normals[start:end],spp,axis=0)*EPS
        if light['type']=='disk':
            t=(c[2]-physical[:,2])/np.where(np.abs(d[:,2])>1e-10,d[:,2],1e-10)
            hit=physical+d*t[:,None];valid=(t>0)&(d[:,2]>0)&(np.linalg.norm(hit[:,:2]-c[:2],axis=1)<radius)
        else:
            oc=physical-c;a=(d[:,:2]**2).sum(1);b=2*(d[:,:2]*oc[:,:2]).sum(1);cc=(oc[:,:2]**2).sum(1)-radius*radius
            discriminant=b*b-4*a*cc;root=np.sqrt(np.maximum(discriminant,0));t=(-b-root)/np.maximum(2*a,1e-20)
            hit=physical+d*t[:,None];normal=hit-c;normal[:,2]=0;front=(normal*(-d)).sum(1)>0
            valid=(discriminant>=0)&(t>0)&front&(np.abs(hit[:,2]-c[2])<light['height']/2)
        ray=mi.Ray3f(mi.Point3f(origin.T),mi.Vector3f(d.T));ray.maxt=mi.Float(np.maximum(t-.002,.00001))
        visible=~sc.ray_test(ray,mi.Bool(valid));dr.eval(visible)
        fraction=(np.asarray(visible)&valid).reshape(end-start,spp).mean(1)
        result[start:end]=fraction[:,None]*np.array(light['radiance'])
    return result


def direct_light(sc,points,normals,light,spp=4096,batch=256,triangles=None):
    import mitsuba as mi
    import drjit as dr
    q,source_n,area=emitter_samples(light,spp);result=np.zeros((len(points),3),np.float32)
    center=np.array(light['center']);delta=center-points
    # Conservative geometric-facing bounds. Cylinder height is included so a
    # partially visible finite emitter is not culled by its central direction.
    extent=np.sqrt(light['radius']**2+(.5*light.get('height',0))**2)
    possible=(delta*normals).sum(1)>-extent
    if light['type']=='disk':possible&=points[:,2]<center[2]+.00001
    near=possible&(np.linalg.norm(delta,axis=1)<3*extent)
    if near.any():result[near]=near_emitter_direct(sc,points[near],normals[near],light)
    possible&=~near
    if triangles is not None:
        candidate=np.flatnonzero(possible);hidden=occluded_cones(sc,points[candidate],normals[candidate],light,triangles)
        rejected=candidate[hidden]
        if len(rejected):
            test=rejected[np.linspace(0,len(rejected)-1,min(128,len(rejected))).astype(int)]
            pp=points[test];nn=normals[test];v=q[None]-pp[:,None];length=np.linalg.norm(v,axis=2)
            ray=mi.Ray3f(mi.Point3f(np.repeat(pp+nn*EPS,spp,axis=0).T),mi.Vector3f((v/np.maximum(length[...,None],1e-12)).reshape(-1,3).T))
            ray.maxt=mi.Float(np.maximum(length-.002,.00001).ravel())
            if np.asarray(~sc.ray_test(ray)).any():raise RuntimeError('Conservative emitter-cone rejection failed independent visibility rays')
        possible[candidate[hidden]]=False
    ids=np.flatnonzero(possible)
    for start in range(0,len(ids),batch):
        ii=ids[start:start+batch];p=points[ii];normal=normals[ii]
        vector=q[None]-p[:,None];r2=(vector*vector).sum(2);d=vector/np.sqrt(np.maximum(r2,1e-12))[...,None]
        factor=np.maximum((d*normal[:,None]).sum(2),0)*np.maximum((-d*source_n[None]).sum(2),0)*area/(np.pi*np.maximum(r2,1e-12))
        ray=mi.Ray3f(mi.Point3f(np.repeat(p+normal*EPS,spp,axis=0).T),mi.Vector3f(d.reshape(-1,3).T))
        ray.maxt=mi.Float((np.sqrt(r2)-.002).clip(.00001).ravel())
        visible=~sc.ray_test(ray);dr.eval(visible)
        incoming=(np.asarray(visible).reshape(len(ii),spp)*factor).mean(1,dtype=np.float64)
        result[ii]=incoming[:,None]*np.array(light['radiance'],np.float32)
    if result.max()>max(light['radiance'])*1.02:raise RuntimeError('Finite-emitter irradiance exceeded its radiance bound')
    return result


def radiosity(matrix,reflectance,direct):
    """Incident irradiance/pi, with up to 8 receiving-surface reflections."""
    rho=np.repeat(reflectance,2,axis=0)
    shape=direct.shape;x=np.asarray(direct,np.float32).reshape(len(rho),-1)
    if x.shape[1]%3:raise ValueError('Expected RGB sources')
    color=np.tile(rho,(1,x.shape[1]//3));total=x.copy();bounce=x.copy()
    for _ in range(7):
        bounce=matrix@(color*bounce);total+=bounce
    if not np.isfinite(total).all() or total.min()<0:raise RuntimeError('Invalid radiosity')
    return total.reshape(shape)


def reciprocal_operator(data,raw):
    """Area-weighted reciprocal exchange with conservative row capacities."""
    tri=data['vertices'][data['faces'][data['inside_faces']]]
    area=np.repeat(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2,2).astype(np.float64)
    exchange=sparse.diags(area)@raw.astype(np.float64)
    exchange=((exchange+exchange.T)*.5).tocsr()
    for _ in range(60):
        row=np.asarray(exchange.sum(1)).ravel();ratio=row/np.maximum(area,1e-20)
        if ratio.max()<=1.0000001:break
        scale=np.minimum(1,np.sqrt(area/np.maximum(row,1e-20)))
        exchange=sparse.diags(scale)@exchange@sparse.diags(scale)
    if (np.asarray(exchange.sum(1)).ravel()>area*1.00001).any():raise RuntimeError('Reciprocal energy correction did not converge')
    matrix=(sparse.diags(1/np.maximum(area,1e-20))@exchange).astype(np.float32).tocsr()
    reciprocal=sparse.diags(area)@matrix;difference=reciprocal-reciprocal.T
    report=dict(method='reciprocal area exchange with conservative capacity scaling',
        rows=len(area),entries=matrix.nnz,maximum_row_sum=float(np.asarray(matrix.sum(1)).max()),
        maximum_reciprocity_error=float(np.max(np.abs(difference.data),initial=0)),
        maximum_flux_transfer_ratio=float(np.max((matrix.T@area)/np.maximum(area,1e-20))),
        guarantees='nonnegative transport; area-weighted reflected flux cannot exceed incident flux')
    atomic_json(OUT/'reciprocity_validation.json',report);np.save(OUT/'coarse_areas.npy',area)
    return matrix


def emission_rays(light,uv):
    count=len(uv);center=np.array(light['center']);radius=light['radius']
    if light['type']=='disk':
        r=radius*np.sqrt(uv[:,0]);phi=2*np.pi*uv[:,1]
        p=center+np.stack([r*np.cos(phi),r*np.sin(phi),np.zeros(count)],axis=1);n=np.tile([0,0,-1.],(count,1))
        area=np.pi*radius*radius
    else:
        phi=2*np.pi*uv[:,0];n=np.stack([np.cos(phi),np.sin(phi),np.zeros(count)],axis=1)
        p=center+radius*n;p[:,2]+=(uv[:,1]-.5)*light['height'];area=2*np.pi*radius*light['height']
    axis=np.tile([0,0,1.],(count,1));axis[np.abs(n[:,2])>.8]=[0,1,0]
    tangent=np.cross(axis,n);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True);bitangent=np.cross(n,tangent)
    r=np.sqrt(uv[:,2]);phi=2*np.pi*uv[:,3]
    d=tangent*(r*np.cos(phi))[:,None]+bitangent*(r*np.sin(phi))[:,None]+n*np.sqrt(1-r*r)[:,None]
    return (p+n*EPS).astype(np.float32),d.astype(np.float32),area


def emitted_direct(data,sc,light,samples=4194304):
    """Deposit bounded emitted flux onto receiver areas, avoiding point spikes."""
    import mitsuba as mi
    import drjit as dr
    areas=np.load(OUT/'coarse_areas.npy');counts=np.zeros(len(areas),np.int64)
    sampler=qmc.Sobol(4,scramble=True,seed=9721+light['id']);escaped=0;absorbed=0
    batch=262144
    for start in range(0,samples,batch):
        uv=sampler.random(min(batch,samples-start));origin,d,source_area=emission_rays(light,uv)
        si=sc.ray_intersect(mi.Ray3f(mi.Point3f(origin.T),mi.Vector3f(d.T)));dr.eval(si.t,si.prim_index)
        valid=np.asarray(si.is_valid());primitive=np.asarray(si.prim_index);hit=np.flatnonzero(valid);face=primitive[hit]
        local=data['face_to_node'][face];side=(data['face_normals'][face]*d[hit]).sum(1)>0;good=local>=0
        ids=2*local[good]+side[good];counts+=np.bincount(ids,minlength=len(counts))
        escaped+=int((~valid).sum());absorbed+=int((~good).sum())
    direct=(counts[:,None]/samples)*(source_area/areas[:,None])*np.array(light['radiance'])
    received=(direct*areas[:,None]).sum(0);emitted=source_area*np.array(light['radiance'])
    ratio=received/np.maximum(emitted,1e-20)
    if ratio.max()>1.000001 or counts.sum()+escaped+absorbed!=samples:raise RuntimeError('Emitter flux budget failed')
    report=dict(light=light['id'],samples=samples,received_fraction_rgb=ratio.tolist(),escape_fraction=escaped/samples,
        outside_receiver_fraction=absorbed/samples,all_emitted_rays_accounted=True,
        maximum_direct_irradiance_over_pi=float(direct.max()))
    atomic_json(OUT/f'lamp_{light["id"]:02}_flux_validation.json',report)
    return direct.astype(np.float32)


def sky_direct(sc,points,normals,environments,spp=1024):
    import mitsuba as mi
    import drjit as dr
    result=np.zeros((len(points),2,3),np.float32)
    for start in range(0,len(points),256):
        end=min(start+256,len(points));origins,d=cosine_rays(points[start:end],normals[start:end],spp,start+1000001)
        visible=~sc.ray_test(mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(d.T)));dr.eval(visible)
        mask=np.asarray(visible)
        si=dr.zeros(mi.SurfaceInteraction3f,len(d));si.wi=mi.Vector3f(-d.T)
        for k,e in enumerate(environments):
            rgb=np.asarray(e.eval(si,visible)).T;rgb[~mask]=0
            result[start:end,k]=rgb.reshape(end-start,spp,3).mean(1,dtype=np.float64)
        if end%16384<256:progress('fine_sky',completed=end,total=len(points))
    return result


def fine_nodes(data):
    active=np.load(OUT/'active_nodes.npy');ids=np.flatnonzero(active)
    p=data['fine_points'][ids//2];n=data['fine_normals'][ids//2]*np.where(ids%2, -1,1)[:,None]
    return ids,p.astype(np.float32),n.astype(np.float32)


def gi_mapping(data,sc,ids,points,normals):
    from scipy.spatial import cKDTree
    import mitsuba as mi
    import drjit as dr
    path=OUT/'fine_gi_binding.npz'
    if path.exists():
        z=np.load(path);return z['indices'],z['weights']
    cp=np.repeat(data['coarse_points'],2,axis=0);cn=np.repeat(data['coarse_normals'],2,axis=0);cn[1::2]*=-1
    tree=cKDTree(cp);indices=np.empty((len(ids),3),np.int32);weights=np.empty((len(ids),3),np.float32)
    for start in range(0,len(ids),8192):
        end=min(start+8192,len(ids));p=points[start:end];n=normals[start:end]
        distance,candidate=tree.query(p,k=48,workers=8)
        compatible=(cn[candidate]*n[:,None]).sum(2)>.8
        compatible&=np.abs(((cp[candidate]-p[:,None])*n[:,None]).sum(2))<.035
        distance=np.where(compatible,distance,np.inf);order=np.argsort(distance,axis=1)[:,:3]
        chosen=np.take_along_axis(candidate,order,axis=1);d=np.take_along_axis(distance,order,axis=1)
        parent=2*data['fine_parent'][ids[start:end]//2]+ids[start:end]%2
        empty=~np.isfinite(d[:,0]);chosen[empty,0]=parent[empty];d[empty,0]=np.linalg.norm(cp[parent[empty]]-p[empty],axis=1)
        # Do not interpolate GI through intervening furniture or a wall.
        origin=np.repeat(p+n*EPS,3,axis=0);target=(cp[chosen]+cn[chosen]*EPS).reshape(-1,3)
        v=target-origin;length=np.linalg.norm(v,axis=1)
        ray=mi.Ray3f(mi.Point3f(origin.T),mi.Vector3f((v/np.maximum(length[:,None],1e-8)).T));ray.maxt=mi.Float(np.maximum(length-.002,.00001))
        blocked=np.asarray(sc.ray_test(ray)).reshape(-1,3);d[blocked]=np.inf
        w=1/np.maximum(d,.005)**2;w[~np.isfinite(w)]=0
        fallback=w.sum(1)==0;chosen[fallback,0]=parent[fallback];w[fallback,0]=1
        w/=w.sum(1,keepdims=True);indices[start:end]=chosen;weights[start:end]=w
    np.savez(path,indices=indices,weights=weights);return indices,weights


def lift_gi(field,indices,weights):
    extra=(None,)*(field.ndim-1)
    return (field[indices]*weights[(slice(None),slice(None))+extra]).sum(1).astype(np.float32)


def sun_state(minute):
    configs=[]
    for name in ['direct','dusk']:
        info=json.loads((ROOT/'experiments/out/nucleus4d_weather_presets'/name/'lighting.json').read_text());configs.append(info)
    t=np.clip((minute-750)/345,0,1);a=np.array(configs[0]['sun_direction']);b=np.array(configs[1]['sun_direction'])
    omega=np.arccos(np.clip(a@b,-1,1));d=(np.sin((1-t)*omega)*a+np.sin(t*omega)*b)/np.sin(omega)
    power=(1-t)*np.array(configs[0]['sun_irradiance'])+t*np.array(configs[1]['sun_irradiance'])
    return d/np.linalg.norm(d),power


def sun_direct(sc,points,normals,minute,spp=1024):
    from nucleus4d_weather_presets import solar_directions
    import mitsuba as mi
    import drjit as dr
    d,power=sun_state(minute);rays=solar_directions(d,spp);result=np.zeros((len(points),3),np.float32)
    ids=np.flatnonzero(normals@d>0)
    for start in range(0,len(ids),256):
        ii=ids[start:start+256];p=points[ii];n=normals[ii];origins=np.repeat(p+n*EPS,spp,axis=0)
        direction=np.tile(rays,(len(ii),1));visible=~sc.ray_test(mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(direction.T)));dr.eval(visible)
        cosine=np.maximum(n@rays.T,0);irradiance=(np.asarray(visible).reshape(len(ii),spp)*cosine).mean(1)/np.pi
        result[ii]=irradiance[:,None]*power
    return result


def bake_sun_keys(data,sc,matrix,ids,indices,weights):
    folder=OUT/'sun_gi';folder.mkdir(exist_ok=True)
    cp=np.repeat(data['coarse_points'],2,axis=0);cn=np.repeat(data['coarse_normals'],2,axis=0);cn[1::2]*=-1
    def sample(minute):
        path=folder/f'coarse_{minute:08.3f}.npy'
        if path.exists():return np.load(path)
        direct=sun_direct(sc,cp,cn,minute);gi=radiosity(matrix,data['coarse_reflectance'],direct)-direct
        np.save(path,gi);return gi
    # Direct sunlight is never interpolated: the browser traces its moving
    # geometric shadow map. Only low-frequency secondary diffuse GI uses keys.
    minutes=list(np.linspace(750,1095,24));values={m:sample(m) for m in minutes};checks=[]
    converged=False
    for depth in range(10):
        additions=[]
        for a,b in zip(minutes[:-1],minutes[1:]):
            mid=(a+b)/2;actual=sample(mid);estimate=(values[a]+values[b])/2
            scale=max(float(actual.max()),float(values[a].max()),float(values[b].max()),1e-6)
            error=float(np.max(np.abs(actual-estimate))/scale)
            checks.append(dict(a=a,b=b,mid=mid,normalized_max_error=error))
            if error>.025:additions.append(mid);values[mid]=actual
        progress('sun_gi_refinement',level=depth,keys=len(minutes),new_keys=len(additions))
        if not additions:converged=True;break
        minutes=sorted(minutes+additions)
        if len(minutes)>1500:raise RuntimeError('Solar GI requires more than 1500 keys at the requested error bound; do not release an unverified interpolation')
    if not converged:raise RuntimeError('Solar GI interpolation has not reached its validation bound')
    fine=[]
    for i,m in enumerate(minutes):
        gi=lift_gi(values[m],indices,weights);np.save(folder/f'fine_{i:03}.npy',gi);fine.append(gi)
    meta=dict(minutes=minutes,states=[dict(minute=m,direction=sun_state(m)[0].tolist(),irradiance=sun_state(m)[1].tolist()) for m in minutes],
        direct_sun='single continuously moving geometric source, not endpoint patches',gi_interpolation_checks=checks,
        maximum_final_interval_error=max((x['normalized_max_error'] for x in checks if x['a'] in minutes and x['b'] in minutes and x['mid'] not in minutes),default=0),
        interpolation_limit=.025,coarse_spp=1024)
    atomic_json(OUT/'sun_keys.json',meta);return fine,meta


def compact_bindings(ids,points):
    binding=OUT/'binding_full.u32'
    source=np.memmap(binding if binding.exists() else OUT/'gaussian_binding.u32',mode='r',dtype='<u4').reshape(-1,4)
    target=np.memmap(OUT/'binding_compact.u32',mode='w+',dtype='<u4',shape=source.shape)
    remap=np.full(2*len(np.load(OUT/'geometry.npz')['fine_points']),0xffffffff,np.uint32);remap[ids]=np.arange(len(ids),dtype=np.uint32)
    for start in range(0,len(source),100000):
        block=np.array(source[start:start+100000],copy=True);valid=block[:,0]!=0xffffffff
        flags=block[valid,0]&np.uint32(0x80000000)
        bound=block[valid,:3].copy();bound[:,0]&=np.uint32(0x7fffffff)
        mapped=remap[bound]
        if (mapped==np.uint32(0xffffffff)).any():raise RuntimeError('Binding references an unbaked receiver')
        mapped[:,0]|=flags;block[valid,:3]=mapped;target[start:start+len(block)]=block
    target.flush();np.asarray(points,dtype='<f4').tofile(OUT/'patch_positions.f32')


def bake_fine(data,sc,environments,matrix):
    ids,p,n=fine_nodes(data);np.save(OUT/'compact_node_ids.npy',ids);compact_bindings(ids,p)
    gi_ids,gi_weights=gi_mapping(data,sc,ids,p,n)
    path=OUT/'fine_sky_direct.npy'
    if path.exists():direct=np.load(path)
    else:direct=sky_direct(sc,p,n,environments);np.save(path,direct)
    coarse=np.load(OUT/'coarse_sky_total.npy')-np.load(OUT/'coarse_sky_direct.npy')
    np.save(OUT/'fine_sky_total.npy',direct+lift_gi(coarse,gi_ids,gi_weights))
    environments.clear()
    import gc
    import drjit as dr
    gc.collect();dr.flush_malloc_cache()
    lights=json.loads((OUT/'lights.json').read_text())['lights']
    for light in lights:
        path=OUT/f'fine_lamp_{light["id"]:02}.npy'
        if not path.exists():
            direct=direct_light(sc,p,n,light,triangles=data['vertices'][data['faces']])
            coarse=np.load(OUT/f'coarse_lamp_{light["id"]:02}.npy')-np.load(OUT/f'coarse_lamp_{light["id"]:02}_direct.npy')
            total=direct+lift_gi(coarse,gi_ids,gi_weights);np.save(path,total)
        progress('fine_lamp',light=light['id']+1,total=len(lights),receivers=len(p))
    suns,sunmeta=bake_sun_keys(data,sc,matrix,ids,gi_ids,gi_weights)
    sky=np.load(OUT/'fine_sky_total.npy');layers=[sky[:,0],sky[:,1]]+[np.load(OUT/f'fine_lamp_{light["id"]:02}.npy') for light in lights]+suns
    width=1024;height=int(np.ceil(len(ids)/width));bank=np.memmap(OUT/'light_bank.rgb.f32',mode='w+',dtype='<f4',shape=(len(layers),height*width,3));bank[:]=0
    for k,layer in enumerate(layers):bank[k,:len(ids)]=layer
    bank.flush();reference=sky[:,1].copy()
    for light in lights:reference+=np.load(OUT/f'fine_lamp_{light["id"]:02}.npy')
    np.asarray(reference,dtype='<f4').tofile(OUT/'reference_irradiance.f32')
    meta=dict(nodes=len(ids),width=width,height=height,layers=len(layers),sky_layers=[0,1],
        lamp_layers=list(range(2,2+len(lights))),sun_gi_layers=list(range(2+len(lights),len(layers))),
        baseline=dict(sky_clear=0,sky_overcast=1,lamps=[1]*len(lights),direct_sun=0),
        direct_sun='runtime geometric shadow map',all_lamps_independent=True,reference_appearance='original capture, exact bypass',
        reference_min_rgb=reference.min(0).tolist(),reference_max_rgb=reference.max(0).tolist(),
        estimated_geometry_and_bounce_reflectance=True)
    atomic_json(OUT/'bank_manifest.json',meta);progress('full_house_bases_ready',nodes=len(ids),sources=len(layers))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu',action='store_true');parser.add_argument('--operator-only',action='store_true');parser.add_argument('--fine',action='store_true')
    args=parser.parse_args()
    if args.cpu:os.environ.setdefault('DRJIT_LIBCUDA_PATH','/dev/null')
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('llvm_ad_rgb' if args.cpu else 'cuda_ad_rgb');dr.set_thread_count(8)
    with np.load(OUT/'geometry.npz') as z:data={k:z[k] for k in z.files}
    sc=make_scene(data)
    need_environment=not (OUT/'coarse_sky_direct.npy').exists() or (args.fine and not (OUT/'fine_sky_direct.npy').exists())
    environments=environment_emitters() if need_environment else []
    matrix_path=OUT/'form_factors.npz';raw_path=OUT/'form_factors_raw.npz'
    if matrix_path.exists() and (OUT/'reciprocity_validation.json').exists():matrix=sparse.load_npz(matrix_path);sky=np.load(OUT/'coarse_sky_direct.npy')
    else:
        if raw_path.exists():raw=sparse.load_npz(raw_path);sky=np.load(OUT/'coarse_sky_direct.npy')
        else:raw,sky=build_operator(data,sc,environments);sparse.save_npz(raw_path,raw,compressed=False)
        matrix=reciprocal_operator(data,raw);sparse.save_npz(matrix_path,matrix,compressed=False)
    np.save(OUT/'coarse_sky_total.npy',radiosity(matrix,data['coarse_reflectance'],sky))
    if args.fine:
        bake_fine(data,sc,environments,matrix);return
    if args.operator_only:return
    lights=json.loads((OUT/'lights.json').read_text())['lights']
    p=np.repeat(data['coarse_points'],2,axis=0);n=np.repeat(data['coarse_normals'],2,axis=0);n[1::2]*=-1
    for light in lights:
        path=OUT/f'coarse_lamp_{light["id"]:02}.npy'
        if not path.exists():
            direct=emitted_direct(data,sc,light);np.save(OUT/f'coarse_lamp_{light["id"]:02}_direct.npy',direct)
            np.save(path,radiosity(matrix,data['coarse_reflectance'],direct))
        progress('coarse_lamp',light=light['id']+1,total=len(lights))
    progress('coarse_bases_ready',lights=len(lights))


if __name__=='__main__':main()
