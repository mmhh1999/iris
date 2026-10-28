"""CPU-only, resumable float32 surface-radiance baking for a fixed light state."""
import json
from pathlib import Path
import time

import numpy as np


def setup(threads=4):
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('llvm_ad_rgb');dr.set_thread_count(threads)
    return mi,dr


def sample_surface_radiance(scene, positions, normals, spp, seed=42):
    import mitsuba as mi
    import drjit as dr
    if spp<1:raise ValueError('Positive sample count required')
    p=np.asarray(positions,dtype=np.float32);n=np.asarray(normals,dtype=np.float32)
    if p.shape!=n.shape or p.ndim!=2 or p.shape[1]!=3:raise ValueError('Expected point and normal triples')
    n=n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
    # A ray hits the actual diffuse surface first; the stock MIS path integrator
    # then includes direct area/environment/directional light and indirect GI.
    # Repeated coincident rays have independent paths, not repeated image pixels.
    origins=np.repeat(p+n*.005,spp,axis=0)
    directions=np.repeat(-n,spp,axis=0)
    ray=mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(directions.T))
    hit=scene.ray_intersect(ray)
    valid=np.asarray(hit.is_valid())&np.isfinite(np.asarray(hit.t))&(np.asarray(hit.t)<.012)
    sampler=mi.load_dict({'type':'independent'});sampler.seed(seed,len(origins))
    result,_,_=scene.integrator().sample(scene,sampler,ray,medium=None,active=mi.Bool(valid))
    dr.eval(result)
    radiance=np.asarray(result).T.reshape(len(p),spp,3).mean(axis=1,dtype=np.float64).astype(np.float32)
    good=valid.reshape(len(p),spp).all(1)
    radiance[~good]=np.nan
    return radiance


def make_scene(vertices, faces, albedo, lamps, sky, sun, blackout=False):
    import mitsuba as mi
    props=mi.Properties();props['bsdf']=mi.load_dict({'type':'twosided','bsdf':{
        'type':'diffuse','reflectance':{'type':'mesh_attribute','name':'vertex_albedo'}}})
    mesh=mi.Mesh('living_room_proxy',len(vertices),len(faces),props)
    params=mi.traverse(mesh);params['vertex_positions']=vertices.ravel();params['faces']=faces.ravel();params.update()
    mesh.add_attribute('vertex_albedo',3,albedo.ravel())
    # Mitsuba depth 2 permits one surface reflection. Depth 9 therefore allows
    # eight surface reflections, matching the requested bounce budget.
    desc={'type':'scene','integrator':{'type':'path','max_depth':9,'rr_depth':5},'surface':mesh,
          'sky':{'type':'constant','radiance':{'type':'rgb','value':sky}}}
    if sun is not None:desc['sun']={'type':'directional','direction':sun['direction'],
        'irradiance':{'type':'rgb','value':sun['irradiance']}}
    for i,light in enumerate(lamps):
        c=np.asarray(light['center'])
        # The proxy covers this room only. Other-room lamps must not shine
        # through the missing rooms into the isolated living-room model.
        if not ((c>vertices.min(0))&(c<vertices.max(0))).all():continue
        desc[f'fixture_{i}']={'type':'sphere','center':light['center'],'radius':light['radius'],
            'bsdf':{'type':'diffuse','reflectance':0},'emitter':{'type':'area',
            'radiance':{'type':'rgb','value':light['radiance']}}}
    if blackout:
        # Covers both known east apertures, spanning their full height/width.
        desc['blackout']={'type':'cube','to_world':mi.ScalarTransform4f.translate([5.94,0,0]).scale([.01,2.5,1.5]),
                         'bsdf':{'type':'diffuse','reflectance':0}}
    return mi.load_dict(desc)


def bake(output, plan, progress, threads=4):
    from scipy.spatial import cKDTree
    from nucleus4d_baked_relight import ROOT,MATERIAL,PHOTOS,atomic_json,atomic_npz
    import sys,datetime
    sys.path.insert(0,str(ROOT))
    from utils.solar_geometry import solar_position,sun_vector_world
    from zoneinfo import ZoneInfo
    setup(threads)
    with np.load(MATERIAL/'surface.npz') as a:vertices=a['vertices'];faces=a['faces']
    with np.load(MATERIAL/'surface_materials.npz') as a:albedo=a['albedo']
    with np.load(output/'mapping.npz') as a:used=a['used_vertices']
    # Use an adjacent real face to avoid casting exactly onto triangle corners.
    import trimesh
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    adjacency=mesh.vertex_faces[used];face_id=np.maximum(adjacency[:,0],0)
    positions=.999*vertices[used]+.001*mesh.triangles_center[face_id]
    normals=np.array(mesh.face_normals[face_id],dtype=np.float32)
    views=json.loads((MATERIAL/'views.json').read_text())
    eyes=np.asarray([v['position'] for v in views if v['split']=='train'])
    _,nearest=cKDTree(eyes).query(positions)
    normals*=np.where(((eyes[nearest]-positions)*normals).sum(1)<0,-1,1)[:,None]
    # Same explicitly synthetic daylight convention as the existing experiment.
    moment=datetime.datetime(2026,3,21,13,tzinfo=ZoneInfo('America/Los_Angeles'))
    solar=solar_position(37.7749,-122.4194,moment)
    direction=-sun_vector_world(solar.azimuth_deg,solar.elevation_deg,90)
    light_doc=json.loads((PHOTOS/'indoor_photo_repair_20261021/indoor_lights.json').read_text())
    lamps=[{k:e[k] for k in ('center','radius','radiance')} for e in light_doc['emitters']]
    lighting=dict(date=moment.isoformat(),latitude=37.7749,longitude=-122.4194,north_bearing=90,
        sky=[.12,.18,.28],sun=dict(direction=direction.tolist(),irradiance=[4.,3.8,3.44]),
        calibration='synthetic example, not calibrated site/location or measured radiometry',
        windows='estimated open apertures; no glass model',fixture_source=str(PHOTOS/'indoor_photo_repair_20261021/indoor_lights.json'))
    atomic_json(output/'lighting.json',lighting)
    scene=make_scene(vertices,faces,albedo,lamps,lighting['sky'],lighting['sun'])
    field_path=output/'surface_radiance.npy';done_path=output/'surface_done.npy'
    n=len(used)
    if field_path.exists()!=done_path.exists():raise RuntimeError('Incomplete bake cache pair; use a fresh directory')
    if field_path.exists():
        field=np.lib.format.open_memmap(field_path,mode='r+')
        done=np.lib.format.open_memmap(done_path,mode='r+')
        if field.shape!=(n,3) or done.shape!=(n,):raise ValueError('Bake cache shape mismatch')
    else:
        field=np.lib.format.open_memmap(field_path,mode='w+',dtype='<f4',shape=(n,3));field[:]=np.nan;field.flush()
        done=np.lib.format.open_memmap(done_path,mode='w+',dtype='u1',shape=(n,));done[:]=0;done.flush()
    started=time.monotonic();completed_before=int(done.sum())
    batch=128
    for start in range(0,n,batch):
        stop=min(start+batch,n)
        if done[start:stop].all():continue
        value=sample_surface_radiance(scene,positions[start:stop],normals[start:stop],plan['spp'],seed=42+start)
        finite=np.isfinite(value).all(1)
        if (value[finite]<0).any():raise RuntimeError('Negative path-traced radiance')
        field[start:stop]=value;field.flush()
        done[start:stop]=1;done.flush() # Write radiance durably before marking work complete.
        completed=int(done.sum());elapsed=time.monotonic()-started
        progress('baking_surface',nodes_completed=completed,nodes_total=n,spp=plan['spp'],bounces=8,
                 elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(n-completed)/max(1,completed-completed_before),
                 completed_invalid_nodes=int((~np.isfinite(field[done.astype(bool)]).all(1)).sum()))
    return np.asarray(field)
