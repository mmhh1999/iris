"""Full-capture transport geometry, separate from the original Gaussian asset.

Only the two previously verified living-room aperture volumes are repaired.
The complete scan remains the visibility caster, including furniture and the
rest of the apartment. Receiver samples cover the captured indoor bounds.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from nucleus4d_baked_relight import ROOT,SOURCE,PHOTOS,atomic_json,file_identity,srgb_to_linear
from nucleus4d_fixture_controls import verified_fixtures

OUT=ROOT/'experiments/out/nucleus4d_transport_v3'
BOUNDS=np.array([[-4.05,-2.70,-1.28],[6.20,11.50,1.65]],np.float32)
EPS=.001


def split_polygon(polygon,axis,bound):
    low=[];high=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        da=a[axis]-bound;db=b[axis]-bound
        (low if da<=0 else high).append(a)
        if (da<0<db) or (db<0<da):
            point=a+(b-a)*(-da/(db-da));low.append(point);high.append(point)
    return np.asarray(low),np.asarray(high)


def subtract_box(polygon,lo,hi):
    pending=[polygon];outside=[]
    for axis in range(3):
        next_pending=[]
        for poly in pending:
            low,high=split_polygon(poly,axis,lo[axis])
            if len(low)>=3:outside.append(low)
            if len(high)>=3:next_pending.append(high)
        pending=[]
        for poly in next_pending:
            low,high=split_polygon(poly,axis,hi[axis])
            if len(high)>=3:outside.append(high)
            if len(low)>=3:pending.append(low)
    return outside


def repaired_scan():
    scan=trimesh.load(ROOT/'experiments/out/nucleus4d_daylight/scan.obj',process=False)
    triangles=np.array(scan.triangles);normals=scan.face_normals
    windows=[]
    for a,b in [(-2.10,-1.38),(.78,1.50)]:
        for lo,hi in [(-.55,.07),(.11,.60)]:windows.append((np.array([5.70,a,lo]),np.array([6.38,b,hi])))
    output=[];changed=0
    for tri,n in zip(triangles,normals):
        pieces=[tri]
        if abs(n[0])>.5:
            for lo,hi in windows:
                if (tri.max(0)<lo).any() or (tri.min(0)>hi).any():continue
                pieces=[part for p in pieces for part in subtract_box(p,lo,hi)];changed+=1
        for p in pieces:
            for j in range(1,len(p)-1):output.append([p[0],p[j],p[j+1]])
    t=np.asarray(output,np.float32);v=t.reshape(-1,3);faces=np.arange(len(v),dtype=np.uint32).reshape(-1,3)
    mesh=trimesh.Trimesh(v,faces,process=True,validate=True)
    return mesh,changed


def source_bounce_colors(points):
    rows=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    positions=rows[:,:12].copy().view('<f4').reshape(-1,3);tree=cKDTree(positions)
    _,indices=tree.query(points,k=12,workers=8)
    color=np.median(srgb_to_linear(rows[indices,24:27].astype(np.float32)/255),axis=1)
    # A bounded, conservative diffuse bounce prior. Visible GS texture is never
    # replaced by this estimate, and its uncertainty is retained in the report.
    return np.clip(.8*np.sqrt(np.maximum(color,0)),.025,.75).astype(np.float32)


def sample_receivers(mesh,inside,spacing=.04):
    triangles=mesh.triangles[inside];normals=mesh.face_normals[inside];parent=np.flatnonzero(inside)
    rng=np.random.default_rng(2601);points=[];which=[]
    # Oversample then retain an actual surface point per 4cm spatial/normal cell.
    for j,(tri,area) in enumerate(zip(triangles,mesh.area_faces[inside])):
        count=max(1,int(np.ceil(area/(spacing*spacing)*2)))
        uv=rng.random((count,2));root=np.sqrt(uv[:,0]);bary=np.stack([1-root,root*(1-uv[:,1]),root*uv[:,1]],axis=1)
        p=bary@tri;points.append(p);which.append(np.full(count,j,np.int32))
    points=np.concatenate(points).astype(np.float32);which=np.concatenate(which)
    normal=normals[which];cell=np.floor(points/spacing).astype(np.int32)
    # Opposite wall sides and hard corners must not share a light sample.
    normal_key=np.rint(normal*4).astype(np.int32)
    keys=np.column_stack([cell,normal_key]);_,ids=np.unique(keys,axis=0,return_index=True)
    ids.sort();return points[ids],normals[which[ids]].astype(np.float32),which[ids],parent


def make_scene(data,blocked=False):
    import mitsuba as mi
    mesh=mi.Mesh('complete_capture_caster',len(data['vertices']),len(data['faces']))
    params=mi.traverse(mesh);params['vertex_positions']=data['vertices'].ravel();params['faces']=data['faces'].ravel();params.update()
    desc={'type':'scene','mesh':mesh}
    if blocked:
        desc['blackout']={'type':'cube','to_world':mi.ScalarTransform4f.translate([5.69,0,0]).scale([.012,2.5,1.5])}
    return mi.load_dict(desc)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mesh,repaired=repaired_scan();c=mesh.triangles_center
    inside=((c>=BOUNDS[0])&(c<=BOUNDS[1])).all(1)
    points,normals,parent,inside_ids=sample_receivers(mesh,inside)
    coarse=c[inside].astype(np.float32);coarse_n=mesh.face_normals[inside].astype(np.float32)
    rho=source_bounce_colors(coarse)
    face_to_node=np.full(len(mesh.faces),-1,np.int32);face_to_node[inside_ids]=np.arange(len(coarse),dtype=np.int32)
    np.savez(OUT/'geometry.npz',vertices=np.asarray(mesh.vertices,np.float32),faces=np.asarray(mesh.faces,np.uint32),
        face_normals=np.asarray(mesh.face_normals,np.float32),face_to_node=face_to_node,
        coarse_points=coarse,coarse_normals=coarse_n,coarse_reflectance=rho,
        fine_points=points,fine_normals=normals,fine_parent=parent,inside_faces=inside_ids)
    # This is a visibility proxy only. The browser continues drawing the full GS.
    np.asarray(mesh.vertices,dtype='<f4').tofile(OUT/'caster_positions.f32')
    np.asarray(mesh.faces,dtype='<u4').tofile(OUT/'caster_indices.u32')
    fixtures,_=verified_fixtures()
    old=json.loads((ROOT/'experiments/out/nucleus4d_weather_controls/lamps/fixtures.json').read_text())
    supported={f['id'] for f in old['fixtures'] if f['controllable']}
    lights=[]
    for f in fixtures:
        if f['id'] not in supported:continue
        f=dict(f);f['source_id']=f['id'];f['id']=len(lights)
        f['type']='cylinder' if f['group']==2 else 'disk'
        if f['type']=='cylinder':f['radiance']=[45.,40.5,36.];f['height']=.26;f['radius']=.04
        f['power_calibration']='estimated relative radiance; not measured lumens'
        lights.append(f)
    atomic_json(OUT/'lights.json',dict(lights=lights,count=len(lights),source='verified photographic fixtures + reviewed pendant GS components'))
    report=dict(source_mesh=file_identity(ROOT/'experiments/out/nucleus4d_daylight/scan.obj'),source_gaussians=file_identity(SOURCE/'full.splat'),
        caster_vertices=len(mesh.vertices),caster_faces=len(mesh.faces),receiver_faces=int(inside.sum()),
        coarse_nodes=2*len(coarse),fine_nodes=2*len(points),receiver_spacing_m=.04,indoor_bounds=BOUNDS.tolist(),
        aperture_triangle_operations=repaired,full_scan_caster=True,includes_furniture_and_other_rooms=True,
        bounce_reflectance='bounded captured-color prior, used for secondary diffuse bounces only',
        visible_source_texture_unchanged=True,not_surveyed_geometry=True)
    atomic_json(OUT/'geometry_report.json',report);print('GEOMETRY',json.dumps(report),flush=True)


if __name__=='__main__':main()
