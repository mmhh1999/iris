"""Locate verified light-emitting GS regions and bake additional living light.

Default power 1 preserves capture exactly. Power 0 dims confirmed emitting
surfaces only: the illumination baked into captured furniture is retained.
Above 1 adds path-traced living-room light; it never substitutes proxy albedo.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates

from nucleus4d_baked_relight import ROOT,SOURCE,PHOTOS,atomic_json,file_identity
from nucleus4d_sun_increment import PLANES,classify,project,scene as room_scene
from nucleus4d_mesh_bake import sample_surface_radiance

OUT=ROOT/'experiments/out/nucleus4d_weather_controls'
LAMPS=OUT/'lamps'


def verified_fixtures():
    source=PHOTOS/'indoor_photo_repair_20261021/indoor_lights.json'
    doc=json.loads(source.read_text());result=[];counts=[0,0]
    for light in doc['emitters']:
        if len({e['source'] for e in light['source_detections']})<4:continue
        if light['mean_ray_error']>light['radius']*.5:continue
        x,y,z=light['center'];living=(-1.8<x<5.98 and -2.5<y<2.45)
        group=0 if living else 1;counts[group]+=1
        result.append(dict(id=len(result)+1,label=('客厅筒灯 ' if living else '其它灯具 ')+str(counts[group]).zfill(2),
            group=group,center=light['center'],radius=light['radius'],radiance=light['radiance'],
            views=len({e['source'] for e in light['source_detections']}),ray_error=light['mean_ray_error'],evidence='multiview photographic triangulation'))
    # The earlier spherical detector missed two tall emitting pendant tubes.
    # These bounds were checked against native-camera projected GS components
    # and their clearly visible tubular light surfaces in the captured model.
    for j,center in enumerate([[.061,-.320,.669],[-.527,-.297,.669]]):
        result.append(dict(id=len(result)+1,label=f'厨房吊灯 {j+1:02}',group=2,center=center,
            radius=.07,radii=[.065,.065,.165],radiance=[0,0,0],views=None,ray_error=None,
            evidence='reviewed native GS projection; pendant_candidates.json components 28 and 19'))
    return result,source


def fixture_mask(points,encoded,scales,fixtures):
    """Only small, bright Gaussians inside verified fixture volumes qualify."""
    p=np.asarray(points);rgb=np.asarray(encoded);result=np.zeros(len(p),np.float32)
    strongest=np.zeros(len(p),np.float32)
    bright=np.clip((rgb.mean(1)-.60)/.24,0,1)*np.clip((rgb.max(1)-.74)/.18,0,1)
    small=np.max(scales,axis=1)<.055
    for fixture in fixtures:
        radii=np.asarray(fixture.get('radii',[fixture['radius']]*3))
        distance=np.linalg.norm((p-np.array(fixture['center']))/radii,axis=1)
        spatial=np.clip((1.42-distance)/.32,0,1)
        w=spatial*bright*small
        use=(w>strongest)&(w>1e-5)
        result[use]=fixture['id']+.5*w[use];strongest[use]=w[use]
    return result


def build_mask(fixtures):
    rows=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    target=np.memmap(LAMPS/'fixture_mask.f32',mode='w+',dtype='<f4',shape=(len(rows),))
    counts=np.zeros(len(fixtures)+1,np.int64)
    for start in range(0,len(rows),100000):
        r=rows[start:start+100000];p=r[:,:12].copy().view('<f4').reshape(-1,3)
        scale=r[:,12:24].copy().view('<f4').reshape(-1,3)
        value=fixture_mask(p,r[:,24:27]/255.,scale,fixtures);target[start:start+len(r)]=value
        counts+=np.bincount(value.astype(int),minlength=len(counts))
    target.flush()
    for f in fixtures:
        f['gaussians']=int(counts[f['id']]);f['controllable']=f['gaussians']>=5
    if int(counts[1:].sum())>len(rows)*.02:raise RuntimeError('Fixture edits extend too far beyond light-emitting surfaces')
    return dict(affected_gaussians=int(counts[1:].sum()),sha256=hashlib.sha256((LAMPS/'fixture_mask.f32').read_bytes()).hexdigest())


def lamp_scene(fixtures):
    import mitsuba as mi
    base=room_scene();desc={'type':'scene','integrator':{'type':'path','max_depth':9,'rr_depth':5}}
    desc.update({f'shape_{i}':shape for i,shape in enumerate(base.shapes())})
    for f in fixtures:
        if f['group']!=0:continue
        # Recessed downlights emit through the lower aperture. A sphere would
        # cast an unobserved strong uplight onto the nearby ceiling.
        desc[f'fixture_{f["id"]}']={'type':'disk',
            'to_world':mi.ScalarTransform4f.translate(f['center']).rotate([1,0,0],180).scale(f['radius']),
            'bsdf':{'type':'diffuse','reflectance':0},
            'emitter':{'type':'area','radiance':{'type':'rgb','value':f['radiance']}}}
    return mi.load_dict(desc)


def bake_fields(sc):
    fields=[];spacing=.04
    for k,plane in enumerate(PLANES):
        u=np.arange(plane['lo'][0]+.025,plane['hi'][0]-.025,spacing,dtype=np.float32)
        v=np.arange(plane['lo'][1]+.025,plane['hi'][1]-.025,spacing,dtype=np.float32)
        xx,yy=np.meshgrid(u,v);p=np.zeros((xx.size,3),np.float32)
        p[:,plane['axis']]=plane['bound'];p[:,plane['uv'][0]]=xx.ravel();p[:,plane['uv'][1]]=yy.ravel()
        path=LAMPS/f'field_{plane["name"]}.npy'
        if path.exists():data=np.load(path)
        else:
            partial=LAMPS/f'partial_{plane["name"]}.npy';done_path=LAMPS/f'done_{plane["name"]}.npy'
            if partial.exists() and done_path.exists():
                values=np.lib.format.open_memmap(partial,mode='r+');done=np.lib.format.open_memmap(done_path,mode='r+')
            else:
                values=np.lib.format.open_memmap(partial,mode='w+',dtype='<f4',shape=(len(p),3));values[:]=0
                done=np.lib.format.open_memmap(done_path,mode='w+',dtype='u1',shape=(len(p),));done[:]=0
            for start in range(0,len(p),256):
                end=min(start+256,len(p))
                if done[start:end].all():continue
                normal=np.tile(plane['n'],(end-start,1))
                value=sample_surface_radiance(sc,p[start:end],normal,4096,230001+k*100000+start)
                values[start:end]=np.where(np.isfinite(value),value/plane['rho'],0)
                if (values[start:end]<0).any():raise ValueError('Negative artificial-light transport')
                values.flush();done[start:end]=1;done.flush()
                status=dict(stage='lamp_transport',plane=plane['name'],completed=int(done.sum()),total=len(p),updated=time.time())
                atomic_json(LAMPS/'status.json',status)
                if int(done.sum())%4096<256:print('LAMPS',json.dumps(status),flush=True)
            data=gaussian_filter(np.asarray(values).reshape(len(v),len(u),3),sigma=(.5,.5,0),mode='nearest');np.save(path,data)
        fields.append(dict(data=data,u0=float(u[0]),v0=float(v[0]),spacing=spacing))
    return fields


def lamp_gain(points,fields,reference):
    index,weight=classify(points);light=np.zeros((len(points),3),np.float32)
    for k,plane in enumerate(PLANES):
        ids=np.flatnonzero(index==k)
        if not len(ids):continue
        uv=project(points[ids],plane)[:,plane['uv']];f=fields[k]
        coords=np.stack([(uv[:,1]-f['v0'])/f['spacing'],(uv[:,0]-f['u0'])/f['spacing']])
        for c in range(3):light[ids,c]=map_coordinates(f['data'][...,c],coords,order=1,mode='nearest')
    return (1+light*weight[:,None]/np.asarray(reference)[np.maximum(index,0),None]).astype(np.float32)


def main():
    os.environ.setdefault('DRJIT_LIBCUDA_PATH','/dev/null')
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('llvm_ad_rgb');dr.set_thread_count(8)
    LAMPS.mkdir(parents=True,exist_ok=True)
    fixtures,source=verified_fixtures();mask=build_mask(fixtures)
    print('VERIFIED_FIXTURES',len(fixtures),'edited_GS',mask['affected_gaussians'],flush=True)
    plan=dict(model='downward-area-discs-v1',emitters=[{k:f[k] for k in ['center','radius','radiance']} for f in fixtures if f['group']==0],
        geometry=file_identity(ROOT/'experiments/out/nucleus4d_daylight/living_scan.obj'),planes=PLANES,spp=4096,surface_reflections=8)
    plan_path=LAMPS/'transport_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise RuntimeError('Transport inputs changed; keep old fields and choose a fresh output directory')
    atomic_json(plan_path,plan)
    sc=lamp_scene(fixtures);fields=bake_fields(sc)
    reference=json.loads((ROOT/'experiments/out/nucleus4d_sun_increment_v2/reference_scale.json').read_text())['incident_over_pi']
    rows=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    gain=np.memmap(LAMPS/'living_gain.rgb.f32',mode='w+',dtype='<f4',shape=(len(rows),3))
    for start in range(0,len(rows),100000):
        r=rows[start:start+100000];p=r[:,:12].copy().view('<f4').reshape(-1,3)
        gain[start:start+len(r)]=lamp_gain(p,fields,reference)
    gain.flush()
    doc=dict(fixtures=fixtures,groups=[dict(id=0,label='客厅筒灯'),dict(id=2,label='厨房吊灯'),dict(id=1,label='其余已定位灯具')],
        count=len(fixtures),controllable_count=sum(f['controllable'] for f in fixtures),source=file_identity(source),mask=mask,
        gain_sha256=hashlib.sha256((LAMPS/'living_gain.rgb.f32').read_bytes()).hexdigest(),
        gain_max_rgb=np.max(gain,axis=0).tolist(),gaussians=len(rows),
        direct_and_indirect_spp=4096,surface_reflections=8,lightmap_spacing_m=.04,
        original_lighting_retained=True,off_scope='emitting GS surface appearance only; captured room illumination retained',
        boost_scope='above 100% adds approximate path-traced living-room light',full_light_removal=False)
    doc['emitter_geometry']='downward area discs for the ten living-room recessed lights'
    atomic_json(LAMPS/'fixtures.json',doc);atomic_json(LAMPS/'status.json',dict(status='ready_for_review',updated=time.time()))
    print('FIXTURE_CONTROLS_READY',flush=True)


if __name__=='__main__':main()
