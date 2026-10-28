"""Precompute HDR window-light variants without replacing captured GS texture.

The approved additional-light experiment remains separate. These presets use
8K linear EXR environments, aperture visibility, a finite solar disc, and eight
surface reflections. They retain captured illumination; they are not an inverse
rendering solution or a geolocated solar simulation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from nucleus4d_baked_relight import (ROOT, SOURCE, PHOTOS, SH_C0, atomic_json,
    read_source_ply, srgb_to_linear, linear_to_srgb, file_identity)
from nucleus4d_sun_increment import PLANES, WINDOWS, classify, project, scene as room_scene

OUT=ROOT/'experiments/out/nucleus4d_weather_presets'
PRESETS=[
    dict(id='overcast',label='阴天',time='14:00 · 示意',description='冷白漫射天空，柔和窗光',
         asset='overcast_soil_puresky',sun=False,azimuth=0.,sky_target=7.,sun_target=None,exposure=-.35),
    dict(id='direct',label='阳光直射',time='12:30 · 示意',description='晴天强日照，清晰窗影',
         asset='kloofendal_48d_partly_cloudy_puresky',sun=True,azimuth=20.,sky_target=None,sun_target=8.,exposure=0.),
    dict(id='dusk',label='黄昏',time='18:15 · 示意',description='低角度暖光，室内整体更暗',
         asset='venice_sunset',sun=True,azimuth=-8.,sky_target=5.,sun_target=None,exposure=-1.1),
]
LUMA=np.array([.2126,.7152,.0722])


def pixel_directions(y,x,h,w):
    theta=(np.asarray(y)+.5)*np.pi/h;phi=(np.asarray(x)+.5)*2*np.pi/w
    return np.stack([np.sin(phi)*np.sin(theta),np.cos(theta),-np.cos(phi)*np.sin(theta)],axis=-1)


def prepare_environment(preset,output):
    import mitsuba as mi
    path=OUT/'hdri'/f'{preset["asset"]}_8k.exr'
    raw=np.asarray(mi.Bitmap(str(path)))[...,:3].copy();h,w=raw.shape[:2]
    if not np.isfinite(raw).all() or raw.min()<0:raise ValueError('Require finite, nonnegative linear HDR')
    lum=raw@LUMA;y,x=np.unravel_index(np.argmax(lum),lum.shape)
    source_sun=pixel_directions(y,x,h,w)
    # Preserve the HDR's elevation. Only rotate its azimuth toward the known
    # apertures, after converting Mitsuba's local Y-up to the scan's Z-up.
    up=Rotation.from_euler('x',90,degrees=True).as_matrix()
    base=up@source_sun
    yaw=np.deg2rad(preset['azimuth'])-np.arctan2(base[1],base[0])
    rotation=Rotation.from_euler('z',yaw).as_matrix()@up
    sun_dir=rotation@source_sun
    energy=np.zeros(3);removed=np.zeros(3)
    if preset['sun']:
        yy,xx=np.meshgrid(np.arange(max(0,y-70),min(h,y+71)),np.arange(x-90,x+91)%w,indexing='ij')
        d=pixel_directions(yy,xx,h,w)
        angle=np.arccos(np.clip(d@source_sun,-1,1))
        ring=(angle>np.deg2rad(1.2))&(angle<np.deg2rad(1.8))
        background=np.median(raw[yy[ring],xx[ring]],axis=0)
        mask=(angle<np.deg2rad(1.15))&(lum[yy,xx]>max(float(background@LUMA)*12,10.))
        sy=yy[mask];sx=xx[mask]
        residual=np.maximum(raw[sy,sx]-background,0)
        domega=(2*np.pi/w)*(np.pi/h)*np.sin((sy+.5)*np.pi/h)
        energy=(residual*domega[:,None]).sum(0)
        weights=residual@LUMA*domega
        centroid=(pixel_directions(sy,sx,h,w)*weights[:,None]).sum(0)
        centroid/=np.linalg.norm(centroid)
        # Match the extracted disc centroid, rather than one hot image pixel.
        base=up@centroid;yaw=np.deg2rad(preset['azimuth'])-np.arctan2(base[1],base[0])
        rotation=Rotation.from_euler('z',yaw).as_matrix()@up;sun_dir=rotation@centroid
        raw[sy,sx]-=residual;removed=energy.copy()
        if float(energy@LUMA)<=0:raise ValueError('HDR solar extraction failed')
    # Normalize each real HDR once, preserving its chromatic and spatial ratios.
    theta=(np.arange(h)+.5)*np.pi/h
    solid=(2*np.pi/w)*(np.pi/h)*np.sin(theta)
    cosine=np.maximum(np.cos(theta),0)
    horizontal=(raw.sum(1,dtype=np.float64)*((solid*cosine)/np.pi)[:,None]).sum(0)
    scale=(preset['sun_target']/float(energy@LUMA) if preset['sun_target'] is not None
           else preset['sky_target']/float(horizontal@LUMA))
    raw*=scale;energy*=scale
    output.mkdir(parents=True,exist_ok=True)
    sky_path=output/'sky_linear_8k.exr'
    mi.Bitmap(raw).write(str(sky_path))
    matrix=np.eye(4);matrix[:3,:3]=rotation
    info=dict(preset,environment_rotation=matrix.tolist(),sun_direction=sun_dir.tolist(),
        sun_elevation_degrees=float(np.rad2deg(np.arcsin(sun_dir[2]))),
        sun_irradiance=energy.tolist(),hdr_scale=float(scale),
        sky_horizontal_irradiance_over_pi=(horizontal*scale).tolist(),
        extracted_sun_energy_unscaled=removed.tolist(),
        source=file_identity(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_url='https://polyhaven.com/a/'+preset['asset'],license='CC0',
        calibration='illustrative time and orientation; captured illumination retained')
    atomic_json(output/'lighting.json',info)
    # A preview is explicitly tone mapped; the bake above uses full linear EXR.
    thumb=raw[::8,::8];thumb=thumb/(1+thumb)
    mi.util.write_bitmap(str(output/'hdr_preview.jpg'),thumb,write_async=False)
    return info


def make_scene(base,info,output,blackout=False):
    import mitsuba as mi
    desc={'type':'scene','integrator':{'type':'path','max_depth':8,'rr_depth':5}}
    desc.update({f'shape_{i}':shape for i,shape in enumerate(base.shapes())})
    desc['sky']={'type':'envmap','filename':str(output/'sky_linear_8k.exr'),
                 'to_world':mi.ScalarTransform4f(np.array(info['environment_rotation']))}
    if info['sun']:
        desc['sun']={'type':'directional','direction':(-np.array(info['sun_direction'])).tolist(),
                     'irradiance':{'type':'rgb','value':info['sun_irradiance']}}
    if blackout:
        desc['blackout']={'type':'cube','to_world':mi.ScalarTransform4f.translate([5.94,0,0]).scale([.01,2.5,1.5]),
            'bsdf':{'type':'diffuse','reflectance':0}}
    sc=mi.load_dict(desc)
    return sc,next(e for e in sc.emitters() if e.class_name()=='EnvironmentMapEmitter')


def sky_direct(sc,environment,points,normal,spp):
    """Unbiased area sampling of both actual apertures, excluding crossbars."""
    import mitsuba as mi
    import drjit as dr
    uv=qmc.Sobol(3,scramble=True,seed=952).random_base2(int(np.log2(spp)))
    if len(uv)!=spp:raise ValueError('Use a power-of-two sample count')
    aperture=np.minimum((uv[:,0]*2).astype(int),1)
    widths=np.array([b-a for a,b in WINDOWS]);height=.55+.60-(.11-.07)
    # The lower/upper vertical intervals are concatenated by their actual area.
    z=-.55+uv[:,2]*height;z+=np.where(z>=.07,.04,0)
    q=np.stack([np.full(spp,5.98),np.array([a for a,b in WINDOWS])[aperture]+uv[:,1]*widths[aperture],z],axis=1)
    area=widths.sum()*height
    p=np.repeat(np.asarray(points,np.float32),spp,axis=0)
    delta=np.tile(q,(len(points),1))-p;distance2=(delta*delta).sum(1)
    d=delta/np.sqrt(distance2)[:,None]
    n=np.asarray(normal,np.float32)
    ray=mi.Ray3f(mi.Point3f((p+n*.00002).T),mi.Vector3f(d.T));ray.maxt=100
    visible=~sc.ray_test(ray)
    si=dr.zeros(mi.SurfaceInteraction3f,len(p));si.wi=mi.Vector3f(-d.T)
    radiance=environment.eval(si);dr.eval(visible,radiance)
    factor=np.maximum(d@n,0)*np.abs(d[:,0])*area/(np.pi*np.maximum(distance2,1e-8))
    value=np.asarray(radiance).T*np.asarray(visible)[:,None]*factor[:,None]
    return value.reshape(len(points),spp,3).mean(1,dtype=np.float64).astype(np.float32)


def indirect(sc,points,normal,spp,seed):
    """Cosine receiver samples that hit another surface: exclude first misses.

    Mean incoming radiance is E/pi under cosine sampling. Stock MIS transport
    begins at the next surface; together with this receiver it permits 8 bounces.
    """
    import mitsuba as mi
    import drjit as dr
    count=len(points)*spp
    sampler=mi.load_dict({'type':'independent'});sampler.seed(seed,count)
    local=mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
    n=mi.Normal3f(normal);direction=mi.Frame3f(n).to_world(local)
    p=np.repeat(np.asarray(points,np.float32)+np.asarray(normal,np.float32)*.00002,spp,axis=0)
    ray=mi.Ray3f(mi.Point3f(p.T),direction)
    active=sc.ray_test(ray)
    value,_,_=sc.integrator().sample(sc,sampler,ray,medium=None,active=active)
    dr.eval(value)
    return np.asarray(value).T.reshape(len(points),spp,3).mean(1,dtype=np.float64).astype(np.float32)


def bake_fields(sc,environment,output,progress):
    fields=[];spacing=.04
    for k,plane in enumerate(PLANES):
        u=np.arange(plane['lo'][0]+.025,plane['hi'][0]-.025,spacing,dtype=np.float32)
        v=np.arange(plane['lo'][1]+.025,plane['hi'][1]-.025,spacing,dtype=np.float32)
        xx,yy=np.meshgrid(u,v);p=np.zeros((xx.size,3),np.float32)
        p[:,plane['axis']]=plane['bound'];p[:,plane['uv'][0]]=xx.ravel();p[:,plane['uv'][1]]=yy.ravel()
        path=output/f'field_{plane["name"]}.npy'
        if path.exists():data=np.load(path)
        else:
            partial=output/f'partial_{plane["name"]}.npy';done_path=output/f'done_{plane["name"]}.npy'
            if partial.exists() and done_path.exists():
                values=np.lib.format.open_memmap(partial,mode='r+');done=np.lib.format.open_memmap(done_path,mode='r+')
            else:
                values=np.lib.format.open_memmap(partial,mode='w+',dtype='<f4',shape=(len(p),2,3));values[:]=0;values.flush()
                done=np.lib.format.open_memmap(done_path,mode='w+',dtype='u1',shape=(len(p),));done[:]=0;done.flush()
            for start in range(0,len(p),256):
                end=min(start+256,len(p))
                if done[start:end].all():continue
                values[start:end,0]=sky_direct(sc,environment,p[start:end],plane['n'],1024)
                values[start:end,1]=indirect(sc,p[start:end],plane['n'],4096,800001+k*100000+start)
                if not np.isfinite(values[start:end]).all() or values[start:end].min()<0:raise ValueError('Invalid transport')
                values.flush();done[start:end]=1;done.flush()
                progress('transport',plane=plane['name'],completed=int(done.sum()),total=len(p))
            direct=np.asarray(values[:,0]).reshape(len(v),len(u),3)
            gi=np.asarray(values[:,1]).reshape(len(v),len(u),3)
            # Only smooth Monte Carlo GI residual, never direct shadows or GS.
            gi=gaussian_filter(gi,sigma=(.65,.65,0),mode='nearest')
            data=direct+gi;np.save(path,data)
        if data.shape!=(len(v),len(u),3):raise ValueError('Cached field dimensions changed')
        fields.append(dict(data=data,u0=float(u[0]),v0=float(v[0]),spacing=spacing))
    return fields


def solar_directions(direction,spp=1024):
    d=np.array(direction,dtype=np.float64);d/=np.linalg.norm(d)
    t=np.cross(d,[0,0,1]);t/=np.linalg.norm(t);b=np.cross(d,t)
    u=(np.arange(spp)+.5)/spp;phi=np.arange(spp)*np.pi*(3-np.sqrt(5))
    r=np.sqrt(u)*np.tan(np.deg2rad(.265))
    rays=d+r[:,None]*(np.cos(phi)[:,None]*t+np.sin(phi)[:,None]*b)
    return (rays/np.linalg.norm(rays,axis=1,keepdims=True)).astype(np.float32)


def gains(points,sc,fields,info,reference):
    import mitsuba as mi
    import drjit as dr
    index,weight=classify(points);light=np.zeros((len(points),3),np.float32)
    direction=np.array(info['sun_direction']);directions=solar_directions(direction)
    for k,plane in enumerate(PLANES):
        ids=np.flatnonzero(index==k)
        if not len(ids):continue
        p=project(points[ids],plane);f=fields[k];uv=p[:,plane['uv']]
        coords=np.stack([(uv[:,1]-f['v0'])/f['spacing'],(uv[:,0]-f['u0'])/f['spacing']])
        for c in range(3):light[ids,c]=map_coordinates(f['data'][...,c],coords,order=1,mode='nearest')
        if not info['sun'] or np.dot(plane['n'],direction)<=0:continue
        distance=(5.98-p[:,0])/direction[0];hit=p+direction*distance[:,None]
        window=np.zeros(len(ids),bool)
        for a,b in WINDOWS:window|=(hit[:,1]>a-.12)&(hit[:,1]<b+.12)
        lit=np.flatnonzero((distance>0)&window&(hit[:,2]>-.67)&(hit[:,2]<.72))
        cosine=np.maximum(directions@plane['n'],0)
        for start in range(0,len(lit),512):
            j=lit[start:start+512]
            origins=np.repeat(p[j]+np.array(plane['n'],np.float32)*.00002,len(directions),axis=0)
            ds=np.tile(directions,(len(j),1))
            ray=mi.Ray3f(mi.Point3f(origins.T),mi.Vector3f(ds.T));ray.maxt=100
            visible=~sc.ray_test(ray);dr.eval(visible)
            irradiance=(np.asarray(visible).reshape(len(j),-1)*cosine).mean(1)/np.pi
            light[ids[j]]+=irradiance[:,None]*np.array(info['sun_irradiance'])
    light*=weight[:,None]
    return (1+light/np.asarray(reference)[np.maximum(index,0),None]).astype(np.float32)


def write_variant(output,sc,fields,info,reference,progress):
    source=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    gain_path=output/'gain.rgb.f32'
    gain=np.memmap(gain_path,mode='w+',dtype='<f4',shape=(len(source),3))
    for start in range(0,len(source),100000):
        end=min(start+100000,len(source));p=source[start:end,:12].copy().view('<f4').reshape(-1,3)
        gain[start:end]=gains(p,sc,fields,info,reference)
        progress('gaussian_light',completed=end,total=len(source))
    gain.flush()
    header,original=read_source_ply(PHOTOS/'source.cache.ply')
    path=output/'scene.partial.ply'
    with path.open('wb') as stream:stream.write(header);stream.truncate(len(header)+original.nbytes)
    target=np.memmap(path,mode='r+',dtype='<f4',offset=len(header),shape=original.shape)
    for start in range(0,len(original),100000):
        end=min(start+100000,len(original));r=original[start:end];target[start:end]=r
        g=gains(r[:,:3],sc,fields,info,reference)
        encoded=np.maximum(.5+r[:,6:9]*SH_C0,0)
        target[start:end,6:9]=(linear_to_srgb(srgb_to_linear(encoded)*g)-.5)/SH_C0
        progress('float32_master',completed=end,total=len(original))
    target.flush();del target;path.replace(output/'scene.ply')
    if not np.isfinite(gain).all() or gain.min()<1:raise ValueError('Invalid gain export')
    report=dict(status='awaiting_visual_review',gaussians=len(source),
        gain_min=float(gain.min()),gain_max_rgb=np.max(gain,axis=0).tolist(),
        affected=int(np.any(gain>1.00001,axis=1).sum()),
        gain_sha256=hashlib.sha256(gain_path.read_bytes()).hexdigest(),
        source_identity=file_identity(SOURCE/'full.splat'),
        geometry_equal=True,captured_lighting_retained=True,full_relighting=False,
        direct_spp=1024,indirect_spp=4096,surface_reflections=8,
        lightmap_spacing_m=.04,light_field_dtype='float32 RGB',exposure_ev=info['exposure'])
    atomic_json(output/'result.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=[p['id'] for p in PRESETS],nargs='+')
    parser.add_argument('--cpu',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    if args.cpu:os.environ.setdefault('DRJIT_LIBCUDA_PATH','/dev/null')
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('llvm_ad_rgb' if args.cpu else 'cuda_ad_rgb');dr.set_thread_count(args.threads)
    base=room_scene()
    reference=json.loads((ROOT/'experiments/out/nucleus4d_sun_increment_v2/reference_scale.json').read_text())['incident_over_pi']
    for preset in PRESETS:
        if args.preset and preset['id'] not in args.preset:continue
        output=OUT/preset['id'];output.mkdir(parents=True,exist_ok=True)
        def progress(stage,**values):
            record=dict(preset=preset['id'],stage=stage,updated=time.time(),**values)
            atomic_json(OUT/'status.json',record)
            if stage!='transport' or values.get('completed',0)%2048<256:print('WEATHER',json.dumps(record),flush=True)
        info_path=output/'lighting.json'
        if info_path.exists():
            info=json.loads(info_path.read_text())
            if any(info.get(k)!=v for k,v in preset.items()):raise ValueError('Preset changed; use a fresh output directory')
        else:info=prepare_environment(preset,output)
        print('HDR_PREPARED',json.dumps(info),flush=True)
        if args.prepare_only:continue
        sc,environment=make_scene(base,info,output)
        fields=bake_fields(sc,environment,output,progress)
        report=write_variant(output,sc,fields,info,reference,progress)
        progress('awaiting_visual_review',**report)


if __name__=='__main__':main()
