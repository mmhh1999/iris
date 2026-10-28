"""Actual full-Gaussian renders and source-isolation checks for transport v3."""
import argparse
import json
import time

import numpy as np

from nucleus4d_baked_relight import SOURCE,PHOTOS,SH_C0,read_source_ply,srgb_to_linear,linear_to_srgb,atomic_json
from nucleus4d_transport_geometry import OUT,EPS,make_scene


def incident(mode,meta):
    from pathlib import Path
    folder=Path(meta.get('field_folder',OUT))
    sky=np.load(folder/'fine_sky_total.npy');field=mode['sky']*((1-mode['cloud'])*sky[:,0]+mode['cloud']*sky[:,1])
    for i,power in enumerate(mode['lamps']):
        if power:field+=power*np.load(folder/f'fine_lamp_{i:02}.npy')
    if mode['sun']:
        keys=json.loads((OUT/'sun_keys.json').read_text());minutes=keys['minutes'];i=int(np.clip(np.searchsorted(minutes,mode['minute'])-1,0,len(minutes)-2))
        t=(mode['minute']-minutes[i])/(minutes[i+1]-minutes[i]);strength=mode['sun']*(1-mode['cloud'])**2
        field+=strength*((1-t)*np.load(OUT/f'sun_gi/fine_{i:03}.npy')+t*np.load(OUT/f'sun_gi/fine_{i+1:03}.npy'))
    return field


def colors_for_mode(original,mapping,mode,meta,visibility_scene,sun_spp=64):
    import mitsuba as mi
    import drjit as dr
    from nucleus4d_transport_bake import sun_state
    from nucleus4d_weather_presets import solar_directions
    if mode['id']=='original':return np.ascontiguousarray(original[:,6:9]),dict(exact_original=True)
    from pathlib import Path
    field=incident(mode,meta);reference=np.memmap(Path(meta.get('field_folder',OUT))/'reference_irradiance.f32',dtype='<f4',mode='r').reshape(-1,3)
    bind=np.memmap(OUT/'binding_compact.u32',dtype='<u4',mode='r').reshape(-1,4)
    normal_path=OUT/'normal_weights_full.f32';normal_path=normal_path if normal_path.exists() else OUT/'gaussian_normal_weights.f32'
    nw=np.memmap(normal_path,dtype='<f4',mode='r').reshape(-1,4)
    positions=np.memmap(OUT/'patch_positions.f32',dtype='<f4',mode='r').reshape(-1,3)
    mask=np.memmap(OUT.parent/'nucleus4d_weather_controls/lamps/fixture_mask.f32',dtype='<f4',mode='r')
    fixtures=json.loads((OUT/'lights.json').read_text())['lights'];face=np.ones(27,np.float32)
    for i,f in enumerate(fixtures):face[f['source_id']]=mode['lamps'][i]
    direction,power=sun_state(mode['minute']);power=power*mode['sun']*(1-mode['cloud'])**2;rays=solar_directions(direction,sun_spp)
    exterior=mode['sky']+.15*float(power@np.array([.2126,.7152,.0722]));colors=np.empty((len(original),3),np.float32);max_ratio=0;zero_ref=0
    for start in range(0,len(original),32768):
        end=min(start+32768,len(original));order=mapping[start:end];b=np.array(bind[order]);a=np.array(nw[order]);valid=b[:,0]!=0xffffffff;volume=(b[:,0]&np.uint32(0x80000000))!=0;b[:,0]&=np.uint32(0x7fffffff)
        ratio=np.full((end-start,3),exterior,np.float32)
        if valid.any():
            ids=b[valid,:3];weights=np.column_stack([b[valid,3].copy().view('<f4'),a[valid,0],np.maximum(0,1-b[valid,3].copy().view('<f4')-a[valid,0])])
            ref=(reference[ids]*weights[:,:,None]).sum(1);light=(field[ids]*weights[:,:,None]).sum(1);normal=a[valid,1:]
            if np.any(power):
                p=(positions[ids]*weights[:,:,None]).sum(1);p[volume[valid]]=original[start:end,:3][valid][volume[valid]]
                d=np.tile(rays,(len(p),1));origin=np.repeat(p+normal*.002,sun_spp,axis=0)
                visible=~visibility_scene.ray_test(mi.Ray3f(mi.Point3f(origin.T),mi.Vector3f(d.T)));dr.eval(visible)
                light+=(np.asarray(visible).reshape(len(p),sun_spp)*np.maximum(normal@rays.T,0)).mean(1)[:,None]*power/np.pi
            zero_ref+=int((ref.min(1)<1e-7).sum());ratio[valid]=light/np.maximum(ref,1e-7)
        encoded=np.maximum(.5+original[start:end,6:9]*SH_C0,0);base=srgb_to_linear(encoded);linear=base*ratio
        packed=mask[order];id=np.floor(packed).astype(int);weight=(packed-id)*2
        linear=(1-weight[:,None])*linear+weight[:,None]*(base*face[id,None]+.08*linear)/1.08
        colors[start:end]=(linear_to_srgb(linear)-.5)/SH_C0;max_ratio=max(max_ratio,float(ratio.max()))
    return colors,dict(maximum_ratio=max_ratio,zero_reference_gaussians=zero_ref,finite=bool(np.isfinite(colors).all()),sun_visibility_samples=sun_spp)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--width',type=int,default=896);parser.add_argument('--spp',type=int,default=8);parser.add_argument('--quick',action='store_true');parser.add_argument('--lamps-only',action='store_true');parser.add_argument('--filtered',action='store_true');parser.add_argument('--sky-mis',action='store_true');parser.add_argument('--emitters',action='store_true');parser.add_argument('--cpu',action='store_true');args=parser.parse_args()
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('llvm_ad_rgb' if args.cpu else 'cuda_ad_rgb');dr.set_thread_count(6)
    from nucleus4d_ptir import shape,sensor,load_integrators
    from nucleus4d_controls_review import source_mapping
    from nucleus4d_weather_review import display_image
    load_integrators();review=OUT/('review_emitters' if args.emitters else 'review_sky_mis' if args.sky_mis else 'review_filtered' if args.filtered else 'review');review.mkdir(exist_ok=True)
    meta=json.loads((OUT/'bank_manifest.json').read_text()) if (OUT/'bank_manifest.json').exists() else {}
    if not meta and not args.lamps_only:raise RuntimeError('Sunlight bank not complete')
    if args.filtered:meta['field_folder']=str(OUT/'filtered_bases')
    if args.sky_mis:meta['field_folder']=str(OUT/'sky_mis_bases')
    if args.emitters:meta['field_folder']=str(OUT/'emitter_bases')
    if not (OUT/'reference_irradiance.f32').exists():
        ref=np.load(OUT/'fine_sky_total.npy')[:,1].copy()
        for i in range(25):ref+=np.load(OUT/f'fine_lamp_{i:02}.npy')
        ref.tofile(OUT/'reference_irradiance.f32')
        from nucleus4d_transport_audit import repair_unlit_bindings
        repair_unlit_bindings(ref,np.memmap(OUT/'patch_positions.f32',mode='r',dtype='<f4').reshape(-1,3))
    _,original=read_source_ply(PHOTOS/'source.cache.ply');mapping=source_mapping(original)
    with np.load(OUT/'geometry.npz') as z:geometry={k:z[k] for k in z.files}
    visibility_scene=make_scene(geometry)
    gs=mi.load_dict(shape(original,material=False));scene=mi.load_dict({'type':'scene','integrator':{'type':'volprim_rf','max_depth':512,'srgb_primitives':True},'shape':gs})
    params=mi.traverse(scene);key=next(k for k in params.keys() if k.endswith('sh_coeffs'))
    def mode(name,**kwargs):return dict(id=name,sky=1.,cloud=1.,sun=0.,minute=750.,lamps=[1.]*25,**kwargs)
    modes=[mode('original'),mode('lamps_off')]
    modes[1]['lamps']=[0.]*25
    for name,minute in ([] if args.lamps_only else [('sun_1230',750),('sun_1600',960),('sun_1815',1095)]):
        m=mode(name);m.update(cloud=0.,sun=1.,minute=minute);modes.append(m)
    m=mode('black');m.update(sky=0.,lamps=[0.]*25);modes.append(m)
    if not args.quick and (not args.lamps_only or args.emitters):
        for f in json.loads((OUT/'lights.json').read_text())['lights']:
            m=mode(f'isolated_{f["id"]:02}');m.update(sky=0.,lamps=[0.]*25);m['lamps'][f['id']]=1.;modes.append(m)
    records=[]
    capture=json.loads((PHOTOS/'dataset.json').read_text())['frames']
    camera_centers=np.asarray([f['pose'] for f in capture],np.float32)[:,:3,3]
    light_catalog=json.loads((OUT/'lights.json').read_text())['lights']
    for m in modes:
        started=time.monotonic();colors,audit=colors_for_mode(original,mapping,m,meta,visibility_scene,16 if args.quick else 64)
        if not np.isfinite(colors).all():raise RuntimeError('Nonfinite Gaussian color')
        params[key]=mi.Float(colors.ravel());params.update()
        views=[-1] if args.quick or m['id'].startswith('isolated') else [-1,4,3]
        for index,view in enumerate(views):
            camera=sensor(args.width,view)
            camera_visibility_verified=None
            if m['id'].startswith('isolated'):
                light=light_catalog[int(m['id'].split('_')[1])];center=np.array(light['center']);distance=np.linalg.norm(camera_centers[:,:2]-center[:2],axis=1)
                candidates=np.argsort(abs(distance-1.4)+abs(camera_centers[:,2]-.45))[:256]
                eye=camera_centers[candidates[0]];target=center.copy();target[2]-=.65;camera_visibility_verified=False
                for drop in [.65,.3,1.]:
                    aim=center.copy();aim[2]-=drop;eyes=camera_centers[candidates];delta=aim-eyes;length=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(length[:,None],1e-8)
                    ray=mi.Ray3f(mi.Point3f((eyes+direction*.002).T),mi.Vector3f(direction.T));ray.maxt=mi.Float(np.maximum(length-.005,1e-5));clear=~visibility_scene.ray_test(ray);dr.eval(clear)
                    valid=np.flatnonzero(np.asarray(clear)&(length>.3))
                    if len(valid):eye=eyes[valid[0]];target=aim;camera_visibility_verified=True;break
                camera=mi.load_dict({'type':'perspective','fov':77.,'fov_axis':'x','to_world':mi.ScalarTransform4f.look_at(origin=eye,target=target,up=[0,0,1]),'near_clip':.01,'far_clip':200.,'sampler':{'type':'independent'},'film':{'type':'hdrfilm','width':args.width,'height':round(args.width*416/640),'pixel_format':'rgb','rfilter':{'type':'box'}}})
            image=np.array(mi.render(scene,sensor=camera,spp=args.spp,seed=117+index),copy=True)
            if not np.isfinite(image).all():raise RuntimeError('Nonfinite native Gaussian image')
            np.save(review/f'{m["id"]}_{index:02}.linear.npy',image)
            mi.util.write_bitmap(str(review/f'{m["id"]}_{index:02}.png'),image if m['id']=='original' else display_image(image,0),write_async=False)
            record=dict(mode=m['id'],view=view,index=index,gaussians=len(original),width=args.width,spp=args.spp,geometry_equal=True,camera_visibility_verified=camera_visibility_verified,seconds=time.monotonic()-started,**audit)
            records.append(record);print('TRANSPORT_IMAGE',json.dumps(record),flush=True)
        if m['id']=='black' and float(image.max())>1e-6:raise RuntimeError('All-zero sources did not produce a black field')
    atomic_json(review/'render_report.json',dict(records=records,all_original_gaussians=True,only_colors_updated=True,physical_ground_truth_verified=False))


if __name__=='__main__':main()
