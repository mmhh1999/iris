"""Render compact daylight bases from the full-resolution native GS scene."""
import datetime
import hashlib
import json
import sys
import time
from zoneinfo import ZoneInfo
import numpy as np
from nucleus4d_ptir import ROOT, OUT, source_arrays, load_integrators, shape, sensor, mi
sys.path.insert(0,str(ROOT))
from utils.solar_geometry import solar_position, sun_vector_world

W=896
H=round(W*416/640)


def main():
    mi.set_variant('cuda_ad_rgb');load_integrators();OUT.mkdir(parents=True,exist_ok=True)
    data,member=source_arrays()
    material_hash=hashlib.sha256((OUT/'materials.npz').read_bytes()).hexdigest()
    gs=mi.load_dict(shape(data,True));camera=sensor(W)
    assert gs.primitive_count()==len(data), 'Renderer did not retain all original Gaussians'
    denoise=mi.OptixDenoiser([W,H],albedo=True)
    manifest={'width':W,'height':H,'gaussians':len(data),'source_member':member,
              'representation':'original full-resolution 3D Gaussians, analytic ray intersections',
              'geometry_unchanged':True,'renderer':'PTIR-Mitsuba with compatibility and window-portal adapter',
              'materials':'photo-prior initialization; partial coverage; no joint inverse optimization',
              'normals':'PCA of original Gaussian surface neighborhoods',
              'windows':'two manually specified clear apertures for illumination rays; no glass refraction or exterior occluders beyond apertures',
              'location':'example San Francisco, 2026-03-21, scene north bearing 90 degrees',
              'interaction':'fixed camera; two-hour precomputed path-traced frames, no real-time ray tracing',
              'material_sha256':material_hash,
              'filter':'OptiX denoiser with albedo guide; raw linear bases retained',
              'skies':{},'sun':[]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

    def scene(light,portals=True):
        return mi.load_dict({'type':'scene','shape':gs,
                            'integrator':{'type':'nucleus_ptir','max_depth':5,'gaussian_max_depth':256,
                              'use_mis':True,'geometry_threshold':.3,'selfocc_offset_max':.05,
                              'window_portals':portals,'separate_direct_indirect':True},**light})

    def render(name,light,spp,portals=True):
        destination=OUT/(name+'_raw.npz')
        tic=time.time()
        cached=False
        if destination.exists():
            with np.load(destination) as f:
                cached='material_hash' in f and str(f['material_hash'])==material_hash
        if cached:
            with np.load(destination) as f:
                total=f['total'].astype('float32');direct=f['direct'].astype('float32');albedo=f['albedo'].astype('float32')
                spp=int(f['spp']) if 'spp' in f else 192
        else:
            s=scene(light,portals);acc=None
            for i in range(0,spp,4):
                batch=min(4,spp-i)
                a=np.array(s.integrator().render(s,sensor=camera,spp=batch,seed=1000+i))*batch
                if not np.isfinite(a).all():raise RuntimeError('nonfinite radiance')
                if i==0:
                    mi.util.write_bitmap(str(OUT/'estimated_albedo.png'),a[:,:,4:7]/batch)
                acc=a if acc is None else acc+a
                if (i+batch)%32==0: print(name,i+batch,'/',spp,round(time.time()-tic,1),'s',flush=True)
            acc/=spp;total=acc[:,:,:3];direct=acc[:,:,9:12];albedo=acc[:,:,4:7]
            np.savez_compressed(destination,total=total.astype('float16'),direct=direct.astype('float16'),albedo=albedo.astype('float16'),spp=spp,material_hash=material_hash)
        record={'spp':spp,'cached':cached,'seconds':time.time()-tic,'mean':float(total.mean()),'direct_mean':float(direct.mean()),
                'indirect_mean':float((total-direct).mean())}
        guide=mi.TensorXf(albedo.copy())
        filtered_direct=np.maximum(np.array(denoise(mi.TensorXf(direct.copy()),albedo=guide)),0)
        filtered_indirect=np.maximum(np.array(denoise(mi.TensorXf(np.maximum(total-direct,0)),albedo=guide)),0)
        for label,filtered in [('total',filtered_direct+filtered_indirect),('direct',filtered_direct)]:
            filename=name+'_'+label+'.bin';filtered.astype('<f2').tofile(OUT/filename);record[label]=filename
            if label=='total':
                exposed=filtered*8;exposed=exposed/(1+exposed)
                mi.util.write_bitmap(str(OUT/(name+'.png')),exposed)
        print('DONE',name,record,flush=True)
        return record

    original=mi.load_dict({'type':'scene','shape':gs,'integrator':{'type':'volprim_rf','max_depth':512,'srgb_primitives':True}})
    original_rgb=np.array(mi.render(original,sensor=camera,spp=8,seed=29))
    mi.util.write_bitmap(str(OUT/'original.png'),original_rgb)
    original_rgb.astype('<f2').tofile(OUT/'original.bin')
    for kind in ['clear','overcast']:
        y=(np.arange(128)+.5)/128;z=np.cos(np.pi*y)[:,None,None]
        if kind=='clear':sky=np.array([.4,.6,1.])[None,None,:]*(.55+.45*np.maximum(z,0))
        else:sky=np.array([.7,.72,.75])[None,None,:]*(1+2*np.maximum(z,0))/3
        sky=np.repeat(np.where(z<0,np.array([.06,.055,.05])[None,None,:],sky),256,axis=1).astype('float32')
        path=OUT/(kind+'.exr');mi.util.write_bitmap(str(path),sky,write_async=False)
        light={'sky':{'type':'envmap','filename':str(path),'to_world':mi.ScalarTransform4f.rotate([1,0,0],90)}}
        manifest['skies'][kind]=render('sky_'+kind,light,96)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for hour in [13,11,15,9,17]:
        dt=datetime.datetime(2026,3,21,hour,tzinfo=ZoneInfo('America/Los_Angeles'))
        sol=solar_position(37.7749,-122.4194,dt);vec=sun_vector_world(sol.azimuth_deg,sol.elevation_deg,90)
        warm=np.clip(sol.elevation_deg/35,0,1);power=5*np.clip(np.sin(np.deg2rad(sol.elevation_deg))/.65,0,1)
        light={'sun':{'type':'directional','direction':(-vec).tolist(),'irradiance':{'type':'rgb','value':(power*np.array([1,.72+.24*warm,.42+.44*warm])).tolist()}},
               'environment':{'type':'constant','radiance':0.}}
        rec=render(f'sun_{hour:02}',light,48);rec.update(hour=hour,azimuth=sol.azimuth_deg,elevation=sol.elevation_deg)
        manifest['sun'].append(rec);manifest['sun'].sort(key=lambda x:x['hour'])
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if hour==13:
            closed=render('portal_disabled_sun_13',light,32,portals=False)
            manifest['validation']={'sun_portals_enabled_mean':rec['mean'],'sun_original_occlusion_mean':closed['mean'],
                                    'original_occlusion_over_portals':closed['mean']/max(rec['mean'],1e-12),
                                    'occlusion_control':'disable assumed clear apertures and restore original Gaussian occlusion; not an opaque-window blackout'}
    black=render('all_lights_off',{'environment':{'type':'constant','radiance':0.}},1)
    manifest['validation']['all_lights_off_mean']=black['mean']
    manifest['complete']=True
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
