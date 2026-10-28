"""Native full-GS reviews of interpolated light and fixture power controls."""
import json
import time

import numpy as np

from nucleus4d_baked_relight import SOURCE,PHOTOS,SH_C0,read_source_ply,srgb_to_linear,linear_to_srgb,atomic_json
from nucleus4d_fixture_controls import OUT,LAMPS
from nucleus4d_weather_presets import OUT as WEATHER
from nucleus4d_weather_review import display_image


def source_mapping(original):
    source=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    points=source[:,:12].copy().view('<f4').reshape(-1,3)
    path=OUT/'source_to_splat.npy'
    if path.exists():mapping=np.load(path,mmap_mode='r')
    else:
        a=np.lexsort((points[:,2],points[:,1],points[:,0]));b=np.lexsort((original[:,2],original[:,1],original[:,0]))
        if not np.array_equal(points[a],original[b,:3]):raise ValueError('Original GS positions do not match')
        mapping=np.empty(len(points),np.int64);mapping[b]=a;np.save(path,mapping)
    for start in range(0,len(points),100000):
        end=min(start+100000,len(points))
        if not np.array_equal(original[start:end,:3],points[mapping[start:end]]):raise ValueError('Stale GS mapping')
    return mapping


def main():
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('cuda_ad_rgb')
    from nucleus4d_ptir import shape,sensor,load_integrators
    load_integrators();review=OUT/'review';review.mkdir(exist_ok=True,parents=True)
    _,original=read_source_ply(PHOTOS/'source.cache.ply');mapping=source_mapping(original)
    config=json.loads((LAMPS/'fixtures.json').read_text())
    mask=np.memmap(LAMPS/'fixture_mask.f32',mode='r',dtype='<f4')[mapping]
    masks=2*(mask-np.floor(mask));ids=np.floor(mask).astype(int)
    fields={p:np.memmap(WEATHER/p/'gain.rgb.f32',mode='r',dtype='<f4',shape=(len(original),3)) for p in ['direct','dusk','overcast']}
    lamp=np.memmap(LAMPS/'living_gain.rgb.f32',mode='r',dtype='<f4',shape=(len(original),3))
    gs=mi.load_dict(shape(original,material=False));scene=mi.load_dict({'type':'scene','integrator':{'type':'volprim_rf','max_depth':512,'srgb_primitives':True},'shape':gs})
    params=mi.traverse(scene);key=next(k for k in params.keys() if k.endswith('sh_coeffs'))
    minute=960;cloud=.45;t=(minute-750)/345;anchor=90/345
    modes=[dict(id='original',weights=[0,0,0],exposure=0,power=1,hdr=False),
        dict(id='mixed',weights=[(1-t)*(1-cloud),t*(1-cloud),cloud],exposure=-1.1*t+cloud*(-.35+1.1*anchor),power=1,hdr=True),
        dict(id='off',weights=[0,0,0],exposure=0,power=0,hdr=False),
        dict(id='boost',weights=[0,1,0],exposure=-1.1,power=2,hdr=True)]
    records=[]
    for mode in modes:
        colors=np.empty((len(original),3),np.float32)
        for start in range(0,len(original),100000):
            end=min(start+100000,len(original));gain=np.ones((end-start,3),np.float32);mp=mapping[start:end]
            for weight,field in zip(mode['weights'],fields.values()):
                if weight:gain+=weight*(field[mp]-1)
            if mode['power']>1:gain+=(mode['power']-1)*(lamp[mp]-1)
            factor=1+masks[start:end]*(.1+.9*mode['power']-1)
            encoded=np.maximum(.5+original[start:end,6:9]*SH_C0,0)
            encoded=linear_to_srgb(srgb_to_linear(encoded)*gain*factor[:,None])
            colors[start:end]=(encoded-.5)/SH_C0
        if mode['id']=='original':colors=np.ascontiguousarray(original[:,6:9])
        params[key]=mi.Float(colors.ravel());params.update()
        for index,view in enumerate([-1,4,3]):
            start=time.monotonic();image=np.array(mi.render(scene,sensor=sensor(896,view),spp=8,seed=117+index),copy=True)
            if not np.isfinite(image).all():raise ValueError('Nonfinite control image')
            np.save(review/f'{mode["id"]}_{index:02}.linear.npy',image)
            displayed=display_image(image,mode['exposure']) if mode['hdr'] else image
            mi.util.write_bitmap(str(review/f'{mode["id"]}_{index:02}.png'),displayed,write_async=False)
            record=dict(mode=mode['id'],view=index,gaussians=len(original),spp=8,width=896,geometry_equal=True,seconds=time.monotonic()-start)
            records.append(record);print('CONTROLS_IMAGE',json.dumps(record),flush=True)
    atomic_json(review/'render_report.json',dict(records=records,geometry_equal=True,only_color_parameters_updated=True))


if __name__=='__main__':main()
