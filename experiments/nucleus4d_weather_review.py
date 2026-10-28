"""Render full, unchanged Gaussian geometry for weather appearance review."""
import argparse
import json
import shutil
import time

import numpy as np

from nucleus4d_baked_relight import PHOTOS,read_source_ply,atomic_json
from nucleus4d_sun_increment import OUT as APPROVED
from nucleus4d_weather_presets import OUT,PRESETS


def display_image(image,ev):
    linear=np.asarray(image)*2.**ev
    peak=linear.max(axis=2,keepdims=True)
    shoulder=.8+(peak-.8)/(1+np.maximum(peak-.8,0)/.2)
    return linear*np.where(peak>.8,shoulder/np.maximum(peak,1e-10),1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=[p['id'] for p in PRESETS],nargs='+')
    parser.add_argument('--width',type=int,default=896)
    parser.add_argument('--spp',type=int,default=8)
    parser.add_argument('--wait',action='store_true',help='Wait for each durable float32 master while baking continues')
    parser.add_argument('--gpu',action='store_true',help='Use the recovered GPU for native Gaussian image review')
    args=parser.parse_args()
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('cuda_ad_rgb' if args.gpu else 'llvm_ad_rgb');dr.set_thread_count(4)
    from nucleus4d_ptir import shape,sensor,load_integrators
    load_integrators()
    review=OUT/'review';review.mkdir(parents=True,exist_ok=True)
    for i in range(3):
        for old,new in [('original','original'),('increment','approved')]:
            shutil.copyfile(APPROVED/'review'/f'{old}_{i:02}.png',review/f'{new}_{i:02}.png')
    _,original=read_source_ply(PHOTOS/'source.cache.ply')
    gs=mi.load_dict(shape(original,material=False))
    scene=mi.load_dict({'type':'scene','integrator':{'type':'volprim_rf','max_depth':512,'srgb_primitives':True},'shape':gs})
    params=mi.traverse(scene);key=next(k for k in params.keys() if k.endswith('sh_coeffs'))
    report_path=review/'render_report.json';records=[]
    if report_path.exists():records=json.loads(report_path.read_text())['records']
    for preset in PRESETS:
        if args.preset and preset['id'] not in args.preset:continue
        folder=OUT/preset['id']
        if args.wait:
            deadline=time.monotonic()+1800
            while not (folder/'scene.ply').exists() and time.monotonic()<deadline:time.sleep(3)
        if not (folder/'scene.ply').exists():raise RuntimeError('Weather bake is incomplete: '+preset['id'])
        _,candidate=read_source_ply(folder/'scene.ply')
        columns=[i for i in range(17) if i not in [6,7,8]]
        if original.shape!=candidate.shape:raise ValueError('Gaussian count changed')
        for start in range(0,len(original),100000):
            if not np.array_equal(original[start:start+100000,columns],candidate[start:start+100000,columns]):
                raise ValueError('Geometry, opacity or rotation changed')
        params[key]=mi.Float(np.ascontiguousarray(candidate[:,6:9]).ravel());params.update()
        for index,view in enumerate([-1,4,3]):
            path=review/f'{preset["id"]}_{index:02}.png'
            if path.exists():continue
            start=time.monotonic()
            image=np.array(mi.render(scene,sensor=sensor(args.width,view),spp=args.spp,seed=117+index),copy=True)
            if not np.isfinite(image).all() or image.min()<-1e-5:raise ValueError('Invalid Gaussian render')
            np.save(review/f'{preset["id"]}_{index:02}.linear.npy',image)
            displayed=display_image(image,preset['exposure'])
            mi.util.write_bitmap(str(path),displayed,write_async=False)
            record=dict(preset=preset['id'],view=index,gaussians=len(original),width=args.width,spp=args.spp,
                seconds=time.monotonic()-start,geometry_equal=True,exposure=preset['exposure'],
                mean=float(displayed.mean()),max=float(displayed.max()),renderer='native volprim_rf')
            records.append(record);print('WEATHER_IMAGE',json.dumps(record),flush=True)
            atomic_json(report_path,dict(records=records,geometry_equal=True))


if __name__=='__main__':main()
