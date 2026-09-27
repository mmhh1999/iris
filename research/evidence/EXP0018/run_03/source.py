"""Export native BSDF angular responses, not fabricated metallic/roughness labels."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import mitsuba as mi
import drjit as dr


def run(args):
    mi.set_variant('cuda_ad_rgb')
    out=Path(args.out); out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'source.py')
    manifest={'protocol':'generator-only native BSDF response at pixel centers',
              'quantity':'BSDF times outgoing cosine; RGB, inverse steradian',
              'limitations':['Not IRIS metallic-roughness parameters',
                  'Delta lobes have zero continuous eval and require separate assessment',
                  'First geometric intersection, not stochastic transparency traversal',
                  'Pixel-center samples do not reproduce filtered rendered pixels'], 'views':[]}
    directions=[]
    for theta in [0,30,60]:
        for phi in ([0] if theta==0 else [0,90,180,270]):
            t,p=np.deg2rad([theta,phi]); directions.append([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    manifest['outgoing_directions_local']=directions
    for room in ['bathroom','kitchen']:
        for view in [0,1]:
            xml=Path(args.source)/room/f'view{view}_sun0.xml'
            scene=mi.load_file(str(xml.resolve()));sensor=scene.sensors()[0]
            w,h=map(int,sensor.film().size()); y,x=np.mgrid[:h,:w]
            ray,_=sensor.sample_ray(0.,0.,mi.Point2f(mi.Float(((x+.5)/w).ravel()),mi.Float(((y+.5)/h).ravel())),mi.Point2f(.5,.5))
            si=scene.ray_intersect(ray);valid=np.asarray(si.is_valid()).reshape(h,w)
            bsdf=si.bsdf(ray); ctx=mi.BSDFContext()
            responses=[]
            for d in directions:
                value=bsdf.eval(ctx,si,mi.Vector3f(*map(float,d)),si.is_valid())
                responses.append(np.array(value).T.reshape(h,w,3))
            response=np.stack(responses)
            if not np.isfinite(response).all() or (response < -1e-6).any():raise ValueError('Invalid response')
            ids=dr.full(mi.Int,-1,w*h); shapes=[]
            for i,shape in enumerate(scene.shapes()):
                ids=dr.select(si.shape==shape,i,ids)
                shapes.append({'index':i,'shape_id':shape.id(),'bsdf_id':shape.bsdf().id(),'bsdf_description':str(shape.bsdf())})
            flags=np.asarray(bsdf.flags()).reshape(h,w)
            delta=((flags & int(mi.BSDFFlags.Delta))!=0)&valid
            null=((flags & int(mi.BSDFFlags.Null))!=0)&valid
            name=f'{room}_view{view}.npz'
            np.savez_compressed(out/name,valid=valid,depth=np.array(si.t).reshape(h,w),
                position=np.array(si.p).T.reshape(h,w,3),normal=np.array(si.sh_frame.n).T.reshape(h,w,3),
                camera_local=np.array(si.wi).T.reshape(h,w,3),uv=np.array(si.uv).T.reshape(h,w,2),shape_index=np.array(ids).reshape(h,w),
                native_response=response,bsdf_flags=flags,has_delta=delta,has_null=null)
            (out/f'{room}_view{view}_shapes.json').write_text(json.dumps(shapes,indent=2))
            row={'room':room,'view':view,'file':name,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest(),
                 'valid_fraction':float(valid.mean()),'delta_pixel_fraction':float(delta.mean()),'null_pixel_fraction':float(null.mean()),
                 'max_response':float(response.max()),'shape_count':len(shapes)}
            manifest['views'].append(row);print(json.dumps(row),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='experiments/out/EXP0017_pbr_solar/run_04')
    p.add_argument('--out',default='experiments/out/EXP0018_native_brdf/run_01');run(p.parse_args())
