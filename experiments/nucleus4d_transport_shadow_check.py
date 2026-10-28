"""Compare actual browser shadow pixels to independent full-mesh ray queries."""
import json
import numpy as np
from scipy.spatial import cKDTree

from nucleus4d_baked_relight import atomic_json
from nucleus4d_transport_geometry import OUT,BOUNDS,make_scene


def main():
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('llvm_ad_rgb');dr.set_thread_count(6)
    from nucleus4d_transport_bake import sun_state
    from nucleus4d_weather_presets import solar_directions
    with np.load(OUT/'geometry.npz') as z:data={k:z[k] for k in z.files}
    scene=make_scene(data);tree=cKDTree(data['fine_points']);records=[]
    for key,minute in [('sun_1230',750),('sun_1600',960)]:
        world=np.load(OUT/f'browser_{key}_world.npy');alpha=world[:,:,3];p=world[:,:,:3]/np.maximum(alpha[:,:,None],1e-8)
        visible=np.load(OUT/f'browser_{key}_visibility.npy')[:,:,0].astype(float)/255
        sun_mask=np.load(OUT/f'browser_{key}_sun_mask.npy')
        valid=(alpha>.98)&(sun_mask>1e-5)&((p>=BOUNDS[0])&(p<=BOUNDS[1])).all(2)
        pixels=np.flatnonzero(valid.ravel());points=p.reshape(-1,3)[pixels];distance,_=tree.query(points,workers=6)
        pixels=pixels[distance<.025];points=p.reshape(-1,3)[pixels]
        if len(points)>1024:
            take=np.linspace(0,len(points)-1,1024).astype(int);pixels=pixels[take];points=points[take]
        direction,_=sun_state(minute);rays=solar_directions(direction,1024)
        predicted=visible.ravel()[pixels];actual=[]
        for start in range(0,len(points),128):
            part=points[start:start+128];origin=np.repeat(part,1024,axis=0);d=np.tile(rays,(len(part),1))
            vis=~scene.ray_test(mi.Ray3f(mi.Point3f(origin.T),mi.Vector3f(d.T)));dr.eval(vis);actual.extend(np.asarray(vis).reshape(len(part),1024).mean(1).tolist())
        actual=np.array(actual);errors=abs(predicted-actual)
        record=dict(minute=minute,pixels=len(points),mean_absolute_error=float(errors.mean()) if len(errors) else None,
                    fully_lit_ray_pixels=int((actual>.99).sum()),fully_blocked_ray_pixels=int((actual<.01).sum()),
                    boundary_ray_pixels=int(((actual>=.01)&(actual<=.99)).sum()),
                    agreement_within_quarter=float((errors<.25).mean()) if len(errors) else 0,
                    method='4096-map PCSS versus 1024 finite-sun mesh visibility rays, nearest-surface opaque pixels')
        records.append(record)
    passed=all(r['pixels']>=64 and r['mean_absolute_error']<.12 and r['agreement_within_quarter']>.85 for r in records)
    atomic_json(OUT/'shadow_validation.json',dict(passed=passed,passes_internal_proxy_check=passed,records=records,
        physical_scene_accuracy_verified=False,absolute_photometric_units_verified=False,
        boundary_accuracy_certified=False,scope='internal same-proxy consistency only; selected near-surface opaque pixels; no real-room ground truth or complete boundary validation'))
    print('SHADOW_VALIDATION',json.dumps(dict(passed=passed,records=records)),flush=True)
    if not passed:raise RuntimeError('Moving sunlight shadow requires correction before release')


if __name__=='__main__':main()
