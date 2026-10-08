"""Check finite lighting-ray segments and preserved primary Gaussian visibility."""
import json
import numpy as np
from nucleus4d_ptir import OUT, load_integrators, mi, dr

mi.set_variant('cuda_ad_rgb');load_integrators()
integrator=mi.load_dict({'type':'nucleus_ptir','window_portals':True})
records=[]
for name,x,y,expected in [('before_window',4.,1.,True),('beyond_window',7.,1.,False),('outside_aperture',7.,0.,True)]:
    scene=mi.load_dict({'type':'scene','blocker':{'type':'ellipsoids','centers':mi.TensorXf([[x,y,0.]]),
        'scales':mi.TensorXf([[.15,.15,.15]]),'quaternions':mi.TensorXf([[0.,0.,0.,1.]]),
        'opacities':mi.TensorXf([[1.]])}})
    ray=mi.Ray3f(mi.Point3f([0.,y,0.]),mi.Vector3f([1.,0.,0.]))
    sampler=mi.load_dict({'type':'independent'});sampler.seed(72,64)
    occluded=np.array(integrator.shadow_ray_test(scene,sampler,None,ray,mi.Bool(True)))
    primary=bool(np.array(scene.ray_intersect(ray).is_valid()).all())
    fraction=float(occluded.mean())
    assert primary,'Primary ray was incorrectly clipped'
    assert (fraction>.98) if expected else (fraction<.02),(name,fraction)
    records.append({'case':name,'lighting_occlusion_fraction':fraction,'primary_hit':primary})
report={'window_ray_tests':records,'scope':'adapter visibility tests, not real-scene radiometric ground truth'}
(OUT/'adapter_validation.json').write_text(json.dumps(report,indent=2));print(report)
