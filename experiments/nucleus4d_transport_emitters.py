"""Independent candidate correcting scanned lamp housings shadowing themselves.

Only the current source's photographed housing volume is skipped for its own
direct/emitted rays. Other surfaces, other lamps, and all sky/solar rays retain
the complete opaque scan. This is an explicit source/scan alignment estimate.
"""
import argparse
import json
import time

import numpy as np
from scipy import sparse

from nucleus4d_baked_relight import atomic_json
from nucleus4d_transport_geometry import OUT as BASE,make_scene
import nucleus4d_transport_bake as bake

CANDIDATE=BASE/'emitter_bases'


class SourceScene:
    def __init__(self,scene,light):self.scene=scene;self.light=light

    def ray_intersect(self,ray,active=True):
        import mitsuba as mi
        import drjit as dr
        origin=mi.Ray3f(ray);current=mi.Ray3f(ray);offset=mi.Float(0)
        si=self.scene.ray_intersect(current,active)
        c=self.light['center'];radius=self.light['radius']+(.025 if self.light['type']=='cylinder' else .04)
        halfheight=self.light.get('height',0)/2+(.025 if self.light['type']=='cylinder' else .10)
        for _ in range(12):
            own=si.is_valid()&((si.p.x-c[0])**2+(si.p.y-c[1])**2<radius**2)&(dr.abs(si.p.z-c[2])<halfheight)
            if not bool(dr.any(own)):return si
            # Advance only the rays hitting this photographed source's housing.
            # Global t and finite maxt are preserved for all downstream checks.
            offset=dr.select(own,si.t+.0005,offset)
            current.o=origin.o+origin.d*offset;current.maxt=dr.maximum(origin.maxt-offset,0)
            next_si=self.scene.ray_intersect(current,own&(current.maxt>0));next_si.t+=offset
            si=dr.select(own,next_si,si)
        raise RuntimeError('Housing visibility exceeded its bounded traversal budget')

    def ray_test(self,ray,active=True):return self.ray_intersect(ray,active).is_valid()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('cuda_ad_rgb');dr.set_thread_count(6)
    with np.load(BASE/'geometry.npz') as z:data={k:z[k] for k in z.files}
    sc=make_scene(data);lights=json.loads((BASE/'lights.json').read_text())['lights']
    ids=np.load(BASE/'compact_node_ids.npy');p=np.repeat(data['fine_points'],2,axis=0)[ids];n=np.repeat(data['fine_normals'],2,axis=0);n[1::2]*=-1;n=n[ids]
    CANDIDATE.mkdir(exist_ok=True)
    if args.probe:
        select=(p[:,0]>5.65)&(p[:,0]<6.3)&(n[:,0]<-.85)&(p[:,2]>.65)&(p[:,2]<1.1)&(abs(p[:,1])<2.4)
        records=[]
        for i in [1,3,8,9]:
            before=bake.direct_light(sc,p[select],n[select],lights[i]);after=bake.direct_light(SourceScene(sc,lights[i]),p[select],n[select],lights[i])
            record=dict(light=i,receivers=int(select.sum()),raw_nonzero=int((before.mean(1)>1e-6).sum()),housing_aware_nonzero=int((after.mean(1)>1e-6).sum()),raw_mean=float(before.mean()),housing_aware_mean=float(after.mean()))
            records.append(record);print('EMITTER_ALIGNMENT_PROBE',json.dumps(record),flush=True)
        atomic_json(CANDIDATE/'alignment_probe.json',dict(records=records,scope='supplied geometry and photographed lamp housing alignment; not measured illumination'))
        return
    bake.OUT=CANDIDATE
    for name in ['coarse_areas.npy','fine_sky_total.npy']:
        link=CANDIDATE/name
        if not link.exists():link.symlink_to('../'+name)
    gi=np.load(BASE/'fine_gi_binding.npz');matrix=sparse.load_npz(BASE/'form_factors.npz');reference=np.load(BASE/'fine_sky_total.npy')[:,1].copy()
    for light in lights:
        i=light['id'];coarse_path=CANDIDATE/f'coarse_lamp_{i:02}.npy';direct_path=CANDIDATE/f'coarse_lamp_{i:02}_direct.npy';fine_path=CANDIDATE/f'fine_lamp_{i:02}.npy';source=SourceScene(sc,light)
        bake.progress('source_housing_candidate',light=i,total=len(lights),current_source=i,phase='coarse' if not coarse_path.exists() else 'fine')
        if not coarse_path.exists():
            direct=bake.emitted_direct(data,source,light);total=bake.radiosity(matrix,data['coarse_reflectance'],direct)
            np.save(direct_path,direct);np.save(coarse_path,total)
        if not fine_path.exists():
            direct=bake.direct_light(source,p,n,light,triangles=data['vertices'][data['faces']])
            indirect=bake.lift_gi(np.load(coarse_path)-np.load(direct_path),gi['indices'],gi['weights']);np.save(fine_path,direct+indirect)
        reference+=np.load(fine_path);bake.progress('source_housing_candidate',light=i+1,total=len(lights))
    reference.tofile(CANDIDATE/'reference_irradiance.f32')
    atomic_json(CANDIDATE/'candidate_report.json',dict(completed_at=time.time(),candidate_only=True,source_geometry_and_textures_unchanged=True,
        own_source_housing_only=True,other_surfaces_and_sources_opaque=True,sky_and_sun_unchanged=True,
        disk_alignment_volume_xy_padding_m=.04,disk_alignment_volume_z_halfheight_m=.10,
        source_scan_alignment_estimate=True,all_original_sampling_and_bounce_settings_preserved=True,manual_visual_review_pending=True))


if __name__=='__main__':main()
