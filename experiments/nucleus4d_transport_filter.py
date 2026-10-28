"""Candidate denoising of diffuse fields, preserving textures and direct lamp shadows.

Neighbors must share a local surface plane/normal and have an unobstructed
connection. This is a separate review candidate, never an automatic release.
"""
import json
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from nucleus4d_baked_relight import atomic_json
from nucleus4d_transport_geometry import OUT,make_scene,EPS
from nucleus4d_transport_bake import lift_gi

CANDIDATE=OUT/'filtered_bases'


def main():
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('cuda_ad_rgb');dr.set_thread_count(6)
    CANDIDATE.mkdir(exist_ok=True)
    with np.load(OUT/'geometry.npz') as z:data={k:z[k] for k in z.files}
    ids=np.load(OUT/'compact_node_ids.npy');p=np.repeat(data['fine_points'],2,axis=0)[ids];n=np.repeat(data['fine_normals'],2,axis=0);n[1::2]*=-1;n=n[ids];sc=make_scene(data)
    path=CANDIDATE/'smoothing.npz'
    if path.exists():operator=sparse.load_npz(path)
    else:
        tree=cKDTree(p);blocks=[]
        for start in range(0,len(p),4096):
            q=p[start:start+4096];normal=n[start:start+len(q)];distance,neighbors=tree.query(q,k=64,distance_upper_bound=.36,workers=6)
            neighbors[~np.isfinite(distance)]=start+np.nonzero(~np.isfinite(distance))[0]
            delta=p[neighbors]-q[:,None];length=np.linalg.norm(delta,axis=2);other_n=n[neighbors]
            compatible=((normal[:,None]*other_n).sum(2)>.95)&(abs((normal[:,None]*delta).sum(2))<.025)&(abs((other_n*delta).sum(2))<.025)&np.isfinite(distance)
            origin=q[:,None]+normal[:,None]*.003;target=p[neighbors]+other_n*.003;direction=target-origin;r=np.linalg.norm(direction,axis=2)
            test=compatible&(r>.004);rows,cols=np.nonzero(test)
            if len(rows):
                ray=mi.Ray3f(mi.Point3f(np.broadcast_to(origin,direction.shape)[rows,cols].T),mi.Vector3f((direction[rows,cols]/r[rows,cols,None]).T));ray.maxt=mi.Float(np.maximum(r[rows,cols]-.003,1e-5))
                visible=~sc.ray_test(ray);dr.eval(visible);compatible[rows,cols]&=np.asarray(visible)
            weight=np.exp(-length**2/(2*.12**2))*compatible;weight/=np.maximum(weight.sum(1,keepdims=True),1e-20)
            row=np.repeat(np.arange(len(q)),64);blocks.append(sparse.csr_matrix((weight.ravel().astype(np.float32),(row,neighbors.ravel())),shape=(len(q),len(p))))
        operator=sparse.vstack(blocks,format='csr');operator.eliminate_zeros();sparse.save_npz(path,operator)
    if abs(np.asarray(operator.sum(1)).ravel()-1).max()>1e-5 or operator.data.min()<0:raise RuntimeError('Invalid diffuse filter weights')
    gi=np.load(OUT/'fine_gi_binding.npz');indices,weights=gi['indices'],gi['weights']
    sky=np.load(OUT/'fine_sky_total.npy');sky_direct=np.load(OUT/'fine_sky_direct.npy')
    # Only the broad sky and indirect terms are denoised. Finite-emitter direct
    # illumination and its geometric shadows remain unchanged.
    sky_out=np.empty_like(sky)
    for k in range(2):sky_out[:,k]=operator@sky_direct[:,k]+operator@(sky[:,k]-sky_direct[:,k])
    np.save(CANDIDATE/'fine_sky_total.npy',sky_out);reference=sky_out[:,1].copy()
    for i in range(25):
        coarse=np.load(OUT/f'coarse_lamp_{i:02}.npy')-np.load(OUT/f'coarse_lamp_{i:02}_direct.npy');indirect=lift_gi(coarse,indices,weights)
        original=np.load(OUT/f'fine_lamp_{i:02}.npy');direct=np.maximum(original-indirect,0)
        total=direct+operator@indirect
        if not np.isfinite(total).all() or total.min()<0:raise RuntimeError('Invalid filtered lighting')
        np.save(CANDIDATE/f'fine_lamp_{i:02}.npy',total);reference+=total
    reference.tofile(CANDIDATE/'reference_irradiance.f32')
    report=dict(completed=time.time(),candidate_only=True,geometry_and_source_textures_unchanged=True,neighbors=64,maximum_radius_m=.36,sigma_m=.12,normal_cosine_min=.95,plane_distance_max_m=.025,neighbor_connection_visibility_tested=True,direct_lamp_shadows_unchanged=True,sky_and_indirect_only=True,manual_visual_review_pending=True)
    atomic_json(CANDIDATE/'filter_report.json',report);print('FILTER_CANDIDATE',json.dumps(report),flush=True)


if __name__=='__main__':main()
