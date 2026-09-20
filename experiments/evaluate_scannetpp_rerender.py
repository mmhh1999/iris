"""Batch the frozen EXP0012 direction search; reuse existing radiance renders."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import mitsuba as mi
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.scannetpp_rerender import score,sha
from utils.multiview_geometry import intersect_rays,aperture_distances
from utils.solar_geometry import sun_vector_world,angular_error_deg

def visibility_batch(scene,points,directions,offset):
    dirs=np.asarray(directions);pred=[]
    for begin in range(0,len(dirs),32):
        rays=np.broadcast_to(dirs[begin:begin+32,None,:],(min(32,len(dirs)-begin),len(points),3)).copy()
        origin=points[None]+offset*rays
        _,hit,_=intersect_rays(scene,origin.reshape(-1,3),rays.reshape(-1,3))
        pred.extend((~hit).reshape(-1,len(points)))
    return np.array(pred)

def run(folder, scene=None):
    out=Path(folder);manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['config']
    mi.set_variant('cuda_ad_rgb');start=time.perf_counter()
    mesh=ROOT/'data_download/scannetpp/data'/cfg['scene']/'scans/mesh_aligned_0.05.ply'
    assert sha(mesh)==manifest['mesh_sha256']
    if scene is None:
        scene=mi.load_dict({'type':'scene','mesh':{'type':'ply','filename':str(mesh)}})
    maps=[np.load(out/f'view_{i}_geometry.npz') for i in range(len(cfg['views']))]
    observations=json.loads((out/'observations.json').read_text())
    g=maps[0];ids=np.flatnonzero(g['floor'].ravel())
    ids=np.random.default_rng(cfg['fit_seed']).choice(ids,min(len(ids),cfg['fit_max_points']),replace=False)
    p=g['position'].reshape(-1,3)[ids]
    labels=(cv2.imread(str(out/'train'/f"view_0_seed_{cfg['seeds'][0]}"/'direct_sun_proxy.png'),0).ravel()>0)[ids]
    def objective(angles):
        pred=visibility_batch(scene,p,[sun_vector_world(a,e) for a,e in angles],cfg['shadow_ray_offset_m'])
        return [score(row,labels)['iou'] for row in pred]
    coarse=[(a,e) for a in range(0,360,cfg['fit_az_step']) for e in range(5,86,cfg['fit_el_step'])]
    values=objective(coarse);a,e=coarse[int(np.argmax(values))]
    print('COARSE',a,e,max(values),flush=True)
    fine=[(float(az%360),float(el)) for az in np.arange(a-10,a+10.1,1) for el in np.arange(max(1,e-5),min(89,e+5)+.1,1)]
    values=objective(fine);fit=fine[int(np.argmax(values))];train=cfg['conditions'][0]
    fit_result=dict(azimuth=fit[0],elevation=fit[1],train_iou=float(max(values)),
        angular_error_deg=angular_error_deg(sun_vector_world(*fit),sun_vector_world(train['azimuth'],train['elevation'])))
    (out/'train_fit.json').write_text(json.dumps(fit_result,indent=2));print('FIT',fit_result,flush=True)
    aps=json.loads((ROOT/'configs/daylight/exp0008_real_visibility.json').read_text())['apertures']
    rows=[];cache={};w,h=cfg['resolution']
    for obs in observations:
        c=next(c for c in cfg['conditions'] if c['name']==obs['condition'])
        key=(obs['condition'],obs['view']);g=maps[obs['view']];floor=g['floor'].ravel();p=g['position'].reshape(-1,3)[floor]
        if key not in cache:
            true=sun_vector_world(c['azimuth'],c['elevation'])
            est=sun_vector_world(fit[0]+c['azimuth']-train['azimuth'],fit[1]+c['elevation']-train['elevation'])
            oracle,pred,wrong=visibility_batch(scene,p,[true,est,sun_vector_world(c['azimuth']+10,c['elevation'])],cfg['shadow_ray_offset_m'])
            through=np.zeros(len(p),bool)
            for ap in aps:through |= np.isfinite(aperture_distances(p,true,ap))
            noisy={}
            for sigma in [.01,.05]:
                pp=p+np.random.default_rng(19).normal(0,sigma,p.shape)
                noisy[str(sigma)]=visibility_batch(scene,pp,[est],cfg['shadow_ray_offset_m'])[0]
            cache[key]=(oracle,pred,wrong,through,noisy)
        oracle,pred,wrong,through,noisy=cache[key]
        truth=(cv2.imread(str(out/obs['path']/'direct_sun_proxy.png'),0).ravel()>0)[floor]
        row=dict(**obs,estimated_sun=score(pred,truth),oracle_sun=score(oracle,truth),wrong_sun_plus10=score(wrong,truth),
            unoccluded_fraction=float(oracle.mean()),fraction_unoccluded_rays_crossing_approximate_windows=float((through&oracle).sum()/max(1,oracle.sum())),
            receiver_noise={s:score(pr,truth) for s,pr in noisy.items()})
        mask=np.zeros(w*h,np.uint8);mask[floor]=pred.astype(np.uint8)*255
        cv2.imwrite(str(out/obs['path']/'predicted_sun_proxy.png'),mask.reshape(h,w));rows.append(row)
    tests=[r for r in rows if r['condition'].startswith('test')]
    summary=dict(fit=fit_result,test_mean_iou=float(np.mean([r['estimated_sun']['iou'] for r in tests])),
        test_oracle_mean_iou=float(np.mean([r['oracle_sun']['iou'] for r in tests])),
        test_wrong_sun_mean_iou=float(np.mean([r['wrong_sun_plus10']['iou'] for r in tests])),
        test_mean_window_crossing_fraction=float(np.mean([r['fraction_unoccluded_rays_crossing_approximate_windows'] for r in tests])),
        test_receiver_noise_iou={s:float(np.mean([r['receiver_noise'][s]['iou'] for r in tests])) for s in ['0.01','0.05']},
        n_renders=len(observations),evaluation_seconds=time.perf_counter()-start,disclosure=cfg['limits'],independent_rooms=1)
    (out/'results.json').write_text(json.dumps(dict(summary=summary,observations=rows,evaluation_source_sha256=sha(__file__)),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');run(p.parse_args().folder)
