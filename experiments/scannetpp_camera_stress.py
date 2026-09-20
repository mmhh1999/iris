"""Frozen-prediction stress test with camera yaw error and surface reintersection.

Unlike arbitrary 3-D point jitter, ray origins are recomputed on actual mesh
surfaces. This avoids interpreting below-surface self-occlusion as pose error.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import mitsuba as mi
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.scannetpp_rerender import sha,score
from experiments.evaluate_scannetpp_rerender import visibility_batch
from utils.geometry_screening import pinhole_rays
from utils.multiview_geometry import intersect_rays
from utils.solar_geometry import sun_vector_world

def run(folder):
    out=Path(folder);m=json.loads((out/'manifest.json').read_text());cfg=m['config']
    fit=json.loads((out/'train_fit.json').read_text());train=cfg['conditions'][0]
    # Fixed symmetric perturbations; no fitting or model selection here.
    angles=[-3.,-1.,-.25,0.,.25,1.,3.]
    mi.set_variant('cuda_ad_rgb')
    mesh=ROOT/'data_download/scannetpp/data'/cfg['scene']/'scans/mesh_aligned_0.05.ply'
    scene=mi.load_dict({'type':'scene','mesh':{'type':'ply','filename':str(mesh)}})
    w,h=cfg['resolution'];yy,xx=np.mgrid[:h,:w];rows=[]
    for view,cam in enumerate(m['cameras']):
        g=np.load(out/f'view_{view}_geometry.npz');floor=g['floor'].ravel()
        o,d=pinhole_rays(np.c_[xx.ravel(),yy.ravel()][floor],cam['K'],cam['c2w'])
        for deg in angles:
            t=np.radians(deg);R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
            positions,valid,_=intersect_rays(scene,o,d@R.T)
            for condition in cfg['conditions'][2:]:
                sun=sun_vector_world(fit['azimuth']+condition['azimuth']-train['azimuth'],fit['elevation']+condition['elevation']-train['elevation'])
                pred=np.zeros(len(o),bool)
                pred[valid]=visibility_batch(scene,positions[valid],[sun],cfg['shadow_ray_offset_m'])[0]
                for seed in cfg['seeds']:
                    path=out/condition['name']/f'view_{view}_seed_{seed}'/'direct_sun_proxy.png'
                    truth=(cv2.imread(str(path),0).ravel()>0)[floor]
                    rows.append(dict(view=view,condition=condition['name'],seed=seed,yaw_error_deg=deg,**score(pred,truth)))
    means={str(deg):float(np.mean([r['iou'] for r in rows if r['yaw_error_deg']==deg])) for deg in angles}
    result=dict(mean_iou_by_yaw_error_deg=means,rows=rows,source_sha256=sha(__file__),
        interpretation='Pose perturbation only, with primary-ray surface reintersection; no retraining and no physical mesh perturbation. One room, two test illuminations, repeated camera views and render seeds.')
    (out/'camera_stress.json').write_text(json.dumps(result,indent=2));print(json.dumps(means,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');run(p.parse_args().folder)
