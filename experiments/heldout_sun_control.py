"""EXP0009: independent rendered changed-illumination control vs frozen appearance.

Known geometry/albedo, metadata-like *relative* solar change supplied at test.
This validates a component only, not IRIS or real-world material recovery.
"""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import experiments.phase_a_sun_recovery as phase
from experiments.real_sun_visibility import metrics
from utils.multiview_geometry import sunlight_visibility
from utils.solar_geometry import sun_vector_world,angular_error_deg

APERTURE={'endpoints_xy':[[-.8,0],[.8,0]],'z_range':[.8,2.]}


def render_observation(az,el,seed,out):
    phase.OUT_DIR=str(out);phase.TRUE_AZ_DEG=az;phase.TRUE_EL_DEG=el
    scene,_=phase.build_scene(spp=32,res=(192,144))
    image=np.array(mi.render(scene,spp=32,seed=seed))
    sensor=scene.sensors()[0];yy,xx=np.mgrid[:144,:192]
    ray,_=sensor.sample_ray_differential(time=0.,sample1=0.,sample2=mi.Point2f(mi.Float((xx.ravel()+.5)/192),mi.Float((yy.ravel()+.5)/144)),sample3=mi.Point2f(.5,.5))
    si=scene.ray_intersect(ray);p=np.column_stack([np.asarray(si.p[i]) for i in range(3)])
    n=np.column_stack([np.asarray(si.n[i]) for i in range(3)])
    floor=np.asarray(si.is_valid())&(np.abs(p[:,2])<.001)&(n[:,2]>.9)
    lum=image.sum(axis=-1).ravel()
    # Fixed radiometric threshold inherited from Phase A (not fit to test images).
    labels=lum[floor]>1.2
    return scene,image,p[floor],labels,floor


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg=dict(experiment_id='EXP0009',train_angles=[[163.7,31.4],[203.2,43.6],[218.4,51.7]],seeds=[0,1],test_delta_az=20,test_delta_el=10,
        spp=32,resolution=[192,144],threshold_rgb_sum=1.2,known_geometry=True,known_albedo=True,
        coarse_grid_az_step=10,coarse_grid_el_step=5,refine_step=1,
        disclosure='Test receives true RELATIVE solar azimuth/elevation change, applied to recovered training direction. Appearance baseline freezes train labels. No fitted material, transmission or IRIS baseline.')
    files=['experiments/heldout_sun_control.py','experiments/phase_a_sun_recovery.py','utils/multiview_geometry.py','utils/solar_geometry.py','experiments/real_sun_visibility.py']
    manifest=dict(config=cfg,source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),mitsuba=mi.__version__,numpy=np.__version__)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.perf_counter();rows=[]
    coarse=[(a,e) for a in range(0,360,10) for e in range(5,90,5)]
    for az,el in cfg['train_angles']:
        for seed in cfg['seeds']:
            case=out/f'az{az}_el{el}_seed{seed}';case.mkdir()
            scene,image,points,labels,floor=render_observation(az,el,seed,case)
            dirs=np.array([sun_vector_world(a,e) for a,e in coarse])
            predictions=sunlight_visibility(points,dirs,scene,[APERTURE],.005)
            scores=np.array([metrics(p,labels)['iou'] for p in predictions]);a,e=coarse[int(scores.argmax())]
            fine=[(float(aa%360),float(ee)) for aa in np.arange(a-10,a+10.1) for ee in np.arange(max(1,e-5),min(89,e+5)+.1)]
            pred=sunlight_visibility(points,[sun_vector_world(*v) for v in fine],scene,[APERTURE],.005)
            score=np.array([metrics(p,labels)['iou'] for p in pred]);index=int(score.argmax());fitaz,fitel=fine[index]
            testscene,testimage,testpoints,testlabels,testfloor=render_observation(az+20,el+10,seed+100,case)
            if not np.array_equal(floor,testfloor):raise ValueError('camera/geometry changed across illumination')
            physical=sunlight_visibility(testpoints,[sun_vector_world(fitaz+20,fitel+10)],testscene,[APERTURE],.005)[0]
            unchanged=sunlight_visibility(testpoints,[sun_vector_world(fitaz,fitel)],testscene,[APERTURE],.005)[0]
            oracle=sunlight_visibility(testpoints,[sun_vector_world(az+20,el+10)],testscene,[APERTURE],.005)[0]
            row=dict(train_az=az,train_el=el,seed=seed,recovered_az=fitaz,recovered_el=fitel,
                angular_error_deg=angular_error_deg(sun_vector_world(az,el),sun_vector_world(fitaz,fitel)),
                train_iou=float(score[index]),heldout_physical=metrics(physical,testlabels),heldout_frozen_appearance=metrics(labels,testlabels),
                heldout_unchanged_sun=metrics(unchanged,testlabels),heldout_oracle_geometry=metrics(oracle,testlabels),n_train_bright=int(labels.sum()),n_test_bright=int(testlabels.sum()))
            rows.append(row)
            for name,img in [('train',image),('heldout',testimage)]:cv2.imwrite(str(case/(name+'.png')),(np.clip(img,0,1)**(1/2.2)*255).astype(np.uint8)[...,::-1])
            panels=[]
            for label in [testlabels,physical,labels]:
                mask=np.zeros((144*192,3),np.uint8);mask[testfloor]=np.where(label[:,None],255,40);panels.append(mask.reshape(144,192,3))
            cv2.imwrite(str(case/'heldout_proxy_physical_memory.png'),np.concatenate(panels,axis=1))
            print(json.dumps(row),flush=True)
            (out/'partial_results.json').write_text(json.dumps(rows,indent=2))
    summary=dict(mean_angular_error_deg=float(np.mean([r['angular_error_deg'] for r in rows])),
        mean_heldout_physical_iou=float(np.mean([r['heldout_physical']['iou'] for r in rows])),
        mean_heldout_frozen_iou=float(np.mean([r['heldout_frozen_appearance']['iou'] for r in rows])),
        n_cases=len(rows),elapsed_seconds=time.perf_counter()-start)
    (out/'results.json').write_text(json.dumps(dict(summary=summary,cases=rows),indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'experiments/out/EXP0009'));run(p.parse_args())
