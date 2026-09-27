"""Assess Monte Carlo convergence and sun intervention; export photo-only inputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np


def read(p):
    x=cv2.imread(str(p),-1)
    if x is None or not np.isfinite(x).all():raise ValueError(str(p))
    return x


def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))


def run(low,high):
    meta=json.loads((high/'manifest.json').read_text())
    low_meta=json.loads((low/'manifest.json').read_text())
    if len(meta['observations'])!=len(low_meta['observations']):raise ValueError('Unmatched runs')
    for a,b in zip(meta['observations'],low_meta['observations']):
        for field in ['room','view','condition','seed','c2w_mitsuba','sun_toward_world']:
            if a[field]!=b[field]:raise ValueError('Mismatched convergence control: '+field)
    metrics=[];panels=[]
    for room in ['bathroom','kitchen']:
        for view in range(2):
            images=[];uncertainties=[]
            for condition in range(3):
                stem=f'view{view}_sun{condition}'
                a=read(high/room/(stem+'_seed101.exr'));b=read(high/room/(stem+'_seed202.exr'))
                l1=read(low/room/(stem+'_seed101.exr'));l2=read(low/room/(stem+'_seed202.exr'))
                noise=rmse(a,b);old_noise=rmse(l1,l2)
                images.append((a+b)/2);uncertainties.append(noise)
                row={'room':room,'view':view,'sun':condition,'seed_pair_rmse_low':old_noise,
                     'seed_pair_rmse_high':noise,'noise_reduction_factor':old_noise/max(noise,1e-12),
                     'high_relative_noise':noise/max(float(np.mean((a+b)/2)),1e-12)}
                metrics.append(row)
            sky=read(high/room/f'view{view}_sky_only.exr')
            for condition,(im,noise) in enumerate(zip(images,uncertainties)):
                row=metrics[-3+condition]
                delta=im-sky
                row['sun_intervention_rmse']=rmse(im,sky)
                row['changed_light_rmse_from_sun0']=rmse(im,images[0])
                row['intervention_to_seed_noise_ratio']=row['sun_intervention_rmse']/max(noise,1e-12)
                # This is an intervention response, not a direct-sun segmentation GT.
                np.save(high/room/f'view{view}_sun{condition}_minus_sky.npy',delta)
            tiles=[]
            for title,im in [('Sky only',sky)]+[(f'Sun {i}',im) for i,im in enumerate(images)]:
                # Fixed +2 EV for all panels; untouched HDR used in metrics/export.
                display=np.uint8(np.clip(im*4,0,1)**(1/2.2)*255)
                cv2.putText(display,f'{room} v{view} {title}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                tiles.append(display)
            panels.append(np.hstack(tiles))
    cv2.imwrite(str(high/'solar_comparison.jpg'),np.vstack(panels))
    # Export only seed101 as observation; seed202 is kept for rendering diagnostics.
    # Folders separate train input from heldout images and oracle rendering files.
    public=high/'estimator_inputs';test=high/'heldout_evaluation';public.mkdir(exist_ok=True);test.mkdir(exist_ok=True)
    input_records=[];heldout_records=[]
    for row in meta['observations']:
        if row['seed']!=101:continue
        dst=test if row['condition']==2 else public
        name=f"{row['room']}_view{row['view']}_condition{row['condition']}.exr"
        shutil.copy2(high/row['path'],dst/name)
        rec={'file':name,'room':row['room'],'view_group':row['view'],'lighting_group':row['condition']}
        (heldout_records if row['condition']==2 else input_records).append(rec)
    for dest,records in [(public,input_records),(test,heldout_records)]:
        (dest/'images.json').write_text(json.dumps(records,indent=2))
    report={'experiment':'EXP0017','low_spp':low_meta['spp'],'high_spp':meta['spp'],
            'display':'fixed +2 EV, gamma 2.2, clipped only for visualization',
            'train_images':len(input_records),'heldout_images':len(heldout_records),
            'sun_truth_in_estimator_inputs':False,'brdf_truth_in_estimator_inputs':False,
            'inverse_estimation_run':False,'input_isolation':'separate whitelist export only; not process-level access isolation',
            'limitations':['artist-authored scenes','2 cameras only, 12 cm translation',
                          'directional point sun and uniform sky','no finite sun disc or real exterior',
                          'sampling convergence is measured but realism is not certified',
                          'source BSDFs are heterogeneous and cannot all use IRIS parameter-error scoring'],
            'metrics':metrics}
    (high/'quality_report.json').write_text(json.dumps(report,indent=2))
    (high/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>太阳重渲染实验</title>
    <style>body{font:18px system-ui;background:#17202a;color:#eee;max-width:1500px;margin:25px auto}img{width:100%}</style>
    <h1>两个完整材质室内场景的太阳重渲染</h1><p>每行相同材质、相同相机：仅天空 / 太阳条件 0 / 条件 1 / 条件 2。统一 +2 EV 显示；原始 HDR 保留。</p>
    <p>人工建模场景，用于受控实验；不是实拍或新方法恢复结果。太阳采用平行光近似，天空为均匀照明。</p>
    <img src="solar_comparison.jpg"><p>已测量独立随机种子差异，并导出不含太阳/材质真值的照片输入。尚未执行逆渲染对照。</p>''')
    print(json.dumps({'mean_noise_reduction_factor':float(np.mean([m['noise_reduction_factor'] for m in metrics])),
                      'min_sun_intervention_to_seed_noise_ratio':min(m['intervention_to_seed_noise_ratio'] for m in metrics),
                      'train_images':len(input_records),'heldout_images':len(heldout_records)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--low',type=Path,default=Path('experiments/out/EXP0017_pbr_solar/run_03'))
    p.add_argument('--high',type=Path,default=Path('experiments/out/EXP0017_pbr_solar/run_04'))
    a=p.parse_args();run(a.low,a.high)
