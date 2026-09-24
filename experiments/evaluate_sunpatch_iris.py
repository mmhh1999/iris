"""Floor-only controlled metrics; never treat pixels as independent rooms."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import argparse,json,sys
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.brdf import NGPBRDF

def load_model(root,stage):
    ck=root/'checkpoints'/('sunpatch_'+root.name)/(stage+'.ckpt')
    bounds=torch.load(root/'bake/vslf.npz',map_location='cpu',weights_only=False)
    m=NGPBRDF(bounds['voxel_min'],bounds['voxel_max']).cuda()
    checkpoint=torch.load(ck,map_location='cpu',weights_only=False)
    if checkpoint['global_step'] < 200:raise ValueError(f'Stale checkpoint: {ck}, step {checkpoint["global_step"]}')
    state=checkpoint['state_dict'];m.checkpoint_step=checkpoint['global_step']
    m.load_state_dict({k.removeprefix('material.'):v for k,v in state.items() if k.startswith('material.')});m.eval()
    return m,ck

def run(a):
    base=Path(a.base);data=base/'data_03';out=base/a.out;out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'manifest.json').read_text());report={'experiment':'EXP0019','stage':a.stage,'scope':'P1 controlled IRIS with neutral prior, 200-step budget per learned stage; not original published setting',
        'metrics':'floor-only. Sun/shadow masks eroded 1 pixel. Uniform true floor; raw mean-channel albedo gap and gap/mean floor albedo.','views':[]}
    pred={}
    for condition in a.conditions:
        m,ck=load_model(base/('iris_'+condition+a.run_suffix),a.stage);pred[condition]={}
        for view in [3,9]:
            g=np.load(data/'generator_truth'/f'{condition}_view{view:02d}.npz');mask=g['floor'];p=g['position'][mask]
            with torch.no_grad():mat={k:v.cpu().numpy() for k,v in m(torch.tensor(p,device='cuda')).items()}
            albedo=np.zeros((*mask.shape,3),np.float32);albedo[mask]=mat['albedo'];pred[condition][view]=albedo
            np.savez_compressed(out/f'{condition}_view{view}.npz',albedo=albedo,floor=mask,roughness=mat['roughness'],metallic=mat['metallic'])
            for target in ['sun_a','sun_b']:
                t=np.load(data/'generator_truth'/f'{target}_view{view:02d}.npz')
                lit=cv2.erode(t['sun_visible'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&mask
                dark=cv2.erode((mask&~t['sun_visible']).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                gray=albedo.mean(-1);gap=float(gray[lit].mean()-gray[dark].mean()) if lit.sum()>=32 and dark.sum()>=32 else None
                row={'fit_condition':condition,'mask_condition':target,'view':view,'checkpoint':str(ck),'checkpoint_step':m.checkpoint_step,
                     'lit_count':int(lit.sum()),'region_eligible':bool(lit.sum()>=32 and dark.sum()>=32),'shadow_count':int(dark.sum()),'albedo_gap':gap,
                     'normalized_gap':None if gap is None else gap/float(gray[mask].mean()),
                     'floor_fraction_mean_albedo_above_095':float((albedo[mask].mean(-1)>.95).mean()),
                     'floor_albedo_mae':float(np.abs(albedo[mask]-np.array([.28,.20,.12])).mean()),
                     'floor_mean_roughness':float(mat['roughness'].mean()),'floor_mean_metallic':float(mat['metallic'].mean())}
                report['views'].append(row)
    if 'sky' in pred:
        report['difference_in_differences']=[]
        for condition in a.conditions:
            if condition=='sky':continue
            for view in [3,9]:
                rows=[r for r in report['views'] if r['mask_condition']==condition and r['view']==view]
                s=next(r for r in rows if r['fit_condition']==condition);c=next(r for r in rows if r['fit_condition']=='sky')
                report['difference_in_differences'].append({'condition':condition,'view':view,
                    'raw_albedo_gap_change':None if s['albedo_gap'] is None else s['albedo_gap']-c['albedo_gap'],
                    'normalized_gap_change':None if s['normalized_gap'] is None else s['normalized_gap']-c['normalized_gap']})
    (out/'report.json').write_text(json.dumps(report,indent=2))
    w,h=128,96;canvas=Image.new('RGB',(w*3,(h+23)*len(a.conditions)*2));draw=ImageDraw.Draw(canvas)
    for ri,(condition,view) in enumerate((c,v) for c in a.conditions for v in [3,9]):
        obs=next(v for c in manifest['conditions'] if c['name']==condition for v in c['views'] if v['view']==view)
        img=Image.open(data/condition/'val/Image'/f"{obs['index']:03d}_0001.png")
        gt=np.load(data/'generator_truth'/f'{condition}_view{view:02d}.npz');truth=np.zeros((h,w,3),np.float32);truth[gt['floor']]=[.28,.20,.12]
        for col,(im,label) in enumerate([(img,f'{condition} v{view} input'),(Image.fromarray(np.uint8(np.clip(pred[condition][view],0,1)**(1/2.2)*255)),'IRIS floor albedo'),(Image.fromarray(np.uint8(truth**(1/2.2)*255)),'True floor albedo')]):
            canvas.paste(im,(col*w,ri*(h+23)+23));draw.text((col*w+2,ri*(h+23)+4),label,fill='white')
    canvas.resize((w*6,(h+23)*len(a.conditions)*4)).save(out/'comparison.png')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='experiments/out/EXP0019_sunpatch');p.add_argument('--run-suffix',default='_v2');p.add_argument('--out',default='eval_02');p.add_argument('--stage',default='last_1');p.add_argument('--conditions',nargs='+',default=['sun_a','sun_b','sky']);run(p.parse_args())
