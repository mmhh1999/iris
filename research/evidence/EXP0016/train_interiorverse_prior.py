"""Small supervised material-prior pilot with scene-disjoint official splits.

Not IRIS, not a solar method, and not a multi-view reconstruction experiment.
The heldout score is scene-macro; no test image chooses a checkpoint.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


class MaterialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
        self.a=block(3,16);self.b=block(16,32);self.c=block(32,64)
        self.d=block(96,32);self.e=block(48,16);self.head=nn.Conv2d(16,5,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
        d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.head(self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))).sigmoid()


def prepare(cfg,cache):
    if digest(cfg['archive'])!=cfg['archive_sha256']:raise ValueError('Archive changed')
    split_paths=[Path(cfg['splits_root'])/(k+'.txt') for k in ['train','val','test']]
    signature={'archive':cfg['archive_sha256'],'resolution':cfg['resolution'],
               'splits':{p.name:digest(p) for p in split_paths},'format':1}
    if cache.exists():
        x=torch.load(cache,weights_only=False,map_location='cpu')
        if x['signature']!=signature:raise ValueError('Cache signature mismatch')
        return x
    splits={p.stem:set(p.read_text().split()) for p in split_paths}
    assert not(splits['train']&splits['val'] or splits['train']&splits['test'] or splits['val']&splits['test'])
    inputs=[];targets=[];masks=[];records=[];excluded=0
    w,h=cfg['resolution']
    with zipfile.ZipFile(cfg['archive']) as z:
        names=sorted(n for n in z.namelist() if n.endswith('_im.exr'))
        for i,name in enumerate(names):
            arrays={}
            for key in ['im','albedo','material','mask']:
                raw=z.read(name.replace('_im.exr','_'+key+'.exr'))
                a=cv2.imdecode(np.frombuffer(raw,'uint8'),-1)
                if a is None:raise ValueError('Decode '+name)
                if a.ndim==3:a=a[...,::-1]
                arrays[key]=cv2.resize(a,(w,h),interpolation=cv2.INTER_NEAREST)
            im=arrays['im'];albedo=arrays['albedo'];mat=arrays['material'];mask=arrays['mask']
            if mask.ndim==3:mask=mask[...,0]
            y=np.concatenate([albedo,mat[...,:2]],-1)
            finite=np.isfinite(im).all(-1)&np.isfinite(y).all(-1)
            valid=(mask>.5)&finite&(y>=0).all(-1)&(y<=1.001).all(-1)
            excluded+=int(((mask>.5)&~valid).sum())
            if not valid.any():raise ValueError('No valid targets '+name)
            x=np.log1p(np.clip(np.nan_to_num(im,nan=0,posinf=64,neginf=0),0,64))/np.log(65.)
            y=np.clip(np.nan_to_num(y),0,1)
            scene=name.split('/')[0]
            split=next((k for k,s in splits.items() if scene in s),None)
            if split is None:raise ValueError('Scene absent from split '+scene)
            inputs.append(torch.from_numpy(x.transpose(2,0,1).copy()).half())
            targets.append(torch.from_numpy(y.transpose(2,0,1).copy()).half())
            masks.append(torch.from_numpy(valid[None].copy()))
            records.append({'scene':scene,'file':name,'split':split})
            if i%100==0:print('PREPARE',i,len(names),flush=True)
    data={'x':torch.stack(inputs),'y':torch.stack(targets),'mask':torch.stack(masks),
          'records':records,'signature':signature,'excluded_valid_label_pixels':excluded}
    cache.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,cache);return data


@torch.no_grad()
def evaluate(model,x,y,mask,ids,records,batch_size,constant=None):
    by_scene=defaultdict(list)
    for start in range(0,len(ids),batch_size):
        ix=ids[start:start+batch_size];a=x[ix].float();b=y[ix].float();m=mask[ix].float()
        p=model(a) if constant is None else constant[None,:,None,None].expand_as(b)
        err=(p-b).square()*m
        raw=(err.sum((2,3))/m.sum((2,3)).clamp_min(1))
        # Evaluation-only scalar alignment, never fed into training.
        scale=((p[:,:3]*b[:,:3]*m).sum((1,2,3))/(p[:,:3].square()*m).sum((1,2,3)).clamp_min(1e-9)).clamp_min(0)
        aligned=(((p[:,:3]*scale[:,None,None,None]-b[:,:3]).square()*m).sum((1,2,3))/(3*m.sum((1,2,3))))
        vals=torch.stack([raw[:,:3].mean(1),raw[:,3],raw[:,4],aligned],1).cpu().numpy()
        for i,v in zip(ix.tolist(),vals.tolist()):by_scene[records[i]['scene']].append(v)
    rooms={s:np.mean(v,axis=0).tolist() for s,v in by_scene.items()}
    mean=np.mean(list(rooms.values()),axis=0)
    return {'scene_count':len(rooms),'image_count':len(ids),'albedo_mse':float(mean[0]),
            'roughness_mse':float(mean[1]),'metallic_mse':float(mean[2]),
            'albedo_scale_aligned_mse':float(mean[3]),'selection_score':float(mean[:3].mean()),'per_scene':rooms}


def run(args):
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source.py');write(out/'config.json',cfg)
    torch.set_num_threads(4)
    data=prepare(cfg,Path(args.cache));records=data['records']
    write(out/'data_manifest.json',{'records':records,'signature':data['signature'],'excluded_valid_label_pixels':data['excluded_valid_label_pixels']})
    device=torch.device(args.device)
    x=data['x'].to(device);y=data['y'].to(device);mask=data['mask'].to(device)
    ids={k:torch.tensor([i for i,r in enumerate(records) if r['split']==k],device=device) for k in ['train','val','test']}
    assert all(len(v)>0 for v in ids.values())
    # Training-pixel mean is the optimum spatially constant predictor for MSE.
    const=(y[ids['train']].float()*mask[ids['train']]).sum((0,2,3))/mask[ids['train']].sum((0,2,3))
    baseline=evaluate(None,x,y,mask,ids['test'],records,cfg['batch_size'],constant=const)
    write(out/'constant_baseline.json',baseline)
    all_runs=[];start=time.monotonic()
    for seed in cfg['seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=MaterialPrior().to(device);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        best=float('inf');history=[];ckpt=out/f'seed_{seed}.pt'
        for epoch in range(cfg['epochs']):
            model.train();order=ids['train'][torch.randperm(len(ids['train']),device=device)]
            losses=[]
            for pos in range(0,len(order),cfg['batch_size']):
                if time.monotonic()-start>cfg['max_runtime_seconds']:raise TimeoutError('Training budget exceeded')
                ix=order[pos:pos+cfg['batch_size']];m=mask[ix].float();pred=model(x[ix].float())
                channel=((pred-y[ix].float()).square()*m).sum((0,2,3))/m.sum().clamp_min(1)
                loss=(channel[:3].mean()+channel[3]+channel[4])/3
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
            model.eval();val=evaluate(model,x,y,mask,ids['val'],records,cfg['batch_size'])
            item={'seed':seed,'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val':val}
            history.append(item);write(out/f'seed_{seed}_history.json',history)
            print('EPOCH',seed,epoch+1,item['train_loss'],val['selection_score'],flush=True)
            if val['selection_score']<best:
                best=val['selection_score'];torch.save({'model':model.state_dict(),'epoch':epoch+1,'seed':seed,'config':cfg},ckpt)
        saved=torch.load(ckpt,weights_only=False,map_location=device);model.load_state_dict(saved['model']);model.eval()
        test=evaluate(model,x,y,mask,ids['test'],records,cfg['batch_size'])
        result={'seed':seed,'selected_epoch':saved['epoch'],'test':test,'checkpoint_sha256':digest(ckpt)}
        write(out/f'seed_{seed}_test.json',result);all_runs.append(result)
    result={'experiment':'EXP0016','role':cfg['role'],'device':str(device),'torch':torch.__version__,
            'elapsed_seconds':time.monotonic()-start,'constant_train_mean_baseline':baseline,'runs':all_runs,
            'supports_solar_method_claim':False,'supports_sim_to_real_claim':False}
    write(out/'results.json',result)
    print('COMPLETE',json.dumps({k:np.mean([r['test'][k] for r in all_runs]) for k in ['albedo_mse','roughness_mse','metallic_mse']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/daylight/exp0016_material_prior.json')
    p.add_argument('--out',default='experiments/out/EXP0016_material_prior/run_01')
    p.add_argument('--cache',default='data_download/interiorverse/material_cache_128.pt')
    p.add_argument('--device',default='cuda');a=p.parse_args();run(a)
