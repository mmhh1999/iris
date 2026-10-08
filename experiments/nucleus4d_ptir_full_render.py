"""Tiled float32 exports from the trained official PTIR renderer, without denoising."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.light import create_environment
from threedgrut.render import Renderer
from threedgrut.datasets.protocols import Batch
from nucleus4d_ptir_full_data import ROOT,LOCAL,STORAGE
from nucleus4d_ptir_full_train import config_for,atomic_json
sys.path.insert(0,str(ROOT))
from utils.solar_geometry import solar_position,sun_vector_world


def camera(width,height,x,y,w,h,pose=None,fov=77):
    eye=np.array([1.4,1.7,.45]); forward=np.array([5.65,-.35,-.45])-eye
    forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    if pose is None:
        pose=np.eye(4,dtype=np.float32)
        pose[:3,:3]=np.stack((right,down,forward),axis=1);pose[:3,3]=eye
    focal=width/(2*math.tan(math.radians(fov)/2));k=[focal,focal,width/2,height/2]
    xx,yy=torch.meshgrid(torch.arange(x,x+w,device='cuda')+.5,
                        torch.arange(y,y+h,device='cuda')+.5,indexing='xy')
    rays=torch.stack(((xx-width/2)/focal,(yy-height/2)/focal,torch.ones_like(xx)),-1)
    rays=torch.nn.functional.normalize(rays,dim=-1)[None]
    return Batch(rays_ori=torch.zeros_like(rays),rays_dir=rays,
                 T_to_world=torch.tensor(pose,device='cuda')[None],intrinsics=k,
                 pixel_coords=torch.stack((xx,yy),-1)[None])


def daylight(kind, scale=1.):
    h,w=1024,2048
    phi=((np.arange(h,dtype=np.float32)+.5)/h-.5)*np.pi
    theta=((np.arange(w,dtype=np.float32)+.5)/w*2-1)*np.pi
    z=-np.sin(phi)[:,None]
    rgb=np.zeros((h,w,3),np.float32)
    if kind=='clear':
        rgb[:]=np.array([.4,.6,1.])*np.maximum(.15,.55+.45*z)[...,None]
        rgb[z[:,0]<0]=[.06,.055,.05]
    elif kind=='overcast':
        rgb[:]=np.array([.7,.72,.75])*np.maximum(.1,(1+2*z)/3)[...,None]
        rgb[z[:,0]<0]=[.06,.055,.05]
    else:
        hour=int(kind.split('_')[1]);dt=datetime.datetime(2026,3,21,hour,tzinfo=ZoneInfo('America/Los_Angeles'))
        sun=solar_position(37.7749,-122.4194,dt)
        direction=np.asarray(sun_vector_world(sun.azimuth_deg,sun.elevation_deg,90),np.float32)
        dot=(np.cos(phi)[:,None]*(np.sin(theta)[None]*direction[0]+np.cos(theta)[None]*direction[1])+z*direction[2])
        # Smooth finite solar disk; unlike a directional delta, this yields penumbrae.
        radius=math.radians(.266); sigma=radius/2
        disk=np.exp(-(np.maximum(0,1-dot))/(sigma*sigma))
        solid_angle=float(disk.sum(axis=1) @ (np.cos(phi)*(2*np.pi/w)*(np.pi/h)))
        power=5*max(0,min(1,np.sin(np.deg2rad(sun.elevation_deg))/.65))
        warmth=max(0,min(1,sun.elevation_deg/35))
        rgb[:]=disk[...,None]*(power/max(solid_angle,1e-10))*np.array([1,.72+.24*warmth,.42+.44*warmth])
    rgba=np.concatenate((rgb*scale,np.ones((h,w,1),np.float32)),axis=-1)
    env=create_environment(device='cuda',environment_type='2d',optimize_environment=False)
    env._set_environment_tensor(torch.from_numpy(rgba).to('cuda'))
    return env


@torch.no_grad()
def render_frame(model, name, args, reference=False):
    out=STORAGE/'renders';out.mkdir(exist_ok=True)
    h=round(args.width*416/640);w=args.width;tile=args.tile
    cache_path=out/f'{name}_record.json'
    cache_key=[args.cache_token,w,h,args.spp,args.bounces,bool(reference)]
    if cache_path.exists():
        old=json.loads(cache_path.read_text())
        keys=['original'] if reference else ['total','direct']
        expected=min(1920,w)*round(h*min(1920,w)/w)*3*4
        if (old.get('cache_key')==cache_key and all(Path(p).exists() for p in old['exr'].values())
            and all((LOCAL/old[k]).exists() and (LOCAL/old[k]).stat().st_size==expected for k in keys)):
            print('CACHED',name,flush=True);return old
    channels=['original'] if reference else ['total','direct']
    arrays={key:np.zeros((h,w,3),np.float32) for key in channels}
    tick=time.monotonic();tiles=math.ceil(w/tile)*math.ceil(h/tile);n=0
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            tw,th=min(tile,w-x),min(tile,h-y)
            outputs=model(camera(w,h,x,y,tw,th),train=False,frame_id=(y*w+x)+42)
            for key in channels:
                val=outputs['pred_rgb' if reference else ('pred_pbr' if key=='total' else 'pred_direct')][0]
                a=val.cpu().numpy()
                if reference:
                    a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
                if not np.isfinite(a).all():raise RuntimeError(f'Nonfinite render {name}')
                arrays[key][y:y+th,x:x+tw]=a
            n+=1
            if n%8==0 or n==tiles:
                atomic_json(LOCAL/'render_status.json',dict(frame=name,tiles=n,total_tiles=tiles,
                            width=w,height=h,spp=1 if reference else args.spp,seconds=time.monotonic()-tick,
                            pid=os.getpid(),updated=time.time()))
    record=dict(name=name,spp=1 if reference else args.spp,width=w,height=h,
                seconds=time.monotonic()-tick,exr={},cache_key=cache_key)
    preview_width=min(1920,w);preview_height=round(h*preview_width/w)
    for key,a in arrays.items():
        path=out/f'{name}_{key}.exr'
        if not cv2.imwrite(str(path),a[...,::-1],[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT]):
            raise IOError(path)
        record['exr'][key]=str(path)
        small=cv2.resize(a,(preview_width,preview_height),interpolation=cv2.INTER_AREA)
        binary='original.bin' if reference else f'{name}_{key}.bin'
        small.astype('<f4').tofile(LOCAL/binary);record[key]=binary
        if key in ('total','original'):
            display=np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,0)**(1/2.4)-.055)
            cv2.imwrite(str(out/f'{name}.png'),np.rint(np.clip(display,0,1)*255).astype('uint8')[...,::-1])
    print('RENDER',json.dumps(record),flush=True)
    atomic_json(cache_path,record)
    return record


def main():
    global LOCAL,STORAGE
    ap=argparse.ArgumentParser();ap.add_argument('--reference',action='store_true')
    ap.add_argument('--width',type=int,default=3840);ap.add_argument('--tile',type=int,default=128)
    ap.add_argument('--spp',type=int,default=1024);ap.add_argument('--bounces',type=int,default=8)
    ap.add_argument('--checkpoint',type=Path,default=STORAGE/'inverse/weights.pt')
    ap.add_argument('--probe',action='store_true')
    args=ap.parse_args();(STORAGE/'renders').mkdir(exist_ok=True,parents=True)
    identity=(ROOT/'experiments/out/nucleus4d_ptir_full/source.cache.ply') if args.reference else args.checkpoint
    stat=identity.stat()
    code_hash=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'experiments/ptir_full_runtime.patch').read_bytes()).hexdigest()
    args.cache_token=f'{identity}:{stat.st_size}:{stat.st_mtime_ns}:{code_hash}'
    if args.probe:
        LOCAL=LOCAL/'render_probe';STORAGE=LOCAL;LOCAL.mkdir(exist_ok=True,parents=True)
    if args.reference:
        cargs=argparse.Namespace(stage='geometry',smoke=True,resume='',steps=1,crop=512,spp=64)
        conf=config_for(cargs);model=MixtureOfGaussians(conf)
        model.init_from_ply(conf.import_ply.path,init_model=False);model.build_acc()
        rec=render_frame(model,'original',args,reference=True)
        atomic_json(LOCAL/'original_4k.json',rec);return
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    conf=checkpoint['config'];OmegaConf.set_struct(conf,False)
    conf.render.render_spp=args.spp;conf.render.spp_chunk=4;conf.render.render_bounces=args.bounces
    conf.render.filter_type='none';conf.model.optimize_environment=False
    model=MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint,setup_optimizer=False);model.requires_grad_(False)
    env=create_environment(device='cuda',environment_type='spherical_gaussian',optimize_environment=False)
    env.load_state_dict(checkpoint['environment_state']);env.configure_optimization(False)
    Renderer._set_model_environment(model,env.get_environment_parameter());model.build_acc()
    del checkpoint
    manifest=dict(width=min(1920,args.width),height=round(min(1920,args.width)*416/640),
                  export_width=args.width,export_height=round(args.width*416/640),gaussians=model.num_gaussians,
                  renderer='official PTIR-GS',joint_optimization=True,denoising=False,
                  preview_dtype='float32',
                  training_views=7446,held_out_views=504,skies={},sun=[],complete=False,
                  lighting_assumptions='Example San Francisco 2026-03-21, north bearing 90; synthetic HDR sky; opaque GS transport, no reconstructed refractive glass or fixture emitters')
    manifest['reconstructed_original']=render_frame(model,'reconstructed_original',args)
    # Preserve the learned illumination's mean scale: absolute radiometric calibration is unavailable.
    scale=max(.001,float(env.get_environment()[...,:3].mean().detach().cpu()))/.4
    for kind in ['clear','overcast','sun_09','sun_11','sun_13','sun_15','sun_17']:
        light=daylight(kind,scale)
        model.renderer.environment_type='2d';conf.environment.type='2d'
        Renderer._set_model_environment(model,light.get_environment_parameter())
        model.environment_alias_table=light.build_alias_table()
        record=render_frame(model,'sky_'+kind if kind in ('clear','overcast') else kind,args)
        if kind in ('clear','overcast'):manifest['skies'][kind]=record
        else:record['hour']=int(kind[-2:]);manifest['sun'].append(record)
        atomic_json(LOCAL/'manifest.json',manifest)
    manifest['complete']=True;atomic_json(LOCAL/'manifest.json',manifest)


if __name__=='__main__':main()
