"""Visual evidence and compact oracle material-family report."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
src=Path('experiments/out/EXP0018_native_brdf/run_04')
out=Path('experiments/out/EXP0018_native_brdf/fit_03')
canvas=Image.new('RGB',(384*4,290*4),(25,25,25));draw=ImageDraw.Draw(canvas)
for row,(room,view) in enumerate([(r,v) for r in ['bathroom','kitchen'] for v in [0,1]]):
    z=np.load(src/f'{room}_view{view}.npz')
    hdr=cv2.imread(f'experiments/out/EXP0017_pbr_solar/run_04/{room}/view{view}_sun1_seed101.exr',-1)[...,::-1]
    valid=z['valid'];eligible=valid&~z['has_delta']&~z['has_null']&(z['camera_local'][...,2]>.1)
    mask=np.zeros((*valid.shape,3),np.uint8);mask[valid]=[150,80,30];mask[eligible]=[45,175,95]
    fit=np.load(out/f'{room}_view{view}_fit.npz');ids=json.loads((out/'report.json').read_text())['heldout_directions']
    error=np.sqrt(((fit['prediction'][ids]-fit['target'][ids])**2).mean((0,2)))
    denom=np.maximum(np.sqrt((fit['target'][ids]**2).mean((0,2))),.01)
    # Sparse evaluated points only; black pixels are unscored, not zero error.
    heat=np.zeros((*valid.shape,3),np.uint8);ratio=np.clip(error/denom,0,1)
    heat.reshape(-1,3)[fit['pixel_index']]=np.stack([255*ratio,255*(1-ratio),np.zeros_like(ratio)],-1).astype(np.uint8)
    imgs=[np.uint8(np.clip(hdr*4,0,1)**(1/2.2)*255),mask,
          np.uint8(np.clip(z['native_response'][0]*3.14159265,0,1)**(1/2.2)*255),heat]
    labels=[f'{room} v{view}: solar image +2EV','Green eligible / brown excluded','Native response at normal x pi','Oracle heldout error: green 0, red >=1']
    for col,(im,label) in enumerate(zip(imgs,labels)):
        canvas.paste(Image.fromarray(im),(col*384,row*290+30));draw.text((col*384+5,row*290+7),label,fill='white')
canvas.save(out/'compatibility.jpg',quality=94)
print(out/'compatibility.jpg')
