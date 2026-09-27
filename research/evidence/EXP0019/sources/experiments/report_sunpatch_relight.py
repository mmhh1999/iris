import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image,ImageDraw
b=Path('experiments/out/EXP0019_sunpatch');out=b/'relight_aligned'
def read(folder,prefix):
 return np.mean([cv2.imread(str(b/folder/f'{prefix}_seed{s}.exr'),-1)[...,::-1] for s in [101,202]],axis=0)
canvas=Image.new('RGB',(128*4,120*2));draw=ImageDraw.Draw(canvas);rows=[]
for row,view in enumerate([3,9]):
 gt=read('relight_aligned',f'truth_view{view}');pred=read('relight_aligned',f'sun_a_view{view}');ctrl=read('relight_aligned',f'sky_view{view}');raw=read('relight_01',f'sun_a_view{view}')
 mask=np.load(b/'data_03/generator_truth'/f'sun_a_view{view:02d}.npz')['sun_visible'];mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
 rows.append({'view':view,'old_patch_interior_pixels':int(mask.sum()),'sun_a_rmse_on_old_patch':float(np.sqrt(((pred[mask]-gt[mask])**2).mean())),'sky_rmse_on_same_old_patch':float(np.sqrt(((ctrl[mask]-gt[mask])**2).mean())),
              'sun_a_signed_mean_residual':float((pred[mask]-gt[mask]).mean()),'sky_signed_mean_residual':float((ctrl[mask]-gt[mask]).mean())})
 for col,(im,label) in enumerate([(gt,'True floor / new sun'),(raw,'Recovered A / raw'),(pred,'Recovered A / scaled'),(ctrl,'No-sun control / scaled')]):
  image=np.uint8(np.clip(im*4,0,1)**(1/2.2)*255);canvas.paste(Image.fromarray(image),(col*128,row*120+24));draw.text((col*128+2,row*120+4),label,fill='white')
canvas.resize((1024,480)).save(out/'comparison.png')
(out/'old_patch_comparison.json').write_text(json.dumps({'scope':'oracle floor-only relight after evaluation-only single-scalar alignment; equal old-sun-A regions for both methods','views':rows},indent=2))
print(json.dumps(rows,indent=2))
