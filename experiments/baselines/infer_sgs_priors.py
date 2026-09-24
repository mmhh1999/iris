"""Image-derived RGB-X priors, training views only; never read material GT."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'third_party/sgs_adapted'))
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from diffusers import DDIMScheduler
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--steps',type=int,default=50);a=p.parse_args()
m=json.loads((a.input/'input.json').read_text());w=json.loads(a.weights.read_text())
pipe=StableDiffusionAOVMatEstPipeline.from_pretrained(w['path'],torch_dtype=torch.float16,local_files_only=True).to('cuda')
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,rescale_betas_zero_snr=True,timestep_spacing='trailing')
pipe.set_progress_bar_config(disable=True)
results=[]
for condition in m['conditions']:
 d=a.input/condition/'priors';d.mkdir(exist_ok=True)
 for row in m['frames']:
  if row['split']!='train':continue
  name=f"view{row['view']:02d}";f=a.input/condition/'train'/f'{name}.png';t=time.monotonic()
  photo=torch.from_numpy(np.array(Image.open(f).convert('RGB')).copy()).permute(2,0,1).float().div(255).pow(2.2)
  photo=torch.nn.functional.interpolate(photo[None],size=(512,512),mode='bilinear',align_corners=False)[0].to('cuda');arrays={}
  for aov,prompt in [('albedo','Albedo (diffuse basecolor)'),('roughness','Roughness')]:
   gen=torch.Generator(device='cuda').manual_seed(42)
   with torch.inference_mode():v=pipe(photo=photo,prompt=prompt,num_inference_steps=a.steps,generator=gen,required_aovs=[aov],height=512,width=512,output_type='pt').images[0]
   v=torch.nn.functional.interpolate(v.float(),size=(m['height'],m['width']),mode='bilinear',align_corners=False)[0].clamp(0,1)
   assert torch.isfinite(v).all()
   arrays[aov]=v.cpu().numpy()
   Image.fromarray((arrays[aov].transpose(1,2,0)*255+.5).astype('uint8')).save(d/f'{name}_{aov}.png')
  np.savez_compressed(d/f'{name}.npz',**arrays)
  rec=dict(condition=condition,view=row['view'],seconds=time.monotonic()-t,image_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  results.append(rec);print(json.dumps(rec),flush=True)
(a.input/'prior_manifest.json').write_text(json.dumps(dict(model=w,steps=a.steps,seed=42,training_only=True,input_color_conversion="LDR ** 2.2, matching public RGB-X demo",results=results),indent=2)+'\n')
