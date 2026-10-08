"""Estimate material priors from undistorted original photographs (not splat colors)."""
import pathlib,json,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/sgs_adapted'))
import torch,numpy as np
from PIL import Image
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from diffusers import DDIMScheduler
P=ROOT/'experiments/out/nucleus4d_material';(P/'priors').mkdir(exist_ok=True)
pipe=StableDiffusionAOVMatEstPipeline.from_pretrained(str(ROOT/'.baseline_deps/hf_cache/hub/models--zheng95z--rgb-to-x/snapshots/b38b3fd73a14ea62f3953fc54bc4ac67b067bae0'),torch_dtype=torch.float16,local_files_only=True).to('cuda')
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,rescale_betas_zero_snr=True,timestep_spacing='trailing');pipe.set_progress_bar_config(disable=True)
report=[]
for r in json.loads((P/'views.json').read_text()):
 i=r['index'];dst=P/'priors'/f'{i:02}.npz'
 if dst.exists():continue
 rgb=np.asarray(Image.open(P/r['photo']).convert('RGB')).copy()/255
 photo=torch.from_numpy(rgb).permute(2,0,1).float().pow(2.2).to('cuda');maps={}
 start=time.monotonic()
 for name,prompt in [('albedo','Albedo (diffuse basecolor)'),('roughness','Roughness'),('metallic','Metallic')]:
  with torch.inference_mode():
   a=pipe(photo=photo,prompt=prompt,num_inference_steps=30,generator=torch.Generator(device='cuda').manual_seed(42),required_aovs=[name],height=512,width=512,output_type='latent').images[0]
  a=(a[0].float()/2+.5).cpu().numpy().transpose(1,2,0);assert np.isfinite(a).all();a=np.clip(a,0,1);maps[name]=a
  display=a**(1/2.2) if name=='albedo' else a
  Image.fromarray(np.rint(display*255).astype('uint8')).save(P/'priors'/f'{i:02}_{name}.png')
 np.savez_compressed(dst,**maps);info={'view':i,'split':r['split'],'seconds':time.monotonic()-start};report.append(info);print(info,flush=True)
(P/'prior_report.json').write_text(json.dumps({'model':'RGB-X rgb-to-x b38b3fd73a14ea62f3953fc54bc4ac67b067bae0','steps':30,'seed':42,'input':'original photos, undistorted, gamma 2.2 inverse','output':'linear decoded channels, latent output then (x/2+0.5), no display gamma','claim':'learned material estimates, not measured BRDF ground truth','views':report},indent=2))
