"""Restartable official RGB-X prior inference for every training view."""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from nucleus4d_ptir_full_data import LOCAL, ROOT, STORAGE, load_image, prior_path
from threedgrut.utils.rgb2x_prior import RGB2X_PROMPTS, _camera_normal_to_world
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from diffusers import DDIMScheduler


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--edge', type=int, default=1024); ap.add_argument('--steps', type=int, default=50)
    args=ap.parse_args()
    info=json.loads((LOCAL/'dataset.json').read_text())
    rows=[r for r in info['frames'] if r['split']=='train']
    # Spread early results across cameras, positions and capture times.
    order=np.random.default_rng(42).permutation(len(rows))
    rows=[rows[int(i)] for i in order]
    if args.limit: rows=rows[:args.limit]
    model=ROOT/'.baseline_deps/hf_cache/hub/models--zheng95z--rgb-to-x/snapshots/b38b3fd73a14ea62f3953fc54bc4ac67b067bae0'
    pending=[]
    for row in rows:
        p=prior_path(row)
        if p.exists():
            with np.load(p) as old:
                if int(old['steps'])>=args.steps and int(old['edge'])>=args.edge: continue
        pending.append(row)
    print(json.dumps(dict(stage='priors', training_views=len(rows), pending=len(pending),
                          edge=args.edge, steps=args.steps)), flush=True)
    if not pending:return
    pipe=StableDiffusionAOVMatEstPipeline.from_pretrained(str(model), torch_dtype=torch.float16,
                                                         local_files_only=True).to('cuda')
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True,
                                            timestep_spacing='trailing')
    pipe.set_progress_bar_config(disable=True)
    STORAGE.mkdir(exist_ok=True, parents=True)
    for i,row in enumerate(pending):
        start=time.monotonic(); dst=prior_path(row); dst.parent.mkdir(exist_ok=True, parents=True)
        photo=load_image(row); h,w=photo.shape[:2]
        # Aspect ratio preserved, all dimensions divisible by the VAE stride.
        scale=args.edge/max(h,w); ph=max(64, int(round(h*scale/8))*8); pw=max(64, int(round(w*scale/8))*8)
        rgb=torch.from_numpy(photo).to('cuda').permute(2,0,1).float()[None]/255
        rgb=F.interpolate(rgb.pow(2.2), (ph,pw), mode='area')
        maps={}
        for aov in ('normal','albedo','roughness'):
            with torch.inference_mode():
                result=pipe(prompt=RGB2X_PROMPTS[aov], photo=rgb[0], num_inference_steps=args.steps,
                            height=ph, width=pw, generator=torch.Generator(device='cuda').manual_seed(42+row['id']),
                            required_aovs=[aov], output_type='pt').images[0][0].float()
            if aov=='normal':
                result=_camera_normal_to_world(result, torch.tensor(row['pose'],device='cuda'),
                                               c2w_uses_opengl_axes=False)*2-1
            elif aov=='albedo':
                # Official pipeline's display-encoded output is converted back to linear RGB.
                result=result.clamp(0,1).pow(2.2)
            else: result=result.mean(dim=0,keepdim=True).clamp(0,1)
            array=result.permute(1,2,0).cpu().numpy()
            if not np.isfinite(array).all():raise ValueError(f'Nonfinite {aov} in {row["name"]}')
            maps[aov]=array.astype(np.float16)
        tmp=dst.with_suffix('.partial.npz')
        np.savez_compressed(tmp, **maps, edge=args.edge, steps=args.steps, source=row['name'])
        os.replace(tmp,dst)
        report=dict(stage='priors', completed=i+1, pending_at_start=len(pending), source=row['name'],
                    seconds=time.monotonic()-start, resolution=[pw,ph], storage_bytes=dst.stat().st_size,
                    pid=os.getpid(), updated=time.time())
        print(json.dumps(report),flush=True)
        tmp_status=LOCAL/'prior_status.tmp';tmp_status.write_text(json.dumps(report));os.replace(tmp_status,LOCAL/'prior_status.json')


if __name__=='__main__':main()
