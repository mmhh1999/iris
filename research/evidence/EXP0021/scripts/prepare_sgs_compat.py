#!/usr/bin/env python3
"""Create a separately labelled SGS portability checkout; never edit the official checkout."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--rgbx',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    src,rgbx,dst=a.source.resolve(),a.rgbx.resolve(),a.output.absolute()
    if dst.exists(): raise FileExistsError(dst)
    shutil.copytree(src,dst,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__'))
    (dst/'rgb2x').unlink()
    shutil.copytree(rgbx/'rgb2x',dst/'rgb2x')
    (dst/'rgb2x/__init__.py').write_text('')
    (dst/'rgb2x/pipeline_rgb2x_myversion.py').write_text('''"""Public RGB-X adapter, NOT recovered author-private implementation.
The SGS caller expects decoded [-1,1] tensors then applies range/gamma conversion.
Public RGB-X output_type='latent' bypasses its image postprocessing; in this
pipeline it returns decoded AOV pixels, not diffusion latent features.
"""
from .pipeline_rgb2x import StableDiffusionAOVMatEstPipeline as PublicPipeline

class StableDiffusionAOVMatEstPipeline(PublicPipeline):
    def __call__(self, *args, **kwargs):
        kwargs['output_type'] = 'latent'
        result = super().__call__(*args, **kwargs)
        if len(result.images) != 1:
            raise ValueError('SGS adapter requires exactly one AOV')
        return result.images[0]
''')
    p=dst/'scene/gaussian_model_sgs.py'
    s=p.read_text().replace('from simple_knn_r3dg._C import distCUDA2','from simple_knn._C import distCUDA2')
    p.write_text(s)
    p=dst/'utils/rgbx_guidance.py';s=p.read_text()
    start=s.index('# Load pipeline');end=s.index('def sd_estimate_aovs_normal')
    original=s[start:end]
    original=original.replace('pipe = StableDiffusion','pipe = StableDiffusion').replace('cache_dir="./rgb2x/model_cache"','cache_dir=os.environ.get("HF_HOME", "./rgb2x/model_cache")')
    s=s[:start]+'pipe = None\n\ndef _get_pipe():\n    global pipe\n    if pipe is None:\n'+''.join('        '+line+'\n' for line in original.splitlines())+'    return pipe\n\n'+s[end:]
    s=s.replace('= pipe(', '= _get_pipe()(')
    p.write_text(s)
    p=dst/'utils/sam_region_infer.py';s=p.read_text()
    start=s.index('sam2_checkpoint =');end=s.index('def extract_number')
    s=s[:start]+'''predictor = None

def _get_predictor():
    global predictor
    if predictor is None:
        checkpoint = os.environ.get("SGS_SAM2_CHECKPOINT")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError("Set SGS_SAM2_CHECKPOINT to sam2.1_hiera_large.pt")
        predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint, device=device)
        for param in predictor.parameters():
            param.requires_grad = False
    return predictor

'''+s[end:]
    s=s.replace('    ensure_dir(save_dir)','    predictor = _get_predictor()\n    ensure_dir(save_dir)')
    p.write_text(s)
    p=dst/'utils/normal_guidance.py';s=p.read_text()
    s=s.replace('predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)', 'predictor = None')
    s=s.replace('def get_normal_tensor(input_tensor):', 'def get_normal_tensor(input_tensor):\n    global predictor\n    if predictor is None:\n        predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)')
    p.write_text(s)
    p=dst/'utils/pose_utils.py';s=p.read_text()
    s=s.replace('    transform = np.linalg.inv(pad_poses(cam2world))  # transform here','  transform = np.linalg.inv(pad_poses(cam2world))  # transform here')
    p.write_text(s)
    sha=lambda path:subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    (dst/'PORTABILITY.json').write_text(json.dumps({'label':'SGS-public-upstream-adaptation','official_sgs':sha(src),'public_rgbx':sha(rgbx),'changes':['public RGB-X raw-AOV return adapter; author private modifications unknown','lazy RGB-X, SAM2 and StableNormal weight load; explicit SAM2 checkpoint environment variable','public simple_knn function for simple_knn_r3dg import','correct upstream recenter_poses indentation error'],'official_reproduction':False},indent=2)+'\n')
    print(dst)

if __name__=='__main__':main()
