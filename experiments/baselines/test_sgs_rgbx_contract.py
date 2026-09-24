#!/usr/bin/env python3
"""Verify public RGB-X raw-AOV adaptation avoids double normalization/gamma."""
import json
import sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'third_party/sgs_adapted'))
from rgb2x.pipeline_rgb2x import VaeImageProcessorAOV
from utils.rgbx_guidance import pipe
assert pipe is None, 'Import unexpectedly allocated/downloaded model'
p=VaeImageProcessorAOV()
x=torch.linspace(-1,1,36).reshape(1,3,3,4)
raw=p.postprocess(x,output_type='latent')
sgs=(raw/2+.5).clamp(0,1).pow(1/2.2)
public=p.postprocess(x,output_type='pt',do_gamma_correction=True)
err=(sgs-public).abs().max().item()
assert err<1e-6
r={'pass':True,'RGBX_lazy_import':True,'postprocess_max_error':err,'checkpoint_inference_tested':False,
   'scope':'decoded values in [-1,1]; interface test, not prediction accuracy'}
print(r)
(root/'experiments/out/EXP0021_sgs_repair/rgbx_contract_test.json').write_text(json.dumps(r,indent=2)+'\n')
