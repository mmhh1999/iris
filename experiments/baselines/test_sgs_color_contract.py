"""Public RGB-X demo input and output contracts, without inference weights."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'third_party/sgs_adapted'))
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline as Public,VaeImageProcessorAOV
from rgb2x.pipeline_rgb2x_myversion import StableDiffusionAOVMatEstPipeline as Adapter
from rgb2x.load_image import load_ldr_image
photo=torch.tensor([0.,.25,.5,.75,1.]).reshape(1,1,1,5).repeat(1,3,1,1)
raw=torch.linspace(-1,1,15).reshape(1,3,1,5)
observed={}
def fake(self,*args,**kwargs):
    observed.update(kwargs)
    return SimpleNamespace(images=[raw])
with patch.object(Public,'__call__',fake):
    result=Adapter.__call__(object.__new__(Adapter),photo=photo)
assert torch.equal(observed['photo'],photo**2.2)
assert observed['output_type']=='latent' and torch.equal(result,raw)
processor=VaeImageProcessorAOV()
err=float(((result/2+.5).clamp(0,1).pow(1/2.2)-processor.postprocess(raw,output_type='pt',do_gamma_correction=True)).abs().max())
assert err<1e-6
# In particular, display gray 0.5 must NOT reach the model as linear gray 0.5.
assert abs(float(observed['photo'][0,0,0,2])-.21763764)<1e-7
report=dict(passed=True,input_gamma=2.2,display_half_gray_to_model=float(observed['photo'][0,0,0,2]),output_postprocess_max_error=err,scope='interface regression; prediction quality evaluated separately')
(root/'experiments/out/EXP0022_sgs_sunpatch/color_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
