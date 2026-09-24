#!/usr/bin/env python3
"""GPU forward/backward contract checks against mathematical invariants."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'third_party/sgs_intrinsic'))
from utils.graphics_utils import getProjectionMatrix
spec=importlib.util.spec_from_file_location('sgs_raster_contract',root/'third_party/sgs_intrinsic/gaussian_renderer/r3dg_rasterization.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from diff_gaussian_rasterization_feature import GaussianRasterizationSettings as FS, GaussianRasterizer as FR

torch.manual_seed(7)
x=torch.tensor([[-.2,0,2.],[.2,.1,2.2],[0,-.2,2.5]],device='cuda',requires_grad=True)
color=torch.tensor([[.2,.4,.6],[.8,.3,.1],[.1,.7,.3]],device='cuda',requires_grad=True)
scale=torch.full((3,3),.15,device='cuda',requires_grad=True)
rotation=torch.tensor([[1.,0,0,0]]*3,device='cuda')
opacity=torch.full((3,1),.65,device='cuda',requires_grad=True)
view=torch.eye(4,device='cuda');proj=getProjectionMatrix(.01,100.,1.,1.).T.cuda()
common=dict(image_height=32,image_width=32,tanfovx=0.5463024898,tanfovy=0.5463024898,
            bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=view,projmatrix=proj,
            sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=False)
settings=m.GaussianRasterizationSettings(**common,cx=16.,cy=16.,backward_geometry=True,computer_pseudo_normal=False)
render=m.GaussianRasterizer(settings)
kw=dict(means3D=x,means2D=torch.zeros_like(x,requires_grad=True),colors_precomp=color,
        opacities=opacity,scales=scale,rotations=rotation)
a=render(**kw,features=torch.ones((3,2),device='cuda'))
assert len(a)==10,len(a)
_,_,rgb,alpha,depth,feature,normal,xyz,weights,radii=a
assert weights.shape==(3,1),weights.shape
assert alpha.sum()>0
mass_error=float((weights.sum()-alpha.sum()).abs())
assert torch.allclose(weights.sum(),alpha.sum(),rtol=1e-4,atol=1e-4),(weights.sum(),alpha.sum())
rgb.sum().backward()
grad_error=float((color.grad-weights.expand(-1,3)).abs().max())
assert torch.allclose(color.grad,weights.expand(-1,3),rtol=1e-4,atol=1e-4)
assert x.grad is not None and torch.isfinite(x.grad).all()
sem=torch.rand((3,32,1),device='cuda',requires_grad=True)
b=FR(FS(**common))(**{k:v.detach().requires_grad_(True) for k,v in kw.items()},semantic_feature=sem)
assert len(b)==4
assert b[1].shape==(32,32,32),b[1].shape
(b[0].sum()+b[1].sum()).backward()
assert sem.grad is not None and torch.isfinite(sem.grad).all() and sem.grad.abs().sum()>0
r={'pass':True,'gpu':torch.cuda.get_device_name(),'r3dg_output_count':len(a),'weight_alpha_mass_error':mass_error,
   'color_gradient_vs_weight_max_error':grad_error,'feature_shape':list(b[1].shape),
   'feature_gradient_finite_nonzero':True}
print(json.dumps(r,indent=2))
(root/'experiments/out/EXP0021_sgs_repair/kernel_test.json').write_text(json.dumps(r,indent=2)+'\n')
