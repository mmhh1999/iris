"""Known-geometry SGS material-stage diagnostic, NOT full published SGS reconstruction.
Reuses upstream train loop, renderer, losses, SAM cross-view and RGB-X self-invariance.
Appearance warmup freezes geometry and omits geometry-stage semantic reconstruction.
"""
import argparse,json,math,os,sys,time,types,random
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scipy.spatial.transform import Rotation,Slerp
root=Path(__file__).resolve().parents[2];repo=root/'third_party/sgs_adapted';sys.path.insert(0,str(repo))
from scene.cameras import Camera,PseudoCamera
from scene import Scene as StockScene
from utils.graphics_utils import BasicPointCloud
from arguments.config import ModelParams,OptimizationParams,PipelineParams
from utils.general_utils import inverse_sigmoid
from gaussian_renderer.render_sgs import render_sgs
from gaussian_renderer import render_fn_dict
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--condition',choices=['sun_a','sky'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--warmup',type=int,default=200);p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=7);p.add_argument('--resume-warmup',type=Path)
a=p.parse_args();a.input=a.input.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
if a.resume_warmup:a.resume_warmup=a.resume_warmup.resolve()
os.chdir(repo);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
m=json.loads((a.input/'input.json').read_text());g=np.load(a.input/'geometry.npz')
start=time.monotonic();live={};phase='';history=[]
# Build the public SAM connected-component postprocessing instead of silently skipping it.
import sam2
from torch.utils.cpp_extension import load
sam2._C=load(name='sgs_sam_connected_components',sources=[str(repo/'sam2/csrc/connected_components.cu')],extra_cuda_cflags=['-O3'],verbose=True)

class KnownScene:
    current=None
    get_canonical_rays=StockScene.get_canonical_rays
    def __init__(self,args,gaussians):
        KnownScene.current=self;self.gaussians=gaussians;self.model_path=args.model_path;self.cameras_extent=1.
        self.train_cameras={1.:[]};self.tests=[];self.pseudo={}
        fx=m['fov_x'];fy=2*math.atan(math.tan(fx/2)*m['height']/m['width'])
        for r in m['frames']:
            pose=np.array(r['c2w']);inv=np.linalg.inv(pose);name=f"view{r['view']:02d}"
            photo=torch.from_numpy(np.array(Image.open(a.input/a.condition/r['split']/(name+'.png')).convert('RGB')).copy()).permute(2,0,1).float()/255
            alb=rough=None
            if gaussians.use_pbr and r['split']=='train':
                pr=np.load(a.input/a.condition/'priors'/(name+'.npz'));alb=torch.from_numpy(pr['albedo']);rough=torch.from_numpy(pr['roughness'][:1])
            cam=Camera(colmap_id=r['view'],R=inv[:3,:3].T,T=inv[:3,3],FoVx=fx,FoVy=fy,image=photo,albedo=alb,roughness=rough,metallic=None,gt_alpha_mask=None,image_name=name,uid=r['view'])
            (self.train_cameras[1.] if r['split']=='train' else self.tests).append(cam)
        gaussians.create_from_pcd(BasicPointCloud(points=g['positions'],colors=np.full_like(g['positions'],.5),normals=g['normals']),1.)
        with torch.no_grad():gaussians._opacity.copy_(inverse_sigmoid(torch.full_like(gaussians._opacity,.95)))
        # Pseudo poses use training cameras only, never heldout RGB or test-derived priors.
        cams=self.getTrainCameras()
        for c in cams:
            center=c.camera_center.detach().cpu().numpy();near=sorted([v for v in cams if v.uid!=c.uid],key=lambda v:float(torch.linalg.norm(v.camera_center-c.camera_center)))[:2]
            pairs=[]
            for other in near:
                c0=c.c2w.detach().cpu().numpy();c1=other.c2w.detach().cpu().numpy();pose=np.eye(4)
                pose[:3,:3]=Slerp([0,1],Rotation.from_matrix(np.stack([c0[:3,:3],c1[:3,:3]])))(.5).as_matrix();pose[:3,3]=(c0[:3,3]+c1[:3,3])/2;inv=np.linalg.inv(pose)
                pairs.append(PseudoCamera(inv[:3,:3].T,inv[:3,3],fx,fy,m['width'],m['height']))
            self.pseudo[c.uid]=pairs
    def getTrainCameras(self):return self.train_cameras[1.]
    def getTestCameras(self):return self.tests
    def sample_random_pseudo_cameras(self,uid):return self.pseudo[uid]
    def save(self,iteration):
        dest=Path(self.model_path)/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply';dest.parent.mkdir(parents=True,exist_ok=True);self.gaussians.save_ply(str(dest))

# Preserve upstream material losses; only scale the self-invariance start for a bounded pilot.
source=(repo/'train.py').read_text().replace('iteration >= 8000',f'iteration >= {a.warmup+2*a.steps//3}').replace('CubemapLight(base_res=256, scale=1.5)','CubemapLight(base_res=128, scale=1.5)')
source=source.replace('        loss.backward()', '''        tb_dict['loss_total_with_invariance'] = float(loss.detach())
        tb_dict['cross_view_active'] = float(is_pbr and iteration % 3 == 0)
        tb_dict['self_invariance_active'] = float(is_pbr and iteration % 3 == 1 and iteration >= SELF_START)
        loss.backward()'''.replace('SELF_START',str(a.warmup+2*a.steps//3)))
module=types.ModuleType('sgs_p1_stock_training');module.__file__=str(repo/'train.py');exec(compile(source,str(repo/'train.py'),'exec'),module.__dict__);module.Scene=KnownScene
original=render_fn_dict['sgs']
def tracking_render(*args,**kwargs):
    live['scene']=kwargs['scene'];live['params']=kwargs['dict_params'];live['pipe']=args[2];live['background']=args[3]
    out=original(*args,**kwargs)
    if 'loss' in out and not torch.isfinite(out['loss']):raise FloatingPointError('Nonfinite SGS loss')
    return out
module.render_fn_dict['sgs']=tracking_render

def report(writer,iteration,stats,scene,*args,**kwargs):
    rec=dict(phase=phase,iteration=iteration,seconds=time.monotonic()-start,**{k:float(v) for k,v in stats.items()})
    history.append(rec)
    with (a.out/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    if iteration%10==0:print('PROGRESS',json.dumps(rec),flush=True)
module.training_report=report

config=dict(protocol='P1 SGS material-stage public adaptation',condition=a.condition,warmup_steps=a.warmup,material_steps=a.steps,seed=a.seed,geometry='shared known mesh ray samples; frozen positions/normals/scales/opacity',geometry_stage_semantics=False,material_cross_view=True,material_self_invariance_start=a.warmup+2*a.steps//3,envmap_resolution=128,priors='RGB-X from training images only',input=str(a.input),official_full_reproduction=False)
(a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for phase in ['warmup','material']:
    if phase=='warmup' and a.resume_warmup:continue
    parser=argparse.ArgumentParser();lp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser);args=parser.parse_args([])
    args.source_path=str(a.input/a.condition);args.model_path=str(a.out/phase);Path(args.model_path).mkdir()
    args.type='render' if phase=='warmup' else 'sgs';args.iterations=a.warmup if phase=='warmup' else a.warmup+a.steps
    args.checkpoint=None if phase=='warmup' else str(a.resume_warmup.resolve() if a.resume_warmup else a.out/'warmup'/f'chkpnt{a.warmup}.pth')
    args.debug_from=-1;args.eval=False;args.test_interval=1000000;args.save_interval=1000000;args.checkpoint_interval=1000000;args.quiet=False;args.n_views=12
    args.position_lr_init=0.;args.position_lr_final=0.;args.normal_lr=0.;args.opacity_lr=0.;args.scaling_lr=0.;args.rotation_lr=0.;args.densify_until_iter=0
    args.lambda_featuregs=0.;args.lambda_normal_render_depth=0.;args.lambda_depth_smooth=0.;args.lambda_depth_var=0.;args.lambda_normal_smooth=0.;args.sh_lr=0.0025 if phase=='warmup' else 0.
    args.pointlight_lr=.05;args.lambda_base_color_smooth=1.;args.lambda_roughness_smooth=.2;args.lambda_pbr=1.;args.lambda_env_smooth=.01
    module.args=args
    module.training(lp.extract(args),op.extract(args),pp.extract(args),is_pbr=phase=='material',is_blender=True)
    if phase=='warmup':
        saved=torch.load(a.out/'warmup'/f'chkpnt{a.warmup}.pth',weights_only=False);assert saved[1]==a.warmup

# Training complete: only now export heldout views (no validation-based selection).
scene=KnownScene.current;d=a.out/'evaluation';d.mkdir()
with torch.no_grad():
 for cam in scene.getTestCameras()+scene.getTrainCameras():
    r=original(cam,scene.gaussians,live['pipe'],live['background'],dict_params=live['params'],scene=scene,is_training=False)
    arrays={k:r[k].detach().cpu().numpy() for k in ['base_color','roughness','pbr','opacity','normal','depth']}
    np.savez_compressed(d/(cam.image_name+'.npz'),**arrays)
    for k in ['base_color','pbr']:
        im=np.uint8(np.clip(arrays[k].transpose(1,2,0),0,1)*255+.5);Image.fromarray(im).save(d/(cam.image_name+'_'+k+'.png'))
result=dict(completed=True,elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1e9,steps_logged=len(history),last_global_step=a.warmup+a.steps,scientific_convergence_established=False)
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
