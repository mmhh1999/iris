"""Run the official PTIR two-stage optimizer on all native Nucleus photographs.

The input adapter, bounded-memory execution and checkpoint/progress handling are local.
Rendering, losses, differentiation, normal supervision and optimizers are upstream.
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from addict import Dict
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.timer import CudaTimer
from threedgrut.model.ptir_helper import linear_to_srgb
from threedgrut.model.losses import ssim
from nucleus4d_ptir_full_data import ROOT, LOCAL, STORAGE, NucleusDataset, prior_path


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp,path)


def complete_stage(trainer):
    # A final checkpoint can exist even if post-training validation failed.
    # Recover that evaluation without any further optimizer step or re-export.
    if trainer.global_step < trainer.conf.n_iterations:
        trainer.run_training()
    else:
        # Resume loads parameter tensors without rebuilding the native BVH;
        # normal training builds it on optimizer iterations that are skipped here.
        trainer.model.build_acc(rebuild=True)
    return trainer.evaluate('after')


def config_for(args):
    OmegaConf.register_new_resolver('int_list',lambda l:[int(x) for x in l],replace=True)
    stage=args.stage
    source_cache=LOCAL/'source.cache.ply'
    source=STORAGE/'source.ply'
    if (stage=='geometry' or args.smoke) and not args.resume:
        if not source_cache.exists() or source_cache.stat().st_size != source.stat().st_size:
            tmp=source_cache.with_suffix('.partial');shutil.copyfile(source,tmp);os.replace(tmp,source_cache)
    name='apps/colmap_3dgrt' if stage=='geometry' else 'inversions/colmap_3dgptir'
    with initialize_config_dir(config_dir=str(ROOT/'third_party/ptir_gs/configs'),version_base=None):
        conf=compose(config_name=name)
    OmegaConf.set_struct(conf,False)
    conf.path=str(STORAGE);conf.out_dir=str(STORAGE);conf.experiment_name=stage
    conf.n_iterations=args.steps or (30000 if stage=='geometry' else 16000)
    conf.num_workers=0;conf.compute_extra_metrics=False;conf.use_wandb=False
    conf.val_frequency=10**9;conf.visualize_frequency=1000;conf.log_frequency=20
    conf.test_last=False;conf.writer.log_image_views=[];conf.checkpoint.iterations=[]
    conf.loss.use_normal_prior_regularization=not args.smoke or getattr(args,'with_priors',False)
    if stage=='geometry':
        conf.import_ply.enabled=True;conf.import_ply.path=str(source_cache)
        # Refine a converged 6.1M-point model without resetting all opacity at step zero.
        conf.strategy.reset_density.start_iteration=-1
        conf.strategy.density_decay.start_iteration=-1
        conf.optimizer.params.positions.lr=0.000016
        conf.optimizer.params.rotation.lr=0.0001
        conf.optimizer.params.scale.lr=0.0005
        conf.optimizer.params.density.lr=0.005
        conf.strategy.densify.start_iteration=500
        conf.strategy.densify.end_iteration=15000
        conf.strategy.densify.max_gaussians=8000000
        conf.strategy.densify.max_growth_per_update=100000
        conf.loss.use_depth_distortion=True
        conf.loss.depth_distortion_start_iteration=5000
    else:
        conf.initialization.path=str(STORAGE/'geometry/weights.pt')
        conf.render.inversion_spp=args.spp
        conf.render.spp_chunk=4
        conf.render.checkpoint_spp_chunks=True
        # Real indoor photographs include unmodelled sky visible through openings.
        conf.render.visualize_lights=True
        conf.render.max_bounces=4
        conf.render.render_spp=1024
        conf.render.render_bounces=8
        conf.render.relight_spp=1024
        # Native-size ray batches are kept bounded; samples still contribute to one loss.
        conf.render.hide_intermediate_outputs=False
        conf.environment.type='spherical_gaussian'
        conf.loss.use_normal_prior_regularization=False
        conf.optimizer.params.material_albedo.lr=0.01
        conf.optimizer.params.material_roughness.lr=0.005
        conf.optimizer.params.environment.lr=0.01
        conf.loss.use_albedo_prior_regularization.max_steps=4000
        conf.loss.use_roughness_prior_regularization.max_steps=6000
    if args.smoke:
        conf.strategy.densify.start_iteration=-1
        conf.strategy.prune.start_iteration=-1
        if not getattr(args,'with_priors',False):
            conf.loss.use_albedo_prior_regularization=False
            conf.loss.use_roughness_prior_regularization=False
        if stage=='inverse':
            conf.import_ply.enabled=True;conf.import_ply.path=str(source_cache)
    if args.resume:conf.resume=args.resume
    return conf


class FullTrainer(Trainer3DGRUT):
    def __init__(self, conf, args):
        self.args=args;self.started=time.monotonic();self.validation_history=[]
        super().__init__(conf)

    def init_dataloaders(self, conf):
        self.train_dataset=NucleusDataset('train', crop=self.args.crop, require_priors=not self.args.smoke)
        if self.args.smoke and getattr(self.args,'with_priors',False):
            self.train_dataset.rows=[r for r in self.train_dataset.rows if prior_path(r).exists()]
            self.train_dataset.poses=np.asarray([r['pose'] for r in self.train_dataset.rows],dtype=np.float32)
            self.train_dataset.require_priors=True
        gen=torch.Generator().manual_seed(42)
        self.train_dataloader=DataLoader(self.train_dataset,batch_size=1,shuffle=True,num_workers=0,generator=gen)
        self.init_validation_loader()

    def init_validation_loader(self):
        self.val_dataset=NucleusDataset('val', crop=256, require_priors=False, limit=24)
        self.val_dataloader=DataLoader(self.val_dataset,batch_size=1,shuffle=False,num_workers=0)

    def init_experiments_tracking(self, conf):
        out=STORAGE/(self.args.stage+('_smoke' if self.args.smoke else ''))
        out.mkdir(exist_ok=True,parents=True)
        OmegaConf.save(conf,out/'parsed.yaml')
        self.tracking=Dict(writer=SummaryWriter(str(out/'events')),output_dir=str(out),
                           object_name='1406-C-int',run_name=self.args.stage)

    def save_checkpoint(self, last_checkpoint=False):
        if self.args.smoke:
            return
        out=Path(self.tracking.output_dir)
        params=self.model.get_model_parameters()
        params.update(self.strategy.get_strategy_parameters())
        params.update(global_step=self.global_step,epoch=self.n_epochs-1,
                      numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state())
        if self.environment is not None:params['environment_state']=self.environment.state_dict()
        tmp=out/'resume.partial.pt';torch.save(params,tmp);os.replace(tmp,out/'resume.pt')
        # A small inference/handoff checkpoint excludes Adam moments and strategy buffers.
        weights={k:v for k,v in params.items() if k not in ('optimizer','numpy_rng','torch_rng')
                 and not k.startswith('densify_')}
        tmp=out/'weights.partial.pt';torch.save(weights,tmp);os.replace(tmp,out/'weights.pt')
        atomic_json(out/'checkpoint.json',dict(step=self.global_step,gaussians=self.model.num_gaussians,
                                             updated=time.time(),precision='float32'))

    @torch.no_grad()
    def evaluate(self, label, count=24):
        # Fixed held-out native pixel regions. No exposure/color fitting to the targets.
        # Upstream on_training_end deletes both datasets and loaders before
        # returning; recreate the same deterministic held-out inputs if needed.
        if not hasattr(self,'val_dataloader') or not hasattr(self,'val_dataset'):
            self.init_validation_loader()
        scores=[];out=Path(self.tracking.output_dir)
        old_spp=self.conf.render.get('render_spp',None)
        if old_spp is not None:self.conf.render.render_spp=64
        for i,batch in enumerate(self.val_dataloader):
            if i>=count:break
            gpu=self.val_dataset.get_gpu_batch_with_intrinsics(batch)
            outputs=self.model(gpu,train=False,frame_id=i)
            pred=linear_to_srgb(outputs['pred_pbr']) if self.args.stage=='inverse' else outputs['pred_rgb']
            mask=gpu.gradient_mask;gt=gpu.rgb_gt
            mse=(((pred-gt)**2)*mask).sum()/(mask.sum()*3).clamp_min(1)
            score=float(-10*torch.log10(mse.clamp_min(1e-12)))
            sim=float(ssim((pred*mask).permute(0,3,1,2),(gt*mask).permute(0,3,1,2)))
            scores.append(dict(index=i,source=self.val_dataset.rows[i]['name'],psnr=score,ssim=sim))
            if i<4:
                from PIL import Image
                pair=torch.cat((gt[0],pred[0]),dim=1).clamp(0,1).cpu().numpy()
                Image.fromarray(np.rint(pair*255).astype('uint8')).save(out/f'{label}_{i:02}.png')
        if old_spp is not None:self.conf.render.render_spp=old_spp
        result=dict(label=label,step=self.global_step,views=scores,
                    psnr=float(np.mean([x['psnr'] for x in scores])),
                    ssim=float(np.mean([x['ssim'] for x in scores])))
        atomic_json(out/f'{label}.json',result);print('VALIDATION',json.dumps(result),flush=True)
        return result

    def run_train_iter(self,*args,**kwargs):
        start=time.monotonic();super().run_train_iter(*args,**kwargs)
        step=self.global_step
        if step==1 or step%20==0:
            metrics=args[3][-1]
            record=dict(stage=self.args.stage,step=step,total=self.conf.n_iterations,
                        gaussians=self.model.num_gaussians,seconds=time.monotonic()-start,
                        elapsed=time.monotonic()-self.started,loss=float(metrics['losses']['total_loss']),
                        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,pid=os.getpid(),updated=time.time())
            if not np.isfinite(record['loss']):raise RuntimeError('Nonfinite training loss')
            atomic_json(LOCAL/'training_status.json',record);print('PROGRESS',json.dumps(record),flush=True)
        save_every=1000
        if step%save_every==0:self.save_checkpoint()
        if not self.args.smoke and step%2000==0:
            self.evaluate(f'validation_{step:06}',count=24)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['geometry','inverse'],required=True)
    ap.add_argument('--steps',type=int,default=0);ap.add_argument('--crop',type=int,default=512)
    ap.add_argument('--spp',type=int,default=64);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',default='');ap.add_argument('--with-priors',action='store_true')
    ap.add_argument('--smoke-save-weights',default='');args=ap.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    # Full-precision arithmetic; TF32 is disabled for the quality run.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    conf=config_for(args)
    trainer=FullTrainer(conf,args)
    print('MODEL',trainer.model.num_gaussians,flush=True)
    if not args.resume:trainer.evaluate('before',count=2 if args.smoke else 24)
    if args.smoke:
        # Prove forward/backward and optimizer updates on the complete original model.
        before=trainer.model.shading_normal.detach().clone()
        trainer.run_training()
        delta=float((trainer.model.shading_normal.detach()-before).abs().max())
        report=dict(stage=args.stage,gaussians=trainer.model.num_gaussians,steps=trainer.global_step,
                    normal_update_max=delta,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                    finite=bool(torch.isfinite(trainer.model.shading_normal).all()))
        atomic_json(LOCAL/f'{args.stage}_gradient_check.json',report)
        print('GRADIENT_CHECK',json.dumps(report),flush=True)
        if args.smoke_save_weights:
            params=trainer.model.get_model_parameters();params.pop('optimizer',None)
            if trainer.environment is not None:params['environment_state']=trainer.environment.state_dict()
            torch.save(params,args.smoke_save_weights)
    else:
        complete_stage(trainer)


if __name__=='__main__':main()
