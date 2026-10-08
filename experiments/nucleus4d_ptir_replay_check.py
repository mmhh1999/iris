"""Compare official gradients with and without sample-chunk recomputation."""
import argparse
import json
import torch
from torch.utils.data import default_collate
from nucleus4d_ptir_full_train import FullTrainer, config_for, atomic_json, LOCAL

args=argparse.Namespace(stage='inverse',steps=1,crop=64,spp=8,smoke=True,resume='')
conf=config_for(args)
# Backgrounds are intentionally hidden in upstream's training default. Enable
# them for the separate direct-environment derivative control below.
conf.render.visualize_lights=True
trainer=FullTrainer(conf,args)
batch=trainer.train_dataset.get_gpu_batch_with_intrinsics(default_collate([trainer.train_dataset[1000]]))
results=[];reference=None
for enabled in (False,True):
    trainer.conf.render.checkpoint_spp_chunks=enabled
    trainer.model.optimizer.zero_grad(set_to_none=True)
    outputs=trainer.model(batch,train=True,frame_id=42)
    loss=trainer.get_pbr_losses(batch,outputs)['total_loss']
    loss.backward()
    gradients={name:p.grad.detach().clone() for name,p in trainer.model.named_parameters()
               if p.requires_grad and p.grad is not None}
    row={'checkpoint':enabled,'loss':float(loss.detach()),'gradient_names':list(gradients)}
    if reference is None:reference=gradients
    else:
        row['comparison']={}
        for name,g in gradients.items():
            ref=reference[name];delta=(g-ref).abs().max().item();scale=ref.abs().max().item()
            row['comparison'][name]={'max_abs_error':delta,'reference_max':scale}
            assert torch.isfinite(g).all()
            assert delta <= 2e-6 + 5e-3*scale, (name,delta,scale)
        assert abs(row['loss']-results[0]['loss'])<1e-5
    results.append(row)
    del outputs,loss,gradients
# A direct view of the environment must carry a nonzero illumination gradient.
# An indoor wall patch can legitimately have no escaping paths at initialization.
trainer.model.optimizer.zero_grad(set_to_none=True)
batch.T_to_world[:, :3, 3]=1000
outputs=trainer.model(batch,train=True,frame_id=51)
print('ENV_CONTROL',float(outputs['pred_pbr'].mean().detach()),
      trainer.model.environment.native_parameters().detach().cpu().tolist()[:2],flush=True)
outputs['pred_pbr'].mean().backward()
env_grad={n:float(p.grad.abs().max()) for n,p in trainer.model.named_parameters()
          if n.startswith('environment.') and p.grad is not None}
assert env_grad and max(env_grad.values())>0, env_grad
atomic_json(LOCAL/'replay_validation.json',{'passed':True,'gaussians':trainer.model.num_gaussians,
                                         'spp':8,'bounces':4,'results':results,
                                         'environment_gradient_control':env_grad})
print(json.dumps(results),flush=True)
