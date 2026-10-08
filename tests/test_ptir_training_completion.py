"""Regression: final evaluation after upstream loader teardown and recovery at target step."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


class FakeDataset:
    def __init__(self, split, crop, require_priors, limit):
        assert (split,crop,require_priors,limit) == ('val',256,False,24)
        self.rows=[{'name':f'held_out_{i}'} for i in range(limit)]

    def get_gpu_batch_with_intrinsics(self, index):
        return SimpleNamespace(gradient_mask=torch.ones(1,4,4,1),rgb_gt=torch.full((1,4,4,3),.5))


class FakeModel:
    def __init__(self):self.ready=False;self.builds=0
    def build_acc(self,rebuild):
        assert rebuild is True
        self.ready=True;self.builds+=1
    def __call__(self,batch,**kwargs):
        assert self.ready, 'A checkpoint has no native acceleration structure until rebuilt'
        return {'pred_rgb':batch.rgb_gt.clone()}


def load_adapter():
    tree=ast.parse((ROOT/'experiments/nucleus4d_ptir_full_train.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
           and n.name in ('atomic_json','complete_stage','FullTrainer')]
    scope=dict(torch=torch,np=np,os=__import__('os'),json=json,Path=Path,
               Trainer3DGRUT=object,NucleusDataset=FakeDataset,
               DataLoader=lambda dataset,**kwargs:list(range(len(dataset.rows))),
               ssim=lambda a,b:torch.tensor(1.))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'training_adapter','exec'),scope)
    return scope['FullTrainer'],scope['complete_stage']


class TrainingCompletionTest(unittest.TestCase):
    def make_trainer(self, output, step, target):
        cls,complete=load_adapter();trainer=cls.__new__(cls)
        trainer.args=SimpleNamespace(stage='geometry')
        trainer.conf=OmegaConf.create(dict(n_iterations=target,render={'render_spp':1024}))
        trainer.global_step=step;trainer.tracking=SimpleNamespace(output_dir=output)
        trainer.model=FakeModel()
        trainer.init_validation_loader()
        trainer.train_dataset=object();trainer.train_dataloader=object()
        return trainer,complete

    def test_normal_completion_evaluates_same_views_after_upstream_cleanup(self):
        with tempfile.TemporaryDirectory() as output:
            trainer,complete=self.make_trainer(output,0,30000)
            expected=[r['name'] for r in trainer.val_dataset.rows]
            def train_and_teardown():
                trainer.global_step=30000
                trainer.model.ready=True
                for name in ('train_dataset','train_dataloader','val_dataset','val_dataloader'):
                    delattr(trainer,name)
            trainer.run_training=train_and_teardown
            with contextlib.redirect_stdout(io.StringIO()):result=complete(trainer)
            self.assertEqual(result['step'],30000)
            self.assertEqual([r['source'] for r in result['views']],expected)
            self.assertTrue(np.isfinite(result['psnr']))
            self.assertEqual(trainer.conf.render['render_spp'],1024)
            self.assertTrue((Path(output)/'after.json').exists())

    def test_finished_checkpoint_recovers_validation_without_training_or_saving(self):
        with tempfile.TemporaryDirectory() as output:
            trainer,complete=self.make_trainer(output,30000,30000)
            def forbidden():self.fail('Completed geometry must not train or overwrite its checkpoint')
            trainer.run_training=forbidden;trainer.save_checkpoint=forbidden
            with contextlib.redirect_stdout(io.StringIO()):result=complete(trainer)
            self.assertEqual(result['step'],30000)
            self.assertEqual(len(result['views']),24)
            self.assertEqual(trainer.model.builds,1)


if __name__=='__main__':unittest.main()
