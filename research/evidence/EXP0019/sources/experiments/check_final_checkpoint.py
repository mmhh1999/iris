"""CPU regression check for final weights when the monitored metric worsens."""
from pathlib import Path
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader,TensorDataset
out=Path('experiments/out/EXP0019_sunpatch/checkpoint_regression');out.mkdir(parents=True,exist_ok=False)
class Model(pl.LightningModule):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
    def training_step(self,batch,index):return self.weight.square()
    def validation_step(self,batch,index):self.log('val/loss',float(self.current_epoch),batch_size=1)
    def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.1)
    def train_dataloader(self):return DataLoader(TensorDataset(torch.ones(1)),batch_size=1)
    def val_dataloader(self):return self.train_dataloader()
m=Model();callback=ModelCheckpoint(out,monitor='val/loss',save_top_k=1,save_last=True)
t=pl.Trainer(accelerator='cpu',devices=1,max_epochs=3,logger=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,callbacks=[callback]);t.fit(m)
t.save_checkpoint(out/'final.ckpt')
f=torch.load(out/'final.ckpt',weights_only=False);last=torch.load(callback.last_model_path,weights_only=False)
assert f['global_step']==t.global_step==3
torch.testing.assert_close(f['state_dict']['weight'],m.weight.detach())
(out/'report.json').write_text(json.dumps({'lightning_version':pl.__version__,'trainer_final_step':t.global_step,'callback_last_step':last['global_step'],'explicit_final_step':f['global_step'],'final_weight_matches_live_model':True},indent=2));print((out/'report.json').read_text())
