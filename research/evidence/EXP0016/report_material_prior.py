"""Summarize the frozen three-seed pilot without selecting on test results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/iris-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.train_interiorverse_prior import MaterialPrior


def run(out):
    results=json.loads((out/'results.json').read_text())
    keys=['albedo_mse','roughness_mse','metallic_mse']
    stats={}
    for k in keys:
        vals=np.array([r['test'][k] for r in results['runs']]);b=results['constant_train_mean_baseline'][k]
        stats[k]={'baseline':b,'mean':float(vals.mean()),'seed_std':float(vals.std(ddof=1)),
                  'relative_mse_reduction':float(1-vals.mean()/b)}
    (out/'summary.json').write_text(json.dumps(stats,indent=2))
    data=torch.load(ROOT/'data_download/interiorverse/material_cache_128.pt',weights_only=False,map_location='cpu')
    # First predeclared seed, not the best test-scoring seed.
    seed=results['runs'][0]['seed'];checkpoint=torch.load(out/f'seed_{seed}.pt',weights_only=False,map_location='cpu')
    model=MaterialPrior();model.load_state_dict(checkpoint['model']);model.eval();torch.set_num_threads(4)
    selected={}
    for i,r in enumerate(data['records']):
        if r['split']=='test' and r['scene'] not in selected:selected[r['scene']]=i
    fig,axes=plt.subplots(len(selected),5,figsize=(14,2.4*len(selected)),squeeze=False)
    for row,(scene,i) in enumerate(selected.items()):
        with torch.no_grad():pred=model(data['x'][i:i+1].float())[0].numpy().transpose(1,2,0)
        gt=data['y'][i].float().numpy().transpose(1,2,0)
        rgb=np.expm1(data['x'][i].float().numpy().transpose(1,2,0)*np.log(65))
        panels=[np.clip(rgb,0,1)**(1/2.2),np.clip(gt[:,:,:3],0,1)**(1/2.2),
                np.clip(pred[:,:,:3],0,1)**(1/2.2),gt[:,:,3],pred[:,:,3]]
        for col,(title,p) in enumerate(zip(['Input HDR preview','GT albedo','Predicted albedo','GT roughness','Predicted roughness'],panels)):
            axes[row,col].imshow(p,cmap='gray',vmin=0,vmax=1);axes[row,col].set_axis_off()
            if row==0:axes[row,col].set_title(title)
        axes[row,0].text(0,-.08,scene[-18:],transform=axes[row,0].transAxes,fontsize=8)
    fig.suptitle('EXP0016: supervised material-prior pilot; heldout scenes; seed 17; not a solar method')
    fig.tight_layout();fig.savefig(out/'heldout_predictions.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for r in results['runs']:
        h=json.loads((out/f'seed_{r["seed"]}_history.json').read_text())
        ax.plot([v['epoch'] for v in h],[v['val']['selection_score'] for v in h],label=f'seed {r["seed"]}')
    ax.set(xlabel='Epoch',ylabel='Validation scene-macro material MSE');ax.legend();fig.tight_layout()
    fig.savefig(out/'validation_curves.png',dpi=150);plt.close(fig)
    evidence=ROOT/'research/evidence/EXP0016';evidence.mkdir(parents=True,exist_ok=True)
    for path in out.glob('*.json'):shutil.copy2(path,evidence/path.name)
    for path in out.glob('*.png'):shutil.copy2(path,evidence/path.name)
    shutil.copy2(out/'source.py',evidence/'train_interiorverse_prior.py')
    shutil.copy2(__file__,evidence/'report_material_prior.py')
    for n in ['status.json','run.log']:shutil.copy2(out.parent/'job_01'/n,evidence/n)
    shutil.copy2(ROOT/'experiments/watch_research_run.py',evidence/'watch_research_run.py')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'experiments/out/EXP0016_material_prior/run_01')
    run(p.parse_args().out)
