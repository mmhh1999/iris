"""Oracle material-family diagnostic. No photo inverse-rendering claim."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.brdf import BaseBRDF


def run(a):
    src=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source.py')
    manifest=json.loads((src/'manifest.json').read_text())
    dirs=torch.tensor(manifest['outgoing_directions_local'],device='cuda',dtype=torch.float32)
    train=[0,1,3,5,7];test=[2,4,6,8];rows=[];model=BaseBRDF()
    for v in manifest['views']:
        z=np.load(src/v['file']);cam=z['camera_local']; mask=z['valid']&~z['has_delta']&~z['has_null']&(cam[...,2]>.1)
        eligible=np.flatnonzero(mask);idx=np.random.default_rng(17).choice(eligible,min(2048,len(eligible)),replace=False)
        target=torch.tensor(z['native_response'].reshape(9,-1,3)[:,idx],device='cuda')
        wo=torch.tensor(cam.reshape(-1,3)[idx],device='cuda');n=len(idx)
        normal=torch.zeros(9*n,3,device='cuda');normal[:,2]=1
        wi=dirs[:,None,:].expand(9,n,3).reshape(-1,3);wo9=wo[None].expand(9,n,3).reshape(-1,3)
        if a.matched_control:
            rng=np.random.default_rng(31)
            truth=torch.tensor(rng.uniform(.1,.9,(n,5)),device='cuda',dtype=torch.float32)
            tm={'albedo':truth[:,:3],'roughness':truth[:,3:4],'metallic':truth[:,4:5]}
            te={k:x[None].expand(9,*x.shape).reshape(-1,x.shape[-1]) for k,x in tm.items()}
            target=model.eval_brdf(wi,wo9,normal,te)[0].reshape(9,n,3).detach()
        scale=target[train].square().mean((0,2)).sqrt().clamp_min(.01)
        best_loss=torch.full((n,),float('inf'),device='cuda');best=None
        for seed in [17,23,41]:
            torch.manual_seed(seed);p=torch.nn.Parameter(torch.randn(n,5,device='cuda')*.5)
            opt=torch.optim.Adam([p],lr=.05)
            for step in range(a.steps):
                mat=p.sigmoid();params={'albedo':mat[:,:3],'roughness':mat[:,3:4]*.98+.02,'metallic':mat[:,4:5]}
                expanded={k:x[None].expand(9,*x.shape).reshape(-1,x.shape[-1]) for k,x in params.items()}
                prediction=model.eval_brdf(wi,wo9,normal,expanded)[0].reshape(9,n,3)
                loss=((prediction[train]-target[train])/scale[None,:,None]).square().mean((0,2))
                if best is None:best=prediction.detach().clone()
                improved=loss.detach()<best_loss
                best[:,improved]=prediction.detach()[:,improved];best_loss=torch.minimum(best_loss,loss.detach())
                opt.zero_grad();loss.mean().backward();opt.step()
            print(v['room'],v['view'],'seed',seed,'finished',flush=True)
        def metrics(ids):
            err=(best[ids]-target[ids]).square().mean((0,2)).sqrt()
            denom=target[ids].square().mean((0,2)).sqrt().clamp_min(.01)
            return {'rmse':float((best[ids]-target[ids]).square().mean().sqrt()),
                    'relative_rmse':float((best[ids]-target[ids]).square().mean().sqrt()/target[ids].square().mean().sqrt().clamp_min(.01)),
                    'median_pixel_relative_rmse':float((err/denom).median()),
                    'p90_pixel_relative_rmse':float(torch.quantile(err/denom,.9))}
        row={'room':v['room'],'view':v['view'],'eligible_fraction':float(mask.mean()),'samples':n,'fit':metrics(train),'heldout_angles':metrics(test)}
        rows.append(row);np.savez_compressed(out/(Path(v['file']).stem+'_fit.npz'),pixel_index=idx,target=target.cpu().numpy(),prediction=best.cpu().numpy())
        print(json.dumps(row),flush=True)
    report={'purpose':'Oracle native-response fit; assesses material-family mismatch only',
            'matched_iris_control':a.matched_control,'uses_ground_truth':True,'is_photo_estimation':False,'is_proven_error_lower_bound':False,
            'train_directions':train,'heldout_directions':test,'steps':a.steps,'seeds':[17,23,41],
            'selection':'per-pixel minimum training loss across steps/seeds; heldout angles not used',
            'excluded':'invalid, any delta/null lobe, camera local z <= 0.1',
            'roughness_range':[.02,1.0],'views':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='experiments/out/EXP0018_native_brdf/run_03');p.add_argument('--out',default='experiments/out/EXP0018_native_brdf/fit_01');p.add_argument('--matched-control',action='store_true');p.add_argument('--steps',type=int,default=400);run(p.parse_args())
