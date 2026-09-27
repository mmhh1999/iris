"""Oracle relight diagnostic: replace only recovered floor, keep other native materials."""
import argparse,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_sunpatch_benchmark import mi,dr,exr
from evaluate_sunpatch_iris import load_model

class Recovered(mi.BSDF):
    def __init__(self,p,material):
        mi.BSDF.__init__(self,p);self.m_flags=mi.BSDFFlags.DiffuseReflection|mi.BSDFFlags.FrontSide;self.m_components=[self.m_flags]
        # Nearest-point LUT: x/z coordinates aligned explicitly with flat array indices.
        self.fields=[mi.Float(material[...,i].ravel()) for i in range(5)];self.res=material.shape[0]
    def eval(self,ctx,si,wo,active=True):
        x=dr.clip((si.p.x+2.54106)/6.05712*self.res,0,self.res-1)
        z=dr.clip((si.p.z+2.53677)/7.98366*self.res,0,self.res-1)
        idx=mi.UInt32(z)*self.res+mi.UInt32(x)
        vals=[dr.gather(mi.Float,f,idx) for f in self.fields];albedo=mi.Color3f(*vals[:3]);r=dr.maximum(vals[3],.02);m=vals[4]
        ci=si.wi.z;co=wo.z;h=dr.normalize(si.wi+wo);aa=r**4
        D=aa/(dr.pi*dr.square(h.z*h.z*(aa-1)+1));k=(r+1)**2/8
        G=1/((ci*(1-k)+k)*(co*(1-k)+k));ks=.04*(1-m)+albedo*m
        F=ks+(1-ks)*(1-dr.dot(wo,h))**5
        value=albedo*(1-m)/dr.pi*co+D*G*F/4*co
        return dr.select(active&(ci>0)&(co>0),value,0)
    def pdf(self,ctx,si,wo,active=True):return dr.select(active&(si.wi.z>0),dr.maximum(wo.z,0)/dr.pi,0)
    def sample(self,ctx,si,s1,s2,active=True):
        b=mi.BSDFSample3f();b.wo=mi.warp.square_to_cosine_hemisphere(s2);b.pdf=self.pdf(ctx,si,b.wo,active);b.eta=1;b.sampled_type=mi.UInt32(mi.BSDFFlags.DiffuseReflection);b.sampled_component=0
        return b,self.eval(ctx,si,b.wo,active)/dr.maximum(b.pdf,1e-8)
    def eval_pdf(self,ctx,si,wo,active=True):return self.eval(ctx,si,wo,active),self.pdf(ctx,si,wo,active)
    def to_string(self):return 'IRIS recovered floor LUT'

def run(a):
    base=Path(a.base);out=base/a.out;out.mkdir(parents=True,exist_ok=False);data=base/'data_03';rows=[]
    res=512;x=-2.54106+(np.arange(res)+.5)/res*6.05712;z=-2.53677+(np.arange(res)+.5)/res*7.98366
    xx,zz=np.meshgrid(x,z);p=np.stack([xx,np.full_like(xx,-.020776),zz],-1).reshape(-1,3).astype(np.float32)
    for condition in a.conditions:
        m,ck=load_model(base/('iris_'+condition+a.run_suffix),'last_1')
        with torch.no_grad():mat=m(torch.tensor(p,device='cuda'));lut=torch.cat([mat['albedo'],mat['roughness'],mat['metallic']],-1).cpu().numpy().reshape(res,res,5)
        scale=1.0
        if a.scale_align:
            numerator=0.;denominator=0.
            manifest=json.loads((data/'manifest.json').read_text())
            views=next(c['views'] for c in manifest['conditions'] if c['name']==condition)
            for v in views:
                if v['split']!='train':continue
                gt=np.load(data/'generator_truth'/f'{condition}_view{v["view"]:02d}.npz')
                mask=gt['floor']&~gt['sun_visible']
                if not mask.any():continue
                with torch.no_grad():al=m(torch.tensor(gt['position'][mask],device='cuda'))['albedo'].cpu().numpy()
                numerator+=float((al*np.array([.28,.20,.12])).sum());denominator+=float((al*al).sum())
            scale=numerator/max(denominator,1e-8);lut[...,:3]=np.clip(lut[...,:3]*scale,0,1)
        np.save(out/(condition+'_floor_lut.npy'),lut)
        mi.register_bsdf('iris_recovered',lambda props:Recovered(props,lut))
        for view in [3,9]:
            source=data/'generator_truth'/f'sun_c_view{view:02d}.xml';root=ET.parse(source)
            node=next(x for x in root.getroot().iter('bsdf') if x.get('id')=='FloorBSDF');node.set('type','iris_recovered')
            xml=out/f'{condition}_view{view}.xml';root.write(xml);scene=mi.load_file(str(xml.resolve()))
            images=[]
            for seed in [101,202]:
                im=np.array(mi.render(scene,spp=a.spp,seed=seed));exr(out/f'{condition}_view{view}_seed{seed}.exr',im);images.append(im)
            # Re-render truth at matched spp/seeds, preserving raw noise estimates.
            truth_scene=mi.load_file(str(source.resolve()));truth=[]
            for seed in [101,202]:
                im=np.array(mi.render(truth_scene,spp=a.spp,seed=seed));exr(out/f'truth_view{view}_seed{seed}.exr',im);truth.append(im)
            gt=np.load(data/'generator_truth'/f'{condition}_view{view:02d}.npz');mask=gt['floor'];old=gt['sun_visible']
            avg=np.mean(images,axis=0);ref=np.mean(truth,axis=0)
            row={'condition':condition,'view':view,'albedo_scale':scale,'floor_hdr_rmse':float(np.sqrt(((avg[mask]-ref[mask])**2).mean())),
                 'old_patch_hdr_rmse':float(np.sqrt(((avg[old]-ref[old])**2).mean())) if old.any() else None,
                 'render_seed_rmse_floor':float(np.sqrt(((images[0][mask]-images[1][mask])**2).mean())),
                 'truth_seed_rmse_floor':float(np.sqrt(((truth[0][mask]-truth[1][mask])**2).mean()))}
            rows.append(row);print(json.dumps(row),flush=True)
    (out/'report.json').write_text(json.dumps({'scope':'oracle floor-only physical relight under heldout sun_c; geometry, other materials, lighting are true; not end-to-end relighting',
        'scale_alignment':bool(a.scale_align),'scale_alignment_source':'single scalar fitted against known floor albedo on training-view non-direct floor pixels, evaluation only',
        'lut_resolution':res,'lut_sampling':'nearest, roughly 1.2x1.6 cm cells','spp':a.spp,'seeds':[101,202],
        'sampling_caveat':'cosine sampling for recovered BRDF; sharp recovered lobes can have high variance','views':rows},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='experiments/out/EXP0019_sunpatch');p.add_argument('--scale-align',action='store_true');p.add_argument('--run-suffix',default='_v2');p.add_argument('--out',default='relight_01');p.add_argument('--conditions',nargs='+',default=['sun_a','sun_b','sky']);p.add_argument('--spp',type=int,default=4096);run(p.parse_args())
