"""Visibility-weighted robust fusion. Predictions are priors, not measured BRDFs."""
import pathlib,json,numpy as np,cv2,warnings
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
P=pathlib.Path(__file__).resolve().parents[1]/'experiments/out/nucleus4d_material'
g=np.load(P/'surface.npz');points=g['vertices'];normals=np.load(P/'surface_normals.npy');views=json.loads((P/'views.json').read_text());train=[v for v in views if v['split']=='train'];N=len(points)
obs=np.full((len(train),N,5),np.nan,'float32');w=np.zeros((len(train),N),'float32')
def samples(v):
 proj=np.load(P/'projections'/f"{v['index']:02}.npz");pr=np.load(P/'priors'/f"{v['index']:02}.npz");uv=proj['uv'];maps=np.concatenate([pr['albedo'],pr['roughness'][:,:,:1],pr['metallic'][:,:,:1]],2).astype('float32');val=np.stack([map_coordinates(maps[:,:,k],[uv[:,1],uv[:,0]],order=1,mode='nearest') for k in range(5)],axis=1)
 return proj['indices'],proj['weights'],val
for j,v in enumerate(train):
 ids,weights,vals=samples(v);ok=weights>.015;ids=ids[ok];w[j,ids]=weights[ok];obs[j,ids]=vals[ok]
with warnings.catch_warnings():
 warnings.simplefilter('ignore',category=RuntimeWarning);median=np.nanmedian(obs,axis=0)
res=np.sqrt(np.nansum((obs[:,:,:3]-median[None,:,:3])**2,axis=2));rw=w/np.maximum(1,res/.15);den=rw.sum(0);counts=(w>0).sum(0);valid=den>0
mat=np.tile([.6,.6,.6,.65,0],(N,1)).astype('float32');mat[valid]=np.nansum(obs*rw[:,:,None],axis=0)[valid]/den[valid,None]
# Fill only small sampling gaps on nearby similarly oriented surfaces.
missing=np.where(~valid)[0];known=np.where(valid)[0];tree=cKDTree(points[known]);dd,kk=tree.query(points[missing],k=12,workers=8);neighbors=known[kk];dot=np.abs((normals[missing,None,:]*normals[neighbors]).sum(2));score=np.where((dd<.18)&(dot>.85),dd,np.inf);best=score.argmin(1);ok=np.isfinite(score[np.arange(len(missing)),best]);filled=missing[ok];src=neighbors[np.arange(len(missing)),best][ok];mat[filled]=mat[src]
confidence=np.minimum(counts/3,1).astype('float32');confidence[filled]=.15
mat[:,:3]=np.clip(mat[:,:3],.01,.95);mat[:,3]=np.clip(mat[:,3],.08,1);mat[:,4]=np.clip(mat[:,4],0,1)
np.savez_compressed(P/'surface_materials.npz',albedo=mat[:,:3],roughness=mat[:,3],metallic=mat[:,4],confidence=confidence,view_count=counts)
report={'method':'RGB-X learned priors + robust visibility-weighted multi-view fusion','train_views':[r['index'] for r in train],'vertices':N,'directly_observed_vertices':int(valid.sum()),'multi_view_vertices':int((counts>=2).sum()),'locally_filled_vertices':len(filled),'unobserved_neutral_vertices':int((confidence==0).sum()),'holdout_consistency':[],'limitations':['No measured albedo or BRDF reference','Does not prove physically unique material-light decomposition','Predictions can hallucinate texture or retain shading','Unobserved surfaces retain neutral fallback','Metallic and roughness are model-dependent priors']}
for v in [v for v in views if v['split']=='holdout']:
 ids,weights,vals=samples(v);ok=valid[ids]&(weights>.015);dif=np.abs(mat[ids[ok]]-vals[ok]);report['holdout_consistency'].append({'view':v['index'],'compared_vertices':int(ok.sum()),'albedo_linear_mae':float(dif[:,:3].mean()),'roughness_mae':float(dif[:,3].mean()),'metallic_mae':float(dif[:,4].mean()),'interpretation':'agreement with an unseen-view model prediction, not ground-truth accuracy'})
(P/'fusion_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
