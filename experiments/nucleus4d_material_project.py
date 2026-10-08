"""Visibility-tested projection of original camera views onto the daylight surface."""
import pathlib,json,numpy as np,trimesh,mitsuba as mi,drjit as dr
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'experiments/out/nucleus4d_material';(P/'projections').mkdir(exist_ok=True)
mi.set_variant('cuda_ad_rgb')
g=np.load(P/'surface.npz');points=g['vertices'];mesh=trimesh.Trimesh(points,g['faces'],process=False);normals=mesh.vertex_normals.astype('float32')
np.save(P/'surface_normals.npy',normals)
scene=mi.load_dict({'type':'scene','scan':{'type':'obj','filename':str(ROOT/'experiments/out/nucleus4d_daylight/scan.obj')}})
rows=json.loads((P/'views.json').read_text());records=[]
for r in rows:
 eye=np.array(r['position'],dtype='float32');R=np.array(r['rotation']);d=points-eye;dist=np.linalg.norm(d,axis=1);v=d/np.maximum(dist[:,None],1e-8);cp=d@R
 uv=cp[:,:2]/np.maximum(cp[:,2:],1e-5)*350+256
 frustum=(cp[:,2]>.15)&(uv[:,0]>3)&(uv[:,0]<508)&(uv[:,1]>3)&(uv[:,1]<508);idx=np.where(frustum)[0]
 ray=mi.Ray3f(mi.Point3f(eye.tolist()),mi.Vector3f(v[idx].T));si=scene.ray_intersect(ray);t=np.array(si.t)
 err=np.abs(t-dist[idx]);valid=np.isfinite(t)&(err<.12);ids=idx[valid]
 weight=np.abs((normals[ids]*v[ids]).sum(1))**2;weight*=np.exp(-(err[valid]/.08)**2);weight/=np.maximum(dist[ids],.5)**.5
 np.savez_compressed(P/'projections'/f"{r['index']:02}.npz",indices=ids,uv=uv[ids].astype('float32'),weights=weight.astype('float32'))
 records.append({'view':r['index'],'visible_vertices':len(ids),'median_depth_error':float(np.median(err[valid])) if len(ids) else None});print(records[-1],flush=True)
(P/'projection_report.json').write_text(json.dumps(records,indent=2))
