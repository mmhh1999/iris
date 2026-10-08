"""Ray-traced material daylight demo. Original 3DGS is preserved separately."""
import os,pathlib,sys,json,numpy as np,mitsuba as mi,drjit as dr,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import nucleus4d_daylight as d
mi.set_variant('cuda_ad_rgb')
P=ROOT/'experiments/out/nucleus4d_material';O=P/'relight';O.mkdir(exist_ok=True)
g=np.load(P/'surface.npz');a=np.load(P/'surface_materials.npz')
props=mi.Properties();props['bsdf']=mi.load_dict({'type':'twosided','bsdf':{'type':'principled','base_color':{'type':'mesh_attribute','name':'vertex_albedo'},'roughness':{'type':'mesh_attribute','name':'vertex_roughness'},'metallic':{'type':'mesh_attribute','name':'vertex_metallic'}}})
mesh=mi.Mesh('estimated_material_surface',len(g['vertices']),len(g['faces']),props,has_vertex_normals=False)
params=mi.traverse(mesh);params['vertex_positions']=g['vertices'].ravel();params['faces']=g['faces'].ravel();params.update()
for name,channels,data in [('albedo',3,a['albedo']),('roughness',1,a['roughness']),('metallic',1,a['metallic']),('confidence',1,a['confidence'])]:mesh.add_attribute('vertex_'+name,channels,data.ravel())
mesh.write_ply(str(P/'estimated_material_mesh.ply'))
d.G={'recovered':mesh};d.OUT=O

def aovs():
 s=d.scene(d.sky('clear'));sensor=s.sensors()[0];u,v=np.meshgrid((np.arange(d.W)+.5)/d.W,(np.arange(d.H)+.5)/d.H)
 ray,_=sensor.sample_ray(0,0,mi.Point2f(np.stack([u.ravel(),v.ravel()])),mi.Point2f(0))
 si=s.ray_intersect(ray);valid=np.array(si.is_valid()).reshape(d.H,d.W);maps={}
 for name,n in [('albedo',3),('roughness',1),('metallic',1),('confidence',1)]:
  val=si.shape.eval_attribute_3('vertex_'+name,si) if n==3 else si.shape.eval_attribute_1('vertex_'+name,si)
  x=np.array(val).T.reshape(d.H,d.W,n);x[~valid]=0;maps[name]=x
  mi.util.write_bitmap(str(O/(name+'.png')),np.repeat(x,3,2) if n==1 else x,write_async=False)
 np.savez_compressed(O/'material_buffers.npz',**maps,valid=valid)
 (O/'visible_coverage.json').write_text(json.dumps({'surface_pixels':int(valid.sum()),'supported_or_local_fill_fraction':float((maps['confidence'][:,:,0][valid]>0).mean()),'direct_evidence_fraction':float((maps['confidence'][:,:,0][valid]>=.333).mean())},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');args=ap.parse_args();aovs()
 if args.quick:
  l,_=d.sun(13);s=mi.load_dict({'type':'scene','integrator':{'type':'path','max_depth':8},'sensor':d.sensor(),'recovered':mesh,'sun':l,'sky':d.sky('clear')});x=np.array(mi.render(s,spp=512,seed=3));d.save('quick',x);mi.util.write_bitmap(str(O/'quick_exposed.png'),x*8/(1+x*8),write_async=False);sys.exit()
 manifest=json.loads((ROOT/'experiments/out/nucleus4d_daylight/manifest.json').read_text());manifest.pop('complete',None);manifest.pop('display_filter',None);manifest['sun']=[];manifest['skies']={};manifest['assumptions']['materials']='RGB-X learned albedo/roughness/metallic priors from 12 original photos, robust multi-view visibility fusion; unobserved surfaces neutral fallback; not measured BRDF';manifest['scene']='1406-C-int — estimated material relighting';manifest['material_report']='../fusion_report.json'
 for kind in ['clear','overcast']:
  l=d.sky(kind);total=d.cached('sky_'+kind,d.scene(l),2048,11);direct=d.cached('sky_'+kind+'_direct',d.scene(l,2),1024,11)
  manifest['skies'][kind]={'total':d.save('sky_'+kind,total),'direct':d.save('sky_'+kind+'_direct',direct)};print('sky',kind,flush=True)
 for j,hour in enumerate(np.arange(8,18.01,.5)):
  l,meta=d.sun(float(hour));total=d.cached(f'sun_{j:02}',d.scene(l),1024,23);direct=d.cached(f'sun_{j:02}_direct',d.scene(l,2),256,23)
  meta.update(total=d.save(f'sun_{j:02}',total),direct=d.save(f'sun_{j:02}_direct',direct));manifest['sun'].append(meta);print('sun',hour,flush=True)
 l,_=d.sun(13);blk=np.array(mi.render(d.scene(l,closed=True),spp=128,seed=23));skyb=np.array(mi.render(d.scene(d.sky('clear'),closed=True),spp=128,seed=23));assert max(blk.max(),skyb.max())<1e-6
 manifest['validation']={'sun_blocked_mean':float(blk.mean()),'sky_blocked_mean':float(skyb.mean())};manifest['complete']=True
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE',flush=True)
