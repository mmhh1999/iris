"""Fixed-view daylight transport prototype on a locally repaired Nucleus4D scan.
All materials, geographic alignment, sky radiance and glazing are assumptions.
Outputs linear HDR lighting bases; the browser combines them before tone mapping.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-nucleus')
import sys,json,pathlib,datetime,argparse
import numpy as np,trimesh,mitsuba as mi,drjit as dr
from zoneinfo import ZoneInfo
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.solar_geometry import solar_position,sun_vector_world
OUT=ROOT/'experiments/out/nucleus4d_daylight';OUT.mkdir(exist_ok=True,parents=True)
mi.set_variant('llvm_ad_rgb');dr.set_thread_count(8)
W,H=640,416
EYE=[1.4,1.7,.45];TARGET=[5.65,-.35,-.45]

def diffuse(rgb):return {'type':'twosided','bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':rgb}}}
def box(lo,hi,rgb):
 lo=np.array(lo);hi=np.array(hi)
 return {'type':'cube','to_world':mi.ScalarTransform4f.translate((lo+hi)/2).scale((hi-lo)/2),'bsdf':diffuse(rgb)}
def geometry():
 m=trimesh.load(OUT/'scan.obj',process=False);c=m.triangles_center
 # Isolate the living room. Retain scan furniture; replace its enclosing shell.
 keep=((c>[-1.65,-2.38,-1.13])&(c<[5.72,2.32,1.30])).all(1)
 m.update_faces(keep);m.remove_unreferenced_vertices();m.export(OUT/'living_scan.obj')
 g={'scan':{'type':'obj','filename':str(OUT/'living_scan.obj'),'bsdf':diffuse([.55,.53,.50])}}
 g['floor']=box([-1.8,-2.5,-1.24],[6.0,2.5,-1.18],[.48,.40,.30])
 g['ceiling']=box([-1.8,-2.5,1.38],[6,2.5,1.44],[.75,.75,.73])
 g['west']=box([-1.86,-2.5,-1.18],[-1.8,2.5,1.44],[.65,.64,.60])
 g['north']=box([-1.8,2.45,-1.18],[6,2.51,1.44],[.68,.67,.64])
 g['south']=box([-1.8,-2.56,-1.18],[6,-2.5,1.44],[.65,.64,.60])
 # Approximate apertures measured from the scan's colored facade elevation.
 g['east_bottom']=box([5.98,-2.5,-1.18],[6.06,2.5,-.55],[.72,.70,.66])
 g['east_top']=box([5.98,-2.5,.60],[6.06,2.5,1.44],[.72,.70,.66])
 for i,(a,b) in enumerate([(-2.5,-2.10),(-1.38,.78),(1.50,2.5)]):
  g[f'east_pier{i}']=box([5.98,a,-.55],[6.06,b,.60],[.72,.70,.66])
 for i,(a,b) in enumerate([(-2.10,-1.38),(.78,1.50)]):
  g[f'crossbar{i}']=box([5.96,a,.07],[6.10,b,.11],[.72,.72,.72])
 return g
G=geometry()
def sensor():return {'type':'perspective','fov':77,'to_world':mi.ScalarTransform4f.look_at(origin=EYE,target=TARGET,up=[0,0,1]),'film':{'type':'hdrfilm','width':W,'height':H,'rfilter':{'type':'box'}},'sampler':{'type':'independent'}}
def scene(light,depth=8,closed=False):
 g=dict(G)
 if closed:g['blackout']=box([5.975,-2.5,-.7],[6.065,2.5,.8],[0,0,0])
 return mi.load_dict({'type':'scene','integrator':{'type':'path','max_depth':depth,'rr_depth':5},'sensor':sensor(),**g,'outside':light})
def save(name,a):
 assert np.isfinite(a).all() and a.min()>-1e-5
 np.maximum(a,0).astype('<f2').tofile(OUT/(name+'.bin'))
 mi.util.write_bitmap(str(OUT/(name+'.png')),a/(1+a))
 return {'file':name+'.bin','mean':float(a.mean()),'max':float(a.max())}
def cached(name,sc,spp,seed):
 p=OUT/(name+'.bin')
 if p.exists() and p.stat().st_size==W*H*3*2:return np.fromfile(p,dtype='<f2').astype('float32').reshape(H,W,3)
 return np.array(mi.render(sc,spp=spp,seed=seed))

def sky(kind):
 yy=(np.arange(128)+.5)/128;z=np.cos(np.pi*yy)[:,None,None]
 if kind=='clear':rgb=np.array([.10,.17,.29])[None,None,:]*(.55+.45*np.maximum(z,0))
 else:rgb=np.array([.23,.24,.25])[None,None,:]*(1+2*np.maximum(z,0))/3
 rgb=np.where(z<0,np.array([.025,.022,.018])[None,None,:],rgb)
 a=np.repeat(rgb,256,axis=1).astype('float32');mi.util.write_bitmap(str(OUT/f'{kind}_sky.exr'),a,write_async=False)
 return {'type':'envmap','filename':str(OUT/f'{kind}_sky.exr'),'to_world':mi.ScalarTransform4f.rotate([1,0,0],90)}
def sun(hour):
 t=datetime.datetime(2026,3,21,int(hour),int(round(hour%1*60)),tzinfo=ZoneInfo('America/Los_Angeles'))
 s=solar_position(37.7749,-122.4194,t);v=sun_vector_world(s.azimuth_deg,s.elevation_deg,90)
 warm=np.clip(s.elevation_deg/35,0,1);color=np.array([1,.72+.24*warm,.42+.44*warm]);power=5*np.clip(np.sin(np.deg2rad(s.elevation_deg))/.65,0,1)
 return {'type':'directional','direction':(-v).tolist(),'irradiance':{'type':'rgb','value':(color*power).tolist()}},{'hour':hour,'azimuth_deg':s.azimuth_deg,'elevation_deg':s.elevation_deg,'sun_vector':v.tolist()}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');args=ap.parse_args()
 manifest={'width':W,'height':H,'format':'little-endian float16 RGB, linear radiance','scene':'1406-C-int living-room geometry prototype','assumptions':{'location':'San Francisco (example only)','date':'2026-03-21','timezone':'America/Los_Angeles','scene_north_bearing_deg':90,'materials':'assigned diffuse colors; not recovered albedo','geometry':'cropped scan furniture plus repaired closed shell and two approximate windows; other rooms excluded','glazing':'open apertures, no glass transmission model','sky':'synthetic linear HDR environment maps, not measured sky','time':'30-minute precomputed light transport samples, slider snaps to samples','rendering':'CPU Mitsuba path tracing; browser combines linear light bases, not real-time ray tracing','camera':{'eye':EYE,'target':TARGET}},'skies':{},'sun':[]}
 if args.quick:
  l,_=sun(13);a=np.array(mi.render(scene(l),spp=64,seed=3));save('quick',a);sys.exit()
 for kind in ['clear','overcast']:
  l=sky(kind);a=cached('sky_'+kind,scene(l),1024,11);b=cached('sky_'+kind+'_direct',scene(l,2),512,11)
  manifest['skies'][kind]={'total':save('sky_'+kind,a),'direct':save('sky_'+kind+'_direct',b)};print('sky',kind,flush=True)
 for j,hour in enumerate(np.arange(8,18.01,.5)):
  l,meta=sun(float(hour));a=cached(f'sun_{j:02}',scene(l),256,23);b=cached(f'sun_{j:02}_direct',scene(l,2),64,23)
  meta.update(total=save(f'sun_{j:02}',a),direct=save(f'sun_{j:02}_direct',b));manifest['sun'].append(meta)
  (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('sun',hour,meta['total']['mean'],flush=True)
 l,_=sun(13);blocked=np.array(mi.render(scene(l,closed=True),spp=256,seed=23));skyblocked=np.array(mi.render(scene(sky('clear'),closed=True),spp=256,seed=23))
 control={'sun_blocked_mean':float(blocked.mean()),'sky_blocked_mean':float(skyblocked.mean()),'reason':'opaque exterior shutter covers both apertures; remaining illumination must be zero','sun_open_mean':manifest['sun'][10]['total']['mean']}
 assert blocked.max()<1e-6 and skyblocked.max()<1e-6,control
 manifest['validation']=control;manifest['complete']=True
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE',control,flush=True)
