"""Geometry-guided linear display filter. Raw radiance bases remain available."""
import pathlib,json,sys,numpy as np,mitsuba as mi
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import nucleus4d_daylight as d
p=d.OUT
if 'display_filter' in json.loads((p/'manifest.json').read_text()):
 print('Display filter already applied');sys.exit()
s=mi.load_dict({'type':'scene','integrator':{'type':'aov','aovs':'normal:sh_normal,depth:depth','integrator':{'type':'path','max_depth':1}},'sensor':d.sensor(),**d.G})
g=np.array(mi.render(s,spp=1,seed=7));print('guidance',g.shape,flush=True)
n=g[:,:,3:6];depth=g[:,:,6];np.savez_compressed(p/'geometry_guidance.npz',normal=n,depth=depth)
weights=[];den=np.zeros((d.H,d.W),np.float32)
for dy in range(-2,3):
 for dx in range(-2,3):
  nn=np.roll(n,(dy,dx),(0,1));dd=np.roll(depth,(dy,dx),(0,1))
  w=np.exp(-(dx*dx+dy*dy)/4-((n-nn)**2).sum(2)/.08-(depth-dd)**2/.0064)
  if dy>0:w[:dy]=0
  elif dy<0:w[dy:]=0
  if dx>0:w[:,:dx]=0
  elif dx<0:w[:,dx:]=0
  weights.append((dy,dx,w));den+=w
for f in sorted(p.glob('*.bin')):
 if f.stem.endswith('_view') or f.stem=='quick':continue
 a=np.fromfile(f,'<f2').astype('float32').reshape(d.H,d.W,3);b=np.zeros_like(a)
 for dy,dx,w in weights:b+=np.roll(a,(dy,dx),(0,1))*w[:,:,None]
 b/=den[:,:,None];b.astype('<f2').tofile(f.with_name(f.stem+'_view.bin'))
m=json.loads((p/'manifest.json').read_text());m['display_filter']='5x5 geometry-guided linear filter; raw bases retained; can soften shadow edges by up to 2 pixels'
for item in [*m['skies'].values(),*m['sun']]:
 for k in ['total','direct']:
  item[k]['raw_file']=item[k]['file'];item[k]['file']=item[k]['file'].replace('.bin','_view.bin')
(p/'manifest.json').write_text(json.dumps(m,indent=2));print('Filtered display bases ready',flush=True)
