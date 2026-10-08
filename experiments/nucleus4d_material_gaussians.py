"""Transfer estimated surface materials to original Gaussian IDs without changing source data."""
import pathlib,json,zipfile,numpy as np,shutil,re
from scipy.spatial import cKDTree
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'experiments/out/nucleus4d_material';V=P/'3dgs';V.mkdir(exist_ok=True)
g=np.load(P/'surface.npz');a=np.load(P/'surface_materials.npz');normals=np.load(P/'surface_normals.npy');tree=cKDTree(g['vertices'])
def match(pos):
 dist,ids=tree.query(pos,k=1,workers=8);region=((pos>[-1.8,-2.6,-1.3])&(pos<[6.2,2.6,1.5])).all(1);ok=region&(dist<.12)&(a['confidence'][ids]>0)
 return ids,ok,dist
parts=[];offset=0
with zipfile.ZipFile(ROOT/'data_download/nucleus4d_20260929/1406-int.zip') as z:
 member=next(n for n in z.namelist() if n.endswith('/point_cloud.ply'))
 with z.open(member) as f:
  while f.readline().strip()!=b'end_header':pass
  while True:
   data=f.read(68*200000)
   if not data:break
   cloud=np.frombuffer(data,'<f4').reshape(-1,17);ids,ok,dist=match(cloud[:,:3]);ii=np.where(ok)[0]
   parts.append({'source_indices':(ii+offset).astype('uint32'),'albedo':a['albedo'][ids[ii]].astype('float16'),'roughness':a['roughness'][ids[ii]].astype('float16'),'metallic':a['metallic'][ids[ii]].astype('float16'),'normal':normals[ids[ii]].astype('float16'),'confidence':(a['confidence'][ids[ii]]*np.exp(-(dist[ii]/.12)**2)).astype('float16')});offset+=len(cloud)
merged={key:np.concatenate([p[key] for p in parts]) for key in parts[0]};np.savez_compressed(P/'gaussian_materials.npz',**merged)
# Preview splats keep their original positions, covariance, rotation and opacity.
source=ROOT/'experiments/out/nucleus4d_3dgs';raw=np.fromfile(source/'scene.splat',dtype='uint8').reshape(-1,32);pos=raw[:,:12].copy().view('<f4').reshape(-1,3);ids,ok,dist=match(pos)
colors=np.full((len(raw),3),.72,dtype='float32');colors[ok]=a['albedo'][ids[ok]]**(1/2.2);out=raw.copy();out[:,24:27]=np.rint(np.clip(colors,0,1)*255).astype('uint8');out.tofile(V/'albedo.splat')
conf=np.zeros(len(raw));conf[ok]=a['confidence'][ids[ok]];rgb=np.tile([.22,.24,.28],(len(raw),1));rgb[conf>.2]=[.12,.8,.48];rgb[(conf>0)&(conf<=.2)]=[.92,.6,.13];out[:,24:27]=(rgb*255).astype('uint8');out.tofile(V/'confidence.splat')
link=V/'original.splat'
if not link.exists():link.symlink_to(source/'scene.splat')
assert np.array_equal(raw[:,:24],out[:,:24]) and np.array_equal(raw[:,27:],out[:,27:])
shutil.copy(source/'LICENSE',V/'LICENSE')
s=(source/'main.js').read_text();eye=np.array([1.4,1.7,.45]);f=np.array([5.65,-.35,-.45])-eye;f/=np.linalg.norm(f);right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(f,right)
camera={'id':0,'img_name':'living','width':1600,'height':1040,'fx':1006,'fy':1006,'position':eye.tolist(),'rotation':np.stack([right,down,f],axis=1).tolist()}
s=re.sub(r'let cameras = \[.*?\];','let cameras = '+json.dumps([camera])+';',s,count=1,flags=re.S)
s=s.replace('params.get("quality") === "full" ? "full.splat" : "scene.splat"','({original:"original.splat",albedo:"albedo.splat",confidence:"confidence.splat"})[params.get("mode")] || "original.splat"');(V/'main.js').write_text(s)
html=(source/'index.html').read_text();html=re.sub(r'<div id="info">.*?</div><div id="progress">','''<div id="info"><h2>1406-C-int · 材质检查</h2><p>真实 3DGS 几何与透明度保持不变<br>反照率是原始照片的模型估计与多视角融合，非实测真值。</p><a href="?mode=original">原始外观</a> <a href="?mode=albedo">估计反照率</a> <a href="?mode=confidence">观测支持</a><p>绿色：照片直接支持 · 金色：邻近补齐<br>灰色：未覆盖，不能宣称已恢复。</p><button onclick="selectView(0)">返回客厅</button> <a href="../relight/">日光滑块</a><p>左键旋转 · 右键移动 · 方向键漫游</p></div><div id="progress">''',html,flags=re.S);(V/'index.html').write_text(html)
report={'source_archive':'1406-int.zip','source_member':member,'source_gaussians':offset,'estimated_gaussians':len(merged['source_indices']),'scope':'living-room Gaussians within 12 cm of supported reconstructed surface','sidecar':'gaussian_materials.npz','indexing':'zero-based original PLY vertex order; not sorted viewer splat order','original_untouched':True,'normal_source':'proxy surface, not measured per-Gaussian normals','preview_covariance_opacity_unchanged':True}
(P/'gaussian_material_report.json').write_text(json.dumps(report,indent=2));print(report)
