"""Build the material version of the daylight interface and filter display bases."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];P=R/'experiments/out/nucleus4d_material';O=P/'relight';base=R/'experiments/out/nucleus4d_daylight'
m=json.loads((O/'manifest.json').read_text());assert m['complete']
if 'display_filter' not in m:
 g=np.load(base/'geometry_guidance.npz');n=g['normal'];depth=g['depth'];weights=[];den=np.zeros_like(depth)
 for dy in range(-2,3):
  for dx in range(-2,3):
   nn=np.roll(n,(dy,dx),(0,1));dd=np.roll(depth,(dy,dx),(0,1));w=np.exp(-(dx*dx+dy*dy)/4-((n-nn)**2).sum(2)/.08-(depth-dd)**2/.0064)
   if dy>0:w[:dy]=0
   elif dy<0:w[dy:]=0
   if dx>0:w[:,:dx]=0
   elif dx<0:w[:,dx:]=0
   weights.append((dy,dx,w));den+=w
 for item in [*m['skies'].values(),*m['sun']]:
  for key in ['total','direct']:
   f=item[key]['file'];a=np.fromfile(O/f,'<f2').astype('float32').reshape(m['height'],m['width'],3);b=np.zeros_like(a)
   for dy,dx,w in weights:b+=np.roll(a,(dy,dx),(0,1))*w[:,:,None]
   b/=den[:,:,None];name=f.replace('.bin','_view.bin');b.astype('<f2').tofile(O/name);item[key]['raw_file']=f;item[key]['file']=name
 m['display_filter']='5x5 geometry-guided linear filter; source radiance bases retained'
 (O/'manifest.json').write_text(json.dumps(m,indent=2))
s=(base/'index.html').read_text().replace('1406 / 日光实验室','1406 / 材质与日光').replace('真实扫描几何 · 光照原型','照片估计材质 · 换光原型').replace('让室外光进入室内','使用估计材质重新受光').replace('● 含多次漫反射','● 含材质反射与间接光').replace('室内间接反射','室内间接反射')
s=s.replace('先验证光怎样进入房间','材质已由原始照片估计并映射回三维')
s=s.replace('这是 1406-C-int 客厅的扫描网格与修正窗洞，使用示意材质。拖动时间观察光斑与阴影；改变天空，观察室内受光和反射变化。','12 张原始照片用于反照率、粗糙度和金属度估计，2 张留作独立检查。拖动时间与天空滑块观察材质重新受光。<br><a href="../3dgs/">查看原始 3DGS / 估计反照率 / 观测支持</a> · <a href="../material_estimates.jpg">照片与材质分解对照</a> · <a href="http://localhost:8767">上一版示意材质</a>')
s=s.replace('材质为手工指定漫反射色，不是已恢复的真实反照率。该图不是原始 3DGS 的完整物理换光结果。','材质来自 RGB-X 模型先验与可见性加权多视角融合，不是实测 BRDF。独立视角的反照率估计仍存在差异（线性 MAE 约 0.11–0.16），并未验证真实物理准确率。<br>当前机位约 61% 表面像素有直接照片支持，约 17% 邻近补齐；其他区域仍用中性回退。反照率、粗糙度、金属度已经用于本页光线追踪。<br>本页使用扫描网格代理进行换光，并非完整 3DGS 的实时光线追踪；原始高斯材质以独立 sidecar 保存，原始 PLY 不变。')
s=s.replace('CPU 路径追踪预计算','GPU 路径追踪预计算').replace('直接光 + 间接反射','直接光 + 材质间接反射')
(O/'index.html').write_text(s)
(P/'index.html').write_text('<!doctype html><meta http-equiv="refresh" content="0;url=relight/"><a href="relight/">打开材质与日光演示</a>')
(P/'serve.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "$0")"\nexec python3 -m http.server 8768 --bind 127.0.0.1\n');(P/'serve.sh').chmod(0o755)
print('Material UI ready')
