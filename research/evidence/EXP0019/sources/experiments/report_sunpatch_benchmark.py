"""Offline Chinese result browser with raw and normalized material views."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
from pathlib import Path
import json,shutil
import cv2
import numpy as np
b=Path('experiments/out/EXP0019_sunpatch');data=b/'data_03';evaluation=b/'eval_02';fig=evaluation/'figures';fig.mkdir(exist_ok=True)
report=json.loads((evaluation/'report.json').read_text());meta=json.loads((data/'manifest.json').read_text())
for cond in ['sun_a','sun_b','sky']:
    for view in [3,9]:
        v=next(v for c in meta['conditions'] if c['name']==cond for v in c['views'] if v['view']==view)
        prefix=f'{cond}_{view}'
        shutil.copy2(data/cond/'val/Image'/f"{v['index']:03d}_0001.png",fig/(prefix+'_input.png'))
        a=np.load(evaluation/f'{cond}_view{view}.npz');floor=a['floor'];rgb=a['albedo']
        cv2.imwrite(str(fig/(prefix+'_albedo.png')),np.uint8(np.clip(rgb,0,1)**(1/2.2)*255)[...,::-1])
        rel=rgb.mean(-1)/max(float(rgb[floor].mean()),1e-8)
        heat=cv2.applyColorMap(np.uint8(np.clip((rel-.6)/.8,0,1)*255),cv2.COLORMAP_TURBO);heat[~floor]=0
        cv2.imwrite(str(fig/(prefix+'_relative.png')),heat)
        gt=np.load(data/'generator_truth'/f'{cond}_view{view:02d}.npz');mask=np.zeros((*floor.shape,3),np.uint8);mask[floor]=[55,55,55];mask[gt['sun_visible']]=[0,210,255]
        cv2.imwrite(str(fig/(prefix+'_mask.png')),mask)
rows=''
for r in report['difference_in_differences']:
    value='像素不足，不评分' if r['normalized_gap_change'] is None else f"{100*r['normalized_gap_change']:.1f} 个百分点"
    rows+=f"<tr><td>{r['condition']}</td><td>{r['view']}</td><td>{value}</td></tr>"
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>太阳光斑 × IRIS 受控实验</title>
<style>body{font:17px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;background:#171b20;color:#edf1f5}h1{font-size:30px}p{line-height:1.7}select{font-size:18px;padding:8px;margin-right:15px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}img{width:100%;image-rendering:auto;background:black}figure{margin:0}figcaption{padding:8px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #58616e;text-align:left;padding:12px}a{color:#8bcaff}.note{color:#c6ced7} @media(max-width:650px){.grid{grid-template-columns:1fr}}</style>
<h1>太阳光斑是否被恢复成了浅色地板？</h1>
<p>真实地板颜色固定。本页比较太阳照片训练得到的材质，与无太阳照片训练得到的材质。<b>这是一个厨房、已知几何、中性先验和每阶段 200 步的受控诊断。</b></p>
<select id="cond"><option>sun_a</option><option>sun_b</option></select><select id="view"><option>3</option><option>9</option></select><select id="mode"><option value="albedo">反照率：固定 gamma 2.2</option><option value="relative">相对亮度图：各自地板均值归一化</option></select>
<div class="grid"><figure><img id="input"><figcaption>输入照片：太阳照在地板上的亮块</figcaption></figure><figure><img id="mask"><figcaption>生成真值：黄色为地板直射可见区，灰色为其他地板</figcaption></figure><figure><img id="pred"><figcaption>太阳照片训练后，IRIS 恢复出的地板材质</figcaption></figure><figure><img id="control"><figcaption>相同位置，无直射太阳照片训练后的材质</figcaption></figure></div>
<p class="note">相对亮度图使用固定色标范围 0.6–1.4（除以各自地板均值），用于观察空间差异，不是原始材质颜色。真实地板 RGB 反照率恒为 [0.28,0.20,0.12]。黑色区域不在地板评分范围。</p>
<p><b>太阳 A 出现局部材质污染；太阳 B 未复现同样局部指标，且额外有 46 个非灯具三角面被提取为发光面。</b></p><h2>扣除无太阳对照后的光斑—阴影材质差</h2><table><tr><th>训练照明</th><th>保留视角</th><th>归一化差值的变化</th></tr>ROWS</table>
<p>正值表示：太阳照片使光斑位置额外变浅。每个模型的差值先除以自己的地板平均反照率，再减去无太阳对照。不是准确率提升，也不是显著性检验。</p>
<p class="note">限制：单房间、单训练随机种子、低分辨率、有限训练步数；未使用原论文 Irisformer 先验。绝对反照率仍有很大尺度偏差。当前结果不能代表所有 IRIS 配置。</p>
<h2>未见太阳条件的物理重渲染</h2><img src="relight_aligned/comparison.png"><p class="note">从左至右：真实地板、A 的原尺度恢复、A 的评估尺度对齐、无太阳对照的评估尺度对齐。全部显示统一 +2EV。只替换地板，其余材质和光照为生成真值，因此不是完整未知光照重渲染。</p><p><a href="../../../research/SUNPATCH_IRIS_BENCHMARK_ZH.md">完整实验记录</a> · <a href="eval_02/report.json">原始指标</a> · <a href="data_03/comparison.png">全部照明对照图</a></p>
<script>function update(){const c=document.getElementById('cond').value,v=document.getElementById('view').value,m=document.getElementById('mode').value;for(const [id,key] of [['input',c+'_'+v+'_input'],['mask',c+'_'+v+'_mask'],['pred',c+'_'+v+'_'+m],['control','sky_'+v+'_'+m]])document.getElementById(id).src='eval_02/figures/'+key+'.png';}document.querySelectorAll('select').forEach(x=>x.onchange=update);update();</script></html>'''.replace('ROWS',rows)
(b/'index.html').write_text(html)
print(b/'index.html')
