"""Static, exportable figures and Chinese gallery from the completed paired run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2];base=root/'experiments/out/EXP0022_sgs_sunpatch';out=base/'evaluation_v2'
r=json.loads((out/'metrics.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(14,4))
summaries={}
for condition in ['sun_a','sky']:
    run=base/f'{condition}_v2_1000_600';h=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()];m=[x for x in h if x['phase']=='material']
    # Each 12-step window sees every training camera exactly once.
    steps=np.array([x['iteration']-1000 for x in m])
    for ax,key,title in zip(axes[:2],['psnr_pbr','l1_albedo'],['Training PBR PSNR (12-step mean)','Albedo prior L1 (12-step mean)']):
        values=np.array([x[key] for x in m]);smooth=np.convolve(values,np.ones(12)/12,mode='valid');ax.plot(steps[11:],smooth,label=condition);ax.set_title(title);ax.set_xlabel('Material iterations');ax.axvline(400,color='gray',ls=':',label='Self-invariance starts' if condition=='sun_a' else None)
    summaries[condition]={key:{'first_60_mean':float(np.mean([x[key] for x in m[:60]])),'previous_60_mean':float(np.mean([x[key] for x in m[-120:-60]])),'last_60_mean':float(np.mean([x[key] for x in m[-60:]]))} for key in ['psnr_pbr','l1_albedo','loss_total_with_invariance']}
for i,c in enumerate(['sun_a','sky']):
    values=[100*next(x for x in r['views'] if x['condition']==c and x['view']==v)['normalized_gap'] for v in [3,9]]
    axes[2].bar(np.arange(2)+(i-.5)*.35,values,width=.35,label=c)
axes[2].set_xticks([0,1],['Heldout v3','Heldout v9']);axes[2].set_ylabel('Normalized lit-shadow albedo gap (%)');axes[2].set_title('Same spatial masks, two lighting fits');axes[2].axhline(0,color='black',lw=.8)
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'training_and_gap.png',dpi=180);plt.close(fig)
(out/'training_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
table=''.join(f"<tr><td>{x['condition']}</td><td>{x['view']}</td><td>{x['albedo_mae_linear']:.4f}</td><td>{x['normalized_gap']*100:.2f}%</td><td>{x['pbr_psnr']:.2f}</td></tr>" for x in r['views'])
diff='；'.join(f"视角 {x['view']}：{x['normalized_gap_change']*100:+.2f} 个百分点" for x in r['difference_in_differences'])
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SGS 太阳光斑对照</title><style>body{{max-width:1200px;margin:32px auto;padding:0 18px;font:17px/1.65 system-ui;background:#f5f5f5;color:#222}}img{{width:100%;background:white}}table{{border-collapse:collapse;background:white}}td,th{{padding:8px 18px;border:1px solid #ddd}}code{{background:#eee}}section{{margin:26px 0}}</style><h1>SGS 太阳光斑对照：实际训练结果</h1><p>已知几何 P1 · 一个合成房间 · 12 个训练视角 / 2 个留出视角 · 同一材质，有 / 无直射太阳。</p><p>SGS 公共上游适配版，1000 步外观初始化 + 600 步材质训练。包含 RGB-X 照片先验、SAM 跨视角约束和光照自一致性。不是官方完整复现，也未证明收敛。</p><p>在完全相同的地板位置比较“原太阳受光区 − 阴影区”的反照率差。太阳组减去无太阳组：{diff}。正值表示太阳组的局部材质变亮更明显；不能仅凭此断言多场景普遍失败。</p><section><img src="comparison.png" alt="输入、重建、反照率、真实地板材质和光斑掩码"></section><table><tr><th>组</th><th>留出视角</th><th>线性反照率 MAE</th><th>归一化亮暗差</th><th>重建 PSNR</th></tr>{table}</table><section><img src="training_and_gap.png" alt="训练曲线与地板亮暗差"></section><p>GT 太阳、材质和地板掩码只用于独立评分。未做新时刻重照明或真实空间验证。与旧 IRIS 结果的色调映射和先验不同，不能直接做性能排名。</p><p><a href="metrics.json">完整指标</a> · <a href="training_summary.json">训练末段变化</a></p></html>'''
(out/'index.html').write_text(html)
print(out/'index.html')
