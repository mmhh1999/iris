"""Plot measured official-checkpoint reconstruction; no generated illustrations."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iris-mpl')
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root = Path(__file__).resolve().parents[1]
out = root / 'experiments/out/EXP0011_baseline'
data = root / 'data_download/iris_official/datasets/fipt/indoor_synthetic/bathroom/val/Image'
render = out / 'render_official/val/rgb'
rows = (render/'metrics.txt').read_text().strip().splitlines()
mean = list(map(float, rows[-1].split(',')[1:]))
metrics = {'scene':'official synthetic bathroom','checkpoint':'official last_1.ckpt',
           'validation_views':13,'SPP':256,'spp':16,'mean_psnr':mean[0],'mean_ssim':mean[1],
           'scope':'pretrained reconstruction; not from-scratch convergence or real cross-time validation'}
(root/'research/evidence/EXP0011/render_metrics.json').write_text(json.dumps(metrics,indent=2))
font = FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
fig, axes = plt.subplots(3,3,figsize=(14,8),layout='constrained')
for row, i in enumerate([0,6,12]):
    gt = np.asarray(Image.open(data/f'{i:03d}_0001.png')).astype(float)/255
    pred = np.asarray(Image.open(render/f'{i:05d}_rgb_full.png')).astype(float)/255
    axes[row,0].imshow(gt);axes[row,1].imshow(pred)
    im=axes[row,2].imshow(np.abs(gt-pred).mean(-1),vmin=0,vmax=.2,cmap='magma')
    for col in range(3):
        axes[row,col].axis('off')
        axes[row,col].set_title(['目标图','原始 IRIS 重建','平均绝对误差（0–0.2）'][col]+f' · 视角 {i}',fontproperties=font)
fig.suptitle(f'官方合成 bathroom：13 个验证视角，PSNR {mean[0]:.2f} dB / SSIM {mean[1]:.3f}\n'
             '官方预训练模型，256 SPP；不代表真实跨时间优势',fontproperties=font,fontsize=16)
fig.colorbar(im,ax=axes[:,2],shrink=.65)
fig.savefig(out/'baseline_comparison.png',dpi=160)
plt.close(fig)
print(json.dumps(metrics,indent=2))
