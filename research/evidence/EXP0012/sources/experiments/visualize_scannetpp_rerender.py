"""Measured real-mesh relighting contact sheet and offline viewer."""
import argparse
import base64
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/iris-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
import numpy as np

def main(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text());cfg=manifest['config']
    font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
    fig,axes=plt.subplots(4,5,figsize=(18,10),layout='constrained')
    titles=['真实照片（仅作空间参照）','训练光照','验证光照','未见光照 A','未见光照 B']
    for v in range(4):
        images=[folder/f'view_{v}_real_reference.jpg']+[folder/c['name']/f'view_{v}_seed_{cfg["seeds"][0]}'/'rgb.png' for c in cfg['conditions']]
        for col,p in enumerate(images):
            axes[v,col].imshow(Image.open(p));axes[v,col].axis('off')
            if v==0:axes[v,col].set_title(titles[col],fontproperties=font,fontsize=12)
            if col==0:axes[v,col].text(0,.03,f'机位 {v}',transform=axes[v,col].transAxes,color='white',fontproperties=font,bbox=dict(facecolor='black',alpha=.6))
    fig.suptitle('ScanNet++ 真实扫描空间 · 同一几何、不同太阳条件\n原始 422 万三角面；指定灰色材质；合成光照，不是真实跨时间照片',fontproperties=font,fontsize=18)
    fig.savefig(folder/'comparison.png',dpi=150);fig.savefig(folder/'comparison.pdf');plt.close(fig)
    def uri(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
    payload=[]
    for v in range(4):
        payload.append([{'label':titles[i+1], 'angle':f"方位 {c['azimuth']}° / 高度 {c['elevation']}°",'image':uri(folder/c['name']/f'view_{v}_seed_{cfg["seeds"][0]}'/'rgb.png'), 'mask':uri(folder/c['name']/f'view_{v}_seed_{cfg["seeds"][0]}'/'direct_sun_proxy.png')} for i,c in enumerate(cfg['conditions'])])
    results=json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else None
    result_text='评估正在运行。' if results is None else ('训练方向误差 %.3f°；未见光照平均阳光斑 IoU %.3f。已知材质与相对太阳变化；不是原始 IRIS 优势对比。'%(results['summary']['fit']['angular_error_deg'],results['summary']['test_mean_iou']))
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实扫描空间重渲染</title>
<style>body{background:#101722;color:#edf2fa;font:17px/1.65 system-ui;max-width:1100px;margin:auto;padding:25px}h1{font-size:28px}select,button,input{font:inherit;margin:8px;padding:5px}#pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;background:#000;border-radius:6px}small{color:#b8c4d6}figure{margin:0}a{color:#a7ceff}</style>
<h1>真实房间形状下的光照变化</h1><p>ScanNet++ 1b379f1114 · 原始扫描网格 · 指定材质 · 合成太阳与天空</p>
<p>机位 <select id="view"><option value="0">0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select>
太阳条件 <input id="light" type="range" min="0" max="3" value="0" step="1"><button id="play">播放</button></p>
<p id="label"></p><div id="pair"><figure><img id="rgb"><figcaption>路径追踪重渲染</figcaption></figure><figure><img id="mask"><figcaption>地面直接阳光区域（单独渲染得到）</figcaption></figure></div>
<p>RESULT_TEXT</p><small>保留扫描缺口；窗户范围是先前的近似标注。原照片只提供几何空间参照，未当作无光照材质。太阳角度使用场景坐标，不代表真实拍摄时间。本实验尚未证明真实照片上的迁移优势。</small>
<script>const data=PAYLOAD;let timer=null;const view=document.getElementById('view'),light=document.getElementById('light');function update(){const x=data[+view.value][+light.value];document.getElementById('rgb').src=x.image;document.getElementById('mask').src=x.mask;document.getElementById('label').textContent=x.label+' · '+x.angle;}view.onchange=update;light.oninput=update;document.getElementById('play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='播放';}else{timer=setInterval(()=>{light.value=(+light.value+1)%4;update();},1300);this.textContent='暂停';}};update();</script></html>'''
    (folder/'index.html').write_text(html.replace('RESULT_TEXT',result_text).replace('PAYLOAD',json.dumps(payload)))
    frames=[Image.open(folder/c['name']/f'view_0_seed_{cfg["seeds"][0]}'/'rgb.png') for c in cfg['conditions']]
    frames[0].save(folder/'sun_changes.gif',save_all=True,append_images=frames[1:],duration=1100,loop=0)
    print(folder/'index.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');main(p.parse_args().folder)
