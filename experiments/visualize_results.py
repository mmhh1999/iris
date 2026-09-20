"""Create offline Chinese visualization from recorded evidence; no new fitting."""
import argparse
import base64
from io import BytesIO
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/iris-matplotlib')
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]


def uri(im,fmt='PNG'):
    stream=BytesIO();im.save(stream,format=fmt);mime='jpeg' if fmt=='JPEG' else 'png'
    return 'data:image/'+mime+';base64,'+base64.b64encode(stream.getvalue()).decode()


def real_assets(row,source):
    path=ROOT/'data_download/scannetpp/data/1b379f1114/dslr/resized_undistorted_images'/row['image']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
        raise ValueError('Source photo differs from evaluated image')
    photo=Image.open(path).convert('RGB');w,h=photo.size;photo=photo.resize((876,round(h*876/w)))
    s=np.load(source/(Path(row['image']).stem+'_samples.npz'))
    pred=s['predicted'].astype(bool);obs=s['proxy_labels'].astype(bool)
    # Check saved visualization samples against published metrics.
    tp=int(np.sum(pred&obs));fp=int(np.sum(pred&~obs));fn=int(np.sum(~pred&obs))
    m=row['metrics']['fitted']
    assert (tp,fp,fn)==(m['tp'],m['fp'],m['fn'])
    px=s['pixels']*876/w
    assets={'photo':photo}
    for mode in ['observed','predicted','errors']:
        im=photo.copy();draw=ImageDraw.Draw(im)
        for (x,y),p,o in zip(px,pred,obs):
            if mode=='errors':
                if p==o:continue
                color='#f29c38' if p else '#bd62ed'
            else:
                if not (o if mode=='observed' else p):continue
                color='#17b5cd' if mode=='observed' else '#f29c38'
            draw.ellipse((x-1.7,y-1.7,x+1.7,y+1.7),fill=color,outline='#172434',width=1)
        assets[mode]=im
    return assets


def synthetic_assets(case,source):
    folder=source/f"az{case['train_az']}_el{case['train_el']}_seed{case['seed']}"
    mask=Image.open(folder/'heldout_proxy_physical_memory.png').convert('RGB');w,h=mask.size;assert w%3==0
    assets={'train':Image.open(folder/'train.png').convert('RGB'),'heldout':Image.open(folder/'heldout.png').convert('RGB'),
            **{key:mask.crop((i*w//3,0,(i+1)*w//3,h)) for i,key in enumerate(['proxy','physical','memory'])}}
    truth=np.asarray(assets['proxy'])[...,0]==255
    for key,metric in [('physical','heldout_physical'),('memory','heldout_frozen_appearance')]:
        prediction=np.asarray(assets[key])[...,0]==255;m=case[metric]
        assert (int(np.sum(prediction&truth)),int(np.sum(prediction&~truth)),int(np.sum(~prediction&truth)))==(m['tp'],m['fp'],m['fn'])
    return assets


def load_font(path):
    if not path.exists():raise FileNotFoundError('Provide a Chinese font with --font')
    font_manager.fontManager.addfont(str(path));name=font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({'font.family':name,'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'#f4f6f9',
                        'axes.facecolor':'#f4f6f9','text.color':'#16283d','axes.labelcolor':'#16283d'})


def image_axis(fig,box,im,title,subtitle=None):
    ax=fig.add_axes(box);ax.imshow(im,interpolation='nearest');ax.set_axis_off();ax.set_title(title,loc='left',fontsize=12,pad=9)
    if subtitle:ax.text(0,-.07,subtitle,transform=ax.transAxes,fontsize=9,va='top',color='#516074')


def build_figures(data,real,synthetic,out):
    f=plt.figure(figsize=(16,12))
    f.text(.035,.955,'日光感知逆渲染  /  阶段成果',fontsize=26,weight='bold')
    f.text(.035,.92,'实际照片与实验输出 · 2026-09-19 · 组件验证，尚未完成完整 IRIS 对比',fontsize=12,color='#516074')
    f.text(.035,.875,'01  真实场景：换视角仍不足以证明物理模型更好',fontsize=16,weight='bold')
    # Fixed first heldout view, the weakest measured view; no favorable-view selection.
    r=data['real'][1];a=real[1]
    image_axis(f,[.035,.605,.29,.235],a['photo'],'未参与拟合的真实照片','DSC09417 · 三个留出视角中表现最弱的一张')
    image_axis(f,[.35,.605,.29,.235],a['errors'],'预测与亮度代理标签的分歧','橙色：预测亮、观测暗；紫色：预测暗、观测亮')
    ax=f.add_axes([.72,.635,.245,.175]);labels=['外观记忆','物理模型','错误方向 +45°'];vals=[data['realMeans'][k] for k in ['memorized_floor_1nn','fitted','wrong_direction_plus45']]
    bars=ax.barh(labels,vals,color=['#668298','#157f82','#b97b5b'],height=.52);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('三个留出视角平均代理 IoU',fontsize=10);ax.set_xticks([0,.5,1]);ax.spines[['top','right','left']].set_visible(False)
    for b,v in zip(bars,vals):ax.text(v+.015,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center',fontsize=11)
    f.text(.685,.565,'结论：0.709 < 0.833，真实优势尚未成立。',fontsize=11,color='#9d532d')
    f.text(.035,.53,'02  合成场景：改变太阳方向后，光斑能够随之移动',fontsize=16,weight='bold')
    c=data['synthetic'][0];sa=synthetic[0]
    titles=['训练光照 · 路径追踪图','新光照 · 路径追踪图','新光照观测代理光斑','物理模型预测','冻结训练外观']
    keys=['train','heldout','proxy','physical','memory']
    for i,(key,title) in enumerate(zip(keys,titles)):
        image_axis(f,[.035+i*.193,.32,.178,.165],sa[key],title)
    f.text(.035,.292,'示例：真实训练太阳 163.7° / 31.4° → 新光照 183.7° / 41.4°；种子 0 / 32 spp。白色表示亮区，灰色表示其余可见地面。',fontsize=10,color='#516074')
    ax=f.add_axes([.075,.09,.50,.145]);xs=np.arange(6);phy=[c['heldout_physical']['iou'] for c in data['synthetic']];mem=[c['heldout_frozen_appearance']['iou'] for c in data['synthetic']]
    ax.bar(xs-.17,phy,width=.32,color='#157f82',label='物理模型');ax.bar(xs+.17,mem,width=.32,color='#668298',label='冻结外观')
    ax.set_ylim(0,1.08);ax.set_ylabel('换光照代理 IoU');ax.set_xticks(xs,[f"{c['train_az']}° / 种子{c['seed']}" for c in data['synthetic']],fontsize=8)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper right',bbox_to_anchor=(1,1.28),ncol=2,frameon=False,fontsize=9)
    summary=data['syntheticSummary']
    f.text(.64,.225,f"{summary['mean_angular_error_deg']:.2f}°",fontsize=27,weight='bold',color='#157f82');f.text(.64,.195,'平均太阳方向误差',fontsize=11)
    f.text(.81,.225,f"{summary['mean_heldout_physical_iou']:.3f}",fontsize=27,weight='bold',color='#157f82');f.text(.81,.195,'换光照平均 IoU',fontsize=11)
    f.text(.64,.12,'边界：已知几何与材质；提供真实相对太阳变化。\n3 个条件 × 2 个随机种子，不是 6 个独立场景。\n尚无真实跨时间重光照验证。',fontsize=10,linespacing=1.7,color='#516074')
    f.text(.035,.025,'证据：EXP0008 / EXP0009 / EXP0010   ·   真实标签为亮度代理，合成标签为固定阈值代理   ·   数值来自已保存结果',fontsize=9,color='#516074')
    f.savefig(out/'overview.png',dpi=150,facecolor=f.get_facecolor())
    with PdfPages(out/'results_zh.pdf') as pdf:pdf.savefig(f,facecolor=f.get_facecolor())
    plt.close(f)


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rs=ROOT/'experiments/out/EXP0008_memory_control';ss=ROOT/'experiments/out/EXP0009_offgrid'
    rawr=json.loads((rs/'results.json').read_text());raws=json.loads((ss/'results.json').read_text());portal=json.loads((ROOT/'research/evidence/EXP0010/results.json').read_text())
    screening=json.loads((ROOT/'research/evidence/EXP0007_expanded/results.json').read_text())['summary']
    real=[real_assets(r,rs) for r in rawr['images']];synth=[synthetic_assets(c,ss) for c in raws['cases']]
    held=[r for r in rawr['images'] if r['split']=='heldout_view']
    means={k:float(np.mean([r['metrics'][k]['iou'] for r in held])) for k in held[0]['metrics']}
    data=dict(real=rawr['images'],synthetic=raws['cases'],realMeans=means,syntheticSummary=raws['summary'],portal=portal,screening=screening)
    load_font(Path(args.font));build_figures(data,real,synth,out)
    evidence={'real_results':str(rs/'results.json'),'synthetic_results':str(ss/'results.json')}
    files=[rs/'results.json',ss/'results.json',ROOT/'research/evidence/EXP0010/results.json',Path(__file__),ROOT/'research/templates/daylight_report.html']
    (out/'visualization_manifest.json').write_text(json.dumps({'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'evidence':evidence,'new_experiments_run':False},indent=2))
    # HTML is fully offline: all image resources are embedded, no CDN or server.
    for r,a in zip(data['real'],real):r['assets']={k:uri(v,'JPEG') for k,v in a.items()}
    for c,a in zip(data['synthetic'],synth):c['assets']={k:uri(v) for k,v in a.items()}
    template=(ROOT/'research/templates/daylight_report.html').read_text()
    rendered=template.replace('__EVIDENCE_DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (out/'index.html').write_text(rendered)
    print('Created',out/'index.html');print('Created',out/'overview.png');print('Created',out/'results_zh.pdf')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'experiments/out/visualization_20260919'))
    p.add_argument('--font',default='/mnt/c/Windows/Fonts/msyh.ttc');main(p.parse_args())
