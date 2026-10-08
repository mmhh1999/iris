"""Local job status, plus a result viewer published only after quality checks."""
import argparse
import ast
from nucleus4d_ptir_full_data import ROOT,LOCAL,STORAGE

ap=argparse.ArgumentParser();ap.add_argument('--results',action='store_true');ap.add_argument('--comparison',action='store_true');ap.add_argument('--probe',action='store_true');args=ap.parse_args()
if args.probe:
    LOCAL=LOCAL/'render_probe';STORAGE=LOCAL
LOCAL.mkdir(exist_ok=True,parents=True)
if args.results and not args.probe and not args.comparison:
    from nucleus4d_ptir_freeview_validate import validate
    validate(STORAGE/'inverse/weights.pt',LOCAL)
    (LOCAL/'results.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>换光后自由漫游</title><style>body{background:#11161c;color:#e8edf2;font:16px/1.8 system-ui;max-width:760px;margin:60px auto;padding:24px}a{color:#9dcdff}</style><h1>换光后自由漫游</h1><p><a href="http://localhost:8770/">打开换光后的场景：旋转、行走、自由换视角</a></p><p>当前相机与连续时间驱动完整 PTIR 路径追踪。3840 × 2496、1024 SPP、8 次反弹、float32、无降噪；完整质量需要等待每次渲染。</p><p><a href="../nucleus4d_3dgs/">原拍摄光照 · 3DGS 漫游</a> · <a href="index.html">训练记录</a></p><p>材质是逆渲染估计；默认日光地点与朝向为示例。服务只有在完整训练及外观检查通过后才接收最终渲染请求。</p></html>''')
elif args.results:
    tree=ast.parse((ROOT/'experiments/nucleus4d_ptir_ui.py').read_text())
    html=next(ast.literal_eval(n.args[0]) for n in ast.walk(tree)
              if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text')
    html=html.replace('原始高斯日光实验','完整逆渲染 · 日光实验')
    html=html.replace('6,100,978 个高斯 · 保留原始几何 · 太阳、HDR 天空与间接光',
                      '原始全量高斯初始化 · 几何、材质与光照联合流程 · 4K / 1024 SPP 导出')
    html=html.replace('材质与法线仍是估计，未完成联合逆渲染优化。墙面可能出现斑驳、残留旧阴影或漏光。两扇窗按手动透光窗口处理，未恢复玻璃、窗外遮挡。地点与朝向为示例，天空为模拟 HDR。使用 GPU 降噪。',
                      '已运行几何/法线优化和材质/照明逆渲染。原光照还原通过预设外观检查；材质仍非实测真值。采用不透明高斯光传输，未恢复折射玻璃和室内灯具发光模型。地点与朝向为示例，天空为模拟 HDR。4K 导出无降噪；此页面为 1920 宽预览。')
    html=html.replace('+4 EV','+0 EV').replace('value="4"','value="0"').replace('exposure:4','exposure:0')
    html=html.replace('Float32Array.from(new Uint16Array(b),half)',"manifest.preview_dtype==='float32'?new Float32Array(b):Float32Array.from(new Uint16Array(b),half)")
    html=html.replace('v=v/(1+v)','v=Math.max(0,v)')
    html=html.replace('</header>', '</header><p><a href="exports/original.png">原始 4K 图</a> · <a href="exports/reconstructed_original.png">原光照还原 4K 图</a> · <a href="index.html">训练记录</a></p>')
    if args.probe:
        html=html.replace('完整逆渲染 · 日光实验','组件联调 · 非最终画质')
    (LOCAL/('comparison.html' if args.comparison else 'results.html')).write_text(html)
    link=LOCAL/'exports'
    if not link.exists():link.symlink_to(STORAGE/'renders',target_is_directory=True)
else:
    (LOCAL/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>1406-C-int · 完整逆渲染进度</title><style>body{margin:40px auto;padding:0 24px;max-width:980px;background:#11161c;color:#e8edf2;font:16px/1.7 system-ui}h1{font-size:26px}p{color:#bac8d6}pre{white-space:pre-wrap;background:#1c2631;padding:20px;border-radius:8px}a{color:#8fc6ff}progress{width:100%;height:20px}</style>
<h1>1406-C-int · 完整逆渲染</h1><p>原始 610 万高斯初始化；7,446 个训练视图，504 个留出视图。先计算学习先验，再优化几何、法线、材质与原光照，通过外观检查后在换光的场景中自由漫游。</p>
<p><a href="http://localhost:8770/">换光后自由漫游（模型训练完成前显示等待状态）</a> · <a href="../nucleus4d_3dgs/">原拍摄光照 · 3DGS 漫游</a></p>
<h2 id="stage">读取进度…</h2><progress id="bar" max="1" value="0"></progress><p id="detail"></p>
<p>配置：原分辨率 512 像素图块训练 · 64 SPP / 4 次反弹 · 4K / 1024 SPP / 8 次反弹导出。训练与 EXR 使用 float32。训练中的高斯可增密，显存预算上限 800 万。</p>
<p>先验保存在 E 盘；照片直接读取原始包，不重复解包。断点保留在 E 盘。当前页面不是训练完成后的画质展示。</p><p id="result"></p><pre id="info"></pre>
<script>
async function get(n){const r=await fetch(n,{cache:'no-store'});return r.ok?r.json():null}
async function update(){try{const s=await get('pipeline_status.json');if(!s){document.getElementById('stage').textContent='准备中';return}let d=null;
if(s.stage==='priors')d=await get('prior_status.json');else if(['geometry','inverse'].includes(s.stage))d=await get('training_status.json');else if(s.stage?.includes('4k'))d=await get('render_status.json');
const loading=s.status==='running'&&s.child_pid&&d?.pid&&d.pid!==s.child_pid;if(loading)d=null;
const names={priors:'计算法线与材质先验',geometry:'几何与法线优化',inverse:'材质与照明联合优化',reference_4k:'原始 4K 参考图',daylight_4k:'4K 日光渲染',viewer:'生成展示页',completed:'流程完成'};
const stale=s.status==='running'&&Date.now()/1000-s.updated>120;
document.getElementById('stage').textContent=(s.status==='paused'?'已暂停：':stale?'进度已停止更新，请检查后台进程：':s.status==='needs_attention'?'流程已停止：':'')+(names[s.stage]||s.stage||'准备中');
let a=d?.completed??d?.step??d?.tiles??0,b=d?.pending_at_start??d?.total??d?.total_tiles??1;document.getElementById('bar').value=a;document.getElementById('bar').max=b;
document.getElementById('detail').textContent=d?`${a} / ${b} · 最近更新 ${new Date(d.updated*1000).toLocaleString()}`:loading?'正在初始化并载入断点，训练步数将在恢复后更新。':'';
if(s.status==='needs_attention')document.getElementById('detail').textContent+=' · '+(s.error||'请检查运行日志');
if(s.status==='paused')document.getElementById('detail').textContent+=' · 进程已冻结，保留内存与显存；请勿关闭 WSL 或重启。';
document.getElementById('info').textContent=JSON.stringify({status:s.status,stage:s.stage,error:s.error,settings:s.settings,progress:d},null,2);
if(s.status==='completed')document.getElementById('result').innerHTML='<a href="http://localhost:8770/">打开换光后自由漫游</a>';
}catch(e){document.getElementById('detail').textContent=String(e)}}update();setInterval(update,15000);
</script>''')
print(LOCAL/('results.html' if args.results else 'index.html'))
