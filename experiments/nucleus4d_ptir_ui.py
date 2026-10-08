"""Small local viewer for full-Gaussian path-traced daylight experiments."""
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'experiments/out/nucleus4d_ptir'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>1406 · 原始高斯日光实验</title><style>
*{box-sizing:border-box}body{margin:0;background:#11161c;color:#e8edf2;font:15px/1.55 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:25px}header{display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:24px;margin:0}p{color:#aab6c2;margin:8px 0 18px}a{color:#8fc6ff}section{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:20px}.picture{background:#05080b;position:relative;align-self:start;border-radius:10px;overflow:hidden}canvas{display:block;width:100%;height:auto}.badge{position:absolute;top:14px;left:14px;background:#10151bcb;padding:6px 12px;border-radius:5px}.controls{background:#1b232d;padding:20px;border-radius:10px}label{display:block;margin:16px 0 4px}input[type=range]{width:100%;accent-color:#9dc9ed}select,button{font:inherit;color:#eaf0f5;background:#303e4b;border:1px solid #546170;border-radius:5px;padding:7px}select{width:100%}button{cursor:pointer;margin-top:15px}output{float:right;color:#a6d1f3}small{color:#aab6c2;display:block;margin-top:15px}details{margin-top:18px;color:#b7c3ce;border-top:1px solid #394550;padding-top:13px}.status{min-height:24px;color:#9fcbab}strong{color:#edf4fa}@media(max-width:850px){section{grid-template-columns:1fr}header{display:block}.controls{display:grid;grid-template-columns:1fr 1fr;gap:0 20px}details,small{grid-column:1/-1}}
</style><main><header><div><h1>1406-C-int · 原始高斯日光实验</h1><p>6,100,978 个高斯 · 保留原始几何 · 太阳、HDR 天空与间接光</p></div><a href="../nucleus4d_3dgs/?quality=full" target="_blank">打开原始全量 3D 漫游 ↗</a></header>
<section><div><div class="picture"><canvas id="canvas" width="896" height="582"></canvas><div id="badge" class="badge">加载中</div></div><p id="status" class="status">读取渲染数据…</p></div>
<div class="controls"><div><label>查看方式</label><select id="mode"><option value="relight">日光重渲染</option><option value="split">左原始 / 右重渲染</option><option value="original">原始拍摄外观</option></select></div>
<div><label>示例日光时间 <output id="timeOut">13:00</output></label><input id="time" type="range" min="9" max="17" step="2" value="13"></div>
<div><label>阴天程度 <output id="cloudOut">0%</output></label><input id="cloud" type="range" min="0" max="1" step=".01" value="0"></div>
<div><label>太阳强度 <output id="sunOut">1.0×</output></label><input id="sun" type="range" min="0" max="2" step=".05" value="1"></div>
<div><label>天空强度 <output id="skyOut">1.0×</output></label><input id="sky" type="range" min="0" max="3" step=".05" value="1"></div>
<div><label>曝光 <output id="exposureOut">+4 EV</output></label><input id="exposure" type="range" min="-1" max="6" step=".1" value="4"></div>
<div><label><input id="gi" type="checkbox" checked> 包含间接反射光</label><button id="reset">恢复默认</button></div>
<small>固定视角，按两小时间隔切换预计算的高斯路径追踪结果。可用“原始全量 3D 漫游”检查完整场景。</small>
<details open><summary>本次试验的边界</summary>材质与法线仍是估计，未完成联合逆渲染优化。墙面可能出现斑驳、残留旧阴影或漏光。两扇窗按手动透光窗口处理，未恢复玻璃、窗外遮挡。地点与朝向为示例，天空为模拟 HDR。使用 GPU 降噪。</details></div></section></main>
<script>
const $=id=>document.getElementById(id),cache=new Map();let manifest,ticket=0;
function half(n){const s=n&32768?-1:1,e=(n>>10)&31,f=n&1023;return e===0?s*Math.pow(2,-14)*f/1024:e===31?(f?NaN:s*Infinity):s*Math.pow(2,e-15)*(1+f/1024)}
async function load(name){if(!cache.has(name))cache.set(name,fetch(name).then(r=>{if(!r.ok)throw Error(name+' 下载失败');return r.arrayBuffer()}).then(b=>Float32Array.from(new Uint16Array(b),half)));return cache.get(name)}
function srgb(v){return v<=.0031308?12.92*v:1.055*Math.pow(v,1/2.4)-.055}
async function draw(){if(!manifest)return;const token=++ticket,time=+$('time').value,cloud=+$('cloud').value,sun=+$('sun').value,sky=+$('sky').value,ev=+$('exposure').value,gi=$('gi').checked,mode=$('mode').value;
$('timeOut').value=time+':00';$('cloudOut').value=Math.round(cloud*100)+'%';$('sunOut').value=sun.toFixed(1)+'×';$('skyOut').value=sky.toFixed(1)+'×';$('exposureOut').value=(ev>=0?'+':'')+ev.toFixed(1)+' EV';
try{const original=await load('original.bin');let a,b,c;const record=manifest.sun.find(s=>s.hour===time);if(mode!=='original'){if(!record||!manifest.skies.overcast){$('status').textContent='这个时间点仍在渲染，可先看原始外观。';return}const field=gi?'total':'direct';[a,b,c]=await Promise.all([load(manifest.skies.clear[field]),load(manifest.skies.overcast[field]),load(record[field])])}if(token!==ticket)return;
const w=manifest.width,h=manifest.height,ctx=$('canvas').getContext('2d');$('canvas').width=w;$('canvas').height=h;const image=ctx.createImageData(w,h);let mean=0;const exposure=Math.pow(2,ev);
for(let p=0;p<w*h;p++){const showOriginal=mode==='original'||mode==='split'&&p%w<w/2;for(let k=0;k<3;k++){const i=p*3+k;let v;if(showOriginal)v=original[i];else{v=Math.max(0,sky*((1-cloud)*a[i]+cloud*b[i])+sun*Math.pow(1-cloud,2)*c[i]);mean+=v;v*=exposure;v=v/(1+v)}image.data[p*4+k]=Math.round(Math.max(0,Math.min(1,srgb(v)))*255)}image.data[p*4+3]=255}ctx.putImageData(image,0,0);
$('badge').textContent=mode==='original'?'原始 3DGS 外观':mode==='split'?'原始外观 ← → 日光重渲染':time+':00 · '+(gi?'直射 + 间接光':'仅直射光');$('status').textContent=w+' × '+h+' · 全量高斯 · '+(manifest.complete?'渲染完成':'其余时间点正在生成');window.lastFrame={time,cloud,sun,sky,gi,mode,mean:mean/(w*h*3)};window.ptirReady=true;
}catch(e){$('status').textContent=e.message;console.error(e)}}
for(const id of ['time','cloud','sun','sky','exposure','gi','mode'])$(id).addEventListener('input',draw);
$('reset').onclick=()=>{for(const [k,v]of Object.entries({time:13,cloud:0,sun:1,sky:1,exposure:4,mode:'relight'}))$(k).value=v;$('gi').checked=true;draw()};
async function refresh(){manifest=await fetch('manifest.json',{cache:'no-store'}).then(r=>r.json());draw();if(!manifest.complete)setTimeout(refresh,10000)}refresh();
</script>''')
print(OUT/'index.html')
