"""Build the independent full-house transport viewer, preserving prior entries."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import numpy as np

from nucleus4d_baked_relight import SOURCE,atomic_json
from nucleus4d_transport_geometry import OUT


def publish():
    meta=json.loads((OUT/'bank_manifest.json').read_text())
    geometry=json.loads((OUT/'geometry_report.json').read_text())
    meta['indoor_bounds']=geometry['indoor_bounds']
    with np.load(OUT/'geometry.npz') as z:
        meta['caster_bounds']=[z['vertices'].min(0).tolist(),z['vertices'].max(0).tolist()]
    meta['normal_asset']='normal_weights_full.f32' if (OUT/'normal_weights_full.f32').exists() else 'gaussian_normal_weights.f32'
    sun=json.loads((OUT/'sun_keys.json').read_text());lights=json.loads((OUT/'lights.json').read_text())['lights']
    source=(OUT.parent/'nucleus4d_sun_increment_v2/main.js').read_text()
    old='''    if (u_addedSun > 0.0) {
        float g=texelFetch(u_gain,ivec2(uint(index)&0x3ffu,uint(index)>>10),0).r;
        if (g > 1.0) vColor.rgb=encodeColor(decodeColor(vColor.rgb)*(1.0+u_addedSun*(g-1.0)));
    }'''
    if source.count(old)!=1:raise RuntimeError('Original draw shader insertion changed')
    source=source.replace(old,'    applyTransport(uint(index),uintBitsToFloat(cen.xyz));')
    shader=Path(__file__).with_name('nucleus4d_transport_shading.glsl').read_text()
    source=source.replace('void main () {',shader+'\nvoid main () {',1)
    source=source.replace('in vec4 vColor;\nin vec2 vPosition;','in vec4 vColor;\nin vec4 vSunColor;\nin vec3 vWorld;\nin vec2 vPosition;',1)
    source=source.replace('out vec4 fragColor;','layout(location=0) out vec4 fragColor;\nlayout(location=1) out vec4 sunColor;\nlayout(location=2) out vec4 worldColor;',1)
    source=source.replace('    fragColor = vec4(B * vColor.rgb, B);','    fragColor = vec4(B * vColor.rgb, B);\n    sunColor = vec4(B*vSunColor.rgb,B);\n    worldColor = vec4(B*vWorld,B);',1)
    runtime=Path(__file__).with_name('nucleus4d_transport_runtime.js').read_text()
    runtime=runtime.replace('/* BANK_META */',json.dumps(meta)).replace('/* SUN_KEYS */',json.dumps(dict(minutes=sun['minutes'],states=sun['states']))).replace('/* LIGHTS */',json.dumps(lights,ensure_ascii=False))
    runtime=runtime.replace('/* PROGRAMS */',Path(__file__).with_name('nucleus4d_transport_programs.js').read_text())
    start=source.index('    window.activeSceneAsset=url.pathname;');end=source.index('    const indexBuffer = gl.createBuffer();',start)
    source=source[:start]+runtime+'\n'+source[end:]
    source=source.replace('    const frame = (now) => {','    const frame = (now) => {\n        transportClock(now);',1)
    old_draw='''            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);'''
    if source.count(old_draw)!=1:raise RuntimeError('Original draw site changed')
    source=source.replace(old_draw,'''            beginTransportFrame();
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            finishTransportFrame();''',1)
    (OUT/'main.js').write_text(source)
    def slider(key,label,low,high,value,step=1,output=''):
        return f'<label>{label}<output id="{key}-value">{output}</output><input disabled id="{key}-slider" type="range" min="{low}" max="{high}" value="{value}" step="{step}"></label>'
    lamps=''.join(f'<label class="lamp">{html.escape(f["label"])}<input disabled id="lamp-{f["id"]}" type="range" min="0" max="200" value="100" oninput="setLamp({f["id"]},this.value/100)"><output id="lamp-value-{f["id"]}">100%</output></label>' for f in lights)
    panel='''<div id="info"><h2>1406 · 全屋光照试验</h2><p class="note">关灯后的墙面斑块仍待改善；原有太阳移动、天气、灯光与漫游功能保留。</p><p id="sun-state">原拍摄光照 · 精确对照</p><p id="loading" role="status">准备光照数据…</p>
<div class="buttons"><button disabled onclick="transportPreset('original')">拍摄原貌</button><button disabled onclick="transportPreset('sunny')">直射阳光</button><button disabled onclick="transportPreset('overcast')">阴天</button><button disabled onclick="transportPreset('dusk')">黄昏</button><button disabled onclick="transportPreset('night')">仅人工灯</button></div>
<div class="buttons"><button disabled id="sun-enable" onclick="setDirectSunEnabled(true)">开启太阳（晴空）</button><button disabled id="sun-disable" onclick="setDirectSunEnabled(false)">关闭太阳</button></div>
<p id="direct-sun-status" class="note" role="status"></p>
TIME<CLOUD><SKY><SUN>
<label>太阳阴影精度<select id="sun-shadow-resolution" disabled><option value="4096">4K · 原版</option><option value="8192">8K · 更高分辨率</option></select></label><EXPOSURE>
<button id="play-time" disabled onclick="toggleTimePlayback()">播放时间</button>
<p class="note">时间范围 12:30–18:15 为示意太阳路径。</p>
<details open><summary>人工灯 · 25 个独立光源</summary><div class="buttons"><button disabled onclick="setLampGroup('all',0)">全关</button><button disabled onclick="setLampGroup('all',1)">全部 100%</button><button disabled onclick="setLampGroup('all',1.5)">全部 150%</button></div>LAMPS</details>
<div class="buttons"><button onclick="selectView(0)">客厅</button><button onclick="selectView(1)">窗边</button><button onclick="selectView(2)">反向</button></div><p>左键旋转 · 右键移动 · 方向键走动</p>
<details><summary>精度与对照</summary><p>保留完整 610 万原始高斯及纹理。单一太阳的几何阴影实时移动；灯具光照及多次漫反射预计算，太阳间接光按通过误差检查的时间节点插值。</p><p>这仍是基于扫描几何的光照估计：97.6% 室内高斯有近邻网格支持，其余使用体积近邻估计；灯功率、反射率、玻璃及镜面材料未实测恢复。原拍摄对照可精确保留原图。</p><p><a href="verification.html">验证记录与画面对照</a> · <a href="../nucleus4d_weather_controls/">已认可的旧版</a></p></details></div><div id="progress">'''
    panel=panel.replace('TIME',slider('time','时间',750,1095,750,output='12:30')).replace('<CLOUD>',slider('cloud','云量',0,100,100,output='100%')).replace('<SKY>',slider('sky','天空强度 %',0,200,100)).replace('<SUN>',slider('sun','太阳强度 %',0,200,0)).replace('<EXPOSURE>',slider('exposure','曝光 EV',-3,3,0,.1)).replace('LAMPS',lamps)
    page=(SOURCE/'index.html').read_text();page=re.sub(r'<div id="info">.*?</div><div id="progress">',lambda _:panel,page,flags=re.S)
    page=page.replace('<title>1406-C-int · 3D Gaussian Splatting</title>','<title>1406 · 全屋光照试验</title>')
    page=page.replace('</style>','''#info{max-width:340px;max-height:calc(100vh - 66px);overflow:auto;padding:16px}h2{font-size:19px}#info label{display:block;margin:10px 0;font-size:13px}input[type=range]{display:block;width:100%;accent-color:#d3b57d}select{display:block;width:100%;padding:7px;margin-top:6px;background:#172b3b;color:#edf1f5;border:1px solid #ffffff40;border-radius:5px}output{float:right;color:#e6c697}.buttons{display:flex;flex-wrap:wrap;gap:5px}button:disabled{opacity:.45}details{border-top:1px solid #ffffff25;padding-top:12px;margin:12px 0}summary{cursor:pointer}.lamp{display:grid!important;grid-template-columns:95px 1fr 40px;gap:8px;align-items:center}.note,#loading{font-size:12px;color:#b6c5cf}</style>''')
    page=page.replace('<script src="main.js"></script>',f'<script src="main.js?v={hashlib.sha256(source.encode()).hexdigest()[:12]}"></script>')
    (OUT/'index.html').write_text(page);shutil.copyfile(SOURCE/'LICENSE',OUT/'LICENSE')
    atomic_json(OUT/'viewer_manifest.json',dict(gaussians=6100978,source_geometry_unchanged=True,source_asset='../nucleus4d_3dgs/full.splat',direct_sun='single moving geometric shadow map, original 4096 default, optional 8192',secondary_diffuse_bounces=8,independent_lights=len(lights),resident_float32_fields=True,experimental=True,visual_review_passed=False,known_issues=['wall mottling after lamp removal','source housing alignment remains approximate'],physical_ground_truth_verified=False,optimization_revision='v3_opt1',default_solar_shadow_resolution=4096,optional_solar_shadow_resolution=8192,sun_controls=['continuous time','power','on/off','playback','sunny/dusk presets','soft shadows'],reference_capture_bypass=True))
    print('VIEWER_BUILT',OUT,flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();publish()
