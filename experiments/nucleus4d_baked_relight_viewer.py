"""Full-count viewer generation, independent of the cached physical bake."""
import json
import shutil
import hashlib


def appearance_released(report):
    """Geometry/navigation checks alone must never release a relit appearance."""
    return all(report.get(key) is True for key in (
        'appearance_passed', 'original_quality_verified',
        'physical_accuracy_verified', 'geometry_bitwise_unchanged', 'coverage_complete'))


def publish_viewer(output, plan, SOURCE, MATERIAL):
    # Reuse the exact same full-count renderer and navigation for both versions.
    source_js=(SOURCE/'main.js').read_text()
    import re
    cameras=json.loads((MATERIAL/'views.json').read_text())
    selected=[cameras[i] for i in [4,3,7]]
    presets=[dict(id=i,img_name='living',width=1600,height=1040,fx=1006,fy=1006,
                  position=v['position'],rotation=v['rotation']) for i,v in enumerate(selected)]
    source_js=re.sub(r'let cameras = \[.*?\];','let cameras = '+json.dumps(presets)+';',source_js,count=1,flags=re.S)
    old='params.get("quality") === "full" ? "full.splat" : "scene.splat"'
    if source_js.count(old)!=1:raise RuntimeError('Original viewer source changed')
    result_path=output/'result.json'
    try:report=json.loads(result_path.read_text())
    except (OSError,ValueError):report={}
    released=appearance_released(report)
    choice=('params.get("mode") === "relit" ? "relit.splat" : "../nucleus4d_3dgs/full.splat"'
            if released else '"../nucleus4d_3dgs/full.splat"')
    source_js=source_js.replace(old,choice)
    source_js=source_js.replace('    const req = await fetch(url, {',
        '    window.activeSceneAsset = url.pathname;\n    const req = await fetch(url, {')
    source_js=source_js.replace('window.selectView = (n) => { viewMatrix = getViewMatrix(cameras[n]); };',
        'window.selectView = (n) => { viewMatrix = getViewMatrix(cameras[n]); };\nwindow.bakedView = () => ({viewMatrix, count: window.bakedCount || 0});')
    source_js=source_js.replace('fps.innerText = Math.round(avgFps) + " fps";',
        'window.bakedCount=vertexCount; window.bakedReady=vertexCount===6100978; fps.innerText = Math.round(avgFps) + " fps";')
    loaded='gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);\n            vertexCount = e.data.vertexCount;'
    if source_js.count(loaded)!=1:raise RuntimeError('Original viewer load handler changed')
    source_js=source_js.replace(loaded,loaded+' window.bakedCount=vertexCount; window.bakedReady=vertexCount===6100978;')
    (output/'main.js').write_text(source_js)
    html=(SOURCE/'index.html').read_text()
    panel='''<div id="info"><h2>1406 · 固定光照烘焙验证</h2>
<p><strong>全量 610 万高斯，几何与透明度保持不变。</strong></p>
<p>客厅受支持的漫反射表面换光；其他房间与不可靠表面保留原光照。当前是局部试验，尚未通过原画质与物理准确性验收。</p>
<a href="?mode=original" id="original-link">原始光照</a> <a href="?mode=relit" id="relit-link">烘焙换光（实验）</a>
<p>左键旋转 · 右键移动<br>方向键走动 · WASD 转向</p>
<button onclick="selectView(0)">客厅一</button> <button onclick="selectView(1)">客厅二</button> <button onclick="selectView(2)">客厅三</button>
<p>日光地点和朝向为示例；玻璃、材质及灯具功率尚未实测校准。画质审核以 float32 主文件为准。</p>
<a href="result.json">计算与覆盖报告</a> · <a href="relit.ply">float32 高斯主文件</a></div><div id="progress">'''
    panel=panel.replace('客厅受支持的漫反射表面换光；',
        f'本版换光覆盖 {plan["mapped_gaussians"]:,} / {plan["gaussians"]:,} 个高斯（{plan["mapped_fraction"]:.1%}），主要为客厅墙面与地面；')
    if not released:
        panel='''<div id="info"><h2>1406 · 原始光照</h2>
<p>换光试验未通过画质验收，已撤下。当前显示原始全量 3DGS，保留原拍摄外观。</p>
<p>左键旋转 · 右键移动<br>方向键走动 · WASD 转向</p>
<button onclick="selectView(0)">客厅一</button> <button onclick="selectView(1)">客厅二</button> <button onclick="selectView(2)">客厅三</button>
<p><a href="../nucleus4d_3dgs/?quality=full">原始全量漫游入口</a></p></div><div id="progress">'''
    html=re.sub(r'<div id="info">.*?</div><div id="progress">',panel,html,flags=re.S)
    # Keep the current world camera when changing the saved lighting version.
    html=html.replace('</html>','''<script>for(const id of ['original-link','relit-link'])document.getElementById(id)?.addEventListener('click',e=>{const v=window.bakedView?.();if(v)e.currentTarget.hash=encodeURIComponent(JSON.stringify(v.viewMatrix));});</script></html>''')
    if '</html>' not in html: # Original document omits its closing html tag.
        html+='''<script>for(const id of ['original-link','relit-link'])document.getElementById(id)?.addEventListener('click',e=>{const v=window.bakedView?.();if(v)e.currentTarget.hash=encodeURIComponent(JSON.stringify(v.viewMatrix));});</script>'''
    revision=hashlib.sha256(source_js.encode()).hexdigest()[:12]
    html=html.replace('src="main.js"',f'src="main.js?v={revision}"')
    (output/'index.html').write_text(html)
    shutil.copyfile(SOURCE/'LICENSE',output/'LICENSE')
