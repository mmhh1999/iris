"""Build an offline review page and freeze pilot acquisition/preparation evidence."""
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/out/EXP0014_eyeful'
EVIDENCE=ROOT/'research/evidence/EXP0014'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    summaries=[];sections=[]
    for scene in ['riverview','apartment']:
        d=OUT/scene
        manifest=json.loads((d/'manifest.json').read_text())
        geometry=json.loads((d/'geometry_report.json').read_text())
        downloads=ROOT/'data_download/eyefultower'/scene/'download_manifest.json'
        files=json.loads(downloads.read_text())
        for f in files:
            if digest(ROOT/f['path'])!=f['sha256']:raise ValueError(f['path'])
        cameras=manifest['cameras']
        ids=[c['camera_id'] for c in cameras]
        assert len(ids)==len(set(ids))==32
        assert sum(c['split']=='test' for c in cameras)==8
        assert all(c['split']=='test' for c in cameras if c['camera_id'].startswith('17/'))
        summary=dict(scene=scene,views=len(cameras),train=24,test=8,
            downloaded_files=len(files),downloaded_bytes=sum(f['bytes'] for f in files),
            sha256_recheck_pass=True,triangles=geometry['triangles'],
            mean_mesh_hit_fraction=sum(x['mesh_hit_fraction'] for x in geometry['views'])/32,
            hdr_max=max(c['raw_max'] for c in cameras),
            max_calibration_roundtrip_px=max(c['calibration_roundtrip_px'] for c in cameras))
        summaries.append(summary)
        dst=EVIDENCE/scene;dst.mkdir(exist_ok=True)
        for path in [downloads,d/'manifest.json',d/'geometry_report.json']:
            shutil.copy2(path,dst/path.name)
        options=''.join(f'<option value="{html.escape(Path(c["file"]).stem)}">{html.escape(c["camera_id"])} / {c["split"]}</option>' for c in cameras)
        first=Path(cameras[0]['file']).stem
        sections.append(f'''<section><h2>{scene}</h2><p>32 张真实 HDR；24 训练 / 8 测试；{geometry['triangles']:,} 个三角形。</p>
        <a href="{scene}/contact_sheet.jpg">全部照片总览</a>
        <p><select onchange="document.getElementById('{scene}').src='{scene}/'+this.value+'_geometry.jpg'">{options}</select></p>
        <p>左：真实 HDR 显示预览　中：扫描纹理投影　右：网格法线。紫色表示没有网格覆盖。</p>
        <img id="{scene}" src="{scene}/{first}_geometry.jpg" style="width:min(100%,900px)"></section>''')
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>真实室内 HDR 数据检查</title>
    <style>body{font:17px system-ui;max-width:1100px;margin:30px auto;background:#161b22;color:#e8edf3;padding:20px}a{color:#9cd4ff}section{border-top:1px solid #536273;margin-top:30px}select{padding:8px}</style>
    <h1>真实室内 HDR 数据检查</h1><p>Eyeful Tower 官方实拍数据。已保留线性 HDR，完成去畸变与相机/网格投影检查。</p>
    <p>这是数据接入与几何检查，尚未进行材质恢复或太阳重照明。扫描纹理含原始阴影，不是反照率真值。</p>'''+''.join(sections))
    (EVIDENCE/'summary.json').write_text(json.dumps(summaries,indent=2))
    for name in ['fetch_eyeful.py','prepare_eyeful.py','check_eyeful_geometry.py','report_eyeful.py']:
        shutil.copy2(ROOT/'experiments'/name,EVIDENCE/'sources'/name)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
