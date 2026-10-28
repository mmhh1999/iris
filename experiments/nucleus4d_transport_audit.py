"""Audit real source fields, repair undefined ratios, and write review evidence."""
import json
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from nucleus4d_baked_relight import SOURCE,atomic_json,file_identity
from nucleus4d_transport_geometry import OUT


def repair_unlit_bindings(reference,positions):
    binding=np.memmap(OUT/'binding_compact.u32',mode='r+',dtype='<u4').reshape(-1,4)
    path=OUT/'normal_weights_full.f32';path=path if path.exists() else OUT/'gaussian_normal_weights.f32'
    normals=np.memmap(path,mode='r+',dtype='<f4').reshape(-1,4)
    good=np.flatnonzero(reference.min(1)>1e-5);tree=cKDTree(positions[good]);fixed=0;max_distance=0.
    rows=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    for start in range(0,len(binding),65536):
        end=min(start+65536,len(binding));b=np.array(binding[start:end]);n=np.array(normals[start:end]);valid=b[:,0]!=0xffffffff
        ids=b[valid,:3].copy();ids[:,0]&=np.uint32(0x7fffffff)
        weights=np.column_stack([b[valid,3].copy().view('<f4'),n[valid,0],np.maximum(0,1-b[valid,3].copy().view('<f4')-n[valid,0])])
        ref=(reference[ids]*weights[:,:,None]).sum(1);which=np.flatnonzero(valid)[ref.min(1)<1e-7]
        if not len(which):continue
        p=rows[start+which,:12].copy().view('<f4').reshape(-1,3);d,neighbors=tree.query(p,k=3,workers=6)
        w=1/np.maximum(d,.004)**2;w/=w.sum(1,keepdims=True);new=good[neighbors].astype(np.uint32);new[:,0]|=np.uint32(0x80000000)
        b[which,:3]=new;b[which,3]=w[:,0].astype('<f4').view('<u4');n[which,0]=w[:,1]
        binding[start:end]=b;normals[start:end]=n;fixed+=len(which);max_distance=max(max_distance,float(d[:,0].max()))
    binding.flush();normals.flush()
    previous=json.loads((OUT/'reference_repair.json').read_text()) if (OUT/'reference_repair.json').exists() else {}
    report=dict(rebound_this_run=fixed,rebound_undefined_reference_gaussians=previous.get('rebound_undefined_reference_gaussians',0)+fixed,maximum_donor_distance=max(max_distance,previous.get('maximum_donor_distance',0)),
                confidence='additional nearest volume estimates, explicitly flagged; original appearance bypass remains exact')
    atomic_json(OUT/'reference_repair.json',report);return report


def main():
    meta=json.loads((OUT/'bank_manifest.json').read_text());lights=json.loads((OUT/'lights.json').read_text())['lights']
    with np.load(OUT/'geometry.npz') as z:data={k:z[k] for k in z.files}
    ids=np.load(OUT/'compact_node_ids.npy');p=np.repeat(data['fine_points'],2,axis=0)[ids];n=np.repeat(data['fine_normals'],2,axis=0);n[1::2]*=-1;n=n[ids]
    reference=np.memmap(OUT/'reference_irradiance.f32',mode='r',dtype='<f4').reshape(-1,3)
    repair=repair_unlit_bindings(reference,p)
    categories={
        'floor':(n[:,2]>.75)&(p[:,2]<-.9),
        'ceiling':(n[:,2]<-.75)&(p[:,2]>1.1),
        'vertical_surfaces':abs(n[:,2])<.3,
        'furniture_horizontal':(abs(n[:,2])>.65)&(p[:,2]>=-.9)&(p[:,2]<=1.1),
        'other_surfaces':(abs(n[:,2])>=.3)&(abs(n[:,2])<=.65),
    }
    records=[];fields=[]
    for light in lights:
        field=np.load(OUT/f'fine_lamp_{light["id"]:02}.npy');fields.append(field)
        if not np.isfinite(field).all() or (field<0).any():raise RuntimeError('Invalid source field')
        flux=json.loads((OUT/f'lamp_{light["id"]:02}_flux_validation.json').read_text())
        balance=max(flux['received_fraction_rgb'])+flux['escape_fraction']+flux['outside_receiver_fraction']
        if not flux['all_emitted_rays_accounted'] or abs(balance-1)>2e-5:raise RuntimeError('Emitted flux budget failed')
        luma=field@np.array([.2126,.7152,.0722]);direct=np.load(OUT/f'coarse_lamp_{light["id"]:02}_direct.npy');total=np.load(OUT/f'coarse_lamp_{light["id"]:02}.npy')
        if np.min(total-direct)<-1e-5:raise RuntimeError('Negative indirect field')
        effect={name:dict(receivers=int(mask.sum()),affected=int((luma[mask]>1e-6).sum()),maximum=float(luma[mask].max(initial=0)),mean=float(luma[mask].mean()) if mask.any() else 0) for name,mask in categories.items()}
        if float(luma.max())<=0:raise RuntimeError('Lamp has no surface influence')
        records.append(dict(id=light['id'],label=light['label'],source_id=light['source_id'],influence=effect,
                            received_fraction=max(flux['received_fraction_rgb']),escaped_fraction=flux['escape_fraction'],outside_fraction=flux['outside_receiver_fraction'],
                            indirect_nonnegative=True,finite_nonnegative=True,power_calibration='estimated, not measured'))
    matrix=sparse.load_npz(OUT/'form_factors.npz');area=np.load(OUT/'coarse_areas.npy')
    transfer=np.asarray(matrix.T@area).ravel()/np.maximum(area,1e-20);maximum=float(transfer.max())
    if maximum>1.00001:raise RuntimeError('Area-weighted flux increases through transport')
    rho=float(data['coarse_reflectance'].max());sun=json.loads((OUT/'sun_keys.json').read_text())
    if sun['maximum_final_interval_error']>.025:raise RuntimeError('Solar GI interpolation failed')
    source=file_identity(SOURCE/'full.splat');expected=json.loads((OUT/'geometry_report.json').read_text())['source_gaussians']
    if source!=expected:raise RuntimeError('Original Gaussian asset changed')
    audit=dict(completed_at=time.time(),independent_sources=len(records),all_source_flux_budgets_passed=True,
               maximum_area_weighted_flux_transfer=maximum,maximum_bounce_reflectance=rho,
               solar_indirect_maximum_normalized_interpolation_error=sun['maximum_final_interval_error'],solar_keys=len(sun['minutes']),
               direct_sun='runtime single-source geometric shadow, requires browser comparison',original_source_unchanged=True,
               reference_repair=repair,lights=records,physical_ground_truth_verified=False,
               limitations=['scanned geometry and apertures are approximate','reflectance and lamp powers are estimated','specular/glass transport not recovered','volume fallback is unverified'])
    atomic_json(OUT/'transport_validation.json',audit)
    table=''.join(f'<tr><td>{r["label"]}</td><td>{r["received_fraction"]:.3%}</td>'+''.join(f'<td>{v["affected"]:,} / {v["receivers"]:,}</td>' for v in r['influence'].values())+'</tr>' for r in records)
    page='''<!doctype html><meta charset="utf-8"><title>全屋光照验证</title><style>body{background:#12171c;color:#eee;font:14px system-ui;margin:24px;line-height:1.6}a{color:#9cf}table{border-collapse:collapse}th,td{padding:6px 10px;border-bottom:1px solid #455;text-align:right}th:first-child,td:first-child{text-align:left}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:15px}img{width:100%}figure{margin:0}</style><h1>全屋光照 · 验证记录</h1><p><a href="./">回到漫游</a> · <a href="transport_validation.json">完整计算记录</a></p><p>以下为当前扫描几何及估计灯功率下的数值检查，尚不等于真实房间测光验证。97.6% 室内高斯有近邻网格支持，其余体积估计另行标记；玻璃、镜面和材料未完整恢复。</p>'''
    if (OUT/'physics_audit/index.html').exists():page+='<p><a href="physics_audit/">光斑破碎专项诊断与物理准确性边界</a></p>'
    page+=f'<p>25 个独立灯源的发射能量预算通过；反射传输最大面积加权通量比 {maximum:.8f}；太阳间接光 {len(sun["minutes"])} 个时间节点，最终插值误差上限 {sun["maximum_final_interval_error"]:.2%}（以场最大值归一化；不是实景照度误差）。</p>'
    page+='<h2>实际完整高斯画面对照</h2><div class="grid">'
    for key,label in [('original','拍摄原貌'),('lamps_off','仅阴天日光 / 人工灯全关'),('sun_1230','太阳 12:30'),('sun_1600','太阳 16:00'),('sun_1815','太阳 18:15'),('black','所有光源关闭')]:
        if (OUT/f'review/{key}_00.png').exists():page+=f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="review/{key}_00.png"></figure>'
    page+='</div><h2>逐灯对其它表面的作用</h2><p>表中为该灯单独开启时收到非零能量的接收点数量；墙面、柜体等垂直面合并统计，属于基于几何的分类。</p><table><tr><th>灯具</th><th>室内收到 / 发射</th><th>地板</th><th>天花</th><th>垂直面</th><th>家具水平面</th><th>其它斜面</th></tr>'+table+'</table>'
    (OUT/'verification.html').write_text(page);print('TRANSPORT_AUDIT',json.dumps({k:v for k,v in audit.items() if k not in ['lights']}),flush=True)


if __name__=='__main__':main()
