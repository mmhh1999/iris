"""Evaluation-only floor truth access; matched masks for sun/sky models."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw

def linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def srgb(x):
    x=np.clip(x,0,1)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--suffix',default='1000_600');p.add_argument('--out',type=Path,required=True);p.add_argument('--input',type=Path);a=p.parse_args()
    inputs=a.input or a.base/'input'
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[];panels=[];gt=np.array([.28,.20,.12]);kernel=np.ones((3,3),np.uint8)
    for view in [3,9]:
        truth=np.load(a.truth/'generator_truth'/f'sun_a_view{view:02d}.npz');floor=truth['floor'].astype(bool)
        interior=cv2.erode(floor.astype(np.uint8),kernel).astype(bool)
        lit=cv2.erode((floor&truth['sun_visible']).astype(np.uint8),kernel).astype(bool)
        shadow=cv2.erode((floor&~truth['sun_visible']).astype(np.uint8),kernel).astype(bool)
        for condition in ['sun_a','sky']:
            run=a.base/f'{condition}_{a.suffix}'
            assert not (run/'INVALID_SCIENTIFIC_RESULT.json').exists(),f'Excluded scientific run: {run}'
            complete=json.loads((run/'result.json').read_text());assert complete['completed']
            r=np.load(run/'evaluation'/f'view{view:02d}.npz');albedo=linear(r['base_color'].transpose(1,2,0));rough=r['roughness'].squeeze();opacity=r['opacity'].squeeze()
            assert np.isfinite(albedo).all() and np.isfinite(rough).all()
            photo=np.array(Image.open(inputs/condition/'val'/f'view{view:02d}.png')).astype(float)/255;pbr=r['pbr'].transpose(1,2,0)
            gray=albedo.mean(-1);gap=float(gray[lit].mean()-gray[shadow].mean()) if min(lit.sum(),shadow.sum())>=32 else None
            scale=float(np.sum(albedo[interior]*gt)/np.sum(albedo[interior]**2))
            row=dict(condition=condition,view=view,lit_count=int(lit.sum()),shadow_count=int(shadow.sum()),floor_interior_count=int(interior.sum()),floor_opacity_mean=float(opacity[interior].mean()),floor_opacity_fraction_above_095=float((opacity[interior]>.95).mean()),albedo_mae_linear=float(np.abs(albedo[interior]-gt).mean()),albedo_scale_aligned_mae=float(np.abs(albedo[interior]*scale-gt).mean()),evaluation_only_albedo_scale=scale,floor_mean_linear_albedo=albedo[interior].mean(0).tolist(),albedo_gap=gap,normalized_gap=None if gap is None else gap/float(gray[floor].mean()),roughness_mean=float(rough[interior].mean()),roughness_mae=float(np.abs(rough[interior]-.9).mean()),pbr_psnr=float(-10*np.log10(np.mean((pbr-photo)**2))),floor_pbr_psnr=float(-10*np.log10(np.mean((pbr[interior]-photo[interior])**2))))
            rows.append(row)
            floor_pred=np.zeros_like(albedo);floor_pred[floor]=srgb(albedo[floor]);floor_gt=np.zeros_like(albedo);floor_gt[floor]=srgb(gt)
            masks=np.zeros_like(albedo);masks[shadow]=[.15,.45,1];masks[lit]=[1,.8,.1]
            panels.append((f'{condition} heldout view {view}',[photo,np.clip(pbr,0,1),srgb(albedo),floor_pred,floor_gt,masks]))
    differences=[]
    for view in [3,9]:
        s=next(r for r in rows if r['condition']=='sun_a' and r['view']==view);c=next(r for r in rows if r['condition']=='sky' and r['view']==view)
        differences.append(dict(view=view,raw_gap_change=None if s['albedo_gap'] is None else s['albedo_gap']-c['albedo_gap'],normalized_gap_change=None if s['normalized_gap'] is None else s['normalized_gap']-c['normalized_gap']))
    report=dict(protocol='P1 SGS public adaptation; fixed known geometry; 12 train / 2 heldout views; one room, one seed; no statistical independence across pixels',truth_in_training=False,color_space='predicted sRGB base color converted to linear for material scoring',geometry_quality_gate=all(r['floor_opacity_fraction_above_095']>.95 for r in rows),convergence_established=False,views=rows,difference_in_differences=differences,comparison_to_EXP0019_IRIS='not directly comparable: different tone mapping, learned prior and optimization protocol')
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    canvas=Image.new('RGB',(128*6,120*4),(25,25,25));draw=ImageDraw.Draw(canvas)
    labels=['Input','PBR reconstruction','SGS albedo','Floor albedo','True floor albedo','Sun/shadow mask']
    for y,(title,ims) in enumerate(panels):
        for x,(label,im) in enumerate(zip(labels,ims)):
            canvas.paste(Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)),(128*x,120*y+24));draw.text((128*x+2,120*y+2),label,fill='white')
        draw.text((2,120*y+13),title,fill='white')
    canvas.resize((1536,960)).save(a.out/'comparison.png')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
