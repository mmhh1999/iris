"""EXP0007: paired 2D/geometry screening audit, no sun labels inferred.

CPU default; uses only train frames, exact undistorted image/calibration pair.
Writes immutable run directory with provenance, selected images and diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import mitsuba as mi
from utils.sun_patch import screen_image_for_sun_patches, filter_candidates_by_geometry
from utils.geometry_screening import scannetpp_pose_in_mesh_frame, validate_scannetpp_bounds

ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def run(args):
    if args.images < 1 or args.samples < 1:
        raise ValueError('positive images/samples required')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    mi.set_variant('llvm_ad_rgb')
    source=['experiments/scannetpp_geometry_audit.py','utils/geometry_screening.py','utils/sun_patch.py','utils/solar_geometry.py','utils/window_geometry.py']
    meta=dict(experiment_id='EXP0007',seed=0,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              sources={s:digest(ROOT/s) for s in source},config=vars(args),
              python=platform.python_version(),mitsuba=mi.__version__,numpy=np.__version__,opencv=cv2.__version__,
              variant=mi.variant(),hypothesis='Mesh support rejects some bright apertures; surviving regions are not certified sun patches.',
              geometry_thresholds=dict(max_invalid_frac=.4,max_depth_factor=1.3),
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    all_rows=[]; scenes=[]; skipped=[]; start=time.perf_counter()
    for sid in args.scenes:
        p=Path(args.data_root)/sid
        tf=p/'dslr/nerfstudio/transforms_undistorted.json'
        mesh=p/'scans/mesh_aligned_0.05.ply'
        d=json.loads(tf.read_text())
        if d.get('camera_model')!='PINHOLE' or any(d.get(k,0)!=0 for k in ('k1','k2','k3','k4','p1','p2')):
            raise ValueError('requires undistorted PINHOLE metadata')
        K=np.array([[d['fl_x'],0,d['cx']],[0,d['fl_y'],d['cy']],[0,0,1.]])
        frames=sorted([f for f in d['frames'] if not f.get('is_bad',False)],key=lambda f:f['file_path'])
        test_names={Path(f['file_path']).name for f in d.get('test_frames',[])}
        frames=[f for f in frames if Path(f['file_path']).name not in test_names]
        frames=[frames[i] for i in np.linspace(0,len(frames)-1,min(args.images,len(frames))).astype(int)]
        scene=mi.load_dict({'type':'scene','mesh':{'type':'ply','filename':str(mesh.resolve())}})
        try:
            validate_scannetpp_bounds(d['aabb_range'], [np.asarray(scene.bbox().min), np.asarray(scene.bbox().max)])
        except ValueError as exc:
            skipped.append(dict(scene=sid,reason=str(exc)))
            (out/'skipped.json').write_text(json.dumps(skipped,indent=2))
            print(sid, 'SKIPPED:', exc, flush=True)
            continue
        diag=float(np.linalg.norm(np.asarray(scene.bbox().max)-np.asarray(scene.bbox().min)))
        scenes.append(dict(scene=sid,mesh_sha256=digest(mesh),transforms_sha256=digest(tf),aabb_diag=diag))
        for frame in frames:
            name=Path(frame['file_path']).name
            path=p/'dslr/resized_undistorted_images'/name
            img=cv2.imread(str(path))
            if img is None or img.shape[:2]!=(d['h'],d['w']):
                raise ValueError(f'missing or mismatched image: {path}')
            cs=screen_image_for_sun_patches(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),top_k=3)
            pose=scannetpp_pose_in_mesh_frame(frame['transform_matrix'])
            filter_candidates_by_geometry(cs,scene,K,pose,img.shape[:2],n_samples=args.samples,aabb_diag=diag)
            row=dict(scene=sid,image=name,image_sha256=digest(path),candidates=[])
            vis=img.copy()
            for i,c in enumerate(cs):
                row['candidates'].append(dict(rank=i,score=c.score,area_px=c.area_px,status=c.geometry_status,**c.geometry_evidence))
                color=(0,200,0) if c.geometry_status=='surface_supported_unverified' else (0,0,255)
                cv2.drawContours(vis,[c.contour],-1,color,4)
                x,y=map(int,c.contour[0,0]);cv2.putText(vis,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
            cv2.imwrite(str(out/f'{sid}_{Path(name).stem}.jpg'),cv2.resize(vis,(600,400)))
            all_rows.append(row)
        (out/"partial_results.json").write_text(json.dumps(dict(scenes=scenes,images=all_rows,skipped=skipped),indent=2))
        print(sid,'completed',len(frames),'frames',flush=True)
    candidates=[c for r in all_rows for c in r['candidates']]
    summary=dict(n_scenes=len(scenes),n_scenes_skipped=len(skipped),n_images=len(all_rows),n_candidates=len(candidates),
                 n_surface_supported=sum(c['status']=='surface_supported_unverified' for c in candidates),
                 n_rejected=sum(c['status']!='surface_supported_unverified' for c in candidates),
                 elapsed_seconds=time.perf_counter()-start,
                 limitation='No labels, precision/recall or semantic sun confirmation. Mesh hits can include windows/glass/exterior surfaces.')
    (out/'results.json').write_text(json.dumps(dict(summary=summary,scenes=scenes,skipped=skipped,images=all_rows),indent=2))
    lines=['| Scene | Images | Candidates | Surface supported (unverified) | Rejected |',
           '|---|---:|---:|---:|---:|']
    for info in scenes:
        rows=[r for r in all_rows if r['scene']==info['scene']]
        cs=[c for r in rows for c in r['candidates']]
        n=sum(c['status']=='surface_supported_unverified' for c in cs)
        lines.append(f"| {info['scene']} | {len(rows)} | {len(cs)} | {n} | {len(cs)-n} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default=str(ROOT/'data_download/scannetpp/data'))
    p.add_argument('--scenes',nargs='+',default=['7b04052ad0','3cb9f85891'])
    p.add_argument('--images',type=int,default=12)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--out',default=str(ROOT/'experiments/out/EXP0007_geometry_corrected'))
    run(p.parse_args())
