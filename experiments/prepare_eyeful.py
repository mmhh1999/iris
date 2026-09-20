"""Undistort a real HDR pilot without removing/baking its illumination.

Output remains linear DCI-P3, with no white balance or exposure normalization.
Preview uses the official white-balance and display curve only.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np

WB = {'riverview': [1.077719, 1., 1.145992], 'apartment': [.726097, 1., 1.741252]}


def preview(rgb, scene):
    x = rgb * np.array(WB[scene])
    x = np.where(x <= .0031308, x*12.92, 1.055*np.maximum(x, 0)**(1/2.4)-.055)
    return np.uint8(np.clip(x*255, 0, 255))[..., ::-1]


def run(root, out, scene):
    source = root / scene
    dest = out / scene
    dest.mkdir(parents=True, exist_ok=True)
    selected = json.loads((source / 'pilot_selection.json').read_text())['cameras']
    splits = json.loads((source / 'splits.json').read_text())
    records = []
    tiles = []
    for c in selected:
        name = c['cameraId']
        img = cv2.imread(str(source / 'images-1k' / (name+'.exr')), -1)
        if img is None or not np.isfinite(img).all():
            raise ValueError(f'Invalid HDR: {name}')
        h, w = img.shape[:2]
        K = np.array(c['K']).T.copy()
        K[0] *= w / c['width']
        K[1] *= h / c['height']
        if c['distortionModel'] != 'RadialAndTangential':
            raise ValueError('Only v2 pinhole pilot is supported')
        D = np.array(c['distortion'])
        # Keep original scaled intrinsics; record padding in a separate mask.
        mx, my = cv2.initUndistortRectifyMap(K, D, None, K, (w,h), cv2.CV_32FC1)
        valid = (mx >= 0) & (mx < w-1) & (my >= 0) & (my < h-1)
        und = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        pose = np.linalg.inv(np.array(c['T']).T)
        R = pose[:3,:3]
        if not np.allclose(R.T@R, np.eye(3), atol=1e-5) or np.linalg.det(R) < .999:
            raise ValueError('Invalid camera rotation')
        # Verify OpenCV undistortion against independent projection at image points.
        px = np.array([[w*.25,h*.25],[w*.5,h*.5],[w*.75,h*.75]], np.float64)
        rays = np.c_[px, np.ones(3)] @ np.linalg.inv(K).T
        distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, D)
        recovered = cv2.undistortPoints(distorted, K, D, P=K).reshape(-1,2)
        error = float(np.max(np.abs(recovered-px)))
        if error > .01:
            raise ValueError(f'Calibration round trip {error}')
        stem = name.replace('/', '_')
        if not cv2.imwrite(str(dest/(stem+'.exr')), und):
            raise IOError(stem)
        cv2.imwrite(str(dest/(stem+'_valid.png')), valid.astype('uint8')*255)
        show = preview(und[...,::-1], scene)
        cv2.imwrite(str(dest/(stem+'.jpg')), show)
        split = next((k for k, v in splits.items() if name in v), None)
        if split is None:
            raise ValueError('Image absent from official split')
        records.append(dict(camera_id=name, file=stem+'.exr', mask=stem+'_valid.png',
                            width=w, height=h, K=K.tolist(), c2w_opencv=pose.tolist(),
                            split=split, valid_fraction=float(valid.mean()),
                            raw_min=float(img.min()), raw_max=float(img.max()),
                            raw_fraction_above_one=float((img>1).mean()),
                            calibration_roundtrip_px=error))
        tile = cv2.resize(show,(171,256))
        cv2.putText(tile, name.split('/')[-1], (4,18), cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
        tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)])
    cv2.imwrite(str(dest/'contact_sheet.jpg'),grid)
    manifest = dict(scene=scene, provenance='real_photographs',
                    linear_color_space='DCI-P3 as provided; no white balance applied',
                    preview='official white balance and sRGB transfer; not training input',
                    world='right handed, Y up, metres', cameras=records,
                    mesh=str((source/'mesh.obj').resolve()),
                    material_ground_truth=False, sunlight_ground_truth=False,
                    strict_photo_only=False)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(scene, len(records), 'finite HDRs, calibration and splits verified',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('data_download/eyefultower'))
    p.add_argument('--out',type=Path,default=Path('experiments/out/EXP0014_eyeful'))
    p.add_argument('--scenes',nargs='+',default=['riverview','apartment'])
    a=p.parse_args()
    for scene in a.scenes:
        run(a.root,a.out,scene)
