"""T11 v2: EXP0035 (calibrated-direction perturbation) and EXP0036 (window-geometry error +
occluder). Same protocol as experiments/solarir_test.py (D0019); only the conditions differ.

EXP0035: clean v1 ground truth; arm C with its calibrated bearing replaced by
         true + delta, delta in {0, 2, 5, 10, 20} deg (the failure mode of calibration is yaw).
EXP0036: ground truth rendered with the true window and a sofa-like box that shades part of
         the sun path; every arm (B, C, C_oracle, A_hi) believes a window that is shifted and
         shrunk by a few centimetres (as a reconstruction would get it) and knows the box.
         B and C re-estimate their sun from the images with the believed geometry.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

import numpy as np
import mitsuba as mi

import disk_guard
import solarir_scene as S
import solarir_test as T
from scipy.ndimage import binary_dilation, binary_erosion

TRUE_WIN = (tuple(S.WIN_X), tuple(S.WIN_Z))
BELIEVED_WIN = ((-0.77, 0.83), (0.83, 1.97))    # pre-registered +-3 cm: whole window 3 cm east, sill +3 cm, head -3 cm
SOFA = ((0.2, 1.4), (1.5, 2.3), 0.45)            # x-range, y-range, height (m): shades the midday path


def write_room(path, win, boxes=()):
    old = S.WIN_X, S.WIN_Z
    S.WIN_X, S.WIN_Z = win
    try:
        S.room_shell_obj(path)
    finally:
        S.WIN_X, S.WIN_Z = old
    if boxes:
        lines = Path(path).read_text().splitlines()
        nv = sum(1 for l in lines if l.startswith('v '))
        verts, faces = [], []
        for (x0, x1), (y0, y1), h in boxes:
            c = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0), (x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]
            b = nv + len(verts) + 1
            verts += c
            for q in [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:   # top + 4 outward sides
                faces += [(b + q[0], b + q[1], b + q[2]), (b + q[0], b + q[2], b + q[3])]
        Path(path).write_text('\n'.join([l for l in lines if l.startswith('v ')]
                                        + ['v %.6f %.6f %.6f' % v for v in verts]
                                        + [l for l in lines if l.startswith('f ')]
                                        + ['f %d %d %d' % f for f in faces]) + '\n')


def traced_region_masks(gt_obj):
    """Region masks from ray tracing the true geometry (the box shades part of the patch)."""
    geom = T.geom_scene(gt_obj)
    p, n = S.floor_texel_points()
    lit_t = [S.sunlit(geom, p, n, S.sun_dir(t))[0].reshape(S.MODEL_TEX, S.MODEL_TEX) for t in S.TRAIN_TIMES]

    def masks(observed, n_=S.MODEL_TEX):
        edge = np.zeros_like(observed)
        for m in lit_t:
            edge |= binary_dilation(m) & ~binary_erosion(m)
        lit = np.any(lit_t, 0)
        return {'all': observed, 'all_texels': np.ones_like(observed), 'lit_interior': observed & lit & ~edge,
                'boundary': observed & edge, 'never_lit': observed & ~lit & ~edge}
    return masks


def run_arm(out, obj, gt, gt_tex, arm, times, dirs, a, seed, name, meta):
    if (out / 'runs' / f'{name}.json').exists():
        return
    disk_guard.check(out, need_gb=0.05)
    hits = S.primary_hits(T.geom_scene(obj))
    res = T.fit(obj, gt, arm, times, dirs, a.iters, a.spp, seed)
    ev = T.evaluate(obj, gt, gt_tex, arm, res, dirs, times, hits)
    rec = T.save_run(out, name, res, ev, dict(meta, arm=arm, seed=seed, iters=a.iters, spp=a.spp))
    f = rec['floor']
    print(f'  {name}: floor MAE_s {f["all"]["mae_scaled"]:.4f} corr {f["all"]["corr"]:.3f} '
          f'patch_bias {f["patch_bias"]:+.4f} | 14:30 PSNR {rec["heldout"]["14:30"]["psnr"]:.2f} '
          f'IoU {rec["heldout"]["14:30"]["patch_iou"]:.3f} | {rec["seconds"]:.0f}s', flush=True)


def exp0035(a):
    """C with a wrong bearing, on the clean v1 data."""
    v1 = Path(a.v1)
    out = Path(a.out) / 'EXP0035'
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    gt, gt_tex = T.load_gt(v1)
    obj = v1 / 'room_shell.obj'
    for seed in a.seeds:
        for delta in a.deltas:
            dirs = {t: S.sun_dir(t, float(delta)) for t in T.ALL_TIMES}
            run_arm(out, obj, gt, gt_tex, 'C', S.TRAIN_TIMES, dirs, a, seed, f'C_bearing{delta:g}_multi_s{seed}',
                    dict(exp='EXP0035', mode='multi', bearing_error_deg=delta,
                         dir_err_deg={t: round(S.angle_deg(dirs[t], S.sun_dir(t)), 3) for t in T.ALL_TIMES}))


def exp0036(a):
    out = Path(a.out) / 'EXP0036'
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    gt_obj, obj = out / 'room_true.obj', out / 'room_believed.obj'
    write_room(gt_obj, TRUE_WIN, [SOFA])
    write_room(obj, BELIEVED_WIN, [SOFA])
    tex = S.floor_texture_gt(0)
    if not (out / 'gt.npz').exists():
        imgs = {}
        for t in T.ALL_TIMES:
            sc = S.make_scene(gt_obj, tex, S.WALL_GT, 'gt', d=S.sun_dir(t))
            imgs[T.key(t)] = np.stack([np.array(mi.render(sc, sensor=i, spp=a.gt_spp, seed=1000 + i))
                                       for i in range(len(S.CAMERAS))]).astype(np.float32)
        np.savez_compressed(out / 'gt.npz', floor=tex, **imgs)
    gt, gt_tex = T.load_gt(out)
    T.region_masks = traced_region_masks(gt_obj)          # truth regions include the box's shadow
    est_path = out / 'estimates.json'
    if not est_path.exists():
        geom = T.geom_scene(obj)                            # estimators only know the believed geometry
        est = {'B': {}, 'C': {}}
        for t in S.TRAIN_TIMES:
            m = S.PatchMatcher(geom, list(gt[t]))
            (az, el), iou = S.search_direction(m)
            est['B'][t] = dict(az=az, el=el, iou=round(iou, 4),
                               err_deg=round(S.angle_deg(S.dir_from_az_el(az, el), S.sun_dir(t)), 3))
        m = S.PatchMatcher(geom, list(gt[S.SINGLE_TIME]))
        bearing, iou = S.search_bearing(m, S.SINGLE_TIME)
        est['C'] = dict(bearing_deg=bearing, iou=round(iou, 4),
                        err_deg={t: round(S.angle_deg(S.sun_dir(t, bearing), S.sun_dir(t)), 3) for t in T.ALL_TIMES})
        est_path.write_text(json.dumps(est, indent=2))
        print('  estimates', json.dumps(est))
    est = json.loads(est_path.read_text())
    for seed in a.seeds:
        for mode in a.modes:
            for arm in a.arms:
                times = T.MODES[mode]
                run_arm(out, obj, gt, gt_tex, arm, times, T.arm_dirs(arm, est, times), a, seed,
                        f'{arm}_{mode}_s{seed}', dict(exp='EXP0036', mode=mode,
                                                      true_window=TRUE_WIN, believed_window=BELIEVED_WIN, box=SOFA))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True, choices=['EXP0035', 'EXP0036'])
    p.add_argument('--v1', default=str(ROOT / 'experiments/out/T11_solarir'))
    p.add_argument('--out', default=str(ROOT / 'experiments/out/T11_solarir_v2'))
    p.add_argument('--arms', type=lambda s: s.split(','), default=['B', 'C', 'C_oracle', 'A_hi'])
    p.add_argument('--modes', type=lambda s: s.split(','), default=['multi', 'single'])
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=[0, 1, 2])
    p.add_argument('--deltas', type=lambda s: [float(x) for x in s.split(',')], default=[0, 2, 5, 10, 20])
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--spp', type=int, default=32)
    p.add_argument('--gt_spp', type=int, default=512)
    a = p.parse_args()
    t0 = time.time()
    {'EXP0035': exp0035, 'EXP0036': exp0036}[a.exp](a)
    print(f'done {a.exp} in {time.time() - t0:.0f}s')
