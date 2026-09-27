"""T11 / EXP0030-EXP0034: SolarIR controlled hypothesis test in Mitsuba 3.

Design and pre-registered rules: research/SOLARIR_HYPOTHESIS_TEST_ZH.md; ID mapping and the
metric clarification made before any arm result: research/DECISIONS.md D0019.
Scene, lighting arms, transport and estimators: experiments/solarir_scene.py (unit tests in
tests/test_solar_transport.py).

Stages (each writes into --out; nothing large is stored, no checkpoints):
  gt        render ground truth (Mitsuba sunsky, ephemeris sun) for train + held-out times
  estimate  arm B: per-time 2-DOF sun direction from the image patch; arm C: 1-DOF scene bearing
            from the 11:30 patch, ephemeris for everything else
  diag      D1: materials learned under the exact GT lighting (pipeline check)
            D2: GT materials fixed, each arm's lighting learned (forward-model adequacy)
  fit       arms x modes x seeds: shared floor texture + wall albedo, per-time lighting
  report    aggregate, apply R1-R3
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
import drjit as dr

mi.set_variant('cuda_ad_rgb')

from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

import disk_guard
import solarir_scene as S

ALL_TIMES = S.TRAIN_TIMES + S.HELDOUT_TIMES
MODES = {'single': [S.SINGLE_TIME], 'multi': S.TRAIN_TIMES}
ARMS = ['W', 'A', 'A_hi', 'B', 'C', 'C_oracle', 'C_oracle16']
# Lighting learning rates chosen per arm on the training-image fit with true materials only (D2);
# never on material error or held-out metrics (D0019).
LR_LIGHT = {'W': 0.1, 'A': 0.1, 'A_hi': 0.4, 'B': 0.1, 'C': 0.1, 'C_oracle': 0.1, 'C_oracle16': 0.1, 'gt': 0.1}
EVAL_SPP = 512


def key(t):
    return t.replace(':', '')


def load_gt(out):
    z = np.load(out / 'gt.npz')
    return {t: z[key(t)] for t in ALL_TIMES}, z['floor']


def geom_scene(obj):
    return S.make_scene(obj, np.full((4, 4, 3), 0.5, np.float32), S.WALL_GT, 'gt', with_emitters=False)


# ---------------------------------------------------------------- stages: gt, estimate

def stage_gt(out, obj, spp):
    tex = S.floor_texture_gt(0)
    imgs = {}
    for t in ALL_TIMES:
        sc = S.make_scene(obj, tex, S.WALL_GT, 'gt', d=S.sun_dir(t))
        imgs[key(t)] = np.stack([np.array(mi.render(sc, sensor=i, spp=spp, seed=1000 + i))
                                 for i in range(len(S.CAMERAS))]).astype(np.float32)
        az, el = S.az_el(S.sun_dir(t))
        print(f'  GT {t}: sun az {az:.1f} el {el:.1f}', flush=True)
    np.savez_compressed(out / 'gt.npz', floor=tex, **imgs)


def stage_estimate(out, obj):
    gt, _ = load_gt(out)
    geom = geom_scene(obj)
    res = {'B': {}, 'C': {}}
    for t in S.TRAIN_TIMES:
        m = S.PatchMatcher(geom, list(gt[t]))
        (az, el), iou = S.search_direction(m)
        taz, tel = S.az_el(S.sun_dir(t))
        res['B'][t] = dict(az=az, el=el, iou=round(iou, 4), true_az=round(taz, 3), true_el=round(tel, 3),
                           err_deg=round(S.angle_deg(S.dir_from_az_el(az, el), S.sun_dir(t)), 3))
        print(f'  B {t}: az {az:.2f} el {el:.2f} (true {taz:.2f} {tel:.2f}) err {res["B"][t]["err_deg"]}  IoU {iou:.3f}')
    m = S.PatchMatcher(geom, list(gt[S.SINGLE_TIME]))
    bearing, iou = S.search_bearing(m, S.SINGLE_TIME)
    res['C'] = dict(calibrated_on=S.SINGLE_TIME, bearing_deg=bearing, iou=round(iou, 4), true_bearing_deg=0.0,
                    err_deg={t: round(S.angle_deg(S.sun_dir(t, bearing), S.sun_dir(t)), 3) for t in ALL_TIMES})
    print(f'  C bearing {bearing:.2f} deg (true 0), IoU {iou:.3f}')
    (out / 'estimates.json').write_text(json.dumps(res, indent=2))
    return res


def arm_dirs(arm, est, train_times):
    """Sun direction the arm uses at every time. B only knows its image estimates at the training
    times it saw; held-out directions come from a least-squares linear trajectory in (az, el) vs
    time (a constant when only one time was seen). C uses the ephemeris rotated by its one
    calibrated bearing; C_oracle the true ephemeris."""
    if arm in ('C_oracle', 'C_oracle16'):
        return {t: S.sun_dir(t) for t in ALL_TIMES}
    if arm == 'C':
        return {t: S.sun_dir(t, est['C']['bearing_deg']) for t in ALL_TIMES}
    if arm == 'B':
        h = np.array([S.hours(t) for t in train_times])
        az = np.array([est['B'][t]['az'] for t in train_times])
        el = np.array([est['B'][t]['el'] for t in train_times])
        dirs = {t: S.dir_from_az_el(est['B'][t]['az'], est['B'][t]['el']) for t in train_times}
        for t in ALL_TIMES:
            if t in dirs:
                continue
            if len(h) == 1:
                dirs[t] = dirs[train_times[0]]
            else:
                pa, pe = np.polyfit(h, az, 1), np.polyfit(h, el, 1)
                dirs[t] = S.dir_from_az_el(np.polyval(pa, S.hours(t)), np.polyval(pe, S.hours(t)))
        return dirs
    return {t: None for t in ALL_TIMES}


# ---------------------------------------------------------------- fitting

def loss_weights(img, c_frac=0.5, sigma=2.0):
    """Fixed per-pixel weights 1/(blurred GT luminance + c)^2, so the squared error is relative:
    the bright patch no longer dominates the walls. Weights come from GT only, so the gradient of
    the weighted MSE stays unbiased under Monte Carlo noise."""
    L = gaussian_filter(S.lum(img), sigma)
    w = 1.0 / (L + c_frac * np.median(L)) ** 2
    return np.repeat((w / w.mean())[..., None], 3, -1).astype(np.float32)


def fit(obj, gt, arm, times, dirs, iters, spp, seed, views_per_time=2, lr_mat=0.05, lr_light=None,
        learn_materials=True, floor_init=None, wall_init=None, log_every=0, loss='mse', decay=True):
    """loss='mse' (linear) is the protocol; 'rel' is kept only to document why it was rejected:
    it stops the generic envmap arm from ever forming a sun (D0019)."""
    rng = np.random.default_rng(seed)
    lr_light = LR_LIGHT[arm] if lr_light is None else lr_light
    tex0 = floor_init if floor_init is not None else \
        np.clip(0.3 + 0.02 * rng.standard_normal((S.MODEL_TEX, S.MODEL_TEX, 3)), 0, 1).astype(np.float32)
    wall0 = list(wall_init) if wall_init is not None else [0.5, 0.5, 0.5]
    st0 = S.init_state(arm) if arm != 'gt' else {}
    scenes = {t: S.make_scene(obj, tex0, wall0, arm, st0, d=dirs[t], integrator='prb') for t in times}
    params = {t: mi.traverse(sc) for t, sc in scenes.items()}
    opt = mi.ad.Adam(lr=lr_mat)
    if learn_materials:
        opt['floor'] = mi.TensorXf(tex0)
        opt['wall'] = mi.Color3f(*wall0)
    for t in times:
        for k, v in st0.items():
            name = f'{k}_{t}'
            if k == 'E':
                opt[name] = mi.Color3f(*np.log(v).tolist())
            elif k == 'win':
                opt[name] = mi.TensorXf(np.log(v))
            else:
                opt[name] = mi.Float(np.log(v).ravel())
            opt.set_learning_rate({name: lr_light})
    up = {'sky': S.SKY[arm][1] if arm in S.SKY else 1, 'env': 1}
    pad = {k: S.env_pad_index(*v.shape[:2], up[k]) for k, v in st0.items() if k in up}

    def push(t):
        p = params[t]
        if learn_materials:
            p['floor.bsdf.reflectance.data'] = opt['floor']
            p['room.bsdf.reflectance.value'] = opt['wall']
        for k, v in st0.items():
            x = dr.exp(opt[f'{k}_{t}'])
            if k == 'E':
                p['sun.irradiance.value'] = x
            elif k == 'win':
                p['window.emitter.radiance.data'] = x
            else:
                p['sky.data'] = S.padded_env(x, *v.shape[:2], pad[k], up[k])
        p.update()

    gt_dev = {t: [mi.TensorXf(im) for im in gt[t]] for t in times}
    w_dev = {t: [mi.TensorXf(loss_weights(im) if loss == 'rel' else np.ones_like(im))
                 for im in gt[t]] for t in times}
    lrs = {k: (lr_mat if k in ('floor', 'wall') else lr_light) for k in opt.keys()}
    hist, t0 = [], time.time()
    n_terms = len(times) * views_per_time
    for it in range(iters):
        if decay:   # constant for 60% of the run, then linear decay to 10%
            f = 1.0 - 0.9 * max(0.0, (it / iters - 0.6) / 0.4)
            opt.set_learning_rate({k: v * f for k, v in lrs.items()})
        total = 0.0
        for ti, t in enumerate(times):
            for v in rng.choice(len(S.CAMERAS), views_per_time, replace=False):
                push(t)
                img = mi.render(scenes[t], params[t], sensor=int(v), spp=spp,
                                seed=(seed * 7919 + it * 131 + ti * 17 + int(v)) % 2**31)
                loss = dr.mean(w_dev[t][v] * dr.square(img - gt_dev[t][v]), axis=None) / n_terms
                dr.backward(loss)
                total += float(np.array(dr.detach(loss)).ravel()[0])
        opt.step()
        if learn_materials:
            opt['floor'] = dr.clip(opt['floor'], 0.0, 1.0)
            opt['wall'] = dr.clip(opt['wall'], 0.0, 1.0)
        for t in times:
            for k in st0:
                opt[f'{k}_{t}'] = dr.clip(opt[f'{k}_{t}'], -12.0, 9.0)
        hist.append(total)
        if log_every and it % log_every == 0:
            print(f'    it {it:4d} loss {total:.3e}  {time.time() - t0:.0f}s', flush=True)

    states = {}
    for t in times:
        states[t] = {}
        for k, v in st0.items():
            states[t][k] = np.exp(np.array(opt[f'{k}_{t}'])).reshape(v.shape).astype(np.float32)
    floor = np.array(opt['floor']) if learn_materials else tex0
    wall = np.array(opt['wall']).ravel() if learn_materials else np.array(wall0)
    return dict(floor=floor, wall=wall, states=states, hist=hist, seconds=time.time() - t0)


# ---------------------------------------------------------------- evaluation

_COVERAGE = {}


def floor_coverage(obj, n=S.MODEL_TEX):
    """Camera pixels (all 6 views) landing in each floor texel."""
    if n not in _COVERAGE:
        cnt = np.zeros((n, n), int)
        for h in S.primary_hits(geom_scene(obj)):
            q = h['p'][h['valid']]
            q = q[np.abs(q[:, 2]) < 1e-4]
            np.add.at(cnt, (np.clip((q[:, 1] / 4 * n).astype(int), 0, n - 1),
                            np.clip(((q[:, 0] + 2) / 4 * n).astype(int), 0, n - 1)), 1)
        _COVERAGE[n] = cnt
    return _COVERAGE[n]


def region_masks(observed, n=S.MODEL_TEX):
    """Regions over observed texels (>= 4 camera pixels; unobserved texels carry no information
    for any arm). 'all_texels' is kept for reference."""
    lit_t = [S.analytic_floor_sunlit(S.sun_dir(t), n) for t in S.TRAIN_TIMES]
    edge = np.zeros((n, n), bool)
    for m in lit_t:
        edge |= binary_dilation(m) & ~binary_erosion(m)            # one texel either side of each patch edge
    lit = np.any(lit_t, 0)
    return {'all': observed, 'all_texels': np.ones((n, n), bool), 'lit_interior': observed & lit & ~edge,
            'boundary': observed & edge, 'never_lit': observed & ~lit & ~edge}


def floor_metrics(est, gt_tex, wall_est, observed):
    """Albedo is identified only up to one global scale shared with the light intensity, so the
    primary numbers are after a least-squares scale fitted on the whole floor (D0019)."""
    gt = S.downsample(gt_tex)
    s = float((est[observed] * gt[observed]).sum() / max((est[observed] ** 2).sum(), 1e-12))
    out = {'scale': round(s, 4)}
    for name, m in region_masks(observed).items():
        e, g = est[m], gt[m]
        out[name] = dict(n=int(m.sum()), mae=round(float(np.abs(e - g).mean()), 5),
                         mae_scaled=round(float(np.abs(s * e - g).mean()), 5),
                         corr=round(float(np.corrcoef(e.mean(-1), g.mean(-1))[0, 1]), 4),
                         bias_scaled=round(float((s * e - g).mean()), 5))
    out['patch_bias'] = round(out['lit_interior']['bias_scaled'] - out['never_lit']['bias_scaled'], 5)
    out['wall_err_scaled'] = round(float(np.abs(s * np.asarray(wall_est) - np.array(S.WALL_GT)).mean()), 5)
    return out


def state_at(states, t, train_times):
    """Lighting for any time: fitted state at a training time; linear interpolation between the
    bracketing training times; nearest training time outside their range."""
    if t in states:
        return states[t]
    tr = sorted(train_times, key=S.hours)
    h = S.hours(t)
    if len(tr) == 1 or h <= S.hours(tr[0]):
        return states[tr[0]]
    if h >= S.hours(tr[-1]):
        return states[tr[-1]]
    for a, b in zip(tr[:-1], tr[1:]):
        if S.hours(a) <= h <= S.hours(b):
            w = (h - S.hours(a)) / (S.hours(b) - S.hours(a))
            return {k: (1 - w) * states[a][k] + w * states[b][k] for k in states[a]}


def tonemap(x, k):
    return np.clip(x * k, 0, 1) ** (1 / 2.2)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def image_metrics(pred, gt_imgs, hits):
    """Room pixels only: pixels looking out of the window show the sky directly, which says
    nothing about the room and which a low-res sky cannot cut sharply at the horizon (D0019)."""
    room = np.stack([h['valid'].reshape(h['shape']) for h in hits])
    k = 1.0 / np.percentile(S.lum(gt_imgs)[room], 99.5)
    s = float((pred[room] * gt_imgs[room]).sum() / max((pred[room] ** 2).sum(), 1e-12))
    obs_gt, thr = S.observed_patch(list(gt_imgs), hits)
    obs_pr, _ = S.observed_patch(list(pred), hits, thr=thr[0], eps=thr[1])
    g, p = np.concatenate(obs_gt), np.concatenate(obs_pr)
    return dict(psnr=round(psnr(tonemap(pred[room], k), tonemap(gt_imgs[room], k)), 3),
                psnr_scaled=round(psnr(tonemap(s * pred[room], k), tonemap(gt_imgs[room], k)), 3),
                patch_iou=round(float((g & p).sum() / max((g | p).sum(), 1)), 4))


def render_all(sc, spp=EVAL_SPP):
    return np.stack([np.array(mi.render(sc, sensor=i, spp=spp, seed=5000 + i)) for i in range(len(S.CAMERAS))])


def evaluate(obj, gt, gt_tex, arm, res, dirs, train_times, hits):
    ev = {'floor': floor_metrics(res['floor'], gt_tex, res['wall'], floor_coverage(obj) >= 4),
          'heldout': {}, 'material_only': {}}
    for t in S.HELDOUT_TIMES:
        st = state_at(res['states'], t, train_times)
        sc = S.make_scene(obj, res['floor'], res['wall'], arm, st, d=dirs[t])
        ev['heldout'][t] = image_metrics(render_all(sc), gt[t], hits)
        if dirs[t] is not None:
            ev['heldout'][t]['sun_dir_err_deg'] = round(S.angle_deg(dirs[t], S.sun_dir(t)), 3)
        sc = S.make_scene(obj, res['floor'], res['wall'], 'gt', d=S.sun_dir(t))   # materials under true light
        ev['material_only'][t] = image_metrics(render_all(sc), gt[t], hits)
    return ev


# ---------------------------------------------------------------- stages: diag, fit, report

def save_run(out, name, res, ev, meta):
    d = out / 'runs'
    d.mkdir(exist_ok=True)
    np.savez_compressed(d / f'{name}.npz', floor=res['floor'], wall=res['wall'],
                        **{f'{k}_{key(t)}': v for t, st in res['states'].items() for k, v in st.items()})
    rec = dict(meta, seconds=round(res['seconds'], 1), loss_first=res['hist'][0], loss_last=res['hist'][-1],
               loss_curve=[round(x, 8) for x in res['hist'][:: max(1, len(res['hist']) // 20)]], **ev)
    (d / f'{name}.json').write_text(json.dumps(rec, indent=2))
    return rec


def stage_diag(out, obj, a):
    gt, gt_tex = load_gt(out)
    hits = S.primary_hits(geom_scene(obj))
    truth = {t: S.sun_dir(t) for t in ALL_TIMES}
    ceiling = {t: image_metrics(render_all(S.make_scene(obj, gt_tex, S.WALL_GT, 'gt', d=truth[t])), gt[t], hits)
               for t in ALL_TIMES}                   # exact scene, independent samples: Monte Carlo noise floor
    (out / 'noise_ceiling.json').write_text(json.dumps(ceiling, indent=2))
    print('  noise ceiling', {t: c['psnr'] for t, c in ceiling.items()})
    res = fit(obj, gt, 'gt', S.TRAIN_TIMES, truth, a.iters, a.spp, 0, log_every=a.iters // 5)
    ev = {'floor': floor_metrics(res['floor'], gt_tex, res['wall'], floor_coverage(obj) >= 4)}
    rec = save_run(out, 'D1_gt_light_multi_s0', res, ev, dict(diag='D1', arm='gt', mode='multi', seed=0))
    print('  D1', json.dumps(rec['floor']['all']), 'patch_bias', rec['floor']['patch_bias'])
    for arm in a.arms:
        dirs = arm_dirs(arm, json.loads((out / 'estimates.json').read_text()), S.TRAIN_TIMES) \
            if arm in S.SUN_ARMS else {t: None for t in ALL_TIMES}
        res = fit(obj, gt, arm, S.TRAIN_TIMES, dirs, a.iters, a.spp, 0, learn_materials=False,
                  floor_init=gt_tex, wall_init=S.WALL_GT)
        fit_q = {}
        for t in S.TRAIN_TIMES:
            sc = S.make_scene(obj, gt_tex, S.WALL_GT, arm, res['states'][t], d=dirs[t])
            fit_q[t] = image_metrics(render_all(sc), gt[t], hits)
        rec = save_run(out, f'D2_{arm}_multi_s0', res, {'train_fit': fit_q}, dict(diag='D2', arm=arm, mode='multi', seed=0))
        print(f'  D2 {arm}:', {t: (q['psnr'], q['patch_iou']) for t, q in fit_q.items()}, flush=True)


def stage_fit(out, obj, a):
    gt, gt_tex = load_gt(out)
    est = json.loads((out / 'estimates.json').read_text())
    hits = S.primary_hits(geom_scene(obj))
    for seed in a.seeds:
        for mode in a.modes:
            for arm in a.arms:
                name = f'{arm}_{mode}_s{seed}'
                if (out / 'runs' / f'{name}.json').exists() and not a.overwrite:
                    continue
                disk_guard.check(out, need_gb=0.05)
                times = MODES[mode]
                dirs = arm_dirs(arm, est, times)
                res = fit(obj, gt, arm, times, dirs, a.iters, a.spp, seed)
                ev = evaluate(obj, gt, gt_tex, arm, res, dirs, times, hits)
                rec = save_run(out, name, res, ev, dict(arm=arm, mode=mode, seed=seed, iters=a.iters, spp=a.spp))
                f = rec['floor']
                print(f'  {name}: floor MAE_s {f["all"]["mae_scaled"]:.4f} corr {f["all"]["corr"]:.3f} '
                      f'patch_bias {f["patch_bias"]:+.4f} | 14:30 PSNR {rec["heldout"]["14:30"]["psnr"]:.2f} '
                      f'IoU {rec["heldout"]["14:30"]["patch_iou"]:.3f} | {rec["seconds"]:.0f}s', flush=True)


def stage_report(out):
    recs = [json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*.json')) if 'diag' not in json.loads(p.read_text())]
    rows = {}
    for r in recs:
        rows.setdefault((r['arm'], r['mode']), []).append(r)
    table = {}
    for (arm, mode), rs in sorted(rows.items()):
        g = lambda f: [f(r) for r in sorted(rs, key=lambda r: r['seed'])]
        table[f'{arm}/{mode}'] = dict(
            seeds=g(lambda r: r['seed']),
            mae_scaled=g(lambda r: r['floor']['all']['mae_scaled']),
            mae_raw=g(lambda r: r['floor']['all']['mae']),
            corr=g(lambda r: r['floor']['all']['corr']),
            mae_scaled_lit=g(lambda r: r['floor']['lit_interior']['mae_scaled']),
            mae_scaled_boundary=g(lambda r: r['floor']['boundary']['mae_scaled']),
            mae_scaled_never=g(lambda r: r['floor']['never_lit']['mae_scaled']),
            patch_bias=g(lambda r: r['floor']['patch_bias']),
            wall_err=g(lambda r: r['floor']['wall_err_scaled']),
            **{f'{m}_{key(t)}': g(lambda r, t=t, m=m: r['heldout'][t][m])
               for t in S.HELDOUT_TIMES for m in ('psnr', 'psnr_scaled', 'patch_iou')},
            **{f'matonly_psnr_scaled_{key(t)}': g(lambda r, t=t: r['material_only'][t]['psnr_scaled'])
               for t in S.HELDOUT_TIMES})
    verdict = {}
    C, B, Cs = table.get('C/multi'), table.get('B/multi'), table.get('C/single')
    if C and B:
        for metric in ('mae_scaled', 'mae_raw'):
            c, b = np.array(C[metric]), np.array(B[metric])
            verdict[f'R1_{metric}'] = dict(C=c.tolist(), B=b.tolist(),
                                           rel_improvement=round(float(1 - c.mean() / b.mean()), 4),
                                           ranges_disjoint=bool(c.max() < b.min()),
                                           holds=bool(c.mean() <= 0.85 * b.mean() and c.max() < b.min()))
        dp = np.mean(C['psnr_1430']) - np.mean(B['psnr_1430'])
        di = np.mean(C['patch_iou_1430']) - np.mean(B['patch_iou_1430'])
        verdict['R2'] = dict(dpsnr_1430=round(float(dp), 3), diou_1430=round(float(di), 4), holds=bool(dp >= 1 and di >= 0.1))
    if C and Cs:
        verdict['R3'] = dict(C_multi=C['mae_scaled'], C_single=Cs['mae_scaled'],
                             holds=bool(np.mean(C['mae_scaled']) < np.mean(Cs['mae_scaled'])))
    rep = dict(table=table, verdict=verdict)
    (out / 'report.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(verdict, indent=2))
    return rep


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    disk_guard.check(out, need_gb=0.3)
    obj = out / 'room_shell.obj'
    S.room_shell_obj(obj)
    t0 = time.time()
    if a.stage in ('gt', 'all'):
        stage_gt(out, obj, a.gt_spp)
    if a.stage in ('estimate', 'all'):
        stage_estimate(out, obj)
    if a.stage in ('diag', 'all'):
        stage_diag(out, obj, a)
    if a.stage in ('fit', 'all'):
        stage_fit(out, obj, a)
    if a.stage in ('report', 'all'):
        stage_report(out)
    print(f'done {a.stage} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['gt', 'estimate', 'diag', 'fit', 'report', 'all'])
    p.add_argument('--out', default=str(ROOT / 'experiments/out/T11_solarir'))
    p.add_argument('--arms', type=lambda s: s.split(','), default=ARMS)
    p.add_argument('--modes', type=lambda s: s.split(','), default=list(MODES))
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=[0, 1, 2])
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--spp', type=int, default=32)
    p.add_argument('--gt_spp', type=int, default=512)
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
