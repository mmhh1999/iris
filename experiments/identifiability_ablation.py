# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EXP0004: does an external, image-independent solar constraint actually
improve material (albedo) recovery under a single static illumination
condition -- or does it only improve pixel-fitting error?

This is a deliberately minimal, controlled stand-in for the "material
vs. static-illumination ambiguity" argument in NOVELTY_GAP.md, NOT a
run of IRIS's actual NGPBRDF/SLF machinery (that requires real data
and the full training pipeline, not yet available -- see
BASELINE_REPRODUCTION.md). What this script tests is the mechanism:

For a Lambertian point observed under ONE static illumination
condition, L(x) = rho(x) * E(x). Multi-view consistency alone cannot
separate rho from E (it only forces L to be view-independent, which
is trivially true regardless of the rho/E split). Any method that
regularizes both rho and E with generic, image-derived priors only
(e.g. spatial smoothness -- a reasonable stand-in for what a
smoothness/sparsity-regularized neural field like NGPBRDF or a voxel
SLF effectively imposes without additional information) is exactly as
capable of explaining a sharp brightness edge as an albedo edge as an
illumination edge, because nothing in the objective favors one
explanation over the other except how "smooth" each field is assumed
to be. Since a hard sun-patch boundary is NOT spatially smooth, a
smoothness-regularized free-form illumination field will underfit it
and push the residual sharp transition into the (also-free) albedo
field -- creating a spurious albedo edge exactly where a sun patch
boundary is, even when the true material is uniform there.

"Ours" replaces the free-form illumination field with a low-dimensional
PHYSICALLY PARAMETERIZED one (ambient + sun*patch_mask(params)), where
patch_mask's shape is a differentiable rectangle initialized near --
not exactly at -- the true patch location (simulating an imperfect
Mode A/B geometric prior, not an oracle). This lets the illumination
side natively produce the sharp edge, so albedo is no longer forced to
explain it.

Two sub-experiments:
  1. Uniform true albedo, one sun patch -- tests whether baseline
     hallucinates a spurious albedo edge at the patch boundary that
     ours does not.
  2. A genuine material stripe elsewhere in the scene, uncorrelated
     with the patch -- tests that "ours" is not simply suppressing all
     high-frequency albedo variation (which would be cheating the
     first test) and still recovers a real edge when there is one.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as NF

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'identifiability')
os.makedirs(OUT_DIR, exist_ok=True)

GRID = 80
E_AMBIENT_TRUE = 0.3
E_SUN_TRUE = 2.5
PATCH_TRUE = dict(cx=50.0, cy=45.0, hw=14.0, hh=11.0)   # true sun-patch rectangle (center, half-extent)
PATCH_PRIOR_OFFSET = dict(cx=4.0, cy=-3.0, hw=1.5, hh=-1.0)  # simulated imperfect geometric prior, not oracle
NOISE_SIGMA = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def soft_rect_mask(cx, cy, hw, hh, sharpness=2.0):
    """Differentiable rectangle indicator over the GRIDxGRID domain."""
    xs = torch.arange(GRID, device=DEVICE, dtype=torch.float32)
    ys = torch.arange(GRID, device=DEVICE, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    mx = torch.sigmoid((hw - (xx - cx).abs()) * sharpness)
    my = torch.sigmoid((hh - (yy - cy).abs()) * sharpness)
    return mx * my


def make_true_irradiance():
    mask = soft_rect_mask(**PATCH_TRUE, sharpness=50.0)  # near-hard edge for ground truth
    return E_AMBIENT_TRUE + E_SUN_TRUE * mask


def make_true_albedo(with_material_edge):
    rho = torch.full((GRID, GRID), 0.5, device=DEVICE)
    if with_material_edge:
        # a real material stripe, deliberately placed away from the sun patch
        rho[:, 10:22] = 0.15
    return rho


def total_variation(field):
    dx = (field[:, 1:] - field[:, :-1]).abs().mean()
    dy = (field[1:, :] - field[:-1, :]).abs().mean()
    return dx + dy


def render_observation(rho_true, E_true):
    R = rho_true * E_true
    R = R + torch.randn_like(R) * NOISE_SIGMA
    return R.clamp_min(0.0)


def fit_baseline(R_obs, steps=2000, lr=0.03, lambda_tv_E=0.08, lambda_tv_rho=0.02, lambda_scale=1.0):
    """No solar constraint: rho and E are both free per-pixel fields,
    regularized only by generic spatial smoothness (total variation) --
    a stand-in for what a smoothness/sparsity-regularized neural field
    effectively imposes absent any external grounding.

    A global-scale anchor (lambda_scale) pins E's mean to the true scene
    mean irradiance for BOTH this method and `fit_ours` equally -- this
    is not spatial information (it says nothing about WHERE the patch
    is), it stands in for the kind of absolute/relative radiometric
    calibration a real system gets from exposure/CRF estimation, and
    without it rho*E has a trivial global scale ambiguity unrelated to
    the spatial disentanglement question this experiment targets."""
    E_mean_true = (E_AMBIENT_TRUE + E_SUN_TRUE * ((2 * PATCH_TRUE['hw']) * (2 * PATCH_TRUE['hh'])) / (GRID * GRID))
    rho = torch.nn.Parameter(torch.full((GRID, GRID), 0.5, device=DEVICE))
    E = torch.nn.Parameter(torch.full((GRID, GRID), float(E_mean_true), device=DEVICE))
    opt = torch.optim.Adam([rho, E], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        pred = rho.clamp(0, 1) * E.clamp_min(0)
        loss = NF.mse_loss(pred, R_obs) + lambda_tv_E * total_variation(E) + lambda_tv_rho * total_variation(rho) \
            + lambda_scale * (E.mean() - E_mean_true) ** 2
        loss.backward()
        opt.step()
    return rho.detach().clamp(0, 1), E.detach().clamp_min(0)


def fit_ours(R_obs, steps=2000, lr=0.03, lambda_tv_rho=0.02, lambda_scale=1.0):
    """Solar-constrained: E is a low-dimensional physical parameterization
    (ambient + sun*patch(params)), patch params initialized near --not
    at-- the true location (PATCH_PRIOR_OFFSET), simulating an imperfect
    Mode A/B geometric prior rather than an oracle. rho is still a free
    per-pixel field with the SAME regularization as the baseline, so any
    improvement comes from constraining E's spatial form, not from extra
    rho supervision or a different scale anchor (same lambda_scale)."""
    E_mean_true = (E_AMBIENT_TRUE + E_SUN_TRUE * ((2 * PATCH_TRUE['hw']) * (2 * PATCH_TRUE['hh'])) / (GRID * GRID))
    rho = torch.nn.Parameter(torch.full((GRID, GRID), 0.5, device=DEVICE))
    E_ambient = torch.nn.Parameter(torch.tensor(float(E_mean_true), device=DEVICE))
    E_sun = torch.nn.Parameter(torch.tensor(1.0, device=DEVICE))
    cx = torch.nn.Parameter(torch.tensor(PATCH_TRUE['cx'] + PATCH_PRIOR_OFFSET['cx'], device=DEVICE))
    cy = torch.nn.Parameter(torch.tensor(PATCH_TRUE['cy'] + PATCH_PRIOR_OFFSET['cy'], device=DEVICE))
    hw = torch.nn.Parameter(torch.tensor(PATCH_TRUE['hw'] + PATCH_PRIOR_OFFSET['hw'], device=DEVICE))
    hh = torch.nn.Parameter(torch.tensor(PATCH_TRUE['hh'] + PATCH_PRIOR_OFFSET['hh'], device=DEVICE))

    opt = torch.optim.Adam([rho, E_ambient, E_sun, cx, cy, hw, hh], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        mask = soft_rect_mask(cx, cy, hw.clamp_min(0.5), hh.clamp_min(0.5), sharpness=2.0)
        E = E_ambient.clamp_min(0) + E_sun.clamp_min(0) * mask
        pred = rho.clamp(0, 1) * E
        loss = NF.mse_loss(pred, R_obs) + lambda_tv_rho * total_variation(rho) \
            + lambda_scale * (E.mean() - E_mean_true) ** 2
        loss.backward()
        opt.step()
    recovered_patch = dict(cx=cx.item(), cy=cy.item(), hw=hw.item(), hh=hh.item())
    return rho.detach().clamp(0, 1), E.detach(), recovered_patch, E_ambient.item(), E_sun.item()


def rmse(a, b):
    return float(torch.sqrt(NF.mse_loss(a, b)).item())


def run_subexperiment(name, with_material_edge):
    rho_true = make_true_albedo(with_material_edge)
    E_true = make_true_irradiance()
    R_obs = render_observation(rho_true, E_true)

    rho_baseline, E_baseline = fit_baseline(R_obs)
    rho_ours, E_ours, recovered_patch, E_amb_rec, E_sun_rec = fit_ours(R_obs)

    result = dict(
        name=name,
        albedo_rmse_baseline=rmse(rho_baseline, rho_true),
        albedo_rmse_ours=rmse(rho_ours, rho_true),
        pixel_fit_rmse_baseline=rmse(rho_baseline * E_baseline, R_obs),
        pixel_fit_rmse_ours=rmse(rho_ours * E_ours, R_obs),
        recovered_patch_ours=recovered_patch,
        true_patch=PATCH_TRUE,
        recovered_E_ambient_ours=E_amb_rec, true_E_ambient=E_AMBIENT_TRUE,
        recovered_E_sun_ours=E_sun_rec, true_E_sun=E_SUN_TRUE,
    )
    print(f'[{name}]', json.dumps(result, indent=2))

    # save visualization
    try:
        import cv2
        def to_img(t, vmin=0, vmax=1):
            a = t.detach().cpu().numpy()
            a = np.clip((a - vmin) / (vmax - vmin), 0, 1)
            return (a * 255).astype(np.uint8)
        panels = [to_img(rho_true), to_img(rho_baseline), to_img(rho_ours), to_img(R_obs, 0, E_AMBIENT_TRUE + E_SUN_TRUE)]
        panels = [cv2.applyColorMap(p, cv2.COLORMAP_VIRIDIS) for p in panels]
        row = np.concatenate(panels, axis=1)
        cv2.imwrite(os.path.join(OUT_DIR, f'{name}.png'), row)
        print(f'  saved visualization (true albedo | baseline albedo | ours albedo | observation) to {name}.png')
    except Exception as e:
        print('  warning: could not save visualization:', e)

    return result


if __name__ == '__main__':
    r1 = run_subexperiment('test1_uniform_albedo', with_material_edge=False)
    r2 = run_subexperiment('test2_real_material_edge', with_material_edge=True)
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump({'test1_uniform_albedo': r1, 'test2_real_material_edge': r2}, f, indent=2)
    print('\nResults written to', os.path.join(OUT_DIR, 'results.json'))
