# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EXP0002 (Phase A): can a known synthetic sun direction be recovered
from a window-cast sun patch, using only geometry (Mode B)?

Two tests, in increasing order of realism:
  1. Analytic self-consistency: predict the patch for the true sun
     direction, then run the Mode B search against that exact
     prediction. This is a sanity check on the search/optimization
     code itself (does it find the global optimum of a function it
     also defines?) -- necessary but not sufficient evidence.
  2. Render-based cross-validation: build the same room+window in
     Mitsuba (llvm_ad_rgb, CPU -- see DECISIONS.md D0004 for why this
     doesn't wait on the CUDA/OptiX fix), render it under the true
     sun direction with a directional+constant emitter pair, extract
     the actual bright floor pixels via ray intersection + brightness
     thresholding, back-project them onto the floor plane, and run
     the Mode B search against *that*. This tests whether the analytic
     projection model agrees with an independent physically-based
     renderer, and whether recovery survives rasterization/thresholding
     noise and the finite pixel grid -- a real (if still synthetic and
     noise-free-lighting) test of the Mode B pipeline.

Per the project brief: if recovery fails here, do not proceed to
real/uncontrolled data.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import mitsuba
mitsuba.set_variant('llvm_ad_rgb')

from utils.solar_geometry import sun_vector_world, vector_to_azimuth_elevation, angular_error_deg
from utils.window_geometry import WindowSurface, Plane, project_window_to_plane, plane_to_2d
from utils.sun_patch import RasterGrid, search_sun_direction, mask_iou

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'phase_a')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Ground truth scene definition (world frame: +X East, +Y scene-north, +Z up;
# matches utils/solar_geometry.py's convention).
# ---------------------------------------------------------------------------
TRUE_AZ_DEG = 200.0   # south-southwest
TRUE_EL_DEG = 40.0
SCENE_NORTH_BEARING_DEG = 0.0  # scene +Y == true north for this synthetic test

WINDOW_X0, WINDOW_X1 = -0.8, 0.8
WINDOW_Z0, WINDOW_Z1 = 0.8, 2.0
ROOM_X = (-2.0, 2.0)
ROOM_Y = (0.0, 4.0)
ROOM_Z_TOP = 2.5


def true_sun_vector():
    return sun_vector_world(TRUE_AZ_DEG, TRUE_EL_DEG, SCENE_NORTH_BEARING_DEG)


def window_surface():
    corners = np.array([
        [WINDOW_X0, 0.0, WINDOW_Z0],
        [WINDOW_X1, 0.0, WINDOW_Z0],
        [WINDOW_X1, 0.0, WINDOW_Z1],
        [WINDOW_X0, 0.0, WINDOW_Z1],
    ])
    return WindowSurface(corners)


def floor_plane():
    return Plane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])


def build_raster_grid(plane, resolution=400, margin=1.0):
    """Grid sized to cover the room floor, derived from the floor's
    own corners rather than hand-picked, so it's robust to changing
    ROOM_X/ROOM_Y above."""
    corners_3d = np.array([
        [ROOM_X[0], ROOM_Y[0], 0.0], [ROOM_X[1], ROOM_Y[0], 0.0],
        [ROOM_X[1], ROOM_Y[1], 0.0], [ROOM_X[0], ROOM_Y[1], 0.0],
    ])
    corners_2d = plane_to_2d(corners_3d, plane)
    lo = corners_2d.min(axis=0) - margin
    hi = corners_2d.max(axis=0) + margin
    extent = float(max(hi - lo))
    return RasterGrid(origin=lo, extent=extent, resolution=resolution)


# ---------------------------------------------------------------------------
# Test 1: analytic self-consistency
# ---------------------------------------------------------------------------

def test_analytic_self_consistency():
    window = window_surface()
    plane = floor_plane()
    grid = build_raster_grid(plane)

    s_true = true_sun_vector()
    patch_3d = project_window_to_plane(window, s_true, plane)
    assert patch_3d is not None, "true sun direction casts no patch onto the floor -- check TRUE_AZ/EL"
    patch_2d = plane_to_2d(patch_3d, plane)
    observed_mask = grid.polygon_to_mask(patch_2d)

    best_az, best_el, best_iou = search_sun_direction(
        window, plane, observed_mask, grid, SCENE_NORTH_BEARING_DEG)

    s_recovered = sun_vector_world(best_az, best_el, SCENE_NORTH_BEARING_DEG)
    err_deg = angular_error_deg(s_true, s_recovered)

    result = dict(true_az=TRUE_AZ_DEG, true_el=TRUE_EL_DEG,
                  recovered_az=best_az, recovered_el=best_el,
                  best_iou=best_iou, angular_error_deg=err_deg)
    print('[Test 1: analytic self-consistency]', json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Test 2: render-based cross-validation
# ---------------------------------------------------------------------------

def build_room_mesh_obj_string():
    """Explicit world-space quads for a closed room with a rectangular
    window opening in the y=0 wall. Winding order chosen per-quad so
    the resulting face normal points INTO the room (see derivation in
    research/PHASE_A_SYNTHETIC.md). Returns an OBJ-format string."""
    verts = []
    faces = []

    def add_quad(v0, v1, v2, v3):
        base = len(verts)
        verts.extend([v0, v1, v2, v3])
        faces.append((base + 1, base + 2, base + 3))
        faces.append((base + 1, base + 3, base + 4))  # OBJ is 1-indexed

    x0, x1 = ROOM_X
    y0, y1 = ROOM_Y
    z1 = ROOM_Z_TOP

    # Floor, normal +Z.
    add_quad((x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0))
    # Ceiling, normal -Z.
    add_quad((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1))
    # Back wall (y=y1), normal -Y.
    add_quad((x0, y1, 0), (x1, y1, 0), (x1, y1, z1), (x0, y1, z1))
    # Side wall x=x0, normal +X.
    add_quad((x0, y0, 0), (x0, y1, 0), (x0, y1, z1), (x0, y0, z1))
    # Side wall x=x1, normal -X.
    add_quad((x1, y0, 0), (x1, y0, z1), (x1, y1, z1), (x1, y1, 0))
    # Window wall (y=0) as 4 strips around the opening, normal +Y.
    def wall_strip(xa, xb, za, zb):
        add_quad((xa, 0, za), (xa, 0, zb), (xb, 0, zb), (xb, 0, za))
    wall_strip(x0, WINDOW_X0, 0, z1)                 # left
    wall_strip(WINDOW_X1, x1, 0, z1)                 # right
    wall_strip(WINDOW_X0, WINDOW_X1, 0, WINDOW_Z0)   # apron (below hole)
    wall_strip(WINDOW_X0, WINDOW_X1, WINDOW_Z1, z1)  # header (above hole)

    lines = ['# Phase A synthetic room, generated by experiments/phase_a_sun_recovery.py']
    for v in verts:
        lines.append('v {:.6f} {:.6f} {:.6f}'.format(*v))
    for f in faces:
        lines.append('f {} {} {}'.format(*f))
    return '\n'.join(lines) + '\n'


def build_scene(spp=256, res=(512, 384)):
    obj_path = os.path.join(OUT_DIR, 'room.obj')
    with open(obj_path, 'w') as f:
        f.write(build_room_mesh_obj_string())

    s_true = true_sun_vector()
    propagation_dir = (-s_true).tolist()

    # Aimed to actually see the floor within the room (see DECISIONS.md
    # D0006): the room only spans y in [0,4], so the camera must look
    # steeply enough downward that its center ray hits z=0 inside that
    # range, not far outside the room.
    cam_origin = [0.0, 3.7, 2.0]  # z < ROOM_Z_TOP=2.5, must stay inside the room
    cam_target = [0.55, 1.2, 0.0]

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 6},
        'sensor': {
            'type': 'perspective',
            'fov': 70,
            'to_world': mitsuba.ScalarTransform4f().look_at(
                origin=cam_origin, target=cam_target, up=[0, 0, 1]),
            'film': {'type': 'hdrfilm', 'width': res[0], 'height': res[1],
                     'pixel_format': 'rgb', 'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': spp},
        },
        'room': {
            'type': 'obj',
            'filename': obj_path,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.6, 0.55, 0.5]}},
        },
        'glossy_object': {
            'type': 'sphere',
            'center': [1.1, 1.6, 0.3],
            'radius': 0.3,
            'bsdf': {'type': 'roughconductor', 'alpha': 0.15},
        },
        'sun': {
            'type': 'directional',
            'direction': propagation_dir,
            'irradiance': {'type': 'rgb', 'value': [6.0, 5.7, 5.0]},
        },
        'sky': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [0.35, 0.4, 0.5]},
        },
    }
    return mitsuba.load_dict(scene_dict), obj_path


def render_and_extract_floor_patch(scene, res=(512, 384), bright_thresh=1.2):
    """Render the scene, then for every pixel whose primary ray hits
    the floor (identified geometrically: near z=0 with an upward
    normal, since the floor is the only such surface in this scene),
    check whether the *rendered pixel* is bright (a proxy for "this is
    part of the directly-lit sun patch, not just ambient sky fill").
    Returns (image_hwc, floor_hit_points_2d_on_plane)."""
    plane = floor_plane()
    sensor = scene.sensors()[0]
    film_size = sensor.film().size()
    w, h = int(film_size[0]), int(film_size[1])

    image = mitsuba.render(scene, spp=256)
    image_np = np.array(image)  # (h, w, 3)

    # Build one primary ray per pixel (pixel centers) to geometrically
    # identify which pixels see the floor.
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    sample = np.stack([(xs.ravel() + 0.5) / w, (ys.ravel() + 0.5) / h], axis=-1)
    rays, _ = sensor.sample_ray_differential(
        time=0.0, sample1=0.0,
        sample2=mitsuba.Point2f(sample[:, 0], sample[:, 1]),
        sample3=mitsuba.Point2f(0.5, 0.5))
    si = scene.ray_intersect(rays)
    valid = np.array(si.is_valid())
    p = np.array(si.p).T          # (N, 3)
    n = np.array(si.n).T          # (N, 3)

    is_floor = valid & (np.abs(p[:, 2]) < 1e-3) & (n[:, 2] > 0.9)

    brightness = image_np.reshape(-1, 3).sum(axis=-1)
    is_bright = brightness > bright_thresh

    floor_bright_img = (is_floor & is_bright).reshape(h, w).astype(np.uint8)

    # Keep only the largest connected bright-floor component in image
    # space. Raw per-pixel thresholding also catches scattered
    # Monte-Carlo global-illumination noise (e.g. bounce light off the
    # glossy sphere) unrelated to the direct sun patch; the patch
    # itself is, by construction, one contiguous region.
    import cv2
    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(floor_bright_img, connectivity=8)
    if n_components > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        floor_bright_filtered = (labels == largest_label)
    else:
        floor_bright_filtered = floor_bright_img.astype(bool)

    floor_bright_flat = floor_bright_filtered.reshape(-1)
    floor_points_3d = p[floor_bright_flat]
    floor_points_2d = plane_to_2d(floor_points_3d, plane) if len(floor_points_3d) else np.zeros((0, 2))

    return image_np, floor_points_2d, is_floor.reshape(h, w), floor_bright_filtered


def save_preview_png(image_np, path):
    ldr = np.clip(image_np, 0, 1) ** (1 / 2.2)
    ldr = (ldr * 255).astype(np.uint8)
    try:
        import cv2
        cv2.imwrite(path, ldr[..., ::-1])
    except Exception as e:
        print('warning: could not save preview PNG:', e)


def test_render_cross_validation():
    window = window_surface()
    plane = floor_plane()
    grid = build_raster_grid(plane)

    scene, obj_path = build_scene()
    image_np, floor_points_2d, floor_mask, bright_mask = render_and_extract_floor_patch(scene)

    preview_path = os.path.join(OUT_DIR, 'render.png')
    save_preview_png(image_np, preview_path)

    n_floor_bright_px = len(floor_points_2d)
    print(f'[Test 2] {n_floor_bright_px} bright floor pixels detected; render saved to {preview_path}')
    if n_floor_bright_px < 50:
        return dict(status='FAILED', reason='too few bright floor pixels detected; '
                    'check emitter intensities / camera framing / threshold',
                    n_floor_bright_px=n_floor_bright_px)

    observed_mask = grid.points_to_mask(floor_points_2d, dilate_px=2)

    # Cross-check: does the analytic prediction for the TRUE direction
    # agree with what the renderer actually produced? (independent
    # validation of the projection math against Mitsuba's own ray tracer)
    s_true = true_sun_vector()
    predicted_true_3d = project_window_to_plane(window, s_true, plane)
    predicted_true_2d = plane_to_2d(predicted_true_3d, plane)
    predicted_true_mask = grid.polygon_to_mask(predicted_true_2d)
    cross_check_iou = mask_iou(predicted_true_mask, observed_mask)

    best_az, best_el, best_iou = search_sun_direction(
        window, plane, observed_mask, grid, SCENE_NORTH_BEARING_DEG)
    s_recovered = sun_vector_world(best_az, best_el, SCENE_NORTH_BEARING_DEG)
    err_deg = angular_error_deg(s_true, s_recovered)

    result = dict(status='OK', true_az=TRUE_AZ_DEG, true_el=TRUE_EL_DEG,
                  recovered_az=best_az, recovered_el=best_el,
                  best_iou=best_iou, angular_error_deg=err_deg,
                  cross_check_iou_at_true_direction=cross_check_iou,
                  n_floor_bright_px=n_floor_bright_px,
                  preview_image=preview_path)
    print('[Test 2: render-based cross-validation]', json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    r1 = test_analytic_self_consistency()
    r2 = test_render_cross_validation()
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump({'test1_analytic_self_consistency': r1,
                   'test2_render_cross_validation': r2}, f, indent=2)
    print('\nResults written to', os.path.join(OUT_DIR, 'results.json'))
