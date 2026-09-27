"""SolarIR controlled testbed: scene, lighting arms, solar transport, sun-direction estimators.

Shared by experiments/solarir_test.py (fits + evaluation) and tests/test_solar_transport.py.
Coordinates: +X east, +Y scene north, +Z up (utils/solar_geometry.py). The window is a hole in
the y=0 wall, so it faces south (-Y). Floor texture: row = world y (row 0 at the window wall),
column = world x (column 0 at x=-2); verified by ray casting.
"""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import mitsuba as mi
import drjit as dr
from scipy.ndimage import median_filter

if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

from utils.solar_geometry import solar_position, sun_vector_world, vector_to_azimuth_elevation

LAT, LON, TZ = 40.44, -79.94, ZoneInfo('America/New_York')
DATE = (2026, 2, 25)
TRAIN_TIMES = ['10:00', '11:30', '13:00']
SINGLE_TIME = '11:30'
HELDOUT_TIMES = ['12:15', '14:30']
ROOM_X, ROOM_Y, ROOM_Z = (-2.0, 2.0), (0.0, 4.0), 2.5
WIN_X, WIN_Z = (-0.8, 0.8), (0.8, 2.0)
RES = (160, 120)
GT_TEX, MODEL_TEX = 128, 64
WIN_RES = (16, 16)
ENV_RES = {'A': (32, 64), 'A_hi': (128, 256)}     # generic arms: free envmap, and a 16x larger capacity control
# Explicit-sun arms: sun + piecewise-constant sky cells ((rows, cols), texels per cell side). Drawing each
# cell as a block keeps the horizon sharp. 4x8 (45 deg cells) is the protocol sky (D0020); C_oracle16 keeps
# the earlier 16x32 sky as a sensitivity arm.
SKY = {'B': ((4, 8), 8), 'C': ((4, 8), 8), 'C_oracle': ((4, 8), 8), 'C_oracle16': ((16, 32), 2)}
MAX_DEPTH = 4                      # same truncation for ground truth and every model
TURBIDITY = 3.0
WALL_GT = (0.62, 0.60, 0.55)
CAMERAS = [  # (origin, target)
    ((-1.5, 3.6, 1.7), (0.0, 1.4, 0.0)), ((1.5, 3.6, 1.7), (0.0, 1.4, 0.0)),
    ((0.0, 3.8, 2.1), (0.0, 1.0, 0.0)), ((-1.7, 1.2, 1.6), (0.8, 1.8, 0.0)),
    ((1.7, 1.2, 1.6), (-0.8, 1.8, 0.0)), ((0.0, 2.8, 2.3), (0.0, 1.2, 0.0)),
]
SUN_ARMS = ('B', 'C', 'C_oracle', 'C_oracle16')


# ---------------------------------------------------------------- solar geometry

def hours(hhmm):
    h, m = map(int, hhmm.split(':'))
    return h + m / 60


def ephemeris(hhmm):
    h, m = map(int, hhmm.split(':'))
    return solar_position(LAT, LON, datetime(*DATE, h, m, tzinfo=TZ))


def sun_dir(hhmm, bearing_deg=0.0):
    """Unit vector toward the sun at `hhmm`, for a scene whose +Y has compass bearing `bearing_deg`."""
    p = ephemeris(hhmm)
    return sun_vector_world(p.azimuth_deg, p.elevation_deg, bearing_deg)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def az_el(v):
    p = vector_to_azimuth_elevation(np.asarray(v, float))
    return p.azimuth_deg, p.elevation_deg


def dir_from_az_el(az, el):
    return sun_vector_world(az, el, 0.0)


def analytic_floor_sunlit(d, n=MODEL_TEX):
    """Floor texel centres lit by a sun toward `d`: the ray x + t d must pass through the window
    rectangle in the y=0 plane. Returns (n, n) bool, row = y, column = x."""
    d = np.asarray(d, float)
    c = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(ROOM_X[0] + c * 4.0, ROOM_Y[0] + c * 4.0)
    if d[1] >= 0 or d[2] <= 0:
        return np.zeros((n, n), bool)
    t = -Y / d[1]
    xw, zw = X + t * d[0], t * d[2]
    return (xw > WIN_X[0]) & (xw < WIN_X[1]) & (zw > WIN_Z[0]) & (zw < WIN_Z[1])


# ---------------------------------------------------------------- scene

def floor_texture_gt(seed=0, n=GT_TEX):
    """Planked floor with structure at scales the cameras resolve: 10 boards (0.4 m) along x,
    each cut into planks with their own tone, low-frequency grain, mild texel noise."""
    rng = np.random.default_rng(seed)
    c = (np.arange(n) + 0.5) / n * 4.0
    X, Y = np.meshgrid(c, c)                          # metres; row = y, column = x
    board = np.minimum((X / 0.4).astype(int), 9)
    lum = np.zeros_like(X)
    for b in range(10):
        edges = np.concatenate([[0.0], np.sort(rng.uniform(0.3, 3.7, 2)), [4.0]])
        tones = rng.uniform(0.15, 0.55, len(edges) - 1)
        m = board == b
        lum[m] = tones[np.clip(np.searchsorted(edges, Y[m], side='right') - 1, 0, len(tones) - 1)]
    grain = 1 + 0.12 * np.sin(2 * np.pi * (Y / 0.35 + 0.3 * np.sin(2 * np.pi * X / 1.3)) + board)
    lum = lum * grain + 0.015 * rng.standard_normal(X.shape)
    tint = np.array([1.0, 0.76, 0.52])
    return np.clip(lum[..., None] * tint, 0.02, 0.9).astype(np.float32)


def downsample(tex, n=MODEL_TEX):
    k = tex.shape[0] // n
    return tex.reshape(n, k, n, k, 3).mean((1, 3))


def room_shell_obj(path):
    """Walls + ceiling with the window cut in the y=0 wall; the floor is a separate textured rectangle."""
    verts, faces = [], []

    def quad(a, b, c, d):
        i = len(verts)
        verts.extend([a, b, c, d])
        faces.extend([(i + 1, i + 2, i + 3), (i + 1, i + 3, i + 4)])

    x0, x1 = ROOM_X
    y0, y1 = ROOM_Y
    z1 = ROOM_Z
    quad((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1))   # ceiling, normal -Z
    quad((x0, y1, 0), (x1, y1, 0), (x1, y1, z1), (x0, y1, z1))     # back wall, normal -Y
    quad((x0, y0, 0), (x0, y1, 0), (x0, y1, z1), (x0, y0, z1))     # side x0, normal +X
    quad((x1, y0, 0), (x1, y0, z1), (x1, y1, z1), (x1, y1, 0))     # side x1, normal -X

    def strip(xa, xb, za, zb):                                     # window wall, normal +Y
        quad((xa, 0, za), (xa, 0, zb), (xb, 0, zb), (xb, 0, za))
    strip(x0, WIN_X[0], 0, z1); strip(WIN_X[1], x1, 0, z1)
    strip(WIN_X[0], WIN_X[1], 0, WIN_Z[0]); strip(WIN_X[0], WIN_X[1], WIN_Z[1], z1)
    lines = ['v %.6f %.6f %.6f' % v for v in verts] + ['f %d %d %d' % f for f in faces]
    Path(path).write_text('\n'.join(lines) + '\n')


def bitmap_tex(arr, filt='bilinear'):
    return {'type': 'bitmap', 'bitmap': mi.Bitmap(np.ascontiguousarray(arr, np.float32)),
            'raw': True, 'filter_type': filt}


ENV_TO_WORLD = mi.ScalarTransform4f().rotate([1, 0, 0], 90)       # envmap top row -> world +Z


def emitters(arm, state, d=None):
    """Emitter dicts for a lighting arm. `state` holds linear-space numpy values.
    gt: Mitsuba sunsky at direction d. B/C/C_oracle: directional sun (irradiance state['E'])
    toward d + low-res sky envmap state['sky']. A / A_hi: free envmap state['env'].
    W: IRIS-like Lambertian window emitter with radiance texture state['win']; no outside light."""
    if arm == 'gt':
        return {'sky': {'type': 'sunsky', 'sun_direction': [float(x) for x in d], 'turbidity': TURBIDITY}}
    if arm in SUN_ARMS:
        return {'sun': {'type': 'directional', 'direction': [float(-x) for x in d],
                        'irradiance': {'type': 'rgb', 'value': [float(x) for x in state['E']]}},
                'sky': {'type': 'envmap', 'bitmap': mi.Bitmap(np.ascontiguousarray(upsample(state['sky'], SKY[arm][1]), np.float32)),
                        'to_world': ENV_TO_WORLD}}
    if arm in ENV_RES:
        return {'sky': {'type': 'envmap', 'bitmap': mi.Bitmap(np.ascontiguousarray(state['env'], np.float32)),
                        'to_world': ENV_TO_WORLD}}
    if arm == 'W':
        cx, cz = sum(WIN_X) / 2, sum(WIN_Z) / 2
        return {'window': {'type': 'rectangle',
                           'to_world': mi.ScalarTransform4f().translate([cx, 0.0, cz]).rotate([1, 0, 0], -90)
                           .scale([(WIN_X[1] - WIN_X[0]) / 2, (WIN_Z[1] - WIN_Z[0]) / 2, 1]),
                           'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0.0}},
                           'emitter': {'type': 'area', 'radiance': bitmap_tex(state['win'])}}}
    raise ValueError(arm)


def init_state(arm, value=0.1):
    if arm in SUN_ARMS:
        return {'E': np.ones(3, np.float32), 'sky': np.full((*SKY[arm][0], 3), value, np.float32)}
    if arm in ENV_RES:
        return {'env': np.full((*ENV_RES[arm], 3), value, np.float32)}
    if arm == 'W':
        return {'win': np.full((*WIN_RES, 3), 1.0, np.float32)}
    raise ValueError(arm)


FLOOR_FILTER = {True: 'bilinear', False: 'bilinear'}   # {is ground truth: filter}; see D0019


def make_scene(obj_path, floor_tex, wall_rgb, arm, state=None, d=None, integrator='path',
               max_depth=MAX_DEPTH, with_emitters=True):
    sc = {
        'type': 'scene', 'integrator': {'type': integrator, 'max_depth': max_depth},
        'room': {'type': 'obj', 'filename': str(obj_path), 'face_normals': True,
                 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [float(x) for x in wall_rgb]}}},
        'floor': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f().translate([0, 2, 0]).scale([2, 2, 1]),
                  'bsdf': {'type': 'diffuse', 'reflectance': bitmap_tex(floor_tex, FLOOR_FILTER[floor_tex.shape[0] == GT_TEX])}},
    }
    for i, (o, t) in enumerate(CAMERAS):
        sc[f'cam{i}'] = {'type': 'perspective', 'fov': 75,
                         'to_world': mi.ScalarTransform4f().look_at(origin=o, target=t, up=[0, 0, 1]),
                         'film': {'type': 'hdrfilm', 'width': RES[0], 'height': RES[1], 'rfilter': {'type': 'box'}},
                         'sampler': {'type': 'independent'}}
    if with_emitters:
        sc.update(emitters(arm, state, d))
    return mi.load_dict(sc)


def upsample(a, up):
    return np.repeat(np.repeat(a, up, 0), up, 1) if up > 1 else a


def env_pad_index(h, w, up=1):
    """Index from an (h, w, 3) parameter array into Mitsuba's envmap `data`, which for an
    (H, W) = (h*up, w*up) bitmap is (H, W+2, 3): [column W-1, columns 0..W-1, column 0]."""
    H, W = h * up, w * up
    cols = np.concatenate([[W - 1], np.arange(W), [0]]) // up
    rows = np.arange(H) // up
    idx = (rows[:, None, None] * w + cols[None, :, None]) * 3 + np.arange(3)[None, None, :]
    return mi.UInt32(idx.ravel().astype(np.uint32))


def padded_env(flat, h, w, idx, up=1):
    return mi.TensorXf(dr.gather(mi.Float, flat, idx), shape=(h * up, w * up + 2, 3))


# ---------------------------------------------------------------- geometry probes

def primary_hits(scene, stride=1):
    """Per camera: world points, normals and validity of the primary hit at pixel centres."""
    out = []
    W, H = RES
    xs, ys = np.meshgrid((np.arange(0, W, stride) + 0.5) / W, (np.arange(0, H, stride) + 0.5) / H)
    for s in scene.sensors():
        ray, _ = s.sample_ray(0.0, 0.0, mi.Point2f(xs.ravel(), ys.ravel()), mi.Point2f(0.5, 0.5))
        si = scene.ray_intersect(ray)
        out.append(dict(p=np.array(si.p).T, n=np.array(si.n).T, valid=np.array(si.is_valid()),
                        shape=(ys.shape[0], xs.shape[1])))
    return out


def sunlit(scene, p, n, dirs, eps=1e-3, chunk=4_000_000):
    """Visibility toward each direction for each point: (K, P) bool. A point is sunlit when it faces
    the direction and the ray toward it leaves the room (only possible through the window)."""
    dirs = np.atleast_2d(np.asarray(dirs, float))
    P, K = len(p), len(dirs)
    res = np.zeros((K, P), bool)
    per = max(1, chunk // max(P, 1))
    for k0 in range(0, K, per):
        dk = dirs[k0:k0 + per]
        o = np.repeat((p + eps * n)[None], len(dk), 0).reshape(-1, 3)
        dd = np.repeat(dk[:, None, :], P, 1).reshape(-1, 3)
        facing = (np.repeat(n[None], len(dk), 0).reshape(-1, 3) * dd).sum(1) > 0
        ray = mi.Ray3f(mi.Point3f(*[o[:, i].astype(np.float32) for i in range(3)]),
                       mi.Vector3f(*[dd[:, i].astype(np.float32) for i in range(3)]))
        blocked = np.array(scene.ray_test(ray))
        res[k0:k0 + len(dk)] = (facing & ~blocked).reshape(len(dk), P)
    return res


def floor_texel_points(n=MODEL_TEX):
    c = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(ROOM_X[0] + c * 4.0, ROOM_Y[0] + c * 4.0)
    p = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    return p, np.tile([0.0, 0.0, 1.0], (len(p), 1))


# ---------------------------------------------------------------- sun-direction estimation from images

def otsu(values, bins=256):
    h, e = np.histogram(values, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def lum(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def observed_patch(images, hits, stride=1, thr=None, eps=None):
    """Bright-patch mask per camera from images alone: Otsu threshold on log luminance of pixels
    that hit room geometry (window pixels excluded since geometry is known). Returns flat masks and
    (threshold, eps) so the same split can be applied to other images of the same time."""
    L = [median_filter(lum(im), size=3)[::stride, ::stride].ravel() for im in images]  # suppress fireflies
    if eps is None:
        eps = 0.25 * np.median(np.concatenate([l[h['valid']] for l, h in zip(L, hits)]))
    L = [np.log(l + eps) for l in L]                          # eps keeps dark Monte Carlo noise from setting the split
    if thr is None:
        thr = otsu(np.concatenate([l[h['valid']] for l, h in zip(L, hits)]))
    return [(l > thr) & h['valid'] for l, h in zip(L, hits)], (thr, eps)


class PatchMatcher:
    """IoU between the observed bright patch and the patch a candidate sun direction predicts."""

    def __init__(self, geom_scene, images, stride=2):
        self.scene = geom_scene
        hits = primary_hits(geom_scene, stride)
        obs, self.threshold = observed_patch(images, hits, stride)
        valid = np.concatenate([h['valid'] for h in hits])
        self.p = np.concatenate([h['p'] for h in hits])[valid]
        self.n = np.concatenate([h['n'] for h in hits])[valid]
        self.obs = np.concatenate(obs)[valid]

    def iou(self, dirs):
        pred = sunlit(self.scene, self.p, self.n, dirs)
        inter = (pred & self.obs[None]).sum(1)
        union = (pred | self.obs[None]).sum(1)
        return inter / np.maximum(union, 1)


def search_direction(matcher):
    """Arm B: free 2-DOF (azimuth, elevation) grid search, coarse to fine."""
    az, el = np.meshgrid(np.arange(0, 360, 2.0), np.arange(2, 82, 2.0))
    best = None
    for step, span in [(None, None), (0.25, 2.5), (0.05, 0.5)]:
        if step is not None:
            a0, e0 = best
            az, el = np.meshgrid(a0 + np.arange(-span, span + 1e-9, step), e0 + np.arange(-span, span + 1e-9, step))
        cand = np.stack([dir_from_az_el(a, e) for a, e in zip(az.ravel(), el.ravel())])
        s = matcher.iou(cand)
        i = int(np.argmax(s))
        best = (float(az.ravel()[i]), float(el.ravel()[i]))
    return best, float(s[i])


def search_bearing(matcher, hhmm):
    """Arm C: ephemeris gives the sun direction up to the scene's compass bearing (1 DOF)."""
    p = ephemeris(hhmm)
    grid = np.arange(-180, 180, 1.0)
    best = None
    for step, span in [(None, None), (0.1, 1.0), (0.02, 0.1)]:
        if step is not None:
            grid = best + np.arange(-span, span + 1e-9, step)
        cand = np.stack([sun_vector_world(p.azimuth_deg, p.elevation_deg, b) for b in grid])
        s = matcher.iou(cand)
        i = int(np.argmax(s))
        best = float(grid[i])
    return best, float(s[i])
