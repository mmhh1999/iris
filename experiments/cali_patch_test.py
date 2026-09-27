"""EXP0037: real multi-time sun-patch prediction on Cali-HDR coda_334 (fixed THETA Z1 tripod).
Pre-registration: research/EXP0037_REAL_PATCH_PREREG_ZH.md.

Stages:
  level     estimate the panorama's true up vector from near-vertical line segments
  levelpano write a levelled, gridded panorama for annotation
  geometry  fit a Manhattan room + window aperture to the frozen annotation JSON
  detect    prediction-blind sun-patch masks from the temporal brightness ratio
  eval      arms B / C / oracle-direction, extrapolation and interpolation splits
"""
import argparse
import glob
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

import disk_guard
from utils.solar_geometry import solar_position, sun_vector_world

DATA = ROOT / 'data_download/cali_hdr/extracted'
OUT = ROOT / 'experiments/out/EXP0037_cali'
EVID = ROOT / 'research/evidence/EXP0037'
ROOM = 'coda_334'
W, H = 1680, 840                      # working resolution (THETA 6720x3360 reduced 4x): 0.214 deg / px
REF_TIME = {'20230706': '07:56', '20230625': '07:56'}


# ---------------------------------------------------------------- io

def time_dirs(day):
    return sorted(glob.glob(str(DATA / day / 'theta' / f'{ROOM}_*')))


def label(d):
    return Path(d).name.split('_')[-1]


def exif(f):
    ex = Image.open(f)._getexif()
    t = {TAGS.get(k, k): v for k, v in ex.items()}
    g = {GPSTAGS.get(k, k): v for k, v in t['GPSInfo'].items()}
    dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    # Time from the camera clock of this very shot (EDT = UTC-4). The GPS stamp is written once per
    # bracket and can be stale (10:58 on 2023-06-25 repeats 10:56's), see D0021.
    local = datetime.strptime(t['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
    utc = (local + timedelta(hours=4)).replace(tzinfo=timezone.utc)
    lat = dms(g['GPSLatitude']) * (1 if g.get('GPSLatitudeRef', 'N') == 'N' else -1)
    lon = dms(g['GPSLongitude']) * (-1 if g.get('GPSLongitudeRef', 'W') == 'W' else 1)
    return float(t['ExposureTime']), utc, lat, lon


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_radiance(d):
    """Approximate relative radiance from the 9-shot JPG bracket: per pixel, the longest exposure
    whose max channel is below 0.92, linearised (sRGB curve) and divided by exposure time."""
    fs = sorted(glob.glob(d + '/*.JPG'))
    meta = [exif(f) for f in fs]
    order = np.argsort([-m[0] for m in meta])                       # longest exposure first
    rad = np.zeros((H, W, 3), np.float32)
    done = np.zeros((H, W), bool)
    for i in order:
        img = cv2.imread(fs[i], cv2.IMREAD_REDUCED_COLOR_4)[..., ::-1].astype(np.float32) / 255
        ok = (img.max(-1) < 0.92) & ~done
        rad[ok] = srgb_to_linear(img[ok]) / meta[i][0]
        done |= ok
    last = order[-1]                                                  # still saturated: shortest shot
    img = cv2.imread(fs[last], cv2.IMREAD_REDUCED_COLOR_4)[..., ::-1].astype(np.float32) / 255
    rad[~done] = srgb_to_linear(img[~done]) / meta[last][0]
    return rad, meta[order[0]][1], meta[0][2], meta[0][3]


# ---------------------------------------------------------------- spherical geometry

def pixel_dirs():
    az = np.radians((np.arange(W) + 0.5) / W * 360 - 180)
    el = np.radians(90 - (np.arange(H) + 0.5) / H * 180)
    A, E = np.meshgrid(az, el)
    return np.stack([np.cos(E) * np.sin(A), np.cos(E) * np.cos(A), np.sin(E)], -1)     # raw camera frame


def level_rotation(up):
    """Rotation R with R @ up = +z (smallest rotation)."""
    up = up / np.linalg.norm(up)
    z = np.array([0, 0, 1.0])
    v = np.cross(up, z)
    s, c = np.linalg.norm(v), up @ z
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def to_pix(d):
    az = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    return (az + 180) / 360 * W - 0.5, (90 - el) / 180 * H - 0.5


def estimate_up(pano_bgr):
    """Up vector from near-vertical line segments in 8 horizontal perspective crops."""
    lsd = cv2.createLineSegmentDetector()
    Hp, Wp = pano_bgr.shape[:2]
    normals, weights = [], []
    N, fov = 1000, 90
    f = N / 2 / np.tan(np.radians(fov / 2))
    u, v = np.meshgrid(np.arange(N) - N / 2 + 0.5, np.arange(N) - N / 2 + 0.5)
    for yaw in range(0, 360, 45):
        fwd = np.array([np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0])
        right = np.array([np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0])
        up = np.array([0, 0, 1.0])
        d = u[..., None] * right - v[..., None] * up + f * fwd
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        az = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
        el = np.degrees(np.arcsin(d[..., 2]))
        mx = ((az + 180) / 360 * Wp - 0.5).astype(np.float32)
        my = ((90 - el) / 180 * Hp - 0.5).astype(np.float32)
        img = cv2.remap(pano_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        lines = lsd.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0]
        for x1, y1, x2, y2 in (lines[:, 0] if lines is not None else []):
            L = np.hypot(x2 - x1, y2 - y1)
            if L < 60 or np.degrees(np.arctan2(abs(x2 - x1), abs(y2 - y1))) > 8:
                continue
            r1 = (x1 - N / 2) * right - (y1 - N / 2) * up + f * fwd
            r2 = (x2 - N / 2) * right - (y2 - N / 2) * up + f * fwd
            n = np.cross(r1, r2)
            normals.append(n / np.linalg.norm(n))
            weights.append(L)
    Nn, w = np.array(normals), np.array(weights)
    up = np.array([0, 0, 1.0])
    for _ in range(5):
        keep = np.abs(Nn @ up) < np.sin(np.radians(3))
        ev, evec = np.linalg.eigh((Nn[keep] * w[keep, None]).T @ Nn[keep])
        up = evec[:, 0] * np.sign(evec[2, 0])
    res = np.degrees(np.arcsin(np.abs(Nn[keep] @ up)))
    return up, dict(segments=int(keep.sum()), tilt_deg=float(np.degrees(np.arccos(up[2]))),
                    residual_median_deg=float(np.median(res)), residual_p90_deg=float(np.percentile(res, 90)))


# ---------------------------------------------------------------- room model

class Room:
    """Manhattan room in a frame rotated by phi about z from the levelled camera frame.
    Floor z = -1 (camera height = 1 unit). Window wall y = D; right wall x = R; left wall x = -L;
    back wall y = -Bk; ceiling z = Cz. Window opening [x0, x1] x [z0, z1] on the window wall,
    with a reveal of depth r beyond it (the ray must also clear the opening at y = D + r)."""

    def __init__(self, g):
        self.__dict__.update(g)
        c, s = np.cos(np.radians(self.phi)), np.sin(np.radians(self.phi))
        self.Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])    # levelled camera -> room

    def cast(self, e):
        """Camera rays (room frame) -> hit point, surface id (0 floor 1 window-wall 2 right 3 left 4 back 5 ceil)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            cand = np.stack([np.where(e[..., 2] < 0, -1 / e[..., 2], np.inf),
                             np.where(e[..., 1] > 0, self.D / e[..., 1], np.inf),
                             np.where(e[..., 0] > 0, self.R / e[..., 0], np.inf),
                             np.where(e[..., 0] < 0, -self.L / e[..., 0], np.inf),
                             np.where(e[..., 1] < 0, -self.Bk / e[..., 1], np.inf),
                             np.where(e[..., 2] > 0, self.Cz / e[..., 2], np.inf)], -1)
        sid = np.argmin(cand, -1)
        t = np.take_along_axis(cand, sid[..., None], -1)[..., 0]
        return e * t[..., None], sid

    def in_window(self, q):
        return (q[..., 0] > self.x0) & (q[..., 0] < self.x1) & (q[..., 2] > self.z0) & (q[..., 2] < self.z1)

    def sunlit(self, p, sid, s, r):
        """p: receiver points (..., 3); s: unit vector toward the sun (room frame)."""
        normals = np.array([[0, 0, 1.0], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]])
        n = normals[sid]
        if s[1] <= 1e-6:
            return np.zeros(p.shape[:-1], bool)
        ok = (n @ s) > 0
        q1 = p + ((self.D - p[..., 1]) / s[1])[..., None] * s
        q2 = p + ((self.D + r - p[..., 1]) / s[1])[..., None] * s
        return ok & self.in_window(q1) & self.in_window(q2) & (sid != 1)



# ---------------------------------------------------------------- geometry fit

def rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def ray(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])


def floor_line_el(room, wall, az):
    """Elevation of the floor-wall junction seen at azimuth az (levelled frame), or None."""
    e = room.Rz @ ray(az, 0.0)
    if wall == 'window' and e[1] > 1e-6:
        t = room.D / e[1]
    elif wall == 'right' and e[0] > 1e-6:
        t = room.R / e[0]
    elif wall == 'left' and e[0] < -1e-6:
        t = -room.L / e[0]
    else:
        return None
    return -np.degrees(np.arctan(1.0 / t))


def stage_geometry(day):
    from scipy.optimize import least_squares
    ann = json.loads((EVID / f'annotation_{day}.json').read_text())
    lev = cv2.imread(str(OUT / f'levelled_{day}_0756.jpg'))
    g = cv2.GaussianBlur(cv2.cvtColor(lev, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    Hb, Wb = g.shape
    ppd = Wb / 360
    fp = lambda az, el: (1 / np.tan(np.radians(-el))) * np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
    P1 = fp(*ann['floor_corners']['window_wall_x_left_wall'])
    P2 = fp(*ann['floor_corners']['window_wall_x_right_wall'])
    v = P2 - P1
    phi = np.degrees(np.arctan2(v[1], v[0]))                        # wall direction angle in the camera frame
    Rz = rotz(-phi)
    q1, q2 = Rz[:2, :2] @ P1, Rz[:2, :2] @ P2
    geo = dict(phi=-phi, D=float((q1[1] + q2[1]) / 2), R=float(q2[0]), L=float(-q1[0]), Bk=6.0, Cz=2.0,
               x0=0, x1=0, z0=0, z1=0)
    spans = {'window': [(-58, -12), (14, 40)], 'right': [(50, 125)], 'left': [(-122, -68)]}
    for it in range(3):
        room = Room(geo)
        pts = []
        for wall, rngs in spans.items():
            for a0, a1 in rngs:
                for az in np.arange(a0, a1, 1.0):
                    el0 = floor_line_el(room, wall, az)
                    if el0 is None:
                        continue
                    col = int((az + 180) * ppd)
                    els = np.arange(el0 - 1.5, el0 + 1.5, 0.05)
                    rows = ((90 - els) * ppd).astype(int)
                    prof = g[rows, col]
                    grad = prof[:-2] - prof[2:]                        # light above, dark below
                    k = int(np.argmax(grad))
                    if grad[k] < 6:
                        continue
                    pts.append((wall, az, float(els[k + 1])))

        def resid(x):
            ph, D, R, L = x
            Rm = rotz(ph)
            r = []
            for wall, az, el in pts:
                q = Rm[:2, :2] @ fp(az, el)
                r.append({'window': q[1] - D, 'right': q[0] - R, 'left': -q[0] - L}[wall])
            return np.array(r)
        sol = least_squares(resid, [geo['phi'], geo['D'], geo['R'], geo['L']], loss='soft_l1', f_scale=0.02)
        geo.update(phi=float(sol.x[0]), D=float(sol.x[1]), R=float(sol.x[2]), L=float(sol.x[3]))
        res = resid(sol.x)
        print(f'  iter {it}: {len(pts)} points, phi {geo["phi"]:.3f} D {geo["D"]:.4f} R {geo["R"]:.4f} '
              f'L {geo["L"]:.4f}  |resid| median {np.median(np.abs(res)):.4f}')
    room = Room(geo)
    wc = ann['window_opening']
    corners = {k: room.Rz @ ray(*wc[k]) for k in ('top_left', 'top_right', 'bottom_left', 'bottom_right')}
    hit = {k: e * (room.D / e[1]) for k, e in corners.items()}
    geo.update(x0=float((hit['top_left'][0] + hit['bottom_left'][0]) / 2),
               x1=float((hit['top_right'][0] + hit['bottom_right'][0]) / 2),
               z0=float((hit['bottom_left'][2] + hit['bottom_right'][2]) / 2),
               z1=float((hit['top_left'][2] + hit['top_right'][2]) / 2),
               n_boundary_points=len(pts), resid_median=float(np.median(np.abs(res))))
    (OUT / f'geometry_{day}.json').write_text(json.dumps(geo, indent=2))
    print(json.dumps(geo))
    # overlay for visual validation
    vis = lev.copy()
    room = Room(geo)
    for wall, rngs in {'window': [(-63, 44)], 'right': [(44, 150)], 'left': [(-150, -63)]}.items():
        for a0, a1 in rngs:
            for az in np.arange(a0, a1, 0.25):
                el = floor_line_el(room, wall, az)
                if el is not None:
                    cv2.circle(vis, (int((az + 180) * ppd), int((90 - el) * ppd)), 1, (0, 0, 255), -1)
    for wall, az, el in pts:
        cv2.circle(vis, (int((az + 180) * ppd), int((90 - el) * ppd)), 2, (0, 255, 0), -1)
    Rinv = room.Rz.T
    for zz in np.linspace(geo['z0'], geo['z1'], 60):
        for xx in (geo['x0'], geo['x1']):
            d = Rinv @ np.array([xx, geo['D'], zz]); a = np.degrees(np.arctan2(d[0], d[1])); e = np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))
            cv2.circle(vis, (int((a + 180) * ppd), int((90 - e) * ppd)), 1, (255, 0, 255), -1)
    for xx in np.linspace(geo['x0'], geo['x1'], 80):
        for zz in (geo['z0'], geo['z1']):
            d = Rinv @ np.array([xx, geo['D'], zz]); a = np.degrees(np.arctan2(d[0], d[1])); e = np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))
            cv2.circle(vis, (int((a + 180) * ppd), int((90 - e) * ppd)), 1, (255, 0, 255), -1)
    cv2.imwrite(str(OUT / f'geometry_overlay_{day}.jpg'), vis[int(Hb * 0.25):int(Hb * 0.85)], [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- patch detection (prediction-blind)

def receivers(day):
    """Levelled-frame pixel directions -> room-frame hit points and surface ids; evaluation mask."""
    geo = json.loads((OUT / f'geometry_{day}.json').read_text())
    room = Room(geo)
    e = levelled_dirs_same_grid() @ room.Rz.T
    p, sid = room.cast(e)
    el = np.degrees(np.arcsin(np.clip(e[..., 2], -1, 1)))
    win_px = (sid == 1) & room.in_window(p)
    # Receivers: floor and side walls. The window wall itself cannot receive direct sun through its own
    # window, so bright regions there are glare / bounce and are excluded (D0021).
    mask = np.isin(sid, [0, 2, 3]) & ~win_px & (el < 10) & (el > -75)
    w = np.cos(np.radians(el))
    return room, p, sid, mask, w


def levelled_dirs_same_grid():
    """Unit directions of the levelled panorama grid (levelled frame)."""
    return pixel_dirs()


DETECT_EXPOSURE = 1 / 60      # one fixed shot per time (D0021): a multi-exposure merge bands at exposure switches


def load_fixed(d, exposure=DETECT_EXPOSURE):
    fs = sorted(glob.glob(d + '/*.JPG'))
    meta = [exif(f) for f in fs]
    i = int(np.argmin([abs(np.log(m[0] / exposure)) for m in meta]))
    img = cv2.imread(fs[i], cv2.IMREAD_REDUCED_COLOR_4)[..., ::-1].astype(np.float32) / 255
    return srgb_to_linear(img).astype(np.float32), meta[i][1], meta[i][2], meta[i][3]


def stage_detect(day):
    room, p, sid, mask, w = receivers(day)
    dirs = time_dirs(day)
    L, meta = [], []
    for d in dirs:
        rad, utc, lat, lon = load_fixed(d)
        radl = resample_levelled(rad, day)
        lum = radl @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        L.append(lum / np.median(lum[mask]))
        meta.append(dict(label=label(d), utc=utc.isoformat(), lat=lat, lon=lon))
    L = np.stack(L)
    base = np.percentile(L, 30, axis=0)
    ratio = np.log(np.maximum(L, 1e-6) / np.maximum(base, 1e-6)[None])
    vals = np.clip(ratio[:, mask], -1, 4).ravel()
    h, edges = np.histogram(vals, 256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    thr_otsu = float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])
    thr = max(thr_otsu, float(np.log(3.0)))    # direct sun must at least triple a pixel's own baseline (D0021)
    masks = []
    k = np.ones((3, 3), np.uint8)
    for i in range(len(dirs)):
        m = ((ratio[i] > thr) & mask).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m, bool)
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] >= 20:
                keep |= lab == j
        masks.append(keep)
        meta[i]['patch_frac'] = float((w * keep).sum() / (w * mask).sum())
    masks = np.stack(masks)
    np.savez_compressed(OUT / f'patches_{day}.npz', masks=masks, eval_mask=mask, threshold=thr)
    (OUT / f'patches_{day}.json').write_text(json.dumps(dict(threshold_log_ratio=thr, otsu=thr_otsu, times=meta), indent=2))
    for m in meta:
        print(f"  {m['label']}  utc {m['utc']}  patch {100 * m['patch_frac']:.2f}%")
    # visual check
    rows = []
    for i, d in enumerate(dirs):
        rad, *_ = load_fixed(d)
        img = resample_levelled(rad, day)
        img = np.clip(img / np.percentile(img[mask], 99) , 0, 1) ** (1 / 2.2)
        img = (img[..., ::-1] * 255).astype(np.uint8)
        img[masks[i]] = (0.5 * img[masks[i]] + np.array([0, 0, 127])).astype(np.uint8)
        cv2.putText(img, meta[i]['label'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        rows.append(cv2.resize(img[int(H * 0.3):int(H * 0.85)], (840, 231)))
    cv2.imwrite(str(OUT / f'patches_{day}.jpg'), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- evaluation

R_GRID = np.round(np.arange(0.0, 0.401, 0.05), 3)


def dir_room(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])


def az_el_room(v):
    return float(np.degrees(np.arctan2(v[0], v[1]))), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


class Scorer:
    def __init__(self, day, step=3):
        room, p, sid, mask, w = receivers(day)
        z = np.load(OUT / f'patches_{day}.npz')
        sl = (slice(None, None, step), slice(None, None, step))
        m = mask[sl]
        self.room, self.p, self.sid, self.w = room, p[sl][m], sid[sl][m], w[sl][m]
        self.dirs = (pixel_dirs() @ room.Rz.T)[sl][m]
        self.obs = z['masks'][(slice(None),) + sl][:, m]
        meta = json.loads((OUT / f'patches_{day}.json').read_text())['times']
        self.labels = [t['label'] for t in meta]
        self.meta = {t['label']: t for t in meta}

    def pred(self, s, r):
        return self.room.sunlit(self.p, self.sid, s, r)

    def iou(self, label, s, r):
        o = self.obs[self.labels.index(label)]
        pr = self.pred(s, r)
        return float((self.w * (o & pr)).sum() / max((self.w * (o | pr)).sum(), 1e-12))

    def centroid_err(self, label, s, r):
        o = self.obs[self.labels.index(label)]
        pr = self.pred(s, r)
        if pr.sum() == 0 or o.sum() == 0:
            return None
        c1 = (self.dirs[o] * self.w[o, None]).sum(0)
        c2 = (self.dirs[pr] * self.w[pr, None]).sum(0)
        return angle(c1, c2)

    def sun_world(self, label):
        t = self.meta[label]
        pos = solar_position(t['lat'], t['lon'], datetime.fromisoformat(t['utc']))
        return sun_vector_world(pos.azimuth_deg, pos.elevation_deg), pos

    def hours(self, label):
        u = datetime.fromisoformat(self.meta[label]['utc'])
        return u.hour + u.minute / 60 + u.second / 3600


def search_dir(sc, labels, r, coarse=2.0):
    """Best room-frame direction for one label (2 DOF)."""
    best = None
    grids = [(np.arange(-88, 88.1, coarse), np.arange(2, 84, coarse))]
    for it in range(3):
        if best is not None:
            (a0, e0), span, st = best[1], [2.0, 0.4, 0.08][it - 1] * 2, [0.4, 0.08, 0.02][it - 1]
            grids = [(np.arange(a0 - span, a0 + span + 1e-9, st), np.arange(e0 - span, e0 + span + 1e-9, st))]
        for az_g, el_g in grids:
            for az in az_g:
                for el in el_g:
                    v = np.mean([sc.iou(l, dir_room(az, el), r) for l in labels])
                    if best is None or v > best[0]:
                        best = (v, (float(az), float(el)))
    return best


def fit_B(sc, train):
    """Per-time 2-DOF directions; reveal depth r shared across training times."""
    out = None
    for r in R_GRID:
        sols = {l: search_dir(sc, [l], r) for l in train}
        score = np.mean([v for v, _ in sols.values()])
        if out is None or score > out['train_iou']:
            out = dict(r=float(r), train_iou=float(score), dirs={l: sols[l][1] for l in train},
                       per_time_iou={l: sols[l][0] for l in train})
    return out


def B_direction(sc, fitB, label, train):
    h = np.array([sc.hours(l) for l in train])
    az = np.array([fitB['dirs'][l][0] for l in train])
    el = np.array([fitB['dirs'][l][1] for l in train])
    if len(train) == 1:
        return dir_room(az[0], el[0])
    pa, pe = np.polyfit(h, az, 1), np.polyfit(h, el, 1)
    return dir_room(np.polyval(pa, sc.hours(label)), np.polyval(pe, sc.hours(label)))


def fit_C(sc, train):
    """One yaw alpha (world -> room) + r, over all training times jointly."""
    sw = {l: sc.sun_world(l)[0] for l in train}
    out = None
    for r in R_GRID:
        best = None
        grid = np.arange(-180, 180, 1.0)
        for it, (span, st) in enumerate([(None, None), (1.5, 0.1), (0.15, 0.02)]):
            if span is not None:
                grid = np.arange(best[1] - span, best[1] + span + 1e-9, st)
            for a in grid:
                v = np.mean([sc.iou(l, rotz(a) @ sw[l], r) for l in train])
                if best is None or v > best[0]:
                    best = (v, float(a))
        if out is None or best[0] > out['train_iou']:
            out = dict(r=float(r), alpha=best[1], train_iou=float(best[0]))
    return out


def stage_eval(day):
    sc = Scorer(day)
    meta = json.loads((OUT / f'patches_{day}.json').read_text())['times']
    patch_times = [t['label'] for t in meta if t['patch_frac'] > 0.002]
    splits = {'extrapolation': (patch_times[:2], patch_times[2:]),
              'interpolation': ([patch_times[0], patch_times[-1]], patch_times[1:-1])}
    report = dict(day=day, patch_times=patch_times, splits={})
    for name, (train, held) in splits.items():
        fb, fc = fit_B(sc, train), fit_C(sc, train)
        rows = {}
        for l in held:
            sB = B_direction(sc, fb, l, train)
            sC = rotz(fc['alpha']) @ sc.sun_world(l)[0]
            orc = None
            for r in R_GRID:
                v = search_dir(sc, [l], r)
                if orc is None or v[0] > orc[0]:
                    orc = (v[0], v[1], float(r))
            sO = dir_room(*orc[1])
            rows[l] = dict(
                B=dict(iou=sc.iou(l, sB, fb['r']), centroid_err=sc.centroid_err(l, sB, fb['r']),
                       dir=az_el_room(sB), angle_to_oracle=angle(sB, sO)),
                C=dict(iou=sc.iou(l, sC, fc['r']), centroid_err=sc.centroid_err(l, sC, fc['r']),
                       dir=az_el_room(sC), angle_to_oracle=angle(sC, sO)),
                oracle=dict(iou=orc[0], dir=orc[1], r=orc[2]))
            print(f"  [{name}] {l}: B IoU {rows[l]['B']['iou']:.3f} (ang {rows[l]['B']['angle_to_oracle']:.2f})  "
                  f"C IoU {rows[l]['C']['iou']:.3f} (ang {rows[l]['C']['angle_to_oracle']:.2f})  oracle {orc[0]:.3f}", flush=True)
        mB = float(np.mean([rows[l]['B']['iou'] for l in held]))
        mC = float(np.mean([rows[l]['C']['iou'] for l in held]))
        mO = float(np.mean([rows[l]['oracle']['iou'] for l in held]))
        report['splits'][name] = dict(train=train, heldout=held, fit_B=fb, fit_C=fc, rows=rows,
                                      mean_iou=dict(B=mB, C=mC, oracle=mO))
        print(f"  [{name}] mean held-out IoU  B {mB:.3f}  C {mC:.3f}  oracle {mO:.3f}; "
              f"B r={fb['r']} C r={fc['r']} alpha={fc['alpha']:.2f}", flush=True)
    (OUT / f'eval_{day}.json').write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------- stages

def stage_level(day):
    d = [x for x in time_dirs(day) if label(x) == REF_TIME[day]][0]
    fs = sorted(glob.glob(d + '/*.JPG'))
    pano = cv2.imread(fs[2], cv2.IMREAD_REDUCED_COLOR_2)
    up, info = estimate_up(pano)
    info.update(day=day, ref=Path(d).name, up=up.tolist())
    (OUT / f'level_{day}.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


def levelled_dirs(day):
    up = np.array(json.loads((OUT / f'level_{day}.json').read_text())['up'])
    return pixel_dirs() @ level_rotation(up).T                       # levelled camera frame per pixel


def resample_levelled(img, day):
    """Image in raw panorama pixels -> levelled panorama pixels."""
    up = np.array(json.loads((OUT / f'level_{day}.json').read_text())['up'])
    Rl = level_rotation(up)
    d_lev = pixel_dirs()
    d_raw = d_lev @ Rl                                                # inverse rotation (R orthonormal)
    mx, my = to_pix(d_raw)
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True)
    p.add_argument('--day', default='20230706')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    disk_guard.check(OUT, need_gb=0.5)
    {'level': stage_level, 'geometry': stage_geometry, 'detect': stage_detect, 'eval': stage_eval}[a.stage](a.day)


if __name__ == '__main__':
    main()
