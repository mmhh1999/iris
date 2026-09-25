"""EXP0028: does ephemeris (EXIF time + GPS) + the camera's compass heading
predict where the sun actually appears in real outdoor 360 panoramas?

Layer-1 check of the "time + place -> lighting" chain (see research/
PANO_SUN_DIRECTION_ZH.md) on real data already on disk (Pano2Pano outdoor
brackets, JPG only -- no RAW extraction). For each outdoor capture position
the shortest exposure of its 9-shot bracket (least bloom) is used to locate
the sun disc; its direction is compared with the pvlib ephemeris direction.

Elevation error is independent of heading and checks time + zenith
correction; azimuth error checks the compass heading (EXIF says magnetic,
GPSImgDirectionRef=M, corrected with WMM2020 declination). Weather layer:
Open-Meteo hourly DNI/DHI/cloud at the capture hour, compared with whether
a sun disc is detected (WMO sunshine threshold DNI >= 120 W/m^2).

Resource-light by design: EXIF read without decoding pixels, one image
decoded at a time at half resolution, only small JSON/CSV/thumbnails written.
"""
import argparse
import csv
import gc
import json
import math
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pygeomag import GeoMag

import disk_guard
from utils.solar_geometry import angular_error_deg, solar_position, sun_vector_world

SUN_MIN_PEAK = 200  # at ~1/8000s only the sun (or its specular glint) is this bright; hazy sun ~240
BLUR_SIGMA_PX = 1   # denoise only: a 0.5deg sun is ~5px at half res, larger blur erases it (v1 bug)
MAX_SUN_DIAM_DEG = 3.0
APERTURE_EXPOSURE_S = 1 / 1000  # bracket shot used to segment the bright window view
APERTURE_THRESH = 128
WMO_SUNSHINE_DNI = 120.0
TZ = ZoneInfo('America/New_York')


def dms(v, ref):
    d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -d if ref in ('S', 'W') else d


def read_meta(path):
    img = Image.open(path)  # lazy: header only
    exif = {TAGS.get(k, k): v for k, v in (img._getexif() or {}).items()}
    gps = {GPSTAGS.get(k, k): v for k, v in (exif.get('GPSInfo') or {}).items()}
    xmp = img.info.get('xmp') or b''
    xmp = xmp.decode('utf8', 'ignore') if isinstance(xmp, bytes) else xmp
    pose = {}
    for tag in ('PoseHeadingDegrees', 'PosePitchDegrees', 'PoseRollDegrees'):
        m = re.search(tag + r'\D*?(-?\d+(?:\.\d+)?)', xmp)
        pose[tag] = float(m.group(1)) if m else None
    meta = dict(path=str(path), size=img.size, exposure=float(exif.get('ExposureTime', 'nan')),
                dto=str(exif.get('DateTimeOriginal')), pose=pose)
    try:
        meta['lat'] = dms(gps['GPSLatitude'], gps.get('GPSLatitudeRef', 'N'))
        meta['lon'] = dms(gps['GPSLongitude'], gps.get('GPSLongitudeRef', 'E'))
    except KeyError:
        pass
    if 'GPSImgDirection' in gps:
        meta['heading'] = float(gps['GPSImgDirection'])
        meta['heading_ref'] = str(gps.get('GPSImgDirectionRef', '?'))
    if 'GPSDateStamp' in gps and 'GPSTimeStamp' in gps:
        h, m, s = (float(x) for x in gps['GPSTimeStamp'])
        d = datetime.strptime(gps['GPSDateStamp'], '%Y:%m:%d')
        meta['gps_utc'] = d.replace(hour=int(h), minute=int(m), second=int(s), tzinfo=timezone.utc)
    img.close()
    return meta


def detect_sun(path):
    """Brightest compact peak (min over RGB, so white not blue sky) in the upper
    hemisphere of the shortest exposure, at half res. Prediction-blind: the
    ephemeris is never used to guide the search."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    H, W = img.shape[:2]
    minch = img.min(axis=2)
    del img
    horizon = int(H * (92 / 180))  # allow 2 deg below horizon
    blur = cv2.GaussianBlur(minch[:horizon].astype(np.float32), (0, 0), BLUR_SIGMA_PX)
    del minch
    v0, u0 = np.unravel_index(np.argmax(blur), blur.shape)
    peak = float(blur[v0, u0])
    out = dict(W=W, H=H, peak_min_channel=round(peak, 1), detected=False)
    if peak < SUN_MIN_PEAK:
        return out
    shift = W // 2 - u0  # roll so the blob never straddles the 360 seam
    rolled = np.roll(blur, shift, axis=1)
    core = (rolled >= 0.9 * peak).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(core, connectivity=8)
    lab = labels[v0, W // 2]
    area = stats[lab, cv2.CC_STAT_AREA]
    cu, cv = cents[lab]
    px_per_deg = W / 360.0
    diam_deg = 2 * math.sqrt(area / math.pi) / px_per_deg
    # other strong peaks >5 deg away = possible glints/reflections competing with the sun
    strong = (rolled >= SUN_MIN_PEAK).astype(np.uint8)
    n2, lab2, st2, c2 = cv2.connectedComponentsWithStats(strong, connectivity=8)
    others = sum(1 for i in range(1, n2) if math.hypot(c2[i][0] - cu, c2[i][1] - cv) > 5 * px_per_deg)
    out.update(detected=True, u=(cu - shift) % W, v=cv, area_px=int(area), diam_deg=round(diam_deg, 2),
               compact=bool(diam_deg <= MAX_SUN_DIAM_DEG), n_competing_peaks=int(others))
    return out


def aperture_top_elevation(path):
    """Highest elevation of the bright window view (the camera sits indoors
    looking out). Heading-independent: a sun above this cannot be visible."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    H = img.shape[0]
    mask = img.max(axis=2)[: H // 2] >= APERTURE_THRESH
    del img
    rows = np.where(mask.sum(axis=1) >= max(3, 0.002 * mask.shape[1]))[0]  # ignore stray pixels
    if len(rows) == 0:
        return None
    return round(90.0 - (rows.min() + 0.5) / H * 180.0, 1)


def aperture_center_lambda(path):
    """Circular-mean image longitude of the bright window view (deg, 0 = image
    center). Used only for the room-name orientation sanity check."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    H, W = img.shape[:2]
    band = img.max(axis=2)[int(H * 30 / 180):int(H * 120 / 180)] >= APERTURE_THRESH  # el +60..-30
    del img
    w = band.sum(axis=0).astype(np.float64)
    if w.sum() == 0:
        return None
    lam = np.deg2rad((np.arange(W) + 0.5) / W * 360.0 - 180.0)
    return round(float(np.rad2deg(np.arctan2((w * np.sin(lam)).sum(), (w * np.cos(lam)).sum()))), 1)


LABEL_AZ = dict(N=0, NE=45, E=90, SE=135, S=180, SW=225, W=270, NW=315)


def label_facing(group):
    m = re.search(r'(?:^|[_\-])(NE|NW|SE|SW|N|E|S|W)(?=$|[_\-])', Path(group).name)
    return LABEL_AZ[m.group(1)] if m else None


def pixel_to_dir(u, v, W, H, heading_true, right_is_clockwise=True):
    lam = (u + 0.5) / W * 360.0 - 180.0
    az = (heading_true + (lam if right_is_clockwise else -lam)) % 360.0
    el = 90.0 - (v + 0.5) / H * 180.0
    return az, el


def dir_to_pixel(az, el, W, H, heading_true):
    lam = ((az - heading_true + 180.0) % 360.0) - 180.0
    return (lam + 180.0) / 360.0 * W - 0.5, (90.0 - el) / 180.0 * H - 0.5


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def weather(lat, lon, when_utc, cache):
    """Open-Meteo archive; radiation is the mean of the PRECEDING hour, so a
    13:17 capture uses the 14:00 entry (covers 13:00-14:00)."""
    key = f'{when_utc:%Y-%m-%d}_{lat:.2f}_{lon:.2f}'
    if key not in cache:
        url = ('https://archive-api.open-meteo.com/v1/archive?'
               f'latitude={lat:.4f}&longitude={lon:.4f}&start_date={when_utc:%Y-%m-%d}&end_date={when_utc:%Y-%m-%d}'
               '&hourly=cloud_cover,shortwave_radiation,direct_normal_irradiance,diffuse_radiation&timezone=UTC')
        with urllib.request.urlopen(url, timeout=30) as r:
            cache[key] = json.load(r)['hourly']
    h = cache[key]
    idx = min(when_utc.hour + (1 if when_utc.minute or when_utc.second else 0), 23)
    return dict(hour_end_utc=h['time'][idx], cloud_cover=h['cloud_cover'][idx], ghi=h['shortwave_radiation'][idx],
                dni=h['direct_normal_irradiance'][idx], dhi=h['diffuse_radiation'][idx])


def draw(path, det, pred_uv, out_path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    s = img.shape[1] / det['W']
    pu, pv = pred_uv
    cv2.circle(img, (int(pu * s), int(pv * s)), 14, (255, 255, 0), 2)  # cyan: ephemeris prediction
    if det['detected']:
        cu, cv_ = int(det['u'] * s), int(det['v'] * s)
        cv2.drawMarker(img, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)  # red: detected sun
    cv2.imwrite(str(out_path), cv2.resize(img, (960, 480)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    del img


def summarize(rows, key):
    xs = [abs(r[key]) for r in rows if r.get(key) is not None]
    if not xs:
        return None
    return dict(n=len(xs), median=round(float(np.median(xs)), 2), mean=round(float(np.mean(xs)), 2),
                max=round(float(np.max(xs)), 2))


def main(a):
    out = Path(a.out)
    (out / 'vis').mkdir(parents=True, exist_ok=True)
    disk_guard.check(out, need_gb=0.1)
    gm = GeoMag(coefficients_file='wmm/WMM_2020.COF')
    cache_path = out / 'weather_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    groups = defaultdict(list)
    for p in sorted(Path(a.data).rglob('*.JPG')):
        if 'out' in str(p.relative_to(a.data)).lower():
            groups[p.parent].append(p)

    rows = []
    for gdir, files in sorted(groups.items()):
        metas = [read_meta(f) for f in files]
        m = min(metas, key=lambda x: x['exposure'])  # shortest exposure: least bloom
        rel = str(gdir.relative_to(a.data))
        row = dict(group=rel, image=Path(m['path']).name, exposure_s=m['exposure'], n_bracket=len(files),
                   pitch=m['pose']['PosePitchDegrees'], roll=m['pose']['PoseRollDegrees'])
        if 'lat' not in m or 'heading' not in m or m['dto'] in ('None', ''):
            row['status'] = 'missing_gps_or_heading_or_time'
            rows.append(row)
            continue
        local = datetime.strptime(m['dto'], '%Y:%m:%d %H:%M:%S').replace(tzinfo=TZ)
        when = local.astimezone(timezone.utc)
        dec = gm.calculate(glat=m['lat'], glon=m['lon'], alt=0.3,
                           time=when.year + (when.timetuple().tm_yday - 0.5) / 365.25).d
        heading_true = (m['heading'] + (dec if m['heading_ref'] == 'M' else 0.0)) % 360.0
        pos = solar_position(m['lat'], m['lon'], when)
        row.update(lat=round(m['lat'], 5), lon=round(m['lon'], 5), local_time=local.isoformat(),
                   gps_minus_camera_s=(m['gps_utc'] - when).total_seconds() if 'gps_utc' in m else None,
                   heading_raw=m['heading'], heading_ref=m['heading_ref'], declination=round(dec, 2),
                   heading_true=round(heading_true, 2), sun_az=round(pos.azimuth_deg, 2),
                   sun_el=round(pos.elevation_deg, 2))
        try:
            row.update({f'wx_{k}': v for k, v in weather(m['lat'], m['lon'], when, cache).items()})
        except Exception as e:  # network is optional for layer 1
            row['wx_error'] = str(e)[:120]
        if pos.elevation_deg <= 0:
            row['status'] = 'sun_below_horizon'
            rows.append(row)
            continue

        m_ap = min(metas, key=lambda x: abs(math.log(x['exposure'] / APERTURE_EXPOSURE_S)))
        ap_top = aperture_top_elevation(m_ap['path'])
        ap_lam = aperture_center_lambda(m_ap['path'])
        row.update(aperture_top_el=ap_top, aperture_center_lambda=ap_lam, label_facing=label_facing(rel),
                   window_facing_compass=round((heading_true + ap_lam) % 360, 1) if ap_lam is not None else None,
                   sun_above_aperture=(ap_top is not None and pos.elevation_deg > ap_top + 1.0))
        det = detect_sun(m['path'])
        row.update(peak_min_channel=det['peak_min_channel'], sun_detected=det['detected'])
        pred_uv = dir_to_pixel(pos.azimuth_deg, pos.elevation_deg, det['W'], det['H'], heading_true)
        if det['detected']:
            s_true = sun_vector_world(pos.azimuth_deg, pos.elevation_deg)
            for conv, rc in (('cw', True), ('ccw', False)):
                az_d, el_d = pixel_to_dir(det['u'], det['v'], det['W'], det['H'], heading_true, rc)
                row[f'ang_err_{conv}'] = round(angular_error_deg(s_true, sun_vector_world(az_d, el_d)), 2)
                row[f'az_err_{conv}'] = round(wrap180(az_d - pos.azimuth_deg), 2)
                row[f'el_err_{conv}'] = round(el_d - pos.elevation_deg, 2)
            az_mag, _ = pixel_to_dir(det['u'], det['v'], det['W'], det['H'], m['heading'], True)
            row['az_err_cw_no_declination'] = round(wrap180(az_mag - pos.azimuth_deg), 2)
            row.update(sun_diam_deg=det['diam_deg'], sun_compact=det['compact'],
                       n_competing_peaks=det['n_competing_peaks'],
                       clean_detection=bool(det['compact'] and det['n_competing_peaks'] == 0))
            # yaw calibrated from the observed sun instead of the compass (one DOF)
            heading_sun = (heading_true - row['az_err_cw']) % 360.0
            row['heading_from_sun'] = round(heading_sun, 1)
            row['compass_minus_sun_heading'] = round(wrap180(heading_true - heading_sun), 1)
            if ap_lam is not None:
                row['window_facing_sun'] = round((heading_sun + ap_lam) % 360, 1)
        row['status'] = 'ok'
        draw(m['path'], det, pred_uv, out / 'vis' / (rel.replace('/', '__').replace(' ', '_') + '.jpg'))
        rows.append(row)
        gc.collect()
        print(f"{rel:55s} el={row['sun_el']:6.1f} det={det['detected']!s:5} "
              f"err_cw={row.get('ang_err_cw', '-')} dni={row.get('wx_dni', '-')}", flush=True)

    cache_path.write_text(json.dumps(cache))
    det_rows = [r for r in rows if r.get('sun_detected')]
    clean = [r for r in det_rows if r.get('clean_detection')]
    sunny = lambda r: r.get('wx_dni') is not None and r['wx_dni'] >= WMO_SUNSHINE_DNI
    ok = [r for r in rows if r.get('status') == 'ok']
    # heading-independent visibility bookkeeping
    geo_possible = [r for r in ok if not r.get('sun_above_aperture')]
    visibility = dict(
        n_sun_above_window_top=sum(1 for r in ok if r.get('sun_above_aperture')),
        n_sun_above_window_top_but_detected=sum(1 for r in ok if r.get('sun_above_aperture') and r.get('sun_detected')),
        n_geometrically_possible=len(geo_possible),
        possible_and_sunny=sum(1 for r in geo_possible if sunny(r)),
        possible_and_sunny_detected=sum(1 for r in geo_possible if sunny(r) and r.get('sun_detected')),
        possible_not_sunny_detected=sum(1 for r in geo_possible if not sunny(r) and r.get('sun_detected')),
        note='below-window-top does not guarantee visibility: azimuth can still fall outside the window, '
             'and azimuth depends on the (unreliable) compass heading')
    labeled = [r for r in ok if r.get('label_facing') is not None and r.get('window_facing_compass') is not None]
    for r in labeled:
        r['label_err_compass'] = round(wrap180(r['window_facing_compass'] - r['label_facing']), 1)
        if r.get('window_facing_sun') is not None:
            r['label_err_sun'] = round(wrap180(r['window_facing_sun'] - r['label_facing']), 1)
    room_label_check = dict(
        n_labeled=len(labeled),
        compass=summarize(labeled, 'label_err_compass'),
        sun_calibrated=summarize([r for r in labeled if 'label_err_sun' in r], 'label_err_sun'),
        note='room names (E/SE/SW/NW) are coarse human labels, +-22.5 deg at best')
    summary = dict(
        n_groups=len(rows), n_ok=len(ok), n_sun_detected=len(det_rows), n_clean_detections=len(clean),
        all_detected_elevation_error_heading_free=summarize(det_rows, 'el_err_cw'),
        all_detected_compass_azimuth_error=summarize(det_rows, 'az_err_cw'),
        room_label_check=room_label_check,
        declination_model='WMM2020 via pygeomag', convention_note='cw: image right = clockwise azimuth (GPano spec)',
        clean_elevation_error_heading_free=summarize(clean, 'el_err_cw'),
        clean_elevation_error_signed_mean=round(float(np.mean([r['el_err_cw'] for r in clean])), 2) if clean else None,
        clean_compass_azimuth_error=summarize(clean, 'az_err_cw'),
        clean_compass_azimuth_error_no_declination=summarize(clean, 'az_err_cw_no_declination'),
        clean_angular_error_cw_vs_ccw={'cw': summarize(clean, 'ang_err_cw'), 'ccw': summarize(clean, 'ang_err_ccw')},
        visibility=visibility,
    )
    (out / 'results.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=2, default=str))
    keys = sorted({k for r in rows for k in r})
    with open(out / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(ROOT / 'data_download/pano2pano/extracted'))
    p.add_argument('--out', default=str(ROOT / 'experiments/out/EXP0028_pano_sun'))
    main(p.parse_args())
