"""EXP0023: real-EXIF ephemeris cross-check for Cali-HDR / Pano2Pano.

Pure computation on top of EXP0022's exif_inventory.csv (no new I/O against
the source zips beyond what EXP0022 already read). For every image that has
both GPS and a UTC timestamp, computes the *true* solar position via pvlib
(the same `utils/solar_geometry.py::solar_position` already validated on
synthetic data in PHASE_A_SYNTHETIC.md) and, where a compass heading is also
present, the signed angle between the camera's facing direction and the true
sun bearing -- a cheap, real-data sanity check of whether "sun in front of
camera" / "sun behind camera" is consistent with each shot's indoor/outdoor
label and its scene's lit/unlit naming, before any pixel-level sun-patch
matching is attempted.

Does not touch pixel data. Does not claim a BRDF/solar recovery result --
this only tests whether the metadata is *physically self-consistent*
(pvlib ephemeris vs. recorded camera heading vs. day/night), which is a
precondition for using these scenes in the real killer-experiment pipeline.
"""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.solar_geometry import solar_position

IN_CSV = "research/evidence/EXP0022/exif_inventory.csv"
OUT_DIR = Path("research/evidence/EXP0023")


def angular_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(IN_CSV)))

    results = []
    skipped = defaultdict(int)
    for r in rows:
        if not r.get("lat") or not r.get("datetime_utc"):
            skipped["no_gps_or_utc"] += 1
            continue
        try:
            when = datetime.strptime(r["datetime_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            lat, lon = float(r["lat"]), float(r["lon"])
            alt = float(r["alt_m"]) if r.get("alt_m") else 0.0
        except Exception:
            skipped["parse_error"] += 1
            continue
        pos = solar_position(lat, lon, when, altitude_m=alt)
        row = {
            "dataset": r["dataset"],
            "date_folder": r["date_folder"],
            "scene_folder": r["scene_folder"],
            "zip_entry": r["zip_entry"],
            "indoor_outdoor": r["indoor_outdoor"],
            "datetime_utc": r["datetime_utc"],
            "datetime_local": r.get("datetime_local"),
            "lat": lat, "lon": lon,
            "sun_azimuth_deg": round(pos.azimuth_deg, 2),
            "sun_elevation_deg": round(pos.elevation_deg, 2),
            "is_daytime": pos.elevation_deg > 0,
        }
        if r.get("camera_heading_deg"):
            heading = float(r["camera_heading_deg"])
            row["camera_heading_deg"] = heading
            row["camera_to_sun_angle_deg"] = round(angular_diff(heading, pos.azimuth_deg), 1)
            row["sun_roughly_ahead_of_camera"] = row["camera_to_sun_angle_deg"] < 90
        results.append(row)

    # Per-scene summary: daytime consistency + heading-vs-sun spread.
    scenes = defaultdict(list)
    for r in results:
        scenes[(r["dataset"], r["date_folder"], r["scene_folder"])].append(r)

    scene_rows = []
    inconsistent_daytime = []
    for key, items in sorted(scenes.items()):
        dataset, date_folder, scene_folder = key
        n = len(items)
        n_daytime = sum(1 for i in items if i["is_daytime"])
        elevations = [i["sun_elevation_deg"] for i in items]
        with_heading = [i for i in items if "camera_to_sun_angle_deg" in i]
        scene_rows.append({
            "dataset": dataset, "date_folder": date_folder, "scene_folder": scene_folder,
            "n_images": n, "n_daytime": n_daytime,
            "elevation_min": round(min(elevations), 1), "elevation_max": round(max(elevations), 1),
            "n_with_heading": len(with_heading),
            "mean_camera_to_sun_angle_deg": round(sum(i["camera_to_sun_angle_deg"] for i in with_heading) / len(with_heading), 1) if with_heading else None,
        })
        # Outdoor folders captured with the sun below the horizon would be a
        # red flag (night-time "outdoor daylight" capture, or a timezone bug
        # in datetime_utc conversion) -- flag for manual review, don't hide.
        if scene_folder and "out" in scene_folder.lower() and n_daytime < n:
            inconsistent_daytime.append({**key_to_dict(key), "n_images": n, "n_daytime": n_daytime})

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "pvlib solar_position() per EXIF (lat, lon, alt, UTC timestamp); pure computation, no pixel access",
        "input": IN_CSV,
        "n_images_with_gps_and_utc": len(results),
        "skipped": dict(skipped),
        "scene_summary": scene_rows,
        "outdoor_scenes_with_nighttime_frames_flagged_for_review": inconsistent_daytime,
    }
    with open(OUT_DIR / "ephemeris_crosscheck_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(OUT_DIR / "ephemeris_crosscheck_per_image.csv", "w", newline="") as f:
        fieldnames = ["dataset", "date_folder", "scene_folder", "zip_entry", "indoor_outdoor",
                      "datetime_utc", "datetime_local", "lat", "lon", "sun_azimuth_deg",
                      "sun_elevation_deg", "is_daytime", "camera_heading_deg",
                      "camera_to_sun_angle_deg", "sun_roughly_ahead_of_camera"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"Computed ephemeris for {len(results)} images across {len(scene_rows)} scenes.")
    print(f"Skipped: {dict(skipped)}")
    print(f"Outdoor scenes with any nighttime-flagged frame: {len(inconsistent_daytime)}")
    for s in scene_rows[:5]:
        print(" ", s)


def key_to_dict(key):
    return {"dataset": key[0], "date_folder": key[1], "scene_folder": key[2]}


if __name__ == "__main__":
    main()
