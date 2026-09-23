"""EXP0022: inventory Cali-HDR / Pano2Pano (Guanzhou Ji et al.) archives.

Reads only the zip central directory plus a small byte-prefix of each JPEG
entry (enough to cover the EXIF/APP1 segment) directly from the source zip,
so the full ~169 GB does not need to be extracted to answer the question
this script asks: do these real captures carry per-image GPS + absolute
timestamp + compass heading, and if so, for how many images / scenes.

Does not touch the source files (read-only), does not claim any BRDF/solar
recovery result -- this is a data-inventory step, evidence for
CALI_HDR_PANO2PANO_ACQUISITION_ZH.md.
"""
import argparse
import csv
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

PREFIX_BYTES = 262144  # 256 KiB: comfortably covers EXIF+thumbnail for these RICOH THETA Z1 / Ricoh GR JPEGs (64 KiB sufficed on samples)


def dms_to_deg(dms, ref):
    d, m, s = dms
    val = float(d) + float(m) / 60.0 + float(s) / 3600.0
    if ref in ("S", "W"):
        val = -val
    return val


def read_exif_from_zip_entry(zf, name):
    with zf.open(name) as f:
        prefix = f.read(PREFIX_BYTES)
    try:
        img = Image.open(io.BytesIO(prefix))
        exif = img._getexif()
    except Exception as e:
        return None, f"parse_error:{e}"
    if not exif:
        return None, "no_exif"
    tags = {TAGS.get(k, k): v for k, v in exif.items()}
    out = {}
    if "DateTimeOriginal" in tags:
        out["datetime_local"] = str(tags["DateTimeOriginal"])
    if "Model" in tags:
        out["camera_model"] = str(tags["Model"])
    gps_raw = tags.get("GPSInfo")
    if gps_raw:
        gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
        try:
            out["lat"] = round(dms_to_deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N")), 6)
            out["lon"] = round(dms_to_deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E")), 6)
        except Exception:
            pass
        if "GPSAltitude" in gps:
            try:
                out["alt_m"] = round(float(gps["GPSAltitude"]), 1)
            except Exception:
                pass
        if "GPSImgDirection" in gps:
            try:
                out["camera_heading_deg"] = round(float(gps["GPSImgDirection"]), 1)
            except Exception:
                pass
        if "GPSDateStamp" in gps and "GPSTimeStamp" in gps:
            try:
                h, m, s = gps["GPSTimeStamp"]
                out["datetime_utc"] = (
                    f"{gps['GPSDateStamp'].replace(':', '-')}T{int(h):02d}:{int(m):02d}:{int(s):02d}Z"
                )
            except Exception:
                pass
    return out, None


def classify_path(rel_path):
    parts = Path(rel_path).parts
    indoor_outdoor = "unknown"
    lower = rel_path.lower()
    if "outdoor" in lower or "fisheye" in lower or lower.split("/")[0:1] and "out" in Path(rel_path).parent.name.lower():
        indoor_outdoor = "outdoor"
    if "indoor" in lower or "pano" in lower or "theta" in lower:
        indoor_outdoor = "indoor"
    if Path(rel_path).parent.name.lower().endswith("-in"):
        indoor_outdoor = "indoor"
    if Path(rel_path).parent.name.lower().endswith("-out"):
        indoor_outdoor = "outdoor"
    date_folder = parts[0] if parts else None
    scene_folder = parts[1] if len(parts) > 1 else None
    return date_folder, scene_folder, indoor_outdoor


def inventory_archive(zip_path, dataset_name, limit_per_dataset=None):
    rows = []
    errors = {"no_exif": 0, "parse_error": 0}
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".jpg") and not n.endswith("/")]
        if limit_per_dataset:
            names = names[:limit_per_dataset]
        total = len(names)
        for i, name in enumerate(names):
            exif, err = read_exif_from_zip_entry(zf, name)
            date_folder, scene_folder, io_class = classify_path(name)
            row = {
                "dataset": dataset_name,
                "zip_entry": name,
                "date_folder": date_folder,
                "scene_folder": scene_folder,
                "indoor_outdoor": io_class,
            }
            if exif:
                row.update(exif)
            else:
                errors[err.split(":")[0]] = errors.get(err.split(":")[0], 0) + 1
                row["error"] = err
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  [{dataset_name}] {i+1}/{total} entries scanned")
    return rows, errors


def summarize(rows):
    scenes = {}
    gps_count = 0
    heading_count = 0
    for r in rows:
        key = (r["dataset"], r["date_folder"], r["scene_folder"])
        s = scenes.setdefault(key, {"n_images": 0, "n_with_gps": 0, "n_with_heading": 0, "indoor_outdoor": set(), "dates_seen": set(), "lat": None, "lon": None})
        s["n_images"] += 1
        s["indoor_outdoor"].add(r["indoor_outdoor"])
        if "lat" in r:
            s["n_with_gps"] += 1
            s["lat"], s["lon"] = r["lat"], r["lon"]
            gps_count += 1
        if "camera_heading_deg" in r:
            s["n_with_heading"] += 1
            heading_count += 1
        if "datetime_local" in r:
            s["dates_seen"].add(r["datetime_local"][:10])
    scene_summary = []
    for (dataset, date_folder, scene_folder), s in sorted(scenes.items()):
        scene_summary.append({
            "dataset": dataset,
            "date_folder": date_folder,
            "scene_folder": scene_folder,
            "n_images": s["n_images"],
            "n_with_gps": s["n_with_gps"],
            "n_with_heading": s["n_with_heading"],
            "indoor_outdoor_present": sorted(s["indoor_outdoor"]),
            "lat": s["lat"],
            "lon": s["lon"],
            "dates_seen": sorted(s["dates_seen"]),
        })
    return scene_summary, gps_count, heading_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calihdr-zip", default="/mnt/c/Users/XMH/Downloads/Cali-HDR Dataset.zip")
    ap.add_argument("--pano2pano-zip", default="/mnt/c/Users/XMH/Downloads/Pano2Pano_Release.zip")
    ap.add_argument("--out-dir", default="research/evidence/EXP0022")
    ap.add_argument("--limit-per-dataset", type=int, default=None, help="debug: cap JPEGs scanned per dataset")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    all_errors = {}
    for name, path in [("cali_hdr", args.calihdr_zip), ("pano2pano", args.pano2pano_zip)]:
        print(f"Scanning {name}: {path}")
        rows, errors = inventory_archive(path, name, args.limit_per_dataset)
        all_rows.extend(rows)
        all_errors[name] = errors
        print(f"  {name}: {len(rows)} JPEGs scanned, errors={errors}")

    scene_summary, gps_count, heading_count = summarize(all_rows)

    with open(os.path.join(args.out_dir, "exif_inventory.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "zip_entry", "date_folder", "scene_folder", "indoor_outdoor",
                      "lat", "lon", "alt_m", "camera_heading_deg", "camera_model",
                      "datetime_local", "datetime_utc", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_zips": {"cali_hdr": args.calihdr_zip, "pano2pano": args.pano2pano_zip},
        "method": "read-only central-directory + per-JPEG 256KiB prefix EXIF parse; no full extraction",
        "total_jpegs_scanned": len(all_rows),
        "total_with_gps": gps_count,
        "total_with_heading": heading_count,
        "errors": all_errors,
        "scene_summary": scene_summary,
    }
    with open(os.path.join(args.out_dir, "exif_inventory_summary.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTotal JPEGs scanned: {len(all_rows)}")
    print(f"With GPS: {gps_count}  With compass heading: {heading_count}")
    print(f"Scenes found: {len(scene_summary)}")
    print(f"Wrote {args.out_dir}/exif_inventory.csv and exif_inventory_summary.json")


if __name__ == "__main__":
    main()
