# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EXP0006: batch-run the sun-patch screening tool (utils.sun_patch) across
a downloaded sample of real ScanNet++ scenes, ranking scenes by likely
daylight/sun-patch content, ahead of the geometry-grounded Mode B analysis
(which needs each candidate scene's window aperture + floor plane from its
reconstructed mesh -- a follow-up step, not done here).

Scene sample: 55 scenes from ScanNet++'s nvs_sem_train split, selected via
metadata/scene_types.json biased toward room types plausibly containing
windows (apartment/office/living-room/classroom/conference-room/kitchen) --
see data_download/download_scannetpp.yml and DATASET_AUDIT.md/
REAL_PHOTO_SCREENING.md for why a random sample of all 856 scenes would be
less informative for this purpose. Downloaded assets are deliberately
lightweight (resized DSLR images + nerfstudio camera transform + train/test
split + mesh only -- no iPhone video/depth/panorama), per
BASELINE_REPRODUCTION.md's data-format mapping.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from utils.sun_patch import screen_image_for_sun_patches

DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'data_download', 'scannetpp', 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'scannetpp_screening')


def list_scenes(data_root):
    if not os.path.isdir(data_root):
        return []
    return sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))


def scene_is_ready(data_root, scene_id):
    img_dir = os.path.join(data_root, scene_id, 'dslr', 'resized_images')
    mesh_path = os.path.join(data_root, scene_id, 'scans', 'mesh_aligned_0.05.ply')
    return os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0 and os.path.isfile(mesh_path)


def screen_scene(data_root, scene_id, max_images=30, high_conf_thresh=0.55):
    img_dir = os.path.join(data_root, scene_id, 'dslr', 'resized_images')
    names = sorted(os.listdir(img_dir))
    if len(names) > max_images:
        # evenly spaced subsample, not just the first N, for better coverage
        idxs = np.linspace(0, len(names) - 1, max_images).astype(int)
        names = [names[i] for i in idxs]

    per_image = []
    for name in names:
        path = os.path.join(img_dir, name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        candidates = screen_image_for_sun_patches(img_rgb, top_k=1)
        score = candidates[0].score if candidates else 0.0
        per_image.append((name, score))

    per_image.sort(key=lambda x: -x[1])
    n_high_conf = sum(1 for _, s in per_image if s >= high_conf_thresh)
    return dict(
        scene_id=scene_id,
        n_images_screened=len(per_image),
        n_high_confidence=n_high_conf,
        max_score=per_image[0][1] if per_image else 0.0,
        top_images=per_image[:5],
    )


def run(data_root, max_images_per_scene, high_conf_thresh, limit_scenes=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    scenes = list_scenes(data_root)
    if limit_scenes:
        scenes = scenes[:limit_scenes]

    results = []
    skipped = []
    for scene_id in scenes:
        if not scene_is_ready(data_root, scene_id):
            skipped.append(scene_id)
            continue
        r = screen_scene(data_root, scene_id, max_images_per_scene, high_conf_thresh)
        results.append(r)
        print(f"[{scene_id}] screened {r['n_images_screened']} images, "
              f"max_score={r['max_score']:.2f}, high_conf={r['n_high_confidence']}")

    results.sort(key=lambda r: -r['max_score'])

    summary = dict(
        n_scenes_total=len(scenes), n_scenes_screened=len(results), n_scenes_skipped_not_ready=len(skipped),
        max_images_per_scene=max_images_per_scene, high_conf_thresh=high_conf_thresh,
        ranked_scenes=results,
    )
    out_path = os.path.join(OUT_DIR, 'ranking.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{len(results)}/{len(scenes)} scenes screened ({len(skipped)} not yet fully downloaded).")
    print("Top 10 candidate scenes by max sun-patch screening score:")
    for r in results[:10]:
        print(f"  {r['scene_id']}: max_score={r['max_score']:.2f}, "
              f"high_conf_images={r['n_high_confidence']}/{r['n_images_screened']}")
    print(f"\nFull ranking written to {out_path}")
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--max_images_per_scene', type=int, default=30)
    p.add_argument('--high_conf_thresh', type=float, default=0.55)
    p.add_argument('--limit_scenes', type=int, default=None)
    args = p.parse_args()
    run(args.data_root, args.max_images_per_scene, args.high_conf_thresh, args.limit_scenes)
