# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EXP0005: test `utils.sun_patch.screen_image_for_sun_patches` on real,
freely-licensed photographs (not synthetic renders) -- the first
component of a real-world validation pipeline, run ahead of having
access to a full posed-multiview + geometry dataset (gated behind
registration/bulk-download barriers, see DECISIONS.md).

This tests ONLY the 2D screening/detection step (no window/floor
geometry available for these uncurated photos, so no Mode B geometric
search is possible here) -- see IDENTIFIABILITY_ABLATION.md and
PHASE_A_SYNTHETIC.md for the geometry-grounded parts of the pipeline,
which need a real posed-multiview scene to run for real.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from utils.sun_patch import screen_image_for_sun_patches

PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'real_photo_test')
OUT_DIR = PHOTO_DIR

# (filename, expected label) -- expected labels are my own annotation based on
# visual inspection when I selected these images, used only to report
# precision/recall here, not used by the algorithm itself.
IMAGES = [
    ('real1.jpg', 'positive: window-mullion grid patch on tile floor, hard edges'),
    ('neg_candidate_2.jpg', 'positive: two window patches on reflective epoxy floor'),
    ('neg_candidate_3.jpg', 'positive (harder): soft diagonal patch through glass doors, mall'),
    ('neg_final.jpg', 'positive: clean rectangular patch on carpet, window out of frame'),
    ('neg_final2.jpg', 'negative: bright window (blinds), but no distinct cast floor patch'),
    ('neg_candidate_1.jpg', 'negative: outdoor photo, not applicable'),
]


def draw_candidates(image_bgr, candidates):
    out = image_bgr.copy()
    for i, c in enumerate(candidates):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.drawContours(out, [c.contour], -1, color, 4)
        x, y, w, h = cv2.boundingRect(c.contour)
        label = f'score={c.score:.2f} rect={c.rectangularity:.2f} v={c.polygon_vertices}'
        cv2.putText(out, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return out


def run():
    results = {}
    for fname, expected in IMAGES:
        path = os.path.join(PHOTO_DIR, fname)
        if not os.path.exists(path):
            print(f'skip {fname}: not found')
            continue
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        candidates = screen_image_for_sun_patches(img_rgb)
        top = candidates[0] if candidates else None

        result = dict(
            expected=expected,
            n_candidates=len(candidates),
            top_score=top.score if top else None,
            top_rectangularity=top.rectangularity if top else None,
            top_polygon_vertices=top.polygon_vertices if top else None,
            top_area_frac=(top.area_px / (img_rgb.shape[0] * img_rgb.shape[1])) if top else None,
        )
        results[fname] = result
        print(f'[{fname}] {expected}')
        print(f'  -> {json.dumps(result, indent=2)}')

        vis = draw_candidates(img_bgr, candidates)
        # downscale for a manageable output file size
        scale = 900.0 / max(vis.shape[:2])
        if scale < 1.0:
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
        out_path = os.path.join(OUT_DIR, f'screened_{fname}')
        cv2.imwrite(out_path, vis)
        print(f'  saved annotated visualization to {out_path}')

    with open(os.path.join(OUT_DIR, 'screening_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults written to', os.path.join(OUT_DIR, 'screening_results.json'))


if __name__ == '__main__':
    run()
