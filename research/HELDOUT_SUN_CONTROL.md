# EXP0009 — Controlled prediction under changed solar illumination

2026-09-19. **Positive component validation under known geometry/materials.**

## Why this experiment

EXP0008 found that simple memorized floor appearance beats the physical model
on same-sequence heldout views. Test the actual differentiating mechanism by
changing illumination in an independently rendered controlled scene.

Reuse the Phase A room, rectangular opening, diffuse surfaces and glossy sphere.
Mitsuba path tracing (32 spp, 192x144) generates train and test RGB images. The
inference code uses geometric visibility through the explicit aperture, including
the sphere's shadow; it does not fit directly to its own predicted visibility.
Both use Mitsuba intersection infrastructure and identical known mesh, so this
is not independent-renderer validation or a test of geometry reconstruction.

Training sun angles deliberately lie **off** the search grid:
(163.7,31.4), (203.2,43.6), (218.4,51.7) degrees, two render seeds each. The earlier
on-grid six-case pilot is preserved under `experiments/out/EXP0009`; it recovered
zero angular error because the true directions were among search candidates.
Do not use that pilot's zero error as an unrestricted accuracy claim.

The test illumination changes azimuth by +20 degrees and elevation by +10 degrees.
This true **relative angular change is supplied**, analogous to trusted solar
metadata, and applied to the recovered training direction. No test-image fitting.
Known geometry and material are shared. Test masks come from a fixed RGB-sum
threshold 1.2, inherited from Phase A, on directly rendered floor pixels. Thus
IoU is against a rendered radiometric proxy, not a separately rendered direct-only
AOV. The unchanged camera means all methods evaluate the same 15,764 floor points.

## Off-grid results

| Train true azimuth / elevation | Recovered | Angular error | Changed-light physical IoU, seeds 0 / 1 | Frozen-appearance IoU, seeds 0 / 1 |
|---|---|---:|---|---|
| 163.7 / 31.4 | 164 / 31 | 0.475° | 0.937 / 0.939 | 0.263 / 0.262 |
| 203.2 / 43.6 | 203 / 44 | 0.425° | 0.955 / 0.964 | 0.128 / 0.125 |
| 218.4 / 51.7 | 218 / 52 | 0.389° | 0.964 / 0.948 | 0.054 / 0.056 |

Mean angular error: **0.430 degrees**. Mean changed-light IoU: **0.951** for the
physical model, **0.148** for frozen training appearance. A second, automated
end-to-end run reproduced these metrics exactly. Six cases are three conditions
with repeated Monte Carlo seeds, not six independent scenes; no inferential
statistics or real-world generalization claim is made.

The physical result uses actual mesh occluders, addressing the earlier Phase A
analytic aperture-only model's unmodeled sphere-shadow limitation. This is an
engineering improvement, not a novelty claim. Runtime is about 13 CPU seconds.

## What this does not establish

No IRIS comparison, learned BRDF, optimized window transmission, unknown camera
response, reconstruction noise, real-time sunlight metadata or real heldout-light
photograph. The frozen baseline deliberately cannot respond to a new sun direction;
its failure tests this mechanism, not superiority over alternative relighting
systems. This is necessary evidence for the intended mechanism, not sufficient
publication evidence.

Run `.venv/bin/python experiments/run_daylight_validation.py`, or run
`experiments/heldout_sun_control.py --out <new-directory>` separately. Exact
configuration, results and source snapshots are in `research/evidence/EXP0009/`.
RGB previews and masks stay in ignored local experiment output directories.
