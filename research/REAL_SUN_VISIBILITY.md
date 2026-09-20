# EXP0008 / EXP0010 — Real-scene directional visibility, with falsification controls

2026-09-19. **Component result; not a reproduced IRIS baseline or real relighting result.**

## Selection and frozen protocol

Expanded EXP0007 screening processed 53/54 locally ready scenes, 424 images,
1,141 bright-region candidates: 916 supported, 225 rejected. `abf29d2474` was
skipped because camera metadata and mesh bounds disagree. The first broad run
aborted at that guard; the second records the skipped scene and checkpoints
completed results. These counts do not measure detection accuracy.

Visual inspection found clear cast furniture/window shadows in `1b379f1114`,
unlike the ambiguous floor highlight in the earlier room. This is exploratory
scene selection, not an unbiased benchmark. The train frame is
`47630f4e_DSC09446.JPG`. Before fitting, heldout views were chosen geometrically
from the same filename sequence: `DSC09417`, `DSC09462`, `DSC09486` (same prefix).
They have camera baselines roughly 0.4–0.9 m and directions within 40 degrees.
They are disjoint views carved from the official training frames, **not the
ScanNet++ official evaluation split**. No cross-time condition is claimed; EXIF
DateTime is absent in these four resized images, so identical illumination is an
assumption, not verified metadata.

`configs/daylight/exp0008_real_visibility.json` freezes the floor height, two
approximate vertical apertures, sampling, split and search settings. Apertures
were manually hypothesized by the agent from train-image frame/mesh intersections.
The right aperture's far endpoint is extrapolated outside the train image and
unverified. The fitted solution uses only the back aperture; removing the right
one does not change predictions. This does not certify either aperture geometry.

All 950 train poses were independently checked against COLMAP: maximum matrix
error 3.55e-15. Source, configuration, image/mask/mesh/pose hashes are archived in
`research/evidence/EXP0008/`, including exact executed source snapshots.

## Method and metric

Sample image pixels every 8 pixels, intersect the fixed scan mesh, retain points
within 2 cm of the manually measured floor height 0.362 m, obey the official
validity mask, and deterministically sample at most 4,000 points per view (seed 0).
Classify grayscale floor brightness using per-view Otsu thresholding. These are
**brightness-proxy labels**, not independently annotated direct-sun masks.

For a candidate sun direction, trace each floor point toward the source. Require
intersection with an explicit aperture before an opaque mesh occluder; a 4 cm
tolerance accommodates near-aperture mesh errors. Search a 10-degree azimuth /
5-degree elevation grid and refine locally at 1 degree. Optimize only train IoU.
Load the heldout image observations after freezing the direction. Scene azimuth
is referenced to mesh axes, not geographic north; no solar-angle ground truth
exists, so no real angular accuracy is reported.

The recovered direction is scene azimuth 240 degrees, elevation 26 degrees.

## Results and the stronger counterexample

| Model | Train proxy IoU | Mean of 3 heldout-view proxy IoUs |
|---|---:|---:|
| Explicit aperture + mesh occluders | 0.8398 | 0.7093 |
| Fixed fitted direction rotated by +45 degrees | 0.1141 | 0.3186 |
| Back aperture only | 0.8398 | 0.7093 |
| Right aperture only | 0.0000 | 0.0000 |
| Apertures translated -10 cm along X, direction fixed | 0.7136 | 0.6205 |
| Apertures translated +10 cm along X, direction fixed | 0.8424 | 0.7278 |
| Apertures translated +1 m along X, direction fixed | 0.7187 | 0.6476 |
| All floor points bright | 0.4537 | 0.5044 |
| Memorize train floor labels, nearest-3D-neighbor transfer | **1.0000** | **0.8333** |
| EXP0010: implicit mesh openings, separately fitted direction | 0.7837 | 0.6243 |

The real physical model's per-view IoUs are 0.5673, 0.7979, 0.7626. All comparisons
use identical evaluated points and labels. The memory baseline is a simple static
appearance control, **not IRIS or a trained neural SLF**. Its better performance
falsifies a claim of demonstrated superiority over static appearance fitting on
this task. It also shows why same-light multiview consistency is insufficient.

The +10 cm aperture shift improves agreement; the geometry is not accurately
calibrated, and the unshifted model was retained without selecting on heldout
scores. Varying ray tolerance from 2 to 8 cm changes heldout IoU only from 0.7090
to 0.7092 here, but this is not a general robustness guarantee.

## EXP0010 interpretation

Trace sun rays to infinity using the original mesh's implicit openings, without
explicit aperture labels, and independently fit using the same train grid/refinement.
It selects a substantially different direction (345 / 44 degrees) and obtains
mean heldout-view proxy IoU 0.6243. **This is an operational model comparison,
not a clean proof of aperture-prior contribution:** the explicit model also
terminates visibility at the hypothesized portal, whereas the implicit model
counts any later mesh hit as occlusion. Exterior/erroneous reconstructed surfaces
and approximate portal positions can therefore explain part of the difference.
Without sun GT, neither fitted direction can be certified as correct.

## Reproduction and remaining gates

```bash
.venv/bin/python experiments/run_daylight_validation.py
# Optional broad screening in the same evidence cycle:
.venv/bin/python experiments/run_daylight_validation.py --screen
```

Each invocation creates a unique output directory and records commands, logs,
failures, source snapshots and findings. A completed run explicitly records
`real_physical_beats_memory: false`; successful execution is not labeled research
success. The real fit and controls took about 3–4 CPU seconds on this machine,
using about 0.8 GiB peak process RSS in the initial run.

Remaining gates: independent annotations; multiview window calibration; testing
whether ignored exterior occlusion explains the alternate directions; full IRIS
baseline reproduction; material/emitter/CRF joint optimization; real heldout
illumination with known timestamps/orientation. Synthetic EXP0009 addresses only
a controlled changed-lighting component, not those real-world gates.
