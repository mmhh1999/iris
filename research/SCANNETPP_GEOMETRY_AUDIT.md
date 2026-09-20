# EXP0007 — Real-data geometry screening audit

2026-09-19. **Mixed result; not a successful sun detector or IRIS reproduction.**

## Question and protocol

Does rejecting bright 2D regions without mesh support resolve the previous
window-versus-receiving-surface confusion? Two locally available scenes from the
previous screening, 12 evenly spaced good training frames each, top 3 candidates
per frame, 256 uniformly sampled pixels per candidate, seed 0. Thresholds:
reject >40% missing intersections, or median depth >1.3 mesh bounding-box diagonals.
These are heuristic thresholds, not learned or calibrated probabilities. Test
frames are excluded; no thresholds were tuned to the resulting counts.

Use only `resized_undistorted_images` with `transforms_undistorted.json` and the
original `mesh_aligned_0.05.ply`. No official data are modified. CPU LLVM Mitsuba
runs successfully; GPU access in this session is blocked by the OS. No full
training run was launched.

## Critical invalidated attempt

The first attempt converted camera axes from OpenGL to OpenCV but omitted the
**world** transform. It reported 55/59 rejections. That result is invalid and
retained under `experiments/out/EXP0007_geometry/INVALIDATED.json`. It must never
be presented as detection improvement.

The official [ScanNet++ exporter](https://github.com/scannetpp/scannetpp/blob/main/common/utils/nerfstudio.py)
swaps world X/Y and negates world Z in addition to flipping camera Y/Z. The
inverse is `T_mesh_camera = A @ T_nerfstudio_camera @ diag(1,-1,-1,1)`, where
`A` maps `(x,y,z)` to `(y,x,-z)`. Undistortion changes intrinsics, not poses:
[official implementation](https://github.com/scannetpp/scannetpp/blob/main/dslr/undistort.py).

Independent verification compares all 3,741 training poses against inverse
original COLMAP extrinsics. Maximum absolute matrix error: 2.22e-15 (1,472 poses,
7b04052ad0), 1.33e-15 (2,269 poses, 3cb9f85891). Runtime also asserts transformed
camera metadata bounds agree with the mesh bounds. This establishes coordinate
consistency; it does not establish photometric or geometric ground truth accuracy.

## Corrected observations

| Scene | Images | Candidates | Surface supported, unverified | Rejected |
|---|---:|---:|---:|---:|
| 7b04052ad0 | 12 | 33 | 33 | 0 |
| 3cb9f85891 | 12 | 26 | 22 | 4 |
| Total | 24 | 59 | 55 | 4 |

About 6 seconds CPU for the first corrected run (including mesh loading, image
screening and diagnostic output). See the manifest/results for exact runtime,
dependency versions, source/data hashes, git commit, dirty state and configuration.
No PSNR, precision, recall or statistical significance is claimed.

Visual diagnostic checks (qualitative observations, not independent annotations):

- `3cb9f85891/5c25b5f5_DSC04201.JPG`: candidate 0 is visibly on the bright balcony
  door/window and has 4.69% mesh-hit support; rejected. Candidate 1 is on the floor,
  has 100% support and is retained. Direct sunlight versus reflection/diffuse light
  remains unverified.
- `7b04052ad0/DSC09999.JPG`: apparent artificial-light/fixture regions remain
  supported by geometry. Thus surviving regions cannot be certified as sunlight.

**Conclusion:** geometry can reject an aperture in this example but is not
sufficient to resolve the semantic ambiguity. The general claim that mesh hits
alone distinguish sunlit surfaces from light sources is not supported.

## Reproduce and verify

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/validate_scannetpp_poses.py
.venv/bin/python experiments/scannetpp_geometry_audit.py --out experiments/out/EXP0007_new_run
```

Output directory must not already exist, to preserve previous runs. Seven tests
cover known intersections/misses, pixel axes, camera/world conversion, mismatched
bounds, invalid inputs, deterministic sampling and preserving the chosen backend.
Float32 geometry assertions use 1e-6 absolute tolerance. Cache-write warnings
under the read-only home directory do not prevent execution. PLY color warnings
do not affect intersection geometry; vertex colors are not used for this test.

Machine-readable evidence is in `research/evidence/EXP0007/`; dataset images and
local diagnostic JPEGs stay under ignored `experiments/out/EXP0007_verified/`.

## Limits and next discriminating experiment

Two selected scenes are not a representative dataset. Sparse candidate sampling
can miss small surface regions. Undistorted border extrapolation and anonymization
masks are not yet incorporated in candidate extraction. Mesh holes cause false
rejections; glass/exterior geometry and light fixtures cause false acceptance.
Bounding boxes only guard a known coordinate mismatch, not full registration.
No solar time/orientation metadata, window annotation or held-out illumination
has been established. Full IRIS baseline and joint optimization remain undone.

Next: annotate a small, frozen set of receiver/window/artificial-light/unknown
regions; estimate a window polygon and receiving planes on one selected room;
test cross-view projected-patch consistency and deliberate wrong-window/direction
controls. Do not train on heuristic detections as if they were labels. A useful
innovation claim still requires independent held-out-lighting evidence.

Correction (EXP0008 continuation): original prose incorrectly said 50% / 1.5;
the executed wrapper defaults were 40% / 1.3, as preserved in source hashes.
Counts are unchanged. Expanded runs now record thresholds explicitly.
