# Failed / Rejected Ideas

Record every idea that was tried and disproven, or considered and rejected before trying, with the reasoning. Never delete entries. Revisit only if genuinely new evidence appears — reference the old entry rather than re-litigating from scratch.

---

(No ideas have been tested and rejected yet — this is session 1, still in the audit/environment-setup phase. Entries will be added as Phase A synthetic experiments and later phases produce negative results. Recorded here now only to establish the file per the project brief's required research-memory structure — see `PROJECT_STATUS.md`.)

## 2026-09-19 / EXP0007: mesh-hit filtering is insufficient for sunlight recognition

Corrected geometry retained 55/59 candidates in 24 images, including apparent
artificial-light regions. It rejected one visually checked bright window region,
but does not establish precision/recall. Do not call survivors high-confidence
sun patches. The earlier 55/59 rejection result was a coordinate-frame bug and
is explicitly invalidated, not favorable evidence. Next constraint must explain
window-to-receiver projection across views, with hard negatives and frozen labels.

## EXP0008: same-light multiview agreement does not beat memorized appearance

The physical model scores mean heldout-view brightness-proxy IoU 0.709, compared
with 0.833 for nearest-neighbor transfer of the training view's floor labels.
Reject a superiority claim based on these same-light views. Direction is unverified
without GT; +10 cm aperture translation improves the proxy, indicating imperfect
geometry. EXP0009's positive changed-light synthetic result is not a substitute
for real heldout illumination or original IRIS baseline comparison.

## EXP0010: explicit-portal vs sky-visibility is not a perfectly isolated ablation

The portal model clips visibility at the aperture and allows 4 cm tolerance;
implicit mesh visibility counts all hits to infinity. Its lower 0.624 heldout
proxy IoU may partly reflect exterior/erroneous geometry, not just absence of the
window label. Preserve this confound and the substantially different fitted sun
(345/44 vs 240/26) rather than declaring real sun recovery correct.

## EXP0028: camera/phone compass heading as the scene-orientation source

Rejected. Ricoh THETA Z1 `GPSImgDirection` (ref = magnetic, corrected with
WMM2020 -9.3 deg) disagreed with the observed sun azimuth by 82.7 deg median,
146.9 deg max on 7 real window-view panoramas, while ephemeris elevation from
the same EXIF time/GPS matched to 0.48 deg median. The failure is the heading,
not the time/GPS or the equirectangular convention (mirror hypothesis ruled
out). Use observed sun / sun patch for yaw (D0016); keep compass at most as a
logged hint. Do not revisit unless a capture with a verified external heading
reference (e.g. surveyed window normal) shows otherwise.

## EXP0027: "explicit sun term + albedo-consistency regularizer beats SGS on sun contamination" (EXP0025/EXP0026)

Retracted. 3-seed ablation shows the regularizer alone gives the entire gap
reduction by flattening the single-segment uniform floor to ~white (absolute
albedo error gets worse, MAE 0.78 vs vanilla 0.62). The sun term at
sun_lr_scale 0.1 / 200 steps barely moved from init, so it contributed nothing.
The uniform-floor lit/shadow gap metric is gameable by flattening; never use it
alone. Revisit only with a textured-floor, floor-segment benchmark and an
active (learned or supplied) sun term -- D0017.
