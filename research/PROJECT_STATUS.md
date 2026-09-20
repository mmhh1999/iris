## EXP0015 — InteriorVerse material data acquired (2026-09-20)

The user-supplied legacy URL returns 404. The author's August 2026 update endorses
the Lez/InteriorVerse backup. Downloaded and SHA-256/CRC-verified its first 2.11 GB
85-degree shard: 100 scenes, 5,988 EXRs. Inspected all six modalities for 26 views
from three scenes. This is synthetic material-map data, not real scans or a
verified solar-rerender asset bundle. No mesh/extrinsics/sun metadata in this shard;
the official README still marks spatially-varying lighting unreleased.
See `INTERIORVERSE_ACCESS_ZH.md` and `evidence/EXP0015/`.

## EXP0014 — real HDR acquisition and geometry audit (2026-09-20)

Downloaded complete meshes/calibration and 32-view HDR pilots for each of two
real Eyeful Tower scenes: riverview and apartment (64 HDRs, 48 train / 16 official
test views; 1.06 GB acquired). All file hashes rechecked, HDRs decoded, distortion
removed, camera round trips checked, and all 64 views ray-traced against the mesh.
Riverview has visible real daylight patches and is the selected first solar scene.
Interactive inspection: `experiments/out/EXP0014_eyeful/index.html`.
Report: `EYEFUL_PILOT_ZH.md`; evidence: `evidence/EXP0014/`.
This is data/geometry preparation, not recovered BRDF or solar rerendering.
TexIR and Cali-HDR remain access-by-request; official instructions saved, no emails sent.

## Current objective — photo-only solar/BRDF evaluation (2026-09-19)

The user clarified the target: generate many photorealistic interiors with retained
solar and material ground truth, hide that truth from estimation, and test whether
recovering explicit sun improves BRDF accuracy. Shadow IoU alone does not answer
this question. See `PHOTO_ONLY_SOLAR_BRDF_ZH.md` for the current protocol and
`evidence/EXP0013_preparation/asset_audit.json` for two acquired, CRC-checked complete
PBR asset packages. Those artist-authored scenes are engineering assets, not real
scans, and are not yet converted into verified solar datasets. Strict photo-only
and known-geometry diagnostic tracks must remain separate. No new BRDF result yet.

## EXP0012 — real scanned geometry rerendering (2026-09-19)

Completed 32 observations on the unchanged 4.22M-triangle ScanNet++ room
`1b379f1114`: four cameras, four simulated sun conditions, two seeds. Known
uniform diffuse material. Train-only direction error 0.486°, heldout direct-sun
proxy IoU 0.900 (oracle 0.980). Camera yaw ±1° reduces mean IoU to 0.822.
Full pipeline rerun reproduced all mask metrics. Not a full IRIS comparison or
real-photo sim-to-real result. See `SCANNETPP_RERENDER_ZH.md` and
`research/evidence/EXP0012/`.

# Project Status

Last updated: 2026-09-19 (EXP0008–EXP0010).

## Baseline progress — EXP0011 (2026-09-19)

Local CUDA 12.8 / GCC 13 toolchain and tiny-cuda-nn now work on RTX 5070 Ti.
Official bathroom data (109 train / 13 validation images) and checkpoint are
downloaded with ZIP CRC verification and SHA-256 manifests. Official-checkpoint
rendering and three actual initialization optimizer steps have passed. Full-split rendering passed: PSNR 28.97625 / SSIM 0.79503. Original BRDF/CRF
and emitter optimization also passed three steps each. The bounded full pipeline
is running stage 10 in a detached process; a separate detached full 6/4/1/4-epoch
run is queued to start only after all bounded stages succeed. This is not yet a
converged from-scratch reproduction. Real cross-time photos are still
missing. See `research/evidence/EXP0011/` and `CROSS_TIME_CAPTURE_ZH.md`.

## Current verified status — 2026-09-19 (EXP0008–EXP0010)

This supersedes conflicting historical status below. Full IRIS baseline is still
**not reproduced at this earlier checkpoint**. GPU access works outside the sandbox (RTX 5070 Ti, 16,303 MiB,
driver 581.80); missing build dependencies were subsequently resolved in EXP0011. CPU component
experiments are functional and cheap.

- Broad screening: 53 scenes / 424 images / 1,141 candidates; one additional scene
  skipped for mismatched geometry bounds. This is screening, not labeled accuracy.
- Real scene `1b379f1114`: direction fitted to one image, evaluated on three frozen
  disjoint views. Mean brightness-proxy IoU 0.709; static appearance memory 0.833.
  **No demonstrated same-light advantage over static appearance.**
- Implicit mesh-opening control obtains 0.624 with a substantially different sun
  direction. Its visibility convention differs; this is not clean causal evidence
  for window labels and no real angular ground truth is available.
- Controlled changed-light test: 3 off-grid conditions x 2 render seeds. Mean
  angular error 0.430 degrees; changed-light proxy IoU 0.951 vs 0.148 frozen
  appearance. Known mesh/material and supplied relative sun change: component
  validation only, not full IRIS or real relighting.
- 13 tests pass; one-command evidence cycle records unique run directories,
  logs, configurations, exact source snapshots, failures and negative findings:
  `.venv/bin/python experiments/run_daylight_validation.py`.

Reports: `REAL_SUN_VISIBILITY.md`, `HELDOUT_SUN_CONTROL.md`. Machine-readable
records: `research/evidence/EXP0008`, `EXP0009`, `EXP0010`, `EXP0007_expanded`.

Remaining scientific gates: independently calibrated windows and sun observations;
isolate exterior-occlusion effects; real heldout illumination; original IRIS
baseline and joint material/light optimization. No formal certification or new
novelty claim has been made. Existing synthetic passes are component evidence.

## Goal

**Revised in session 2 — see D0007.** Not "add explicit sun+sky to IRIS" (found, session 2, to be already substantially done single-image by the same lab — Li et al., ECCV 2022, `LITERATURE.md` §1a). Now: investigate whether **solar-geometry and window-projection constraints (jointly with a solar-ephemeris prior) resolve material-illumination ambiguity in multi-view indoor inverse rendering**, evaluated by the central test of whether the recovered decomposition **predicts a real, independently captured photograph under an unseen solar condition** (not just training-condition reconstruction quality). See `NOVELTY_GAP.md` for the full reframing and the "killer experiment" design, `TECHNICAL_PROPOSAL.md` for the (mostly unchanged) engineering plan, and `IRIS_ARCHITECTURE_AUDIT.md` for the original confirmed gap this all still rests on (IRIS has zero environment/sky illumination term, full stop — that finding is unaffected by the reframing).

## Branch

`research/daylight-aware-iris`, branched from `main` @ `d2d4381` (clean tree at branch time). Baseline behavior on `main` is untouched.

## Current phase

**Phase A/B/C core claims validated on synthetic data (PASSED); GPU/OptiX environment now working; research question reframed after a critical literature finding.** Three major developments since the branch was created, all in this single continuous session: (1) Phase A/B/C synthetic validation passed, (2) the WSL2/OptiX GPU blocker was resolved end-to-end (user applied the fix, verified with an actual GPU-rendered image), (3) the user identified — and I independently verified — a directly on-point prior paper that required substantially revising the novelty claim (D0007). Next: verify IRIS's own drjit-0.4.x-era code against the now-working drjit 1.5.0 stack, then a Phase 0 real-room capture (per the user's refined `DATASET_AUDIT.md` protocol) as the next go/no-go gate, in parallel with broadening the synthetic sweep.

## What's done

1. **Architecture audit complete** — `IRIS_ARCHITECTURE_AUDIT.md`. Confirmed with file:line evidence: no environment/sky radiance anywhere (`model/emitter.py` explicitly comments "assume zero background lighting"); BRDF is purely opaque reflective (no IOR/transmission); no window/opening entity in scene loading; no solar/temporal metadata in any dataset loader; emitter extraction (`extract_emitter_ldr.py`) flags triangles as emitters purely by saturated raw-LDR-pixel statistics, which would misclassify a sun patch as a "glowing floor."
2. **Compute environment blocker RESOLVED** — see `BASELINE_REPRODUCTION.md` EXP0001. Local machine is WSL2 + RTX 5070 Ti (Blackwell). Repo's pinned `mitsuba==3.5.0`/`drjit==0.4.4` cannot initialize CUDA at all on this GPU (confirmed permanent limitation of that old stack, not fixed). `mitsuba==3.9.1`/`drjit==1.5.0` + the correct WSL2 OptiX driver files (now installed) gives a **fully working GPU-accelerated Mitsuba** — verified with an actual path-traced render, not just scene loading. Not yet verified: whether IRIS's own code (`utils/path_tracing.py` etc., written against drjit 0.4.x's API) runs unmodified against drjit 1.5.0 — this is the immediate next step.
3. **Literature survey + novelty/gap analysis complete** — `LITERATURE.md` (~40 methods), `NOVELTY_GAP.md`. Key finding: sun+sky decomposition (SG-sun + SH-sky) is mature for **outdoor** scenes (NeRF-OSR→SOL-NeRF→ROS-GS/GaRe), window-aperture geometry recovery from indoor point clouds is a solved classical problem, transmissive-material recovery is mature for discrete objects — but no system combines all three with indoor SVBRDF/emitter co-optimization. `TECHNICAL_PROPOSAL.md` updated to adopt the SOL-NeRF/ROS-GS sky parameterization and Mitsuba's built-in thin-dielectric BSDF rather than inventing new machinery.
4. **Dataset audit complete** — `DATASET_AUDIT.md`. Evaluated 18 dataset/families; verdict: no existing real-world dataset combines geometry + multiview + confirmed windows + confirmed sun patches + multi-time variation (structural gap across the field, not a close call). Concludes targeted capture (3 rooms x 3 time-of-day sessions) is necessary for real-world Phase E/F evaluation. **This is a real-world action item for the user, not something I can do — flagged to user, not yet scheduled/blocking.** In the meantime, `OpenRooms` and `I²-SDF` (synthetic, explicit window-emitter ground truth) are recommended for component-level validation, and IRIS's own 8 scenes remain the baseline-reproduction target.
5. **Phase A + B (intensity) synthetic validation PASSED** — `PHASE_A_SYNTHETIC.md`, EXP0002. Built `utils/solar_geometry.py` (Mode A, pvlib-backed, self-tested), `utils/window_geometry.py` (window-to-plane sun-patch projection), `utils/sun_patch.py` (IoU-based Mode B geometric search). Recovered a known synthetic sun direction (ground truth az=200/el=40 deg) to 2.5 deg angular error and sun intensity to 4.1% mean relative error, from an independent Mitsuba (`llvm_ad_rgb`, CPU) render's actual sun patch — not just a circular self-consistency check, and intensity recovery chained off the (imperfect) recovered direction, not ground truth. Residual direction error traced to an unmodeled occluder (a glossy sphere's self-shadow biting into the patch), confirmed not Monte-Carlo noise via an spp ablation. Phase C's core claim (window-aware hard-shadow patch rendering with correct geometric alignment) is also substantially covered by the same experiment. This is a real, physically-grounded pass of the project brief's Phase A/B/C gates, run entirely on CPU without waiting for the GPU/OptiX fix (D0004). Not yet done: sensitivity to material-estimation error (vs. exact GT albedo used here), multi-condition sweep (only one sun angle tested so far) — see `PHASE_A_SYNTHETIC.md` Limitations.

## What's blocked / waiting

- No IRIS datasets downloaded yet (box.com links in README; ~8 scenes). Needed for full baseline reproduction, not yet done.
- Whether IRIS's actual codebase (drjit 0.4.x-era API) runs against the now-working drjit 1.5.0/mitsuba 3.9.1 stack is untested — immediate next step.
- Real-world Phase 0 capture (`DATASET_AUDIT.md`'s refined protocol): a real-world action item for the user (needs a room with a window, a day with usable sun, ~1 hour across 3 sessions). Not blocking synthetic/engineering work.
- Finding a collaborator with differentiable-rendering/light-transport depth: per the user's own networking plan (not recorded in detail here — see memory note below), this is entirely the user's own action (warm intros, CMU graphics seminar, etc.), not something I act on. Worth noting: the "what you need before reaching out" checklist the user described (failure example, method figure, synthetic demo, 1-page proposal) is **already substantially satisfied** by `IRIS_ARCHITECTURE_AUDIT.md` + `TECHNICAL_PROPOSAL.md` + `PHASE_A_SYNTHETIC.md`'s render — missing piece is a real-room predicted-vs-observed sun patch figure, which the Phase 0 capture above would directly produce.

## What's next (in order)

1. ~~Verify IRIS's own `utils/path_tracing.py` against drjit 1.5.0~~ **Done, EXP0003/D0008** — found and fixed two real API breaks (one silent-shape-transpose bug), verified geometrically correct, not just error-free.
2. ~~Identifiability mechanism test~~ **Done, EXP0004, PASSED** — `IDENTIFIABILITY_ABLATION.md`. Controlled proxy experiment (not full IRIS) directly supports Contribution One: the solar constraint improves recovered albedo (2.3x / 1.8x RMSE reduction across two sub-tests) while *not* improving pixel-fit error — the improvement is in disentanglement, not reconstruction, which is the core defensible claim.
3. ~~Real-dataset accessibility investigation + a real-photo screening tool~~ **Done, EXP0005** — `REAL_PHOTO_SCREENING.md`, `DECISIONS.md` D0009. No autonomously-fetchable real posed-multiview+geometry dataset found (box.com, ScanNet++, IVGM's Baidu Pan bulk archive, and a HF aggregator all checked and gated). Built and honestly evaluated the 2D sun-patch screening component on real photos instead: works better than brightness-only on 3/4 positives, with two found limitations (mullion-grid fragmentation; can't always distinguish a light source from a lit surface without 3D geometry) — the latter is itself informative evidence for why Mode B needs real geometry, not just appearance cues.
4. Broaden the Phase A/B synthetic sweep (multiple sun angles, material-estimation-error sensitivity) and the identifiability ablation (noise/config sweep) — cheap, addresses "single favorable data point" concerns, doesn't depend on the user.
5. **Depends on the user now**: either (a) complete ScanNet++ registration, (b) attempt/facilitate the IVGM Baidu Pan download (they may have better access to it than this environment), or (c) the Phase 0 real-room capture (`DATASET_AUDIT.md`) — any one unblocks the actual geometry-grounded real-data comparison, which is the evidence that would really matter for a paper (re-running the identifiability test with IRIS's actual `NGPBRDF`, running the killer experiment).
6. Phase D: joint optimization integrated into IRIS's actual training stages (GPU confirmed working, core ray-intersection primitive confirmed compatible — the remaining engineering, `model/external_lighting.py` / `model/transmission.py` per `TECHNICAL_PROPOSAL.md`, not yet built).

## Open scientific questions (not yet resolved by evidence)

- Does the ephemeris+window-projection prediction actually match a real observed sun patch on real (imperfect) reconstructed geometry, or does real-world mesh/window-reconstruction noise dominate? (Phase 0's whole purpose.)
- Does the recovered decomposition predict a real held-out time-of-day photograph better than baseline IRIS's own relighting edits — the killer experiment (`NOVELTY_GAP.md`)? Nothing else matters as much as this one.
- Can sun direction be recovered from window+sun-patch geometry alone (Mode B) with useful accuracy on real data, or is metadata (Mode A) required to disambiguate? (Synthetic evidence so far: yes, 2.5 deg error, but only one condition tested.)

## Links

- [[architecture-audit]] `IRIS_ARCHITECTURE_AUDIT.md`
- [[baseline-reproduction]] `BASELINE_REPRODUCTION.md`
- [[decisions-log]] `DECISIONS.md`
- [[experiments-registry]] `EXPERIMENTS.md`
- [[failed-ideas]] `FAILED_IDEAS.md`
- [[literature-survey]] `LITERATURE.md`
- [[novelty-gap]] `NOVELTY_GAP.md`
- [[dataset-audit]] `DATASET_AUDIT.md`
- [[technical-proposal]] `TECHNICAL_PROPOSAL.md`
- [[phase-a-synthetic]] `PHASE_A_SYNTHETIC.md`
