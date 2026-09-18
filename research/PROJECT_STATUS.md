# Project Status

Last updated: 2026-09-18 (session 1).

## Goal

Investigate whether explicit modeling of solar geometry, window geometry, and light transmission can improve inverse rendering and relighting of real indoor scenes under daylight, addressing IRIS's lack of any outdoor/environment illumination term. See `IRIS_ARCHITECTURE_AUDIT.md` for the confirmed gap and `../` project brief (conversation) for full scope.

## Branch

`research/daylight-aware-iris`, branched from `main` @ `d2d4381` (clean tree at branch time). Baseline behavior on `main` is untouched.

## Current phase

**Phase A/B/C core claims validated on synthetic data (PASSED).** Phase 0 (audit), Phase A (direction recovery), Phase B (intensity recovery), and Phase C's core geometric-alignment claim are done on one synthetic scene/condition; see below and `EXPERIMENTS.md` EXP0002 / `PHASE_A_SYNTHETIC.md`. Next: either (a) broaden the synthetic sweep (multiple sun angles, material-error sensitivity) before trusting these numbers as representative, or (b) move to Phase D (integrate into IRIS's actual training pipeline) once the GPU/OptiX environment is confirmed. Leaning toward (a) first since it's cheap and directly addresses the "single favorable data point" limitation already flagged, while (b) is blocked on the user's WSL2 fix anyway.

## What's done

1. **Architecture audit complete** — `IRIS_ARCHITECTURE_AUDIT.md`. Confirmed with file:line evidence: no environment/sky radiance anywhere (`model/emitter.py` explicitly comments "assume zero background lighting"); BRDF is purely opaque reflective (no IOR/transmission); no window/opening entity in scene loading; no solar/temporal metadata in any dataset loader; emitter extraction (`extract_emitter_ldr.py`) flags triangles as emitters purely by saturated raw-LDR-pixel statistics, which would misclassify a sun patch as a "glowing floor."
2. **Compute environment blocker found and being resolved** — see `BASELINE_REPRODUCTION.md` EXP0001. Local machine is WSL2 + RTX 5070 Ti (Blackwell). Repo's pinned `mitsuba==3.5.0`/`drjit==0.4.4` cannot initialize CUDA at all on this GPU. Newer `mitsuba==3.9.1`/`drjit==1.5.0` initializes CUDA JIT but needs OptiX, which needs manual driver-file setup under WSL2 (user is performing this now, requires a WSL restart outside my control).
3. **Literature survey + novelty/gap analysis complete** — `LITERATURE.md` (~40 methods), `NOVELTY_GAP.md`. Key finding: sun+sky decomposition (SG-sun + SH-sky) is mature for **outdoor** scenes (NeRF-OSR→SOL-NeRF→ROS-GS/GaRe), window-aperture geometry recovery from indoor point clouds is a solved classical problem, transmissive-material recovery is mature for discrete objects — but no system combines all three with indoor SVBRDF/emitter co-optimization. `TECHNICAL_PROPOSAL.md` updated to adopt the SOL-NeRF/ROS-GS sky parameterization and Mitsuba's built-in thin-dielectric BSDF rather than inventing new machinery.
4. **Dataset audit complete** — `DATASET_AUDIT.md`. Evaluated 18 dataset/families; verdict: no existing real-world dataset combines geometry + multiview + confirmed windows + confirmed sun patches + multi-time variation (structural gap across the field, not a close call). Concludes targeted capture (3 rooms x 3 time-of-day sessions) is necessary for real-world Phase E/F evaluation. **This is a real-world action item for the user, not something I can do — flagged to user, not yet scheduled/blocking.** In the meantime, `OpenRooms` and `I²-SDF` (synthetic, explicit window-emitter ground truth) are recommended for component-level validation, and IRIS's own 8 scenes remain the baseline-reproduction target.
5. **Phase A + B (intensity) synthetic validation PASSED** — `PHASE_A_SYNTHETIC.md`, EXP0002. Built `utils/solar_geometry.py` (Mode A, pvlib-backed, self-tested), `utils/window_geometry.py` (window-to-plane sun-patch projection), `utils/sun_patch.py` (IoU-based Mode B geometric search). Recovered a known synthetic sun direction (ground truth az=200/el=40 deg) to 2.5 deg angular error and sun intensity to 4.1% mean relative error, from an independent Mitsuba (`llvm_ad_rgb`, CPU) render's actual sun patch — not just a circular self-consistency check, and intensity recovery chained off the (imperfect) recovered direction, not ground truth. Residual direction error traced to an unmodeled occluder (a glossy sphere's self-shadow biting into the patch), confirmed not Monte-Carlo noise via an spp ablation. Phase C's core claim (window-aware hard-shadow patch rendering with correct geometric alignment) is also substantially covered by the same experiment. This is a real, physically-grounded pass of the project brief's Phase A/B/C gates, run entirely on CPU without waiting for the GPU/OptiX fix (D0004). Not yet done: sensitivity to material-estimation error (vs. exact GT albedo used here), multi-condition sweep (only one sun angle tested so far) — see `PHASE_A_SYNTHETIC.md` Limitations.

## What's blocked / waiting

- Baseline reproduction (README stage 1-9 pipeline run) blocked on the OptiX/WSL2 fix above. User is applying the documented fix; I'll re-verify once they confirm WSL has restarted.
- No IRIS datasets downloaded yet (box.com links in README; ~8 scenes). Not yet needed for GPU-independent work.

## What's next (in order, per project brief §24 and §4-6)

1. Once GPU CUDA+OptiX is confirmed working: minimal smoke test (load a trivial mesh, run `ray_intersect`, run one path-tracing step) before attempting any full pipeline stage.
2. Literature survey (`LITERATURE.md`) and novelty framing (`NOVELTY_GAP.md`) — GPU-independent, can proceed now.
3. Dataset audit (`DATASET_AUDIT.md`) — GPU-independent, can proceed now: survey real-world daylight-indoor datasets against the required-properties table (windows, sun patches, multiview, poses, geometry, timestamps, geo/orientation, HDR/RAW, license).
4. Download at least one IRIS-provided scene and attempt full baseline reproduction (needs GPU fix + dataset download).
5. Phase A: tiny synthetic room+window+sun scene to validate solar-geometry recovery before touching real data or the main training pipeline.

## Open scientific questions (not yet resolved by evidence)

- Does explicit sun/sky modeling actually improve *relighting* metrics, or only training-view PSNR (which would be a false positive per project brief §16)?
- Can sun direction be recovered from window+sun-patch geometry alone (Mode B) with useful accuracy, or is metadata (Mode A) required?
- Is any candidate real dataset actually adequate, or will targeted capture be necessary (see `DATASET_AUDIT.md` once written)?

## Links

- [[architecture-audit]] `IRIS_ARCHITECTURE_AUDIT.md`
- [[baseline-reproduction]] `BASELINE_REPRODUCTION.md`
- [[decisions-log]] `DECISIONS.md`
- [[experiments-registry]] `EXPERIMENTS.md`
- [[failed-ideas]] `FAILED_IDEAS.md`
