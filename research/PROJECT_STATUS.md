# Project Status

Last updated: 2026-09-18 (session 1).

## Goal

Investigate whether explicit modeling of solar geometry, window geometry, and light transmission can improve inverse rendering and relighting of real indoor scenes under daylight, addressing IRIS's lack of any outdoor/environment illumination term. See `IRIS_ARCHITECTURE_AUDIT.md` for the confirmed gap and `../` project brief (conversation) for full scope.

## Branch

`research/daylight-aware-iris`, branched from `main` @ `d2d4381` (clean tree at branch time). Baseline behavior on `main` is untouched.

## Current phase

**Phase 0: audit + environment setup.** Not yet started: Phase A (synthetic sun-through-window validation) per the project brief's development order.

## What's done

1. **Architecture audit complete** — `IRIS_ARCHITECTURE_AUDIT.md`. Confirmed with file:line evidence: no environment/sky radiance anywhere (`model/emitter.py` explicitly comments "assume zero background lighting"); BRDF is purely opaque reflective (no IOR/transmission); no window/opening entity in scene loading; no solar/temporal metadata in any dataset loader; emitter extraction (`extract_emitter_ldr.py`) flags triangles as emitters purely by saturated raw-LDR-pixel statistics, which would misclassify a sun patch as a "glowing floor."
2. **Compute environment blocker found and being resolved** — see `BASELINE_REPRODUCTION.md` EXP0001. Local machine is WSL2 + RTX 5070 Ti (Blackwell). Repo's pinned `mitsuba==3.5.0`/`drjit==0.4.4` cannot initialize CUDA at all on this GPU. Newer `mitsuba==3.9.1`/`drjit==1.5.0` initializes CUDA JIT but needs OptiX, which needs manual driver-file setup under WSL2 (user is performing this now, requires a WSL restart outside my control).
3. **Literature survey + novelty/gap analysis complete** — `LITERATURE.md` (~40 methods), `NOVELTY_GAP.md`. Key finding: sun+sky decomposition (SG-sun + SH-sky) is mature for **outdoor** scenes (NeRF-OSR→SOL-NeRF→ROS-GS/GaRe), window-aperture geometry recovery from indoor point clouds is a solved classical problem, transmissive-material recovery is mature for discrete objects — but no system combines all three with indoor SVBRDF/emitter co-optimization. `TECHNICAL_PROPOSAL.md` updated to adopt the SOL-NeRF/ROS-GS sky parameterization and Mitsuba's built-in thin-dielectric BSDF rather than inventing new machinery.
4. **Dataset audit in progress** (background agent running) — `DATASET_AUDIT.md`, not yet complete as of this update.

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
