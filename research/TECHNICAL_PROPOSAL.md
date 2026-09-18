# Technical Proposal: Daylight-Aware IRIS (v1, pre-Phase-A)

Status: draft, written after architecture audit (`IRIS_ARCHITECTURE_AUDIT.md`) and before any code changes. Will be revised once literature/dataset surveys land and Phase A synthetic results come in — see `DECISIONS.md` for the revision log going forward.

## Problem, restated precisely

IRIS's renderer (`utils/path_tracing.py`, `model/emitter.py`) returns exactly zero radiance for any camera or bounce ray that fails to intersect the reconstructed mesh. All illumination is represented by two geometry-local terms: per-triangle constant emitter radiance (`AreaEmitter`) and a voxel-cached surface light field (`VoxelSLF`). Under daylight, this forces two failure modes, both confirmed in code (audit §4):
1. Saturated sun patches get flagged as emitter triangles by `extract_emitter_ldr.py`'s `is_emitter = mean_LDR_max_channel > 0.99` criterion — the floor is fit as literally glowing.
2. Sub-saturation daylight gets absorbed into `albedo`/`roughness`/`metallic` (via `NGPBRDF`) and/or the SLF radiance cache — a raw baked radiance value with no reflectance/illumination decomposition.

Both are camera-view-averaged, direction-agnostic, and fixed at training time. Neither supports physically meaningful relighting under a changed sun position, because neither ever represented "external light modulated by geometry" — only "this triangle/voxel emits/reflects this fixed RGB."

## Proposed minimal decomposition

```
L_observed(x, wo) = L_indoor_emitters(x, wo)          [existing: AreaEmitter/SLFEmitter, unchanged]
                   + T_window(x, wo) * L_outdoor(wi)    [new]
                   + L_indirect(x, wo)                  [existing path tracer, extended to query L_outdoor on ray miss]
    then: BSDF(x, wi, wo) applied per existing NGPBRDF,
    then: CRF(., exposure) applied per existing EmorCRF   [existing, unchanged]
```

`L_outdoor(wi) = E_sun * delta(wi, s) + L_sky(wi)`: a directional sun term (direction `s`, RGB intensity `E_sun`) plus a low-dimensional sky term `L_sky`. **Updated per `LITERATURE.md` §3**: use SOL-NeRF's/ROS-GS's validated minimal parameterization — a single spherical-Gaussian lobe for the sun (already what the delta/near-delta term above is, with a tunable concentration parameter for soft shadows) plus a low-order spherical-harmonics term for `L_sky` (not a full environment map). This is not a novel choice on my part — it is the field's already-converged-on minimal-complexity design for exactly this decomposition (outdoor daylight only, to date), reused here rather than reinvented, per project brief §6's instruction not to overcomplicate the first prototype. IRIS's own SLF is architecturally a plausible attachment point for a cached "sky visibility through aperture" field later, per `LITERATURE.md`'s transfer-opportunity notes, but is not needed for the minimal Phase A/B prototype.

`T_window(x, wo)`: transmittance of the window aperture nearest along the ray from `x` toward the sky. Start as a scalar/RGB constant per detected window surface (project brief §8's suggested first approximation), not a full dielectric BSDF — defer IOR/Fresnel/roughness to a later phase, contingent on Phase A/B evidence that the scalar model is insufficient.

## Exactly where this integrates into existing code (not hypothetical — audited line numbers)

- **Ray-miss hook**: `utils/path_tracing.py::ray_intersect` (L17-48) already computes `valid = ~ts.isinf()`. Every call site that currently does `idx[~valid] = -1` and then feeds `triangle_idx` into `emitter_net.eval_emitter(...)` is a candidate integration point. The cleanest change is *not* inside `ray_intersect` itself, but in `AreaEmitter.eval_emitter` / `SLFEmitter.eval_emitter` (`model/emitter.py` L69-98, L180-221): where they currently do `Le = Le*vis[...,None]` (hardcoded zero for `~vis`), add `Le[~vis] += L_outdoor(wi[~vis])` before returning. This is additive and touches exactly the "assume zero background lighting" lines the audit flagged — nothing else in the path tracer needs to change, since `path_tracing()`/`trace_indirect()` already thread `eval_emitter`'s return value through MIS weighting generically (they don't assume `Le=0` anywhere themselves, they just happen to always receive it).
- **New module**: `model/external_lighting.py` (new) — `SunLight`, `SkyLight` as thin `nn.Module`s producing `L_outdoor(wi)` given a batch of ray directions; a `DaylightModel` wrapper combining them plus a learnable/prior-constrained sun direction parameter (see below). This is additive, does not modify `model/brdf.py`/`model/slf.py`.
- **Window transmittance**: needs a per-position-or-per-triangle transmittance lookup, closely mirroring how `AreaEmitter`/`SLFEmitter` already look up per-triangle constant values via `is_emitter`/`triangle_idx` (`model/emitter.py` L38-45's `emitter_idx` mapping is a direct template). Proposed: `model/transmission.py` (new) — a `WindowSurface` registry keyed by triangle index (same pattern as the existing emitter mask), holding a learnable/prior scalar transmittance per flagged window triangle. This deliberately reuses IRIS's existing "flag a subset of triangles, attach a learned per-triangle parameter" pattern rather than inventing a new geometric primitive — lower risk, and directly testable against the existing emitter-extraction machinery for comparison.
- **Solar geometry**: `utils/solar_geometry.py` (new), pure-function module, no torch dependency required for Mode A (metadata-derived), torch-differentiable-friendly for Mode B (patch-based search/optimization). Validated against a reference solar-position library (e.g. `pvlib` or NOAA's algorithm) rather than hand-derived trig, per project brief §7's explicit caution.
- **Window/sun-patch geometry utilities**: `utils/window_geometry.py`, `utils/sun_patch.py` (new) — window aperture extraction/projection and candidate sun-patch detection, feeding Mode B search. Not needed until Phase B/C; will only be built once Phase A validates the representation on synthetic ground truth.
- **Dataset metadata**: `utils/dataset/daylight_real.py` (new adapter) or, more conservatively, additive optional fields on the existing loaders (`utils/dataset/real_ldr.py`, `utils/dataset/scannetpp/dataset.py`) for EXIF timestamp / geo / orientation, since audit §7 confirms none of the current loaders read any such field and the file formats they already parse (`transforms.json`-style) could plausibly be extended rather than replaced.

## Why additive, not a rewrite

Every proposed change either (a) fills in a currently-hardcoded zero (`eval_emitter`'s missed-ray branch) or (b) reuses an existing per-triangle-flag-plus-learned-parameter pattern that `AreaEmitter`/`is_emitter` already establishes. No existing file's *opaque-reflective* material path (`NGPBRDF`, `FIPTBSDF`'s `DiffuseReflection`-only flags) needs to change for the first prototype, because the window-transmittance term is modeled as a separate multiplicative gate on the *outdoor* radiance reaching a point, not as a new BSDF lobe on the room's own material network — deferring the harder "transmission as a BSDF lobe" work (project brief §9) until there's evidence the simpler gate is insufficient.

## Ambiguity controls (project brief §13), concretely

- **Solar-position prior**: Mode A (metadata) gives a hard/soft prior on `s`; even without metadata, `s` is a 2-parameter (azimuth, elevation) quantity, not a free 3-vector, and can be shared across all frames of one capture session (physically, the sun barely moves during a single indoor capture) — a strong low-dimensionality prior vs. IRIS's existing per-triangle/per-voxel emitter parameters.
- **Window transmission bounds**: constrain `T_window \in [0,1]^3` via sigmoid parameterization (same pattern `NGPBRDF` already uses for albedo/roughness/metallic, `model/brdf.py` L255-259).
- **Emitter/daylight disambiguation prior**: modify (or add an alternative to) `extract_emitter_ldr.py`'s saturation threshold so that triangles behind/near a detected window are *not* auto-flagged as emitters, directly countering the misclassification mechanism the audit found in §4(a). This is a deliberately small, targeted patch to an identified concrete bug-like behavior, not a redesign of emitter extraction.
- **Sun directional sharpness**: keep the sun as a true delta/near-delta direction (very low roughness kernel when soft-shadow blur is needed for differentiability) rather than a broad lobe, to preserve the sharp-shadow evidence that makes Mode B (geometric sun-patch matching) informative in the first place.

## Phases (mirrors project brief §12, restated against this codebase)

- **Phase A**: synthetic room+window+sun+sky+one-glossy-object scene with known ground truth (sun direction, window geometry, materials). Test: can `SunLight`+`SkyLight`+`WindowSurface` recover the known sun direction and window transmittance when material/geometry are held fixed? **Gate: do not proceed to Phase B on real data until this works.**
- **Phase B**: sun direction/intensity inference on Phase A's scene with geometry/material approximately fixed; angular-error metric.
- **Phase C**: render hard/soft sun patches through the window; check geometric alignment against synthetic ground truth mask.
- **Phase D**: joint optimization — integrate into IRIS's actual training stages (likely inserted between existing stages 3 and 6, i.e. alongside `initialize.py`/`train_brdf_crf.py`, since those are where BRDF and emitter radiance are currently jointly fit) with the ambiguity controls above active; check no single term (BRDF, indoor emitter, sun, sky, window) silently absorbs all residual.
- **Phase E**: real scene(s) — pending `DATASET_AUDIT.md` verdict on whether an existing dataset suffices or targeted capture is needed.
- **Phase F**: relighting demonstration (sun removed / moved / dimmed; different time-of-day approximation) as the central falsifiable test (project brief §10, §16) — if changing the sun parameter doesn't produce a physically plausible change in rendered sun-patch position/shape, the decomposition has failed regardless of reconstruction PSNR.

## What would falsify this approach (commit to checking, per project brief §16)

- If Phase A cannot recover a known synthetic sun direction to reasonable angular accuracy, do not proceed — this is a hard gate, not a soft one.
- If ablating "sun+sky" back down to "bigger SLF capacity only" achieves equivalent training-view PSNR *and* equivalent relighting quality, the explicit solar/window structure is not earning its complexity — report this as a negative result in `FAILED_IDEAS.md`, not a footnote.
- If the window-transmittance gate, once fit, is not physically plausible (e.g. converges to `T≈1` uniformly, indicating it's not distinguishing glazing from open air, or to values uncorrelated with any independently-verifiable window location), that specific mechanism should be rejected even if overall image metrics improve, since brief §10/§16 explicitly prioritize physically meaningful decomposition over metric gaming.

## Open items pending survey results

- ~~Final choice of sky parameterization~~ **Resolved**: SG-sun + low-order-SH-sky, per `LITERATURE.md` §3 (SOL-NeRF/ROS-GS precedent).
- Window transmittance material: `LITERATURE.md` §6 confirms Mitsuba (IRIS's own renderer) already ships an analytic thin-dielectric BSDF suited to a planar glass pane — cheaper to reuse than building a custom transmission model. Revise `model/transmission.py`'s plan above from "custom scalar gate" to "thin-dielectric Mitsuba BSDF wrapped the same way `FIPTBSDF` wraps `NGPBRDF`" if Phase C evidence shows the scalar gate is insufficient; start with the scalar gate regardless (still the right Phase-A/B minimal test) since it's simpler to fit and debug first.
- Window geometry: `LITERATURE.md` §5 confirms indoor point-cloud opening-detection (ISPRS energy-function method) is a solved classical preprocessing step directly applicable to IRIS's existing reconstructed mesh input — use this rather than inventing new window-detection machinery, deferring semantic/learned detection unless the classical method proves insufficient on real scenes.
- Mode B (geometry-only sun estimation): `LITERATURE.md` §4 confirms SUNDIAL's multi-view shadow-ray-casting achieves high-accuracy sun-direction recovery in a related but distinct domain (satellite/open-ground shadows) — the window-aperture-as-pinhole cue proposed in the project brief is a plausible adaptation of this, not something directly reusable off the shelf; treat as needing its own validation in Phase B, not assumed to work by analogy alone.
- Whether any candidate dataset supports held-out-illumination evaluation (Phase E/F's most important metric per brief §15) — pending `DATASET_AUDIT.md` verdict; if no suitable real dataset exists, Phase E/F evidence will initially be synthetic-only plus qualitative real-scene relighting, with held-out-illumination real evaluation deferred to a targeted-capture dataset (see brief §5's capture protocol, to be designed in `DATASET_AUDIT.md` if warranted).
