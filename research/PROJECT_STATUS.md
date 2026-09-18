# Project Status

Last updated: 2026-09-18 (session 2).

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

1. Verify IRIS's own `utils/path_tracing.py` etc. against drjit 1.5.0 (API compatibility check — Mitsuba-side calls already spot-checked OK; `.torch()` interop and the full training pipeline not yet tried).
2. Broaden the Phase A/B synthetic sweep (multiple sun angles, material-estimation-error sensitivity) — cheap, addresses the "single favorable data point" limitation, doesn't depend on the user.
3. Real-world Phase 0 capture (user-driven, whenever feasible) — the next real go/no-go gate per the reframed research question.
4. Once both above land: Phase D (joint optimization integrated into IRIS's actual training stages, GPU now available).

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
