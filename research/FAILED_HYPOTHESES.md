# Hypothesis ledger (H1–H5) — what is supported, what failed, what is untested

Companion to `FAILED_IDEAS.md`, which records individual ideas and methods.
This file tracks the five hypotheses of the 2026-09-27 research brief.
Statuses change only with evidence; the history of each status is kept.

Status values:
- `supported (scope)`
- `not supported (scope)`
- `failed (scope)`
- `untested`
- `pending`

| # | Hypothesis | Status | Evidence | Scope / caveats |
|---|---|---|---|---|
| H1 | An explicit solar state reduces material–illumination ambiguity compared with generic lighting of similar or greater capacity | **supported (synthetic, exact geometry); reversed under ±3 cm window error (EXP0036, 1 seed)** | T11 single-time floor correlation: W 0.17, A 0.28, A_hi 0.51 vs B 0.62, C 0.53, C_oracle 0.66 (3 seeds each, tight). Multi-time: W 0.29, A 0.49, A_hi 0.72 vs B 0.89, C 0.87, C_oracle 0.93. EXP0036 (window ±3 cm): B 0.39, C 0.35 vs A_hi 0.63 | One procedural room, Lambertian. The explicit-sun arms' sky was chosen with oracle knowledge (D0020). Mechanism found: the free sky, not the sun, carries the ambiguity |
| H2 | Known solar direction improves recovery of the time-invariant BRDF | **not supported vs image-estimated direction** (R1 failed: 6% MAE, B better corr); holds only vs no sun at all | EXP0035: at a 1.6° direction error, floor corr falls from 0.93 to 0.36 | "Known" must mean sub-degree. A one-time patch calibration (0.37°) is less accurate than per-time image estimates (0.2–0.26°) in clean images |
| H3 | Better generalisation to unseen solar conditions than generic lighting of similar or greater capacity | **supported (synthetic, R2: +12.7 dB at 14:30); real R2 did not replicate (EXP0037)** | Single-time, 14:30 patch IoU: C 0.63, C_oracle 0.83; every non-ephemeris arm 0.03 | The single-time comparison is partly by construction: arms without an ephemeris cannot move the sun at all |
| H4 | Multiple solar conditions provide structured supervision | **supported (synthetic, all arms; R3)** | MAE_s single → multi: C 0.130 → 0.034, B 0.104 → 0.036, A_hi 0.111 → 0.056 | Known outdoor negative result: within one clear day the sun path is near-planar (Hold-Geoffroy 2015) |
| H5 | The benefit holds on real scenes | **not supported for patch prediction (EXP0037: replication failed); material part untested** | EXP0037: 07-06 C 0.38 vs B 0.00 IoU; 06-25 C 0.18 vs B 0.20 (pre-registered replication failed) | No real multi-view, multi-time data locally. The material part of H5 is **untested** |

## History

- 2026-09-27: file created; all statuses pending or interim.
- 2026-09-27 evening: updated after T11 v1, EXP0035, EXP0036 (seed 0) and EXP0037.
