# Identifiability Ablation: Does the Solar Constraint Improve Material Recovery, or Only Pixel Fit?

Status: **PASSED, first pass.** EXP0004. Code: `experiments/identifiability_ablation.py`. This is the direct experimental test of "Contribution One" from the novelty discussion (`NOVELTY_GAP.md`): that an external, image-independent solar constraint breaks a *structural* material-illumination ambiguity, which generic capacity (a bigger/smoother free-form illumination field) cannot break regardless of how well it fits pixels.

## Why this experiment, and why it's deliberately not a full IRIS run

For a Lambertian point observed under **one static illumination condition**, `L(x) = rho(x) * E(x)`. Multi-view consistency only forces `L` to be view-independent — true for *any* `(rho, E)` split, so it adds no information about the split itself. Any method regularizing both sides with generic, image-derived priors (spatial smoothness, sparsity) has no way to prefer "uniform material, sharp illumination edge" over "material edge, smooth illumination" except via how smooth each field is assumed to be — and a hard sun-patch boundary is not smooth, so a smoothness-regularized free-form illumination field will underfit it and push the sharp transition into the (also free) albedo field instead. This is the generalized mechanism behind the concrete bug `IRIS_ARCHITECTURE_AUDIT.md` §4 already found (`extract_emitter_ldr.py`'s saturation threshold flagging a sun patch as a "glowing floor").

Running IRIS's actual `NGPBRDF`/`VoxelSLF` on this would be the gold-standard version of this test, but requires real (or at least full-pipeline synthetic) data and the complete training pipeline, not yet available (`BASELINE_REPRODUCTION.md`). This experiment instead builds the **minimal controlled version of the same mechanism** — a 2D radiometric optimization, no rendering, no neural network — specifically so the result is interpretable and the failure mode is unambiguous. It is explicitly a mechanism demonstration, not the final paper-grade experiment; see Limitations.

## Setup

An 80x80 grid representing a floor. Ground truth: a rectangular "sun patch" region gets extra irradiance (`E_sun_true=2.5`) on top of ambient (`E_ambient_true=0.3`); everywhere else gets ambient only. Observed radiance `R = rho_true * E_true + noise`. Two sub-experiments:
- **Test 1 (uniform albedo, `rho_true=0.5` everywhere)**: isolates whether a method *hallucinates* a spurious albedo edge at the patch boundary when none exists.
- **Test 2 (a genuine material stripe elsewhere, uncorrelated with the patch)**: checks a method isn't simply suppressing all high-frequency albedo variation (which would trivially "win" test 1 by cheating) — it must still recover the real edge.

Two recovery methods, both jointly optimizing per-pixel `rho(x,y)` (free, total-variation regularized, **identical regularization for both methods**) and `E(x,y)`, differing only in how `E` is parameterized:
- **Baseline**: `E(x,y)` fully free per-pixel, total-variation regularized (a stand-in for what a smoothness/sparsity-regularized neural field, like `NGPBRDF`/`VoxelSLF`, effectively imposes with no external grounding).
- **Ours**: `E(x,y) = E_ambient + E_sun * mask(x,y; cx,cy,hw,hh)`, a 6-parameter physical form (ambient level, sun intensity, rectangle center/half-extents), with the rectangle **initialized offset from the true patch location** (`PATCH_PRIOR_OFFSET`, not an oracle) to simulate an imperfect Mode A/B geometric prior rather than assuming perfect knowledge.

**Fairness controls, both important:**
- Both methods get the **same global radiometric scale anchor** (`E`'s spatial mean pinned toward the true mean) — without this, `rho*E` has a trivial global scale ambiguity (`(rho/k, E*k)` fits identically for any `k`) that swamped the first run's results with a boring, well-known degeneracy unrelated to the *spatial* question this experiment targets. This anchor gives no spatial information (doesn't say where the patch is), representing the kind of radiometric calibration a real system gets from CRF/exposure estimation, which both methods get equally.
- Both methods get the **same TV regularization strength on `rho`** — the only difference between methods is `E`'s parameterization, not extra supervision or different rho priors.

## Results

| | Test 1 (uniform albedo) | Test 2 (real material edge) |
|---|---|---|
| Albedo RMSE, baseline | 0.226 | 0.220 |
| Albedo RMSE, ours | **0.098** | **0.124** |
| Pixel-fit RMSE, baseline | 0.0102 | 0.0104 |
| Pixel-fit RMSE, ours | 0.0106 | 0.0108 |
| Recovered patch center (true: 50,45) | (52.3, 45.0) | (52.0, 45.0) |
| Recovered `E_ambient` (true 0.3) | 0.342 | 0.370 |
| Recovered `E_sun` (true 2.5) | 1.83 | 1.58 |

Raw output: `experiments/out/identifiability/results.json`. Visualizations: `experiments/out/identifiability/test{1,2}_*.png` (panels: true albedo | baseline albedo | ours albedo | observation).

**The visualizations make the mechanism directly visible**: in both tests, the baseline's recovered albedo shows a clear, bright, rectangular artifact exactly at the sun-patch location — a hallucinated "high albedo" region that isn't there in the ground truth. Ours shows a much fainter version of the same artifact (consistent with the RMSE gap, not a perfect fix — see Limitations). In test 2, the genuine material stripe is visible in **both** methods' recovered albedo, confirming ours is not simply suppressing all spatial variation (which would be a cheap, uninteresting way to "win" test 1).

## Interpretation

**The central claim holds in this controlled setting**: albedo RMSE improves substantially with the solar constraint (2.3x in test 1, 1.8x in test 2) while pixel-fit RMSE is *not* better for "ours" — if anything marginally worse (0.0106 vs 0.0102, 0.0108 vs 0.0104). This is the cleanest possible version of the argument from `NOVELTY_GAP.md`'s reviewer-2 test: **the improvement is not coming from fitting the image better — it's coming from resolving which part of the fit is material and which part is illumination.** A method judged only on reconstruction PSNR would not distinguish baseline from ours here; a method judged on whether it recovered the *correct decomposition* clearly would.

The recovered patch position is close to the true one (a few grid units off, better than the 4-6 unit initial offset in most parameters) — the optimization pulls the initially-offset geometric prior toward the location the photometric data actually supports, which is itself a reasonable stand-in for what Phase D's joint optimization should do with a real Mode A/B prior. Recovered `E_sun` is noticeably off (1.58-1.83 vs true 2.5) even though `E_ambient` is close (0.34-0.37 vs true 0.3) and the patch position/size is close — most likely because the ground-truth patch edge was generated near-hard (`sharpness=50`) while the fitted mask uses a softer edge (`sharpness=2.0`) for optimization stability, creating an amplitude/edge-softness trade-off (a softer, slightly larger patch at lower peak intensity can integrate to a similar total energy as a sharper, smaller, more intense one). This is a real, specific, checkable limitation of this minimal experiment's parameterization, not swept under the rug — worth revisiting (e.g. an annealed sharpness schedule) if intensity recovery accuracy from this style of joint optimization becomes important later.

## Limitations (being explicit)

- **This is not IRIS.** It's a minimal, controlled stand-in for the mechanism, chosen deliberately so the result is interpretable without the confounds of a full neural field, real path tracing, or real (noisy, imperfectly-reconstructed) geometry. The equivalent experiment should eventually be run with IRIS's actual `NGPBRDF` once real data + the full pipeline are available (Phase D) — this result is evidence the mechanism is real and worth building Phase D around, not a substitute for testing it on the real system.
- **Single noise level, single patch configuration, single random seed.** Not yet swept.
- **The TV-regularization baseline is a reasonable but specific choice** of "generic prior" — it is not literally IRIS's `VoxelSLF` (an unconstrained per-voxel cache with no explicit smoothness term at all, arguably an even *weaker* prior than TV-regularized `E`, meaning IRIS's real vulnerability to this ambiguity could plausibly be *worse* than what the baseline here shows, not better) or `NGPBRDF` (a hash-grid MLP whose effective smoothness comes from its architecture, not an explicit term). Chosen as an interpretable, defensible proxy, not because it's known to be the closest possible match to IRIS's own inductive bias.
- **The remaining albedo error for "ours" (0.098, 0.124) is not zero** — the constraint helps substantially but does not perfectly resolve the ambiguity in this noisy, imperfect-prior setting. This is the honest, expected outcome, not a failure to report around.

## Status and next step

This passes as a first-pass mechanism demonstration and directly supports proceeding with the solar-constrained approach. Recorded in `EXPERIMENTS.md` as EXP0004. Next: (a) a small parameter sweep (noise level, prior offset magnitude, patch size) to check robustness rather than one lucky configuration; (b) eventually, the same comparison using IRIS's actual `NGPBRDF` once Phase D is reachable — this remains the evidence that would actually matter for the paper, not this proxy.
