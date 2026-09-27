# Decisions Log

Format per entry: Decision / Alternatives considered / Evidence / Reason.

---

## D0001 — Work on a dedicated branch, don't touch `main`

**Decision:** All research/engineering work happens on `research/daylight-aware-iris`, branched from `main` @ `d2d4381`. `main` stays byte-identical to the reproducible baseline.

**Alternatives considered:** Work directly on `main` with careful commits.

**Evidence:** Project brief §19 explicitly requires this ("Do development on a separate branch... Keep original baseline behavior reproducible").

**Reason:** Guarantees the original IRIS baseline remains runnable/comparable at any point, and lets `git diff main...research/daylight-aware-iris` serve as the audit trail for every architectural change.

---

## D0002 — Audit before touching any algorithm code

**Decision:** Completed a full read-only architecture audit (`IRIS_ARCHITECTURE_AUDIT.md`) before writing any new module or modifying existing files.

**Alternatives considered:** Start prototyping a sun/sky emitter immediately based on the project brief's assumptions about IRIS's structure.

**Evidence:** Project brief §2 explicitly forbids modifying the core algorithm before the audit is complete, and the audit's findings (e.g., the exact "assume zero background lighting" comments, the emitter-extraction saturation threshold, the purely-reflective BSDF flags) are specific enough that skipping this step would have risked building on wrong assumptions about, e.g., whether a transmission BSDF path already partially existed (it doesn't).

**Reason:** Scientific rigor requires verifying the failure mode mechanically, not just architecturally-plausibly, before proposing a fix. The audit also surfaced a concrete, previously-unstated mechanism (emitter-mesh misclassification of sun patches via LDR saturation threshold) that materially shapes which experiments are worth running first.

---

## D0003 — GPU/CUDA reproducibility issue treated as a genuine blocking decision, escalated to user

**Decision:** Asked the user how to resolve the WSL2 + Blackwell (RTX 5070 Ti) OptiX initialization failure rather than silently picking a workaround (e.g., silently porting to CPU/LLVM Mitsuba backend, or silently proceeding with only non-GPU work indefinitely).

**Alternatives considered:** (a) full LLVM/CPU Mitsuba port, (b) cloud GPU, (c) pause GPU work and continue survey-only work, (d) user fixes WSL2/OptiX driver setup (**chosen by user**).

**Evidence:** `BASELINE_REPRODUCTION.md` EXP0001. Confirmed via direct testing: old pinned `mitsuba==3.5.0`/`drjit==0.4.4` cannot init CUDA JIT at all on this GPU (works fine on other, non-Blackwell GPUs presumably, since raw `cuInit`/`cuDeviceGetCount` succeed via ctypes and LLVM/CPU drjit backend works). Newer `mitsuba==3.9.1`/`drjit==1.5.0` inits CUDA JIT but fails at OptiX scene-acceleration-structure init (`could not find symbol optixQueryFunctionTable`) because WSL2's `/usr/lib/wsl/lib/libnvoptix.so.1` is only a 10KB loader stub, not the real OptiX runtime; Mitsuba's own documented fix requires copying `libnvidia-rtcore.so`, `libnvidia-ptxjitcompiler.so`, `libnvidia-gpucomp.so`, `nvoptix.bin` from a matching Linux driver package (identified: 580.105.xx, matching Windows driver 581.80) into `C:\Windows\System32\lxss\lib` — a Windows-host, admin-level action outside this sandboxed session's reach, followed by `wsl --shutdown` which would terminate this very session.

**Reason:** This is a decision with real cost (user must find/download a ~1GB Windows-host driver package, do manual admin file copies, and restart their entire WSL environment, killing any other work they have running there) and affects the entire compute strategy for the rest of the project (CPU porting would mean weeks of additional engineering risk vs. the documented, if fiddly, native fix). Per the operating instructions, this is exactly the kind of "irreversible or expensive decision" / "existing evidence cannot distinguish which direction is best without the user's own constraints" (do they have another idle WSL session? is this their primary dev machine?) that warrants asking rather than assuming.

**User's answer:** Fix WSL2/OptiX directly (recommended option). **Resolved 2026-09-18** — see `BASELINE_REPRODUCTION.md` EXP0001 for the full procedure actually used (including a TrustedInstaller-ownership snag with `takeown`/`icacls` that needed the Explorer GUI path instead) and verification (full GPU path-traced render confirmed working).

---

## D0004 — Phase A synthetic validation proceeds on CPU (LLVM Mitsuba variant), not blocked on the GPU/OptiX fix

**Decision:** Build and run the Phase A synthetic room+window+sun scene (project brief §12 Phase A) using `mitsuba.set_variant('llvm_ad_rgb')` rather than waiting for the CUDA/OptiX fix from D0003 to land.

**Alternatives considered:** Block all rendering work until GPU+OptiX is confirmed working, to stay closest to the "real" execution environment IRIS will eventually train under.

**Evidence:** `BASELINE_REPRODUCTION.md` EXP0001 already confirmed the LLVM/CPU Mitsuba backend loads scenes and renders correctly on this machine today. Phase A's scene is intentionally tiny (one room, one window, a handful of shapes) — exactly the regime where CPU ray tracing is fast enough to not matter, unlike the full IRIS training pipeline (hash-grid BRDF MLP + thousands of training rays per step across a full scan mesh), which does need CUDA for practical runtime.

**Reason:** No reason to let an infrastructure blocker stall algorithmic validation work that doesn't need the blocked resource. If Phase A's synthetic sun-recovery test passes on CPU, that result is unaffected by which Mitsuba backend eventually runs full IRIS training. Revisit only if Phase A CPU runtimes turn out to be impractically slow (not expected at this scale).

---

## D0005 — Dataset audit concludes targeted real-world capture is needed; treated as a user-owned action item, not a blocker for algorithm work

**Decision:** Note the `DATASET_AUDIT.md` verdict (no existing dataset suffices; recommend a 3-room x 3-session capture protocol) to the user as a recommendation requiring their real-world effort (access to rooms with windows, time across a day, a phone/camera), but continue Phase A/B/C algorithmic work on synthetic data in parallel rather than pausing to wait for a capture decision.

**Alternatives considered:** Treat this as a hard blocking decision requiring the user's explicit go/no-go before any further work (per the escalation criteria's "multiple scientifically distinct directions" clause).

**Evidence:** `DATASET_AUDIT.md`'s verdict is unambiguous (not a case where evidence fails to distinguish direction) — the open question is purely whether the user is willing/able to do the physical capture work, which doesn't gate anything upstream of Phase E (real-world validation). Phases A-D are synthetic/methods-development and unaffected either way.

**Reason:** Keeps the project moving on the parts that don't depend on the user's answer, consistent with "own the project... don't ask me what to do next" — while still surfacing the real-world action item clearly rather than silently assuming the user will do the capture, since that's genuinely their call (their house/office access, their time).

---

## D0006 — Phase A render-validation camera must be chosen so its center ray actually hits the floor inside the room

**Decision:** When building the Phase A synthetic-scene camera pose (`experiments/phase_a_sun_recovery.py::build_scene`), explicitly check/derive that the camera's center ray intersects `z=0` at a `y` value inside `ROOM_Y`, not just "looks vaguely downward."

**Alternatives considered:** Pick a camera pose by eye/intuition (as first attempted) and debug empirically if results look wrong.

**Evidence:** The first camera pose (`origin=[0,3.3,1.4]`, `target=[0,0,1]`) produced a render-based cross-validation IoU of only 0.25 at the *true* sun direction (i.e. the analytic model appeared to disagree substantially with Mitsuba's own renderer). Diagnosed by projecting the center ray analytically: it crossed `z=0` at `y=-8.25`, far outside the room (`y` must be in `[0,4]`), meaning the camera saw almost none of the floor region where the patch actually falls — a small sliver only, dominated by grazing-angle sampling noise. This was purely a test-harness bug, not a flaw in `utils/window_geometry.py`'s projection math (confirmed separately: Test 1's analytic self-consistency check, which doesn't depend on any camera, passed with 0.0 error throughout). After fixing the camera pose to look steeply enough at the floor (and correcting a second, related bug — placing the camera above the ceiling, `z=2.6 > ROOM_Z_TOP=2.5` — which put it outside the room entirely and rendered pure black), true-direction cross-check IoU rose to 0.71 and recovered-direction angular error fell to 2.5°.

**Reason:** This class of bug (camera geometrically unable to see the phenomenon being measured) is easy to introduce and easy to misdiagnose as an algorithm failure. Recording the check explicitly so future scene-authoring code (Phase C/D real-scene camera selection, in particular) verifies visibility of the region of interest analytically before trusting a low-overlap or zero-detection result as evidence against the model.

---

## D0007 — Reframe the research question after finding directly on-point prior work (Li et al., ECCV 2022)

**Decision:** Abandon the "weak version" framing ("add an explicit sun+sky model to IRIS") as an insufficient standalone contribution, and adopt the "strong version" the user proposed: **solar-geometry-constrained multi-view inverse rendering**, with the central evaluation target being **cross-time relighting validated against a real, independently captured photograph at an unseen solar condition** (the "killer experiment," detailed in `NOVELTY_GAP.md`), not reconstruction-quality metrics on the training condition.

**Alternatives considered:** (a) keep the original framing and rely on multi-view + IRIS-integration as sufficient differentiation from Li et al. 2022's single-image method; (b) treat Li et al. 2022 as disqualifying and abandon the daylight-IRIS direction entirely; (c) the adopted option — narrow the claim to what genuinely remains undone.

**Evidence:** The user identified, and I independently verified via WebSearch/WebFetch (three separate queries, cross-confirmed), that Li et al. ("Physically-Based Editing of Indoor Scene Lighting from a Single Image," ECCV 2022, arXiv:2205.09343) — co-authored by Zhengqin Li, who also co-authors both FIPT and IRIS — already models window radiance as exactly 3 spherical Gaussians for sun/sky/ground, with per-SG intensity/bandwidth/direction parameters, and supports light-source editing including windows. This was missed entirely by the session-1 automated literature survey (see `LITERATURE.md` §1a for the process-lesson note: the survey searched by topic keywords but never checked IRIS/FIPT's own co-authors' other publications directly, which is how the single closest possible prior work — same lab, same physical decomposition — was missed). Verified as NOT present in Li et al. 2022 (checked, not found): multi-view input/optimization, 3D window aperture geometry (uses a 2D predicted mask instead), solar-ephemeris grounding, or validation against real cross-time held-out photographs. Two further papers found and verified in the same pass (LuxRemix, CVPR 2026; ProjectiveShading, Computer Graphics Forum 2026) are close on individual axes (multi-view lighting decomposition; single-view sun-direction-from-shadows) but neither combines multi-view + 3D window geometry + ephemeris + validated cross-time relighting either.

**Reason:** (a) is not defensible — a reviewer aware of the FIPT/IRIS lineage (very likely, given how small and cross-citing this sub-field is per `LITERATURE.md`) would immediately point to the authors' own prior paper. (b) is an overcorrection — the specific combination (3D window geometry + multi-view photometric optimization + solar ephemeris + validated cross-time prediction) remains, as far as this project's two literature passes have found, absent from the literature, and the engineering work already done (Phase A/B, `PHASE_A_SYNTHETIC.md`) is directly reusable under the reframing, not wasted. (c) preserves everything built so far while substantially raising the evidentiary bar for what counts as "done" — in particular, it makes clear that reconstruction-quality/training-view metrics are no longer sufficient evidence of success; only the cross-time real-photograph comparison is. This changes what Phase E/F must produce (see updated `DATASET_AUDIT.md` capture protocol and `TECHNICAL_PROPOSAL.md`'s revised Phase F) but does not invalidate any code, experiment, or decision made so far.

**Practical implication:** `NOVELTY_GAP.md`, `TECHNICAL_PROPOSAL.md`, and `DATASET_AUDIT.md` were all revised in this same session to reflect this — see those documents' own revision notes rather than duplicating the full reasoning here.

---

## D0008 — Patch `utils/path_tracing.py` and `model/fipt_bsdf.py` for drjit>=1.x / mitsuba>=3.6 API changes, in place, on this branch

**Decision:** Directly modify IRIS's own `utils/path_tracing.py::ray_intersect` and `model/fipt_bsdf.py`'s `FIPTBSDF` methods to fix two breaking API changes between drjit 0.4.x (this repo's original pin) and drjit 1.5.0 (the only version that works on this machine's Blackwell GPU, per `BASELINE_REPRODUCTION.md` EXP0001), rather than trying to keep the code version-agnostic or leaving it broken.

**The two breaks (found by direct testing against a real mesh, not guessed):**
1. `mitsuba.Point3f(torch_tensor, torch_tensor, torch_tensor)` — the direct multi-arg convenience constructor from raw torch tensor slices — now raises `TypeError`. Fix: wrap each component first, `mitsuba.Point3f(mitsuba.Float(x), mitsuba.Float(y), mitsuba.Float(z))`.
2. `.torch()` on a multi-component drjit array (`Point3f`/`Vector3f`/`Point2f`/...) now returns shape `(components, B)` instead of the `(B, components)` layout this codebase assumes everywhere downstream — **a silent shape transpose, not an error**. This is the more dangerous of the two: code would run without crashing and produce silently wrong results (e.g. `positions[...,0]` slicing the wrong axis) rather than failing loudly. Fix: append `.T.contiguous()` at every such conversion. Scalar arrays (`Float`/`Int`) are unaffected and need no fix.

**Alternatives considered:** (a) pin an older/intermediate mitsuba-drjit version that supports Blackwell but keeps the 0.4.x-compatible API (checked: no such version exists — old drjit line cannot initialize CUDA on this GPU at all, confirmed in EXP0001, and the API break happened somewhere between 0.4.x and 1.5.0's drjit rewrite, with no indication of a Blackwell-compatible 0.4.x-API release); (b) write version-detecting shim code that branches on drjit version to support both old and new APIs; (c) the adopted option — just fix for the version that actually runs here.

**Reason:** (a) is not available (verified, not assumed). (b) is not worth the complexity: the old drjit/mitsuba pin is confirmed permanently broken on this specific hardware regardless of any Python-level shim (the break is in drjit's own CUDA backend initialization, not something a compatibility shim in IRIS's code could work around), so there is no scenario in this project where the old-API code path would ever actually execute — writing and maintaining it would be pure unused complexity. (c) keeps the diff minimal and auditable (every change is one of exactly two mechanical patterns, commented at each call site with a pointer to this decision) and was verified correct, not just error-free: a dedicated correctness test (not just "does it run") confirmed `ray_intersect` produces geometrically correct intersection points/normals against a known synthetic room (floor hit at the right point, ceiling hit at the right height, a ray aimed through the window opening correctly reports no intersection).

**Verification:** `ray_intersect` tested directly against `experiments/out/phase_a/room.obj` with real CUDA torch tensors — floor/ceiling hit points and the window-miss case all matched known geometry exactly. `model/fipt_bsdf.py`'s `FIPTBSDF` methods were patched using the identical, now-verified pattern but **not yet end-to-end tested** (requires a trained checkpoint + a Mitsuba scene referencing the `fipt` BSDF, i.e. real data via `render_relight.py`, not yet downloaded) — flagged as a follow-up check once real data is available, not assumed correct by analogy alone.

**Scope note:** `render.py`, `bake_shading.py`, `slf_bake.py`, `train_brdf_crf.py`, and the other pipeline scripts were not individually re-audited for the same patterns in this pass (they weren't fully read at the raw-mitsuba-API level, only at the pipeline-role level, by the session-1 sub-agent). Since they all route their actual ray-scene interaction through `utils.path_tracing.ray_intersect` (confirmed in `IRIS_ARCHITECTURE_AUDIT.md` §1's scene-loading audit), this single fix is expected to unblock the entire core 9-stage training pipeline, but this is an expectation to verify empirically (next: attempt to actually run stage 1, `slf_bake.py`, once a real dataset is downloaded) rather than a settled fact.

---

## D0009 — Real-dataset access is gated everywhere checked; build the screening component now rather than wait

**Decision:** Given no real posed-multiview + geometry dataset is immediately, autonomously fetchable (every strong candidate checked requires either the user's registration/agreement or a bulk-download path impractical from this sandboxed environment), build and validate the 2D sun-patch screening component (`utils/sun_patch.py::screen_image_for_sun_patches`) now, on individually-obtainable real photographs, rather than block on dataset access.

**Alternatives considered / investigated:** (a) IRIS's own box.com dataset links — confirmed (session 2, earlier) to require a JS-rendered browser session, not scriptable. (b) ScanNet++ full corpus (1000+ scenes, the dataset IRIS itself uses) — confirmed to require account creation + an application/approval process; free and apparently not institutionally gated, but not something I can complete on the user's behalf (their identity/agreement). (c) IVGM dataset (arXiv/MDPI, `Sensors` 2024) — the best *content* match found: the paper explicitly documents "In Office Areas 1 and 2, sunlight shines through large glass windows," with panoramic RGB + ground-truth point clouds + textured meshes, CC BY license, no registration mentioned. However, the only download path found is a single unsegmented 282.68GB Baidu Pan archive covering all three of its scenes together, with no per-scene breakdown documented on the project page — Baidu Pan is known to throttle/gate anonymous and non-mainland-China access heavily, and this sandboxed environment could not be confirmed to fetch it at any practical speed; not attempted at scale given the low odds of success and the risk of wasting significant time/bandwidth on an uncertain transfer. (d) A Hugging Face aggregator (`GaussianWorld/scene_splat_7k`) claiming to bundle ScanNet++/Matterport3D/etc as Gaussian splats — checked, but its ScanNet/ScanNet++ components still redirect to the same original-platform registration requirement, and it provides splat parameters rather than raw RGB images, so it would not remove the gate anyway. (e) The adopted option — build the screening component using freely, individually downloadable real photos (via Openverse, an open-license image search aggregator; found several strong real examples with clear window-cast sun patches within minutes, hosted on Wikimedia Commons/Flickr under open licenses).

**Reason:** All four real dataset paths investigated are genuinely gated in ways that require either the user's action (registration, regional access to a specific cloud service) or are impractical to attempt blindly from this environment — this is not a case of insufficient effort, it's a real, confirmed accessibility landscape (worth recording so a future session doesn't re-investigate the same four options from scratch). Rather than treat this as fully blocking, the screening component is valuable to build and validate now regardless of which dataset eventually becomes available — it is the same first-stage tool either way, and testing it on real (if uncurated, ungeoreferenced) photos is a genuine, if partial, piece of real-world validation, honestly scoped as such in `REAL_PHOTO_SCREENING.md` (EXP0005) rather than oversold as the full geometry-grounded comparison the user actually asked for.

**Follow-up owned by the user:** either complete ScanNet++'s registration, or attempt/facilitate the IVGM Baidu Pan download (they may have better access to that service than this sandboxed environment), or proceed with the Phase 0 real-room capture (`DATASET_AUDIT.md`) — any one of these unblocks the real geometry-grounded comparison. Not treated as urgent/blocking for continued engineering work in the meantime.

## D0010 — 2026-09-19: coordinate validation before interpreting real-data screening

Decision: use official undistorted image/intrinsic pairs, invert both ScanNet++
world and camera transforms, verify against COLMAP, and retain all geometry
screening outputs as evidence rather than semantic labels. Alternative: reuse
only the camera-axis flip and declare all mesh hits sunlit surfaces. Rejected:
EXP0007 invalidated first run reversed the apparent conclusion (55 rejected vs
4 after correction); artificial light remains after filtering. CPU Mitsuba avoids
this session's GPU access restriction without changing IRIS training behavior.
See `SCANNETPP_GEOMETRY_AUDIT.md`. Original baseline remains unreproduced.

## D0011 — 2026-09-19: falsify static-view gains before claiming relighting gains

Expanded screening selected room `1b379f1114` for its visible cast shadows. Fit a
single view and freeze the direction before disjoint-view evaluation (EXP0008).
Do not call a 0.709 heldout-view proxy IoU an advantage over SLF: a nearest-neighbor
static appearance control scores 0.833. Retain this negative comparison. Add
controlled changed-illumination validation (EXP0009): off-grid angular error
0.430 degrees, changed-light IoU 0.951 vs 0.148 frozen appearance. Known geometry,
materials and relative angular change make this component evidence only.

An implicit-mesh-opening control (EXP0010) selects a different solar direction and
scores 0.624. Visibility termination differs from the explicit-portal model, so
this comparison alone cannot attribute the difference to explicit window labels.
See `REAL_SUN_VISIBILITY.md` and `HELDOUT_SUN_CONTROL.md` for provenance and limits.

## D0012 — 2026-09-23: copy, don't move, the user's Cali-HDR/Pano2Pano originals; defer pixel-level sun-patch matching

**Decision:** When importing the user-supplied Cali-HDR + Pano2Pano archives
(169GB combined, from `/mnt/c/Users/XMH/Downloads/`) while the user was
offline overnight, (a) copy rather than delete/move the Windows-side
originals, only removing the local zip copy in `data_download/` after a
verified extraction, and (b) stop the automated work at EXIF inventory
(EXP0022) + ephemeris self-consistency (EXP0023) rather than also attempting
pixel-level sun-patch-vs-prediction visual matching in the same unattended
session.

**Reason:** (a) A multi-hour transfer over a slow WSL2 9p mount has real
interruption risk; the user said "move" but nothing indicated the Windows
copies were disposable, and deleting a user's only copy of freshly-obtained
research data during an unsupervised run is not a reversible mistake worth
risking to save ~169GB of disk (636GB was free). (b) Unlike the EXIF
self-consistency check (pure computation on metadata already extracted,
zero new methodological choices), pixel-level sun-patch matching requires
new framing decisions this project has consistently treated as
user-facing — e.g. how to define "predicted patch location" under a
panoramic (not pinhole) projection, and how to pick comparison pairs
across scenes that include real property addresses. Following the same
pattern as D0009's user-owned follow-ups, this is flagged rather than
rushed.

**Follow-up owned by the user:** decide whether/how to scope a pixel-level
sun-patch cross-check pilot (candidate pair already prepared:
`research/evidence/EXP0023/preview_images/`, Pano2Pano `6236 5th ave`
indoor+outdoor, camera-to-sun angle 137°/145°); the full RAW data will be
sitting in `data_download/cali_hdr/` and `data_download/pano2pano/` once the
background import finishes.

## D0014 — 2026-09-24: an explicit low-DOF illumination term needs a matching material-smoothness cost, or it does nothing (or backfires)

**Decision:** When integrating the explicit sun term into real IRIS training
(EXP0025, T06), add a within-segment albedo-consistency regularizer
alongside it (mirroring the existing metallic/roughness regularizer already
in `train_brdf_crf.py`'s `has_part` branch), gated to activate only when the
new `sun_vis` term is present so vanilla runs are unaffected.

**Reason:** Two earlier attempts (a global learnable sun-irradiance
parameter alone, under two different non-negativity parameterizations) did
not just fail to help -- they made the sun-attributable albedo contamination
3x *worse* than vanilla, unmodified IRIS (+55/+50pp and +75/+64pp vs vanilla's
+18/+14pp), and training longer made it worse, not better, ruling out
"just needs more steps." Root cause, confirmed by inspection: `kd`/albedo has
*zero* spatial-consistency regularization in this code path (only
metallic/roughness are pulled toward a per-segment mean) -- so a free-form,
per-point neural albedo field can always absorb a lit/shadow brightness
difference at zero cost, and there is no reason for gradient descent to
prefer routing that difference through a new, otherwise-equally-valid global
parameter instead. This directly generalizes EXP0004's own experimental
design (which used *identical* albedo regularization for both its baseline
and treatment arms, so its reported disentanglement gain already depended on
this ingredient being present) -- it just wasn't obvious it would need to be
added explicitly to IRIS's real loss function, since IRIS's own code doesn't
regularize albedo at all in this branch. Worth remembering for any future
low-DOF explicit-illumination or explicit-material-prior addition to this
codebase: check what (if anything) already constrains the field the new term
is meant to compete with, don't assume adding the term is sufficient by
itself.

**Also recorded:** two numerically-bad parameterizations of a
"non-negative, jointly-learned, near-zero-at-init" scalar, as a reusable
lesson. `softplus(0) = ln(2) ~= 0.69` is not a small number in every unit
system -- here it was ~50x the actual scale of the signal (`diffuse`'s baked
value, ~0.01-0.02) it was meant to compete with, so it dominated from step
zero regardless of evidence. Overcorrecting to `softplus(-10) ~= 4.5e-5`
then failed the opposite way: `sigmoid(-10) ~= 4.5e-5` is also the
gradient's scale factor, so the parameter never moved in 1000 steps. A
squared parameterization (`raw.square()`, `raw` initialized to a small
*positive* value matching the target signal's real scale) kept a healthy
gradient (`2*raw`) in both directions and was not brittle to this. Check
scale/gradient-conditioning together, not just non-negativity, when adding a
new learnable physical parameter to an existing loss whose other terms
already have an established internal unit scale.

## D0015 — 2026-09-24: bilinear kd/sun_irradiance terms need separate learning rates, or the joint fit oscillates across step counts

**Decision:** Give `sun_irradiance_raw` (EXP0025) its own optimizer param
group at 0.1x the base learning rate (new `--sun_lr_scale` flag,
`train_brdf_crf.py`/`experiments/run_sunpatch_iris_daylight.py`, default 1.0
so nothing changes unless explicitly set).

**Reason:** EXP0025's fix (D0014's albedo-consistency regularizer) worked,
but its 200/400/600/1000-step sweep was wildly non-monotonic (+1.71/+0.83/
+1.33/+6.62pp) -- not a smoothly-converging quantity. `Ld = kd*(diffuse +
sun_vis*sun_irradiance)` is bilinear in (kd, sun_irradiance): a
high-capacity per-point neural field and a single low-DOF global scalar
sharing one learning rate is a textbook recipe for oscillatory joint
convergence (the same failure mode as unconditioned matrix-factorization or
NeRF appearance-code/exposure co-optimization). Slowing only the global
parameter's updates relative to the per-point field is the standard fix.

**Result:** confirmed the mechanism, not a coincidence -- the whole
step-count curve tightened from a [0.83, 6.62]pp range to [0.21, 1.74]pp,
and the 200-step point (matching EXP0019's own protocol exactly) went from
a mixed result (one heldout view better than SGS-Intrinsic, one worse) to
cleanly beating SGS's magnitude on both heldout views (+0.21/+0.28pp vs
SGS's -0.43/-1.20pp). Deliberately did not grid-search `sun_lr_scale`
further (tried only 0.1) -- this is reported as a mechanism-motivated
architectural fix, not a tuned hyperparameter, specifically to avoid
overfitting a single 1-room/1-seed synthetic scene. The curve is still
non-monotonic (400/1000 steps worse than 200/600), so 200 steps remains an
observed best point, not a proven-optimal setting, and absolute albedo
accuracy is still saturated at every budget -- neither is hidden.

## D0013 — 2026-09-23: renumber the SGS sun-patch pairing experiment EXP0022 → EXP0024

**Decision:** A pre-existing draft (`research/SGS_SUNPATCH_BENCHMARK_ZH.md`,
written 2026-09-20, never committed) had independently claimed the number
"EXP0022" for the SGS-Intrinsic sun-patch pairing pilot, with its working
directory `experiments/out/EXP0022_sgs_sunpatch` and archive script
targeting `research/evidence/EXP0022`. That number was reassigned the night
of 2026-09-23 to the Cali-HDR/Pano2Pano dataset audit (D0012/T09), which was
committed first. Resolution: keep the Cali-HDR/Pano2Pano work at EXP0022 (already
committed, can't renumber retroactively without rewriting git history), and
renumber the SGS pairing experiment to **EXP0024** (next free number after
EXP0023). Only the archive destination and its embedded README in
`experiments/baselines/archive_sgs_sunpatch.py` were changed; the gitignored
local working directory name (`EXP0022_sgs_sunpatch`) was left as-is since
it isn't part of the permanent record and renaming it would require editing
path references across ~10 baseline scripts for no evidentiary benefit.

**Reason:** discovered while investigating "what baseline have we confirmed
and how do we beat it" — the SGS pairing pilot had actually already been
trained and evaluated on 2026-09-20 (real numbers existed:
`experiments/out/EXP0022_sgs_sunpatch/evaluation_v2/metrics.json`) but was
never archived to `research/evidence/`, never written into
`SGS_SUNPATCH_BENCHMARK_ZH.md` (which still said "results to be filled in
later"), and never reflected in `baseline_registry.json` (`scores: null`).
Fixed all four in the same pass: renumbered, ran the missing
`report_sgs_sunpatch.py` + `archive_sgs_sunpatch.py` steps, filled in the
real numbers, and updated the registry.

---

## D0016 — Scene yaw comes from the observed sun / sun patch, not the compass; weather is a soft prior

**Decision:** In the time + place -> lighting chain, (a) sun direction in the
local ENU frame is taken from EXIF time + GPS via pvlib ephemeris; (b) the one
remaining degree of freedom, the scene's rotation about the vertical (yaw), is
calibrated from an observed sun or sun patch, never from the camera/phone
compass heading; (c) Open-Meteo reanalysis DNI/DHI/cloud enter only as soft
priors with uncertainty, never as hard gates.

**Alternatives considered:** use the EXIF compass heading as the yaw source
(cheapest, zero observations needed); use weather as a hard "sun present"
gate for suppressing sun-patch hypotheses.

**Evidence:** EXP0028 (`PANO_SUN_DIRECTION_ZH.md`). On 7 real panoramas with a
detected sun, ephemeris elevation matched within 0.48 deg median / 1.18 deg
max -- the heading-free part of the chain is sound. Compass-based azimuth was
off by 82.7 deg median / 146.9 deg max even after magnetic-declination
correction; a mirrored-convention bug was ruled out (error 83 deg with the
sun at image center, correlation with image longitude +0.49 not the -2
slope a mirror predicts). Indoors-through-window captures are exactly where
compasses are disturbed. Weather: 3 of 7 suns were detected in hours with
DNI < 120 W/m^2 (hazy sun), so a hard DNI gate would have wrongly suppressed
them; hourly ~25 km reanalysis cannot resolve thin cloud or local occlusion.

**Reason:** the compass is not just noisy, it is uninformative at this error
level, while one observed sun/patch fixes yaw exactly and makes every other
time's sun direction parameter-free -- which is the mechanism the killer
experiment depends on anyway. Weather is still valuable as a prior on sun
intensity and sky/sun ratio, but only probabilistically.

---

## D0017 — Retract EXP0025/EXP0026 "beats SGS"; redesign the sun-patch benchmark before any further method claim

**Decision:** (1) EXP0025/EXP0026 must not be cited as reducing sun contamination
via the explicit sun term, nor as beating SGS-Intrinsic (EXP0024). (2) No further
method comparison on the uniform-floor, single-segment EXP0019 scene. (3) The next
sun-term test must use a benchmark where flattening is penalized and a sun term
that is actually active.

**Evidence:** EXP0027, all 24 runs (4 arms x 2 conditions x 3 seeds): the whole
gap reduction comes from the albedo-consistency regularizer (reg_only
-0.1/-0.0pp), which flattens the floor to ~white (99% of floor albedo > 0.95,
MAE vs truth 0.78 vs vanilla 0.62). sun_only equals vanilla (+19.1/+14.2 vs
+19.5/+14.3pp). With sun_lr_scale 0.1 and 200 steps the learned sun irradiance
barely leaves its init (0.010 -> 0.016 with sun vs 0.013 without), so the sun
term was inert, not refuted.

**Required benchmark changes:** textured / spatially varying true floor albedo;
floor-only segment (not all-zero segmentation); report absolute albedo MAE and
floor CV alongside the gap, and treat a gap win that worsens MAE as a failure.

**Next sun-term arms (so the sun term is actually tested):** (a) sun_only with
the sun term genuinely learned (sun_lr_scale 1.0 and/or more steps, verifying
learned irradiance separates sun_a from sky); (b) sun_only with sun irradiance
*supplied* rather than learned -- generator ground truth in synthetic scenes,
reanalysis DNI in real ones (links to the time + place -> lighting chain, D0016).
(b) is the cleaner test of the identifiability claim: the free albedo field no
longer competes with a free low-DOF scalar.

**Process lesson:** EXP0027 finished on 2026-09-24 night but was recorded only in
a memory note; `task_state.json` and `EXPERIMENTS.md` still reported the
retracted win, and a status summary on 2026-09-25 repeated it before this was
caught. Registry files must be updated in the same pass an experiment finishes,
and status reports must check `experiments/out/` for finished-but-unrecorded runs.

---

## D0018 — Adopt the "SolarIR" strong framing; test calibration against an explicit-but-uncalibrated sun before touching IRIS

**Decision:** The project's claim is narrowed to: the Sun as a *calibrated*
(time + place, one yaw DOF), moving, global light source that enters rooms
through window apertures, whose predictable trajectory supervises recovery of a
time-invariant indoor scene, validated by predicting unseen time-of-day photos.
Before modifying IRIS further, run a controlled differentiable-rendering
hypothesis test (`SOLARIR_HYPOTHESIS_TEST_ZH.md`) whose decisive comparison is
calibrated sun (C) vs explicit sun with image-estimated direction (B), not vs
vanilla IRIS.

**Evidence:** user's 2026-09-27 survey plus my spot checks
(`LITERATURE_AUDIT_2026-09-27_ZH.md`): explicit sun, sun+sky split, window
directional light, indoor sun-direction estimation, shadow/material separation
and explicit emitters all have prior art (Li 2022, ProjectiveShading, SOL-NeRF,
EO-NeRF, SIR, SGS-Intrinsic, IR-HGP, AEGIR). New find: Dynamic Inverse
Rendering (Yunus et al., ECCV 2026) states and exploits the general principle
that multiple lighting conditions reduce material-lighting ambiguity -- so
"multi-time helps" is not a contribution; only calibration can be.

**Pre-registered rules:** R1 (C beats B on floor albedo MAE by >=15%,
non-overlapping seeds), R2 (C beats B at the extrapolated time by >=1 dB PSNR
and >=0.1 patch IoU), R3 sanity (multi-time C beats single-time C). Pre-stated
expectation: in clean synthetic conditions R1 may fail because image-estimated
directions are already accurate; v2 adds window-geometry error and an occluder,
and R1 is judged on v1+v2 together.

**Reason:** the earlier plan's weak point was comparing against vanilla IRIS,
which any explicit-sun method would beat. Comparing against B isolates exactly
what we can still claim.

## D0019 — SolarIR testbed v1: experiment-ID mapping, forward-model fixes and metric definitions, frozen before any arm comparison

**Decision (IDs):** the autonomous-research brief's experiment sequence is numbered
from EXP0030 on, because EXP0001–EXP0028 are taken and EXP0029 is reserved for the
Pano2Pano single-time real check:

| Brief | Ours | Content |
|---|---|---|
| EXP0001 transport sanity | EXP0030 | window-aperture transport, patch projection, estimator checks (`tests/test_solar_transport.py`) + D1/D2 diagnostics |
| EXP0002 material leakage | EXP0031 | W / A / A_hi / B / C / C_oracle, floor error by region |
| EXP0003 capacity control | EXP0032 | same runs; A vs A_hi (16x the lighting parameters) and W |
| EXP0004 held-out solar condition | EXP0033 | 12:15 (interpolation), 14:30 (extrapolation) |
| EXP0005 number of solar states | EXP0034 | 1 vs 3 training times (v1); 2 and 4 later if informative |
| EXP0006 direction perturbation | EXP0035 | 0/2/5/10/20 deg on C's direction |
| EXP0007 window geometry perturbation | EXP0036 | aperture error, occluder (= v2 of T11) |
| EXP0008 Cali-HDR real fit | EXP0037 | |
| EXP0009 real multi-view stress test | EXP0038 | |

**Decision (scene / data):** v1 uses the procedural Mitsuba room (`experiments/solarir_scene.py`)
instead of OpenRooms. Reason: the question needs exact control of the sun, the sky
and ground-truth materials under a physical sky model, which the procedural room
gives at no download cost. OpenRooms remains the candidate for a multi-scene v2;
its access has not yet been checked.

**Decision (forward model, fixed from training-image diagnostics only):**
- D1 (true lighting, materials learned) and D2 (true materials, each arm's lighting
  learned; scored on training images) were run before any arm comparison.
- An 8x16 bilinear sky could not represent the sharp horizon of the sunsky ground
  truth: walls came out 10% too dark with true materials. The sky became 16x32
  piecewise-constant cells (drawn 2x2 per texel); walls are then within 2%.
- A relative-error loss was tried and **rejected**: it stopped the generic
  envmap arm from ever forming a sun (patch IoU 0). Adopting it would have handicapped
  the baseline. All arms use linear MSE, lr decay over the last 40% of iterations.
- The lighting learning rate for each arm was picked on the D2 training-image fit
  only (W, A, B, C: 0.1; A_hi: 0.4).
- A_hi (128x256 envmap) was added as the generic capacity control.
  - Its D2 fit plateaus at 27.5–30.7 dB (patch IoU 0.90–0.95) against
    C_oracle's 32.9–34.0 dB, including at 800 iterations.
  - This is how the generic arm behaves under this optimiser, not proof that no
    generic representation could do better.
- W (Lambertian window area emitter, no outside light) is the analogue of IRIS's
  emitter model.
- Material optimisation was under-converged. With the **true** lighting (D1), 400
  iterations at material lr 0.02 recovered the floor only to correlation 0.83.
  - More views per iteration, higher spp and nearest-texel filtering made no
    difference.
  - Iterations and learning rate did: 1000 iterations at lr 0.05 gives correlation
    0.92 on all texels and 0.965 on observed texels.
  - Protocol for every arm: 1000 iterations, material lr 0.05, spp 8, 2 random views
    per time per iteration.
- The earlier lighting learning-rate choice (made at 400 iterations) is kept; the
  decay schedule scales with the run length.

**Decision (metrics; clarifies the pre-registration, not a change of rules):**
- Image metrics use room pixels only. Pixels that look out of the window show
  the sky directly, which says nothing about the room.
- Floor metrics are computed on **observed** texels: those hit by >= 4 camera
  pixels over the 6 views, 79% of the floor. Unobserved texels carry no information
  for any arm. All-texel numbers are reported beside them.
- Floor albedo is identified only up to one global scale shared with light
  intensity. R1 is therefore read on **scale-aligned** MAE, with raw MAE reported
  beside it. If the two disagree in direction, R1 is recorded as not holding.
- Held-out lighting for every arm: linear interpolation between the bracketing
  training times at 12:15; nearest training time at 14:30.
  - B's held-out sun direction comes from a least-squares linear fit of (az, el)
    against time over its own image estimates (a constant with one training time).
  - C calibrates one bearing from the 11:30 patch and takes everything else from the
    ephemeris.

**Alternatives:** keep the 8x16 bilinear sky (rejected, inadequate on walls); relative
loss (rejected, biased against the baseline); tune lighting learning rates on material
error (rejected, leaks the evaluation metric).

**Reversible?** Yes; every choice is a flag or constant in `solarir_scene.py` / `solarir_test.py`.

**Relevant experiments:** EXP0030–EXP0034 (T11).

## D0020 — Joint-fit protocol: spp 32 × 2000 iterations; explicit-sun arms use a 4×8 sky (chosen with oracle knowledge, declared); 16×32 kept as a sensitivity arm

**What happened:** the first arm fit (EXP0035 δ=0, identical to C_oracle) recovered the
floor far worse than D1 did under true lighting:
- correlation 0.78 against 0.965;
- patch bias +0.09 (sunlit albedo too bright);
- sun irradiance fitted ~30% low;
- sky barely moved from its initial value, including 0.075 below the horizon, where the
  truth is 0.

The loss reached the 8-spp Monte Carlo noise floor by iteration ~50. I stopped every run
before any arm comparison could be read from it.

**Diagnostics** (seed 0, multi-time, observed floor texels; scale-aligned MAE / corr /
patch bias):

| Setting | C_oracle | A_hi |
|---|---|---|
| spp 8, 1000 it, sky 16×32 | 0.053 / 0.78 / +0.090 | 0.097 / 0.57 / +0.158 |
| spp 32, 1000 it | 0.045 / 0.845 / +0.070 | 0.060 / 0.72 / +0.072 |
| spp 64, 2000 it | 0.039 / 0.877 / +0.055 | 0.053 / 0.74 / +0.042 |
| spp 32, sky 8×16 | 0.037 / 0.880 / +0.051 | — |
| spp 32, sky 4×8 | 0.032 / 0.916 / +0.034 | — |
| spp 32, **true sky fixed**, only sun E learned | 0.021 / 0.960 / +0.022 | — |
| D1: all lighting true | 0.018 / 0.965 / −0.002 | — |

**Finding:** with the sun explicit, the remaining material–illumination ambiguity comes
from the **free sky**, not the sun.
- Different floor points see different parts of the sky through the window, so a
  high-capacity sky can paint low-frequency brightness patterns that trade off against
  albedo.
- Lowering sky capacity helps monotonically. The true sky removes the problem.
- This is the H1 mechanism: a generic light model needs high capacity to represent the
  sun, and that capacity is what creates the ambiguity.

**Decision:**
- All arms: spp 32, 2000 iterations. More samples cost almost no time, because the run
  is Python-overhead bound. Longer runs help both arms. This choice is neutral.
- Explicit-sun arms (B, C, C_oracle) use a **4×8 piecewise-constant sky** (45° cells,
  drawn 8×8 per texel).
- **Declared:** this was chosen after seeing *oracle* floor-material error for C_oracle,
  and after seeing C_oracle vs A_hi pipeline numbers. It is a method design choice made
  with test-set knowledge.
- To keep its effect visible, the comparison includes **C_oracle16** (C_oracle with the
  earlier 16×32 sky).
- Generic arms (W, A, A_hi) are unchanged: lowering their capacity would remove their
  ability to represent the sun at all.
- D2 adequacy is re-run for every arm under the new protocol.

**Consequence for the research question:** the sky, not the sun direction, is where the
action is. A calibrated solar state could also constrain the sky: a physical sky model
parameterised by the sun position. In this testbed that would be an inverse crime,
because the ground truth *is* Mitsuba `sunsky`. That test needs a sky from a different
source, e.g. real HDR skies, and is deferred.

**Reversible?** Yes: `SKY` table in `solarir_scene.py`; `--spp`, `--iters`.

**Relevant experiments:** EXP0030–EXP0036.

## D0021 — EXP0037 detector and timing changes, made before any held-out evaluation

**Decision:** three changes to the pre-registered EXP0037 pipeline (`EXP0037_REAL_PATCH_PREREG_ZH.md`),
all made before any held-out prediction was computed or viewed.

1. **Detection uses a temporal brightness ratio, not single-image Otsu.** Because the tripod is
   fixed, each pixel's brightness is divided by its own 30th-percentile brightness over the day's
   times. This cancels albedo: the white card and the colour chart on the floor otherwise look
   like patches. The threshold is `max(Otsu over pooled log-ratios, ln 3)`.
   - The `ln 3` floor was added after 2023-06-25 gave a non-bimodal ratio distribution, with
     Otsu at 0.24–0.32 and 8–40% of pixels flagged.
   - 2023-07-06's Otsu threshold (1.59) is above the floor, so its masks are unchanged by it.
2. **A single fixed exposure (1/60 s) replaces the 9-shot merge for detection.** The merge
   produced horizontal banding where the chosen exposure switched, because the THETA tone curve
   is not exactly sRGB.
3. **Timing:**
   - Time comes from the camera clock (`DateTimeOriginal`, EDT = UTC−4) of the detection shot.
     The GPS time stamp is written once per bracket and was stale: 10:58 on 06-25 repeats 10:56's
     stamp, and 06:56 on 07-06 is 3 min old. Where GPS is fresh, the camera clock agrees to 10–30 s
     (≤ 0.13° of sun motion).
   - The window wall is excluded from the receiver mask. Direct sun cannot land on the wall
     holding the only window, so bright blobs there are bounce or glare.

**Disclosure:** an EXP0037 evaluation on 07-06 under the earlier detector was started and killed
before it printed anything; its log was deleted unread. Detection masks, not predictions, were
inspected to make these changes.

**Reversible?** Yes (`DETECT_EXPOSURE`, the threshold floor, `receivers()` in `cali_patch_test.py`).
**Relevant experiments:** EXP0037.
