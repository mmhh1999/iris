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
