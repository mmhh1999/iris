## EXP0011 — official bathroom baseline on Blackwell (2026-09-19)

This supersedes earlier missing-toolchain and inaccessible-Box conclusions.
The public archives were range-downloaded directly, with ZIP CRC checking and
SHA-256 manifests. No private account or manual download was needed.

- Official synthetic bathroom: 109 training views, 13 validation views, 640×320.
- Official `last_1.ckpt`, original 256 SPP / 16 samples per batch, all 13 validation
  views: **PSNR 28.97625 dB, SSIM 0.79503** (mean per image, original metric code).
- Original initialization passed three optimizer steps; finite losses and state,
  nonzero optimizer momentum in material and emitter parameters; checkpoint saved.
- From-scratch SLF baking and emitter extraction passed on all 109 training views.
- Original BRDF/CRF and emitter optimization each passed three steps.
- The bounded pipeline is running stage 10 (indirect shading refinement) in a
  detached process. It will then run final BRDF/CRF optimization and rendering.
- A detached full 6/4/1/4-epoch run is queued after all bounded stages succeed.
  This is execution validation in progress, not completed converged retraining.
- No real same-room cross-time comparison is possible with the available data yet.

Environment: torch 2.11.0+cu128, torchvision 0.26.0+cu128, Mitsuba 3.9.1 /
Dr.Jit 1.5.0, Lightning 2.6.6, tiny-cuda-nn 2.0 (commit recorded in evidence),
local CUDA 12.8 and GCC 13.4. OpenCV 5 lacked EXR support; switched to 4.11.0.86.
Original pinned torch/Mitsuba versions are not used on this Blackwell machine.
Compatibility changes: explicit Lightning Trainer construction; optional training
step/render frame limits. Our first automation copied `last.ckpt`, unlike the official
scripts which move it. Lightning versioned new files and later stages read stale
weights. Parameter-difference checks detected this; the runner now moves the file,
and affected stages 07 onward were invalidated and restarted from the true
stage-06 checkpoint. The official pretrained evaluation was unaffected. Previously verified Dr.Jit ray conversion fixes remain.
No BRDF architecture, loss, or lighting-model change is part of this baseline.

Evidence: `research/evidence/EXP0011/`; outputs and logs:
`experiments/out/EXP0011_baseline/`. Comparison figure:
`experiments/out/EXP0011_baseline/baseline_comparison.png`.

Reproduction helpers: `experiments/fetch_official_baseline.py`,
`experiments/build_tcnn_local.py`, `experiments/run_original_iris.py`.
The runner defaults to bounded execution; `--full-training` selects the original
6/4/1/4 epoch budgets. Use a fresh output directory for each run.

# Baseline Reproduction Log

Tracks environment setup and baseline reproduction attempts for IRIS on `main` @ `d2d4381`. See `EXPERIMENTS.md` for the experiment index.

## Target hardware/software (from repo)

- README's tested config: Ubuntu 22.04.4 LTS, RTX 4090, driver 535, CUDA 12.2, nvcc 11.7.
- `environment.yml` (default): Python 3.8, `torch==1.13.1+cu117`, `pytorch-lightning==1.9.0`, `mitsuba==3.5.0`, `drjit==0.4.4`.
- `environment_cuda118.yml` (present in repo, pre-existing before this session — see commit `0afb475` "5070ti"): Python 3.10, `torch>=2.1.0`+cu118, `pytorch-lightning==2.1.1`, same `mitsuba==3.5.0`/`drjit==0.4.4` pins. This appears to be a prior attempt (by the repo owner, before this research effort started) to get a newer-GPU-compatible torch while keeping the same old Mitsuba/drjit — which EXP0001 shows does **not** fix the actual blocker (drjit's CUDA backend, not torch, is what fails).

## Local machine (this session)

- OS: Linux 6.18.33.2-microsoft-standard-WSL2 (WSL2 under Windows)
- GPU: NVIDIA GeForce RTX 5070 Ti, 16303 MiB, driver 581.80 (Windows-side) / WSL shim reports `580.105.07`, CUDA 13.0 (`nvidia-smi`)
- This is a **Blackwell** architecture GPU (compute capability ~sm_120), notably newer than anything the pinned 2023-era `mitsuba==3.5.0`/`drjit==0.4.4` stack was built/tested against.
- No system CUDA toolkit installed (`nvcc` not found), no conda (`conda` not found). Python tooling available: system `python3` (3.14.4), `uv` (fast venv/pip), `pip3`.
- No IRIS datasets present locally (`/home/ubuntu/datasets/data`, referenced in a pre-existing `scripts/run_test.sh` from commit `22eb79d` "fk nvidia", does not exist on this machine — that script/path was evidently written for a different, prior machine).

---

## EXP0001 — Can the repo's pinned environment initialize Mitsuba's CUDA backend on this GPU?

- **Git commit:** branch `research/daylight-aware-iris`, no code changes yet (pure environment test, done outside the repo in `/tmp/.../scratchpad/envtest`)
- **Dataset / scene:** none (isolated Mitsuba/drjit smoke test, no scene data needed)
- **Config:** two isolated `uv venv` Python 3.10 environments
- **Seed:** n/a
- **Hypothesis:** `mitsuba==3.5.0` + `drjit==0.4.4` (the repo's pin) will successfully call `mitsuba.set_variant('cuda_ad_rgb')` and load a trivial scene on the local RTX 5070 Ti.
- **Method:**
  1. Created venv, installed exactly `mitsuba==3.5.0` `drjit==0.4.4` (no torch, to isolate Mitsuba/drjit from PyTorch's own CUDA stack).
  2. `import mitsuba; mitsuba.set_variant('cuda_ad_rgb')` → **fails**: `AttributeError: jit_init_thread_state(): the CUDA backend hasn't been initialized. Make sure to call jit_init(JitBackend::CUDA) to properly initialize this backend.` Failure happens deep inside Mitsuba's own `mitsuba.ad.integrators` import chain, i.e. it happens on any use of the `cuda_ad_rgb` variant, not something avoidable by deferring integrator use.
  3. Isolated whether this is a drjit-CUDA-specific problem or a system-level CUDA problem:
     - `from drjit.llvm import Float; Float(1.0,2.0,3.0)` → **works**. drjit's CPU/LLVM JIT backend is fully functional.
     - Raw `ctypes.CDLL('libcuda.so.1'); lib.cuInit(0)` → returns `0` (success); `cuDeviceGetCount` → returns 1 device. The system CUDA driver API itself is healthy and sees the GPU.
     - Conclusion: the failure is specific to **drjit 0.4.4's CUDA JIT backend initialization**, not the system driver, not the LLVM path. Most likely cause: drjit 0.4.4 (built ~2023) has no notion of the Blackwell (sm_120) architecture in its internal compute-capability/kernel-targeting tables, so its `jit_init(JitBackend::CUDA)` call fails silently and every subsequent CUDA-variant use raises the generic "not initialized" error.
  - **Result:** Confirmed failure of the exact pinned stack on this GPU. **Status: FAILED (root cause identified).**

  4. Retested with the current latest release: `mitsuba==3.9.1`, `drjit==1.5.0` (same isolated-venv method).
     - `mitsuba.set_variant('cuda_ad_rgb')` → **succeeds**, `mitsuba.Point3f(1,2,3)` (a CUDA-backed tensor) constructs fine. So drjit 1.5.0's CUDA JIT backend *does* recognize this GPU.
     - `mitsuba.load_dict({'type': 'scene'})` (a scene with zero shapes, i.e. no ray tracing needed yet) → **fails**: `jit_optix_api_init(): libnvoptix.so.1 could not be loaded -- disabling OptiX backend!` then `RuntimeError: ... Could not initialize OptiX!`. Mitsuba's `cuda_ad_rgb` variant requires OptiX for any scene (even an empty one) because scene acceleration-structure construction goes through OptiX unconditionally in this variant.
     - Found `/usr/lib/wsl/lib/libnvoptix.so.1` does exist (10056 bytes — a thin loader stub, `libnvoptix_loader.so.1` is a symlink to the same file), but isn't on the default library search path. Setting `LD_LIBRARY_PATH=/usr/lib/wsl/lib:...` gets past the "could not be loaded" warning but hits a new error: `jit_optix_api_init(): could not find symbol optixQueryFunctionTable` — i.e. the stub loads but can't resolve the actual OptiX entry point, because (per Mitsuba's own WSL2 OptiX setup docs, fetched and confirmed) WSL2's default driver package does **not** ship the real OptiX runtime (`libnvidia-rtcore.so`, `libnvidia-ptxjitcompiler.so`, `libnvidia-gpucomp.so`, `nvoptix.bin`) — only a stub. The documented fix requires downloading a **matching-version Linux driver .run package** (identified: driver branch `580.105.xx`, matching this machine's Windows driver `581.80`), extracting it without installing (`bash NVIDIA-Linux-x86_64-*.run -x --target driver`), and manually copying those specific files into `C:\Windows\System32\lxss\lib` (a Windows-host, admin-privileged path) via Explorer, then running `wsl --shutdown` from PowerShell to reload WSL's GPU paravirtualization layer.
     - This file-copy + WSL-restart step requires Windows-host access and would terminate any live WSL session (including this one) — outside what this sandboxed session can perform directly. Surfaced to the user as a blocking decision (see `DECISIONS.md` D0003); user chose to apply the fix themselves. **Awaiting confirmation of restart + re-verification.**
  5. Confirmed a viable fallback path exists if the OptiX fix doesn't pan out: `mitsuba.set_variant('llvm_ad_rgb'); mitsuba.load_dict({'type': 'scene'})` → succeeds fully (CPU-side ray tracing via drjit's LLVM JIT, no OptiX needed). Not pursued further this session since the user opted for the native GPU fix, but documented as a known-working alternative should the GPU fix fail or need revisiting. Note: switching to `llvm_ad_rgb` throughout IRIS would additionally require porting `path_tracing.py`/`fipt_bsdf.py`/etc. from drjit 0.4.x's API to 1.x's (breaking changes across that major version bump — not yet audited in detail since this path wasn't chosen).
- **Interpretation:** The repository's pinned dependency versions predate this GPU generation and cannot be used as-is on Blackwell hardware. The forward-compatible dependency set (`mitsuba==3.9.1`/`drjit==1.5.0`, likely paired with a matching modern PyTorch/cu12x build) is necessary regardless of the OptiX/WSL2 fix, which raises a **second, separate open question for later**: does the IRIS codebase (written against drjit 0.4.x's API — e.g. `.torch()` interop methods, `mitsuba.Int(...)`, `mitsuba.math.RayEpsilon`, `mitsuba.OptixDenoiser`) run unmodified against drjit 1.5.0/mitsuba 3.9.1, or does it need API porting? **Not yet tested** — blocked on first getting a working OptiX scene load, at minimum, to test `utils/path_tracing.ray_intersect` against a real mesh.
- **Status: RESOLVED / PASSED (2026-09-18).** User downloaded the matching Linux driver package (`NVIDIA-Linux-x86_64-580.105.08.run`, confirmed matching the Windows 581.80 driver via NVIDIA's own version-correspondence docs), extracted it (`bash ... -x --target driver`), and — with my help preparing the exact file set (`libnvoptix.so.1`, `libnvidia-ptxjitcompiler.so.1`, `libnvidia-rtcore.so.580.105.08`, `libnvidia-gpucomp.so.580.105.08`, `nvoptix.bin`) — copied them into `C:\Windows\System32\lxss\lib`. This required taking ownership of that directory (`C:\Windows\System32\lxss\lib` is TrustedInstaller-owned; a first attempt via `takeown`/`icacls` in an ostensibly-elevated PowerShell failed with "current user does not have the take-ownership privilege," resolved via Explorer's GUI "Advanced Security Settings -> Change Owner -> replace owner on subcontainers" path instead — worth remembering if this comes up again on another machine). After `wsl --shutdown` + restart, re-verified from scratch:
  - `mitsuba.set_variant('cuda_ad_rgb'); mitsuba.load_dict({'type':'scene'})` with a real shape — succeeds, no OptiX errors.
  - A full `mitsuba.render(...)` of a lit diffuse rectangle, spp=32, 256x256 — completes in 0.25s and produces a correct-looking image (mean 0.60, max ~1.0 for a scene lit to that level). **GPU-accelerated ray tracing is fully functional on this machine now**, using `mitsuba==3.9.1`/`drjit==1.5.0` (not the repo's originally pinned `mitsuba==3.5.0`/`drjit==0.4.4`, which — per the diagnosis above — cannot initialize CUDA on this Blackwell GPU at all, independent of the OptiX/WSL2 issue).

### Next steps (updated 2026-09-19)

1. ~~Re-run the smoke test~~ **Done, passed.**
2. ~~Build a minimal real-mesh test~~ **Done, EXP0003, `DECISIONS.md` D0008.** Found and fixed two real drjit-1.x API breaks in `utils/path_tracing.py::ray_intersect` (not just "ran without error" — verified geometrically correct against known synthetic geometry). Same pattern applied to `model/fipt_bsdf.py`. `torch==2.11.0+cu128` also confirmed working on this GPU (CUDA matmul test passed).
3. Box.com dataset links (README) require a JS-rendered browser session to actually traverse/download — not reliably scriptable from this sandboxed environment (confirmed: direct `curl`/`wget` against the share URL only returns the Box web-app shell, not file content). **This needs the user to download at least one scene manually** — a much lighter task than the Phase 0 real-room capture, whenever convenient. Not currently blocking other work.
4. Once a real dataset is available: attempt full pipeline stage 1 (`slf_bake.py`).
5. **`torch_scatter` and `pytorch_lightning` installed and verified** (2026-09-19): `torch_scatter` needed a prebuilt-wheel index (`https://data.pyg.org/whl/torch-2.11.0+cu128.html`) since this sandboxed environment has no C/C++ compiler (`g++`/`gcc`/`clang` all absent) and no passwordless `sudo` to install one via `apt`; confirmed working on this GPU with a real `scatter` call, not just import-tested.
6. **`tinycudann` (needed by `model/brdf.py::NGPBRDF`) is blocked**: it has no prebuilt-wheel distribution (unlike `torch_scatter`) and must compile CUDA/C++ source at install time, which needs a host compiler this environment doesn't have. `nvidia-cuda-nvcc-cu12` (a pip-installable nvcc) is available and would solve half the problem, but a host C++ compiler (`g++`) is still missing and would need `apt-get install build-essential` (or similar), which needs `sudo` — a system-level change I can't make without the user, and haven't asked for since it's not on the critical path yet (see reason below). Confirmed blocked by directly attempting the install (hit a build-isolation error before even reaching the compile step), not just inferred.
   - **This does not block near-term real-data work**: emitter-misclassification checking (`extract_emitter_ldr.py`, pure mesh + image statistics, no neural network), window-geometry detection, and Mode B geometric analysis (`utils.sun_patch`/`utils.window_geometry`) are all pure Python/torch/mitsuba and need none of `NGPBRDF`/`tinycudann`. Only Phase D's actual joint material optimization needs it — revisit (ask the user for `sudo`, or find/request a prebuilt `tinycudann` wheel, or use a Docker/conda environment with the toolchain preinstalled) once that phase is actually reached.

### Data format IRIS's `Scannetpp` loader actually expects (reviewed 2026-09-19, ahead of the user's ScanNet++ download landing)

`utils/dataset/scannetpp/dataset.py::Scannetpp.__init__` does **not** consume ScanNet++'s raw download format directly — it expects a pre-processed layout:
```
{dataset_root}/data/{scene_id}/
    psdf/
        images/{name}.png          <- RGB frames (PNG, despite the class docstring saying .exr -- that's stale)
        transforms_all.json        <- NeRF/NeRFstudio-style: fl_x, fl_y, cx, cy, h, w, and a `frames` list each
                                       with file_path + a 4x4 transform_matrix (OpenGL-style c2w; the loader
                                       flips columns 1:3 to convert to OpenCV convention itself, dataset.py:135)
        train_test_lists.json      <- {"train": [...names...], "test": [...names...]}
        render_traj.npy            <- optional, only needed for render_video.py/render_relight.py trajectories
    scans/
        scene.ply                  <- the reconstructed mesh (referenced from initialize.py and others, not
                                       from this file directly)
```
`transforms_all.json`'s format is exactly what NeRFstudio's `ns-process-data` (COLMAP wrapper) produces — consistent with the README's "Customized Data" section recommending that exact tool. ScanNet++'s own official download/toolkit ships pre-computed COLMAP poses and the laser-scanned mesh directly, but not necessarily already in this exact `transforms_all.json` shape or this directory layout — **a conversion step will very likely be needed** once the raw download lands: (a) export ScanNet++'s COLMAP poses to a NeRFstudio/NeRF-style `transforms.json` (either via `ns-process-data` re-running COLMAP from scratch on the DSLR images, or a direct COLMAP->transforms.json converter reusing ScanNet++'s already-computed poses, if one is readily available from ScanNet++'s own toolkit — to be checked once the data is in hand), (b) place/rename the mesh at `scans/scene.ply`, (c) generate or reuse a `train_test_lists.json` (ScanNet++ itself ships an official per-scene train/test split which may be directly reusable here). No exposure/CRF/timestamp handling needed from the raw data beyond what's already hardcoded (`exposures = np.ones(...)`, a fixed mean EMoR CRF curve, both already noted as ScanNet++ limitations in `IRIS_ARCHITECTURE_AUDIT.md` §7).

This mapping is recorded now, ahead of having the actual data, specifically so the conversion script can be written and run immediately once the download completes rather than requiring another investigation pass then.

### 2026-09-19 autonomous continuation: current access check

`nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader`
executed outside the restricted sandbox reports RTX 5070 Ti / 581.80 / 16,303 MiB
(total) / 2,999 MiB (used at check time). Therefore the earlier sandbox NVML error
is not evidence of unavailable hardware. `tinycudann` remains absent;
`gcc`, `g++`, `nvcc`, `cmake` and `ninja` are absent from PATH. Torch, torch_scatter
and pytorch_lightning are installed. No compiler installation or full baseline
training was performed in this continuation. EXP0008–EXP0010 are CPU geometry
and rendering component experiments and must not be recorded as IRIS reproduction.
