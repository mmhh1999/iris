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
- **Status:** IN PROGRESS. Root cause of both failure layers (old-drjit/Blackwell incompatibility; WSL2 OptiX stub) identified and documented; fix for the second applied by user out-of-band.

### Next steps once GPU fix is confirmed

1. Re-run the exact smoke test above (`mitsuba==3.9.1`/`drjit==1.5.0`, `cuda_ad_rgb`, `load_dict({'type':'scene'})`) to confirm OptiX now initializes.
2. Build a minimal real-mesh test: load a tiny `.obj`/`.ply`, call `utils.path_tracing.ray_intersect`, confirm no drjit 0.4.x→1.x API breakage in that function specifically (it's the most heavily used low-level primitive across every training stage).
3. If breakage is found, decide: pin an intermediate mitsuba/drjit version that both (a) supports Blackwell and (b) is close enough to 0.4.x API to need minimal porting, vs. (c) do the full 1.x port. Record as a new decision in `DECISIONS.md`.
4. Only then attempt full pipeline stage 1 (`slf_bake.py`) on a downloaded IRIS scene.
