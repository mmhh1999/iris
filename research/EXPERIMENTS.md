# Experiments Registry

Every experiment gets a unique ID (EXP0001, EXP0002, ...), never reused, never deleted (even if superseded or negative). Full detail for each experiment lives in `BASELINE_REPRODUCTION.md` (baseline/environment experiments) or a future `research/PHASE_A_SYNTHETIC.md` / `research/PHASE_B_...md` etc. as phases progress; this file is the index.

| ID | Date | Git commit | Phase | Hypothesis | Status | Result summary |
|---|---|---|---|---|---|---|
| EXP0001 | 2026-09-17/18 | (branch `research/daylight-aware-iris`, pre-code-changes) | Phase 0 (env) | The repo's pinned CUDA/Mitsuba/drjit stack (`environment.yml`) will initialize and run on the local RTX 5070 Ti (Blackwell) GPU under WSL2. | **Failed** (with identified fix, in progress) | Old pinned stack cannot init CUDA JIT at all. Newer mitsuba/drjit inits CUDA JIT but OptiX scene-init fails without manual WSL driver-file setup. Root cause and fix identified (see `BASELINE_REPRODUCTION.md`); user applying fix. LLVM/CPU Mitsuba backend confirmed as a working fallback if needed. |

Future rows will be appended here as experiments run; do not renumber or reorder existing rows.
