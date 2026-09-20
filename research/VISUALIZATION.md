# 中文阶段成果可视化 · 2026-09-19

Generated from existing EXP0008, EXP0009 off-grid and EXP0010 evidence; no new
fitting or generated imagery. Output: `experiments/out/visualization_20260919/`.

- `index.html`: offline, embedded photos and results; 4 real views × 4 display
  modes; 6 synthetic cases; train/test RGB wipe; prediction/observation/memory
  masks; all numerical controls and explicit evidence limits.
- `overview.png`: Chinese scientific overview, 2400×1800.
- `results_zh.pdf`: matching standalone PDF overview.
- `visualization_manifest.json`: source and result hashes.

Rebuild: `.venv/bin/python experiments/visualize_results.py`.
Chinese font defaults to the locally available Microsoft YaHei; use `--font`
with another Chinese font on a different machine. Font files are not bundled.

Verification: visual inspection of PNG; per-view and synthetic mask TP/FP/FN
checked against recorded metrics; 46 embedded images decoded; HTML IDs and JSON
validated; interaction JS executed in V8 with a minimal DOM for all 16 real
view/mode combinations, 6 synthetic selections and slider endpoints. This is
logic verification, not a full browser layout/accessibility test.

The default real view is the weakest heldout view (DSC09417). RGB before/after
in the synthetic slider are both path-traced observations; model predictions
are binary light-support masks, not recovered-material RGB renders. Actual
32-spp noise remains visible. Static-memory superiority on real same-light views
is retained, and synthetic gains are not described as real IRIS improvements.
