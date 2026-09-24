## 2026-09-24：EXP0026 — 分离学习率修复非单调性，200 步首次干净超过 SGS

在 EXP0025 基础上继续修复。发现 `Ld = kd*(diffuse + sun_vis*sun_irradiance)`
是双线性形式（高容量逐点场 + 低自由度全局标量共享同一学习率），是已知会
导致震荡、非单调收敛的组合——正好解释了 EXP0025 的 200/400/600/1000 步
扫描为什么完全不是平滑曲线（1.71/0.83/1.33/6.62pp）。给 `sun_irradiance`
单独一个慢 10 倍的学习率（新增 `--sun_lr_scale`，默认 1.0 不影响任何既有
行为）后，整条曲线收紧到 [0.21, 1.74]pp，且 **200 步（与 EXP0019 完全
同预算）这一点上，两个留出视角都干净小于 SGS-Intrinsic**（本方法
+0.21/+0.28pp vs SGS −0.43/−1.20pp）——不再是 EXP0025 那种一个视角赢
一个视角输的混合结果。刻意没有对这个学习率再做网格搜索（只试了 0.1 这
一个值），因为这是有机制解释的架构修复，不是在一个极小场景上暴力调参；
曲线仍非单调、绝对反照率精度仍饱和，如实记录未隐藏。详见
[显式太阳项方法与结果](SUNPATCH_DAYLIGHT_METHOD_ZH.md)（已更新）、
`research/evidence/EXP0026/`、`DECISIONS.md` D0015。

## 2026-09-24：EXP0025 — 显式太阳项首次接入真实 IRIS 训练（T06 首个真实结果）

用户明确要求"不要问许可，自己计划自己做下去，想办法用我们的方法超过 SGS"。
在真实 `train_brdf_crf.py`（不是 EXP0004 的简化 2D 代理）里加入了一个显式、
几何驱动、联合学习的外部太阳辐照项（新脚本 `bake_sun_term.py`），复用
EXP0019 的确切场景/切分/预算做对照。

**过程如实记录了两次失败**：第一版参数化（`softplus` 从零初始化）让污染
反而**比原版 IRIS 差 3 倍**（+55.5pp，训练更久还更差，排除了"没训够"）；
诊断后发现 `softplus(0)≈0.69` 是 `diffuse` 烘焙真实量级的 ~50 倍，从第
0 步就淹没了正常光照。第二版改成从极小值初始化又导致梯度消失、参数
1000 步内基本不动，等于该项从未参与优化。第三版换成数值条件更好的平方
参数化后仍不起作用——根因定位到 `train_brdf_crf.py` 里反照率场完全没有
任何空间一致性正则（只有 metallic/roughness 有），优化器没有理由把亮度
差异分配给"免费"的新太阳项。加上一个和 metallic/roughness 同款的
反照率一致性正则后（仅在新太阳项存在时生效，不影响任何既有复现）：

**太阳导致的归一化亮暗差变化：原版 IRIS +18.4/+14.0pp → 本方法在
200/400/600 步测得 +1.71/+0.83/+1.33pp（v3 视角，v9 视角同量级）**，
400 步这一个点上 v9 视角（0.74pp）量级已经**小于** SGS-Intrinsic 的
−1.20pp，v3 视角（0.83pp）仍大于 SGS 的 −0.43pp——混合结果，不是干净的
"赢了"。四个预算点非单调（1000 步回退到 +6.62/+5.40pp），400 步是本次
探索性扫描里观测到的最优点，不是经过验证/调参确认的推荐配置，如实全部
列出未挑最好看的数字汇报。绝对反照率精度在所有测试预算下都有明显代价
（预测饱和），太阳方向仍是生成器真值（oracle），非几何/星历恢复；仅
1 房间 1 seed。详见
[显式太阳项方法与结果](SUNPATCH_DAYLIGHT_METHOD_ZH.md)，
`research/evidence/EXP0025/`；调试过程记为 `DECISIONS.md` D0014。

## 2026-09-23：EXP0024 — SGS-Intrinsic 基线配对结果补齐归档

核实发现 EXP0021 修好入口后，SGS-Intrinsic 的太阳光斑配对试验（已知几何 P1，
1 房间 1 seed，与 EXP0019 用同一 Country Kitchen 场景）其实早在 2026-09-20
就跑完并评测过，但从未归档进 `research/evidence/`、从未写进报告、
`baseline_registry.json` 的 `scores` 也一直是 `null`——本次补上全部四步。
原草稿把这个实验也编号为"EXP0022"，与今晚新分配的 Cali-HDR/Pano2Pano 审计
撞号，已改为 **EXP0024**（D0013）。

结果：两组（有太阳 sun_a / 无太阳 sky）在留出视角上，归一化亮暗差都是**负值**
（−1.4% 到 −2.6%，受光区比阴影区更暗，与太阳无关的基线暗化），太阳额外带来的
偏移很小（−0.43 到 −1.20 个百分点）——**符号相反、量级比原版 IRIS
（EXP0019：+14 到 +18 个百分点）小一个数量级**。样本极小（1 房间 1 seed）
不能当普遍结论，但提示:如果这个趋势在更大样本下站得住，"材质污染指标"
未必是我们方法明显赢过 SGS 的地方，真正的差异化优势更可能在 SGS 结构上
做不到的事——它的重打光是换任意环境贴图，不是基于星历的物理真实光照预测，
这正是 `NOVELTY_GAP.md` killer experiment 的核心论点。
详见 [SGS 太阳光斑对照](SGS_SUNPATCH_BENCHMARK_ZH.md)、
`research/evidence/EXP0024/evaluation/index.html`。

## 2026-09-23：EXP0022/EXP0023 — Cali-HDR + Pano2Pano（Guanzhou Ji 组）已获取并完成 EXIF 审计

用户获得并提供了 Cali-HDR Dataset（88.97GB，14 场景，2022–2023）与其后续
扩展 Pano2Pano_Release（80.21GB，9 场景，2024）——此前 `DATASET_AUDIT.md`
已识别但标记为"仅限邮件申请，未发邮件"的同一课题组数据集。
只读扫描（zip 中心目录 + 每张 JPG 前 256KiB，不解压 169GB 全量数据）发现：
3879 张 JPG 中 73.6% 带 EXIF GPS，55.9% 带罗盘朝向，坐标均落在匹兹堡。
用已在合成数据验证过的 `utils/solar_geometry.py`（pvlib）对 2853 张带
GPS+UTC 时间戳的图像计算真实太阳位置，35 个场景 0 例昼夜不一致，高度角
范围与匹兹堡纬度/季节物理吻合——这是纯自洽性检验，不是方法优势结论。
解锁了 `DATASET_AUDIT.md`"星历预测是否匹配真实观测"这一未决问题的
真实（非合成）数据来源；**不改变**该审计"无数据集同时具备多视角几何+
真实窗口+太阳斑"的核心结论——两者仍是单视点全景，不能替代 Phase 0 实地
采集。169GB 原始数据的复制解压已作为独立后台任务启动（不依赖本次会话），
完成后详见 `data_download/import_ji_datasets_status.json`。
像素级太阳斑视觉核验有意未做，留待下次会话决定框架。
详见 [Cali-HDR/Pano2Pano 获取报告](CALI_HDR_PANO2PANO_ACQUISITION_ZH.md)。

## 2026-09-20：EXP0019 太阳光斑数据与 IRIS 对照已完成

已生成固定哑光地板厨房数据（14 视角×4 照明，56 HDR），三组 IRIS 全 12 阶段完成。
修复并回归验证“训练到 200 步但交接 last 仅第 18 步”的保存问题。
太阳 A 的受光—阴影 albedo 差扣除无太阳对照后增加 14–18 个百分点；
太阳 B 未复现该局部指标，并出现 46 个非灯具三角面被提取为发光面。
仅为已知几何、中性先验、200 步、单房间受控诊断，非原论文默认配置或方法优势。
[报告](SUNPATCH_IRIS_BENCHMARK_ZH.md)；本地入口 `experiments/out/EXP0019_sunpatch/index.html`。

## 2026-09-20：EXP0018 原生材质诊断

已导出两房间四视角的 25 方向 BSDF 响应及几何/材质标识。
直接调用原版 IRIS BRDF 做真值拟合，并运行同模型对照；发现少量高光方向
预测仍不稳定，验收未通过。不能把拟合参数当作 GT，也不能据此声称太阳提升。
详见 [原生材质诊断](NATIVE_BRDF_COMPATIBILITY_ZH.md)。

## EXP0017 — full-material solar-render pilot (2026-09-20)

Two authored textured PBR rooms rendered under three known sun directions, two
cameras and two seeds (24 HDRs), plus four sky-only controls. First lateral camera
move failed in bathroom; preserved and corrected with forward motion and ray-distance
checks. Final low/high-SPP runs have matching cameras/lighting. 4096 vs 256 SPP
reduces paired-seed RMSE by 4.107x on average; minimum sun intervention / seed-noise
ratio is 15.615. Residual relative noise is 3.9–9.3%, so this is not a converged
high-precision BRDF benchmark. Exported eight train and four heldout-light HDRs
without sun/BRDF truth. No inverse method run yet. See `PBR_SOLAR_PILOT_ZH.md`.

## Active research workflow and EXP0016 (2026-09-20)

Current task archive: `ACTIVE_RESEARCH_TASK_ZH.md`, machine state `task_state.json`.
User capture contract: `REAL_CAPTURE_3DGS_ZH.md` (preserve photos/time/exposure/poses,
not only a 3DGS PLY). First InteriorVerse supervised material-prior pilot completed
on 82/14/4 scene-disjoint rooms, three fixed seeds. Compared to a weak constant
training-mean baseline: albedo MSE -29.4%, roughness -12.9%, metallic +0.3%.
Visual predictions are oversmoothed; prior quality gate failed and it is not adopted.
No solar-method or sim-to-real claim. See `MATERIAL_PRIOR_PILOT_ZH.md`.

Baseline correction: EXP0011 stage10 had no logged progress for about 13 hours;
the owned child was stopped with SIGTERM and the full-run prerequisite wait had
already timed out. Previous "running/queued" descriptions below are historical,
not current. Failure evidence preserved. New experiments use bounded watchdogs.
17 unit tests pass, including watchdog success and timeout termination.

## EXP0015 — InteriorVerse material data acquired (2026-09-20)

The user-supplied legacy URL returns 404. The author's August 2026 update endorses
the Lez/InteriorVerse backup. Downloaded and SHA-256/CRC-verified its first 2.11 GB
85-degree shard: 100 scenes, 5,988 EXRs. Inspected all six modalities for 26 views
from three scenes. This is synthetic material-map data, not real scans or a
verified solar-rerender asset bundle. No mesh/extrinsics/sun metadata in this shard;
the official README still marks spatially-varying lighting unreleased.
See `INTERIORVERSE_ACCESS_ZH.md` and `evidence/EXP0015/`.

## EXP0014 — real HDR acquisition and geometry audit (2026-09-20)

Downloaded complete meshes/calibration and 32-view HDR pilots for each of two
real Eyeful Tower scenes: riverview and apartment (64 HDRs, 48 train / 16 official
test views; 1.06 GB acquired). All file hashes rechecked, HDRs decoded, distortion
removed, camera round trips checked, and all 64 views ray-traced against the mesh.
Riverview has visible real daylight patches and is the selected first solar scene.
Interactive inspection: `experiments/out/EXP0014_eyeful/index.html`.
Report: `EYEFUL_PILOT_ZH.md`; evidence: `evidence/EXP0014/`.
This is data/geometry preparation, not recovered BRDF or solar rerendering.
TexIR and Cali-HDR remain access-by-request; official instructions saved, no emails sent.

## Current objective — photo-only solar/BRDF evaluation (2026-09-19)

The user clarified the target: generate many photorealistic interiors with retained
solar and material ground truth, hide that truth from estimation, and test whether
recovering explicit sun improves BRDF accuracy. Shadow IoU alone does not answer
this question. See `PHOTO_ONLY_SOLAR_BRDF_ZH.md` for the current protocol and
`evidence/EXP0013_preparation/asset_audit.json` for two acquired, CRC-checked complete
PBR asset packages. Those artist-authored scenes are engineering assets, not real
scans, and are not yet converted into verified solar datasets. Strict photo-only
and known-geometry diagnostic tracks must remain separate. No new BRDF result yet.

## EXP0012 — real scanned geometry rerendering (2026-09-19)

Completed 32 observations on the unchanged 4.22M-triangle ScanNet++ room
`1b379f1114`: four cameras, four simulated sun conditions, two seeds. Known
uniform diffuse material. Train-only direction error 0.486°, heldout direct-sun
proxy IoU 0.900 (oracle 0.980). Camera yaw ±1° reduces mean IoU to 0.822.
Full pipeline rerun reproduced all mask metrics. Not a full IRIS comparison or
real-photo sim-to-real result. See `SCANNETPP_RERENDER_ZH.md` and
`research/evidence/EXP0012/`.

# Project Status

Last updated: 2026-09-19 (EXP0008–EXP0010).

## Baseline progress — EXP0011 (2026-09-19)

Local CUDA 12.8 / GCC 13 toolchain and tiny-cuda-nn now work on RTX 5070 Ti.
Official bathroom data (109 train / 13 validation images) and checkpoint are
downloaded with ZIP CRC verification and SHA-256 manifests. Official-checkpoint
rendering and three actual initialization optimizer steps have passed. Full-split rendering passed: PSNR 28.97625 / SSIM 0.79503. Original BRDF/CRF
and emitter optimization also passed three steps each. The bounded full pipeline
is running stage 10 in a detached process; a separate detached full 6/4/1/4-epoch
run is queued to start only after all bounded stages succeed. This is not yet a
converged from-scratch reproduction. Real cross-time photos are still
missing. See `research/evidence/EXP0011/` and `CROSS_TIME_CAPTURE_ZH.md`.

## Current verified status — 2026-09-19 (EXP0008–EXP0010)

This supersedes conflicting historical status below. Full IRIS baseline is still
**not reproduced at this earlier checkpoint**. GPU access works outside the sandbox (RTX 5070 Ti, 16,303 MiB,
driver 581.80); missing build dependencies were subsequently resolved in EXP0011. CPU component
experiments are functional and cheap.

- Broad screening: 53 scenes / 424 images / 1,141 candidates; one additional scene
  skipped for mismatched geometry bounds. This is screening, not labeled accuracy.
- Real scene `1b379f1114`: direction fitted to one image, evaluated on three frozen
  disjoint views. Mean brightness-proxy IoU 0.709; static appearance memory 0.833.
  **No demonstrated same-light advantage over static appearance.**
- Implicit mesh-opening control obtains 0.624 with a substantially different sun
  direction. Its visibility convention differs; this is not clean causal evidence
  for window labels and no real angular ground truth is available.
- Controlled changed-light test: 3 off-grid conditions x 2 render seeds. Mean
  angular error 0.430 degrees; changed-light proxy IoU 0.951 vs 0.148 frozen
  appearance. Known mesh/material and supplied relative sun change: component
  validation only, not full IRIS or real relighting.
- 13 tests pass; one-command evidence cycle records unique run directories,
  logs, configurations, exact source snapshots, failures and negative findings:
  `.venv/bin/python experiments/run_daylight_validation.py`.

Reports: `REAL_SUN_VISIBILITY.md`, `HELDOUT_SUN_CONTROL.md`. Machine-readable
records: `research/evidence/EXP0008`, `EXP0009`, `EXP0010`, `EXP0007_expanded`.

Remaining scientific gates: independently calibrated windows and sun observations;
isolate exterior-occlusion effects; real heldout illumination; original IRIS
baseline and joint material/light optimization. No formal certification or new
novelty claim has been made. Existing synthetic passes are component evidence.

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

1. ~~Verify IRIS's own `utils/path_tracing.py` against drjit 1.5.0~~ **Done, EXP0003/D0008** — found and fixed two real API breaks (one silent-shape-transpose bug), verified geometrically correct, not just error-free.
2. ~~Identifiability mechanism test~~ **Done, EXP0004, PASSED** — `IDENTIFIABILITY_ABLATION.md`. Controlled proxy experiment (not full IRIS) directly supports Contribution One: the solar constraint improves recovered albedo (2.3x / 1.8x RMSE reduction across two sub-tests) while *not* improving pixel-fit error — the improvement is in disentanglement, not reconstruction, which is the core defensible claim.
3. ~~Real-dataset accessibility investigation + a real-photo screening tool~~ **Done, EXP0005** — `REAL_PHOTO_SCREENING.md`, `DECISIONS.md` D0009. No autonomously-fetchable real posed-multiview+geometry dataset found (box.com, ScanNet++, IVGM's Baidu Pan bulk archive, and a HF aggregator all checked and gated). Built and honestly evaluated the 2D sun-patch screening component on real photos instead: works better than brightness-only on 3/4 positives, with two found limitations (mullion-grid fragmentation; can't always distinguish a light source from a lit surface without 3D geometry) — the latter is itself informative evidence for why Mode B needs real geometry, not just appearance cues.
4. Broaden the Phase A/B synthetic sweep (multiple sun angles, material-estimation-error sensitivity) and the identifiability ablation (noise/config sweep) — cheap, addresses "single favorable data point" concerns, doesn't depend on the user.
5. **Depends on the user now**: either (a) complete ScanNet++ registration, (b) attempt/facilitate the IVGM Baidu Pan download (they may have better access to it than this environment), or (c) the Phase 0 real-room capture (`DATASET_AUDIT.md`) — any one unblocks the actual geometry-grounded real-data comparison, which is the evidence that would really matter for a paper (re-running the identifiability test with IRIS's actual `NGPBRDF`, running the killer experiment).
6. Phase D: joint optimization integrated into IRIS's actual training stages (GPU confirmed working, core ray-intersection primitive confirmed compatible — the remaining engineering, `model/external_lighting.py` / `model/transmission.py` per `TECHNICAL_PROPOSAL.md`, not yet built).

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
