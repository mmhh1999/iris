# IRIS Architecture Audit

Status: initial audit, pre-modification. Read-only inspection of `main` @ `d2d4381` (branch point for `research/daylight-aware-iris`).

Purpose: establish, with file:line evidence, exactly how IRIS represents illumination and materials, and precisely where/why it cannot account for outdoor daylight entering through windows. This document is the basis for the daylight-aware extension proposed in `research/NOVELTY_GAP.md`.

## 1. Pipeline overview (9 training stages + rendering, per README)

| # | Script | Produces | Role |
|---|---|---|---|
| 1 | `slf_bake.py` | `checkpoints/{exp}/bake/vslf.npz` | Bake voxel Surface Light Field (SLF) from LDR training images |
| 2 | `extract_emitter_ldr.py` (export) | `checkpoints/{exp}/bake/emitter.pth` | Flag mesh triangles as emitters by saturated-pixel statistics |
| 3 | `initialize.py` | `init.ckpt` | Jointly initialize BRDF (`NGPBRDF`) + emitter radiance (`SLFEmitterLearn`) |
| 4 | `extract_emitter_ldr.py` (update) | updated `emitter.pth` | Copy learned emitter radiance back into the emitter file |
| 5 | `bake_shading.py` | `outputs/{exp}/shading` | Bake per-pixel diffuse/specular shading maps from SLF |
| 6 | `train_brdf_crf.py` | `last_0.ckpt` | Optimize BRDF + camera CRF against cached shading + LDR images |
| 7 | `slf_refine.py` | `vslf_0.npz` | Re-bake SLF using the learned CRF (LDR→HDR inversion) |
| 8 | `train_emitter.py` | `last_0_emitter.ckpt` | Refine emitter radiance via full path tracing (BRDF/CRF frozen) |
| 9 | `refine_shading.py` | refined shading maps | Re-bake shading with full recursive path tracing (`indir_depth=5`) |
| — | `render.py` | RGB/BRDF/emission maps | Render train/test/video splits |
| — | `render_relight.py` | relit videos | Mitsuba-native scene-dict rendering with edited/added emitters |

Every stage loads scene geometry as a single opaque triangle mesh:
```python
mitsuba.load_dict({'type': 'scene', 'shape_id': {'type': mesh_type, 'filename': mesh_path}})
```
repeated identically in `bake_shading.py`, `refine_shading.py`, `slf_bake.py`, `slf_refine.py`, `extract_emitter_ldr.py`, `train_brdf_crf.py`, `train_emitter.py`, `render.py`, `initialize.py:57-63`, and `render_relight.py` (mesh passed into `main_scene`). There is exactly one shape in the scene graph; no separate glazing/window shape ever exists.

## 2. Illumination model — what exists

IRIS represents outgoing radiance as the sum of exactly two terms, both **local to reconstructed geometry**:

1. **Mesh-triangle area emitters** (`model/emitter.py:15-131`, `AreaEmitter`): a boolean per-triangle `is_emitter` mask with per-triangle constant RGB `radiance`. Learnable via `SLFEmitterLearn.radiance = nn.Parameter(...)` (`model/emitter.py:257-269`).
2. **Surface light field (SLF)** (`model/slf.py:16-70`, `VoxelSLF`): a sparse voxel grid caching accumulated diffuse radiance leaving *occupied* voxels (i.e. on/near the reconstructed surface), used as a path-tracing termination shortcut for rough/diffuse bounces (`model/emitter.py:180-221`, `trace_roughness` gate) and, in `bake_shading.py`/`refine_shading.py`, as a cached irradiance source.

Both are queried only when a ray successfully intersects scene geometry. `SLFEmitter.forward`/`eval_emitter` and `AreaEmitter.forward`/`eval_emitter` are the **sole sources of outgoing radiance** in the renderer.

## 3. Illumination model — what is absent (the core finding)

**There is no environment/sky/background radiance term anywhere in the codebase.** Verified by:

- `utils/path_tracing.py:ray_intersect` (`L17-48`): a ray that does not hit the mesh gets `idx[~valid] = -1` (`L46`) — it is simply marked invalid, carrying no direction-dependent radiance.
- `model/emitter.py`, `AreaEmitter.eval_emitter` (`L69-98`): `vis = triangle_idx != -1`; `Le = Le*vis[...,None]`, with the **literal comment "assume zero background lighting"** (`L65`, `L93`). Any ray with `triangle_idx == -1` (missed the mesh) contributes exactly `Le = [0,0,0]`.
- `model/emitter.py`, `SLFEmitter.eval_emitter` (`L180-221`): identical structure, identical comment (`L205`), identical zero-radiance behavior for missed rays.
- `utils/path_tracing.py`, `path_tracing()` (`L214-318`) and `trace_indirect()` (`L409-502`): every bounce terminates on `valid_next` derived from `eval_emitter`'s `vis` flag; there is no fallback branch, no `mitsuba` `envmap`/`constant`/`directional` emitter ever instantiated, and no code path that queries anything for a ray direction once geometry is missed.
- Repo-wide grep (performed by a delegated sub-audit, corroborated) for `sun|sky|environment ?map|envmap|outdoor|window|daylight|sunlight|hdri|ibl` across all `.py/.yaml/.yml/.md/.sh/.json`: **zero relevant hits**. The only matches are an unrelated CLI arg `--window` (temporal smoothing width in `utils/video.py:128,145`).
- `render_relight.py`'s `light_cfg` YAML is a literal Mitsuba scene dict (`OmegaConf.load` → `mitsuba.load_dict`), so Mitsuba's built-in `envmap`/`directional`/`constant` emitter plugins are *technically reachable* by hand-authoring a YAML entry — but no shipped config, script, or training stage ever constructs one, and (critically) the custom `FIPTBSDF` used for the room mesh only declares `mitsuba.BSDFFlags.DiffuseReflection` (`model/fipt_bsdf.py:41-43`), so even a hand-added envmap would only illuminate other Mitsuba-native shapes in the scene, not interact with the learned room BRDF through any code path that was designed for it.

**Consequence:** a ray traced from a camera, through a window opening, into open space (sky) returns pure black. The only way IRIS's optimization can explain an observed bright pixel behind/through a window — a sun patch on the floor, a bright sky seen through glass — is by pushing intensity into one of the two local terms above, at the point where the ray *does* terminate on real geometry (the floor, wall, or window-frame mesh itself).

## 4. How daylight actually gets absorbed today (failure mode mechanics)

Traced two concrete absorption pathways, both confirmed in code:

**(a) Emitter misclassification.** `extract_emitter_ldr.py` (`--mode export`, pipeline stage 2) computes, per mesh triangle, the mean-over-training-frames max-channel **raw LDR pixel value** (not inverse-CRF-mapped to HDR — unlike `slf_bake.py`/`slf_refine.py`, which do call the CRF inverse before accumulating) and flags `is_emitter = triangle_radiance_mean > threshold` (default `threshold=0.99`, i.e. saturated/near-white pixels). A sun patch on a floor that appears overexposed in enough training frames triggers exactly this criterion. That triangle then receives a learned constant `emitter_radiance` (refined in stages 3, 4, 8) — i.e. **the floor is modeled as physically glowing**, not as a reflective surface lit by an external directional source. `utils/extract_emitter_mesh.py` later exports these flagged triangles as a standalone emitter mesh for object-insertion/relighting, so a misclassified sun patch becomes a literal permanent light fixture embedded in the floor geometry.

**(b) SLF/albedo absorption for sub-threshold brightness.** Any bright-but-not-saturated daylight-lit region (soft shadow penumbra, indirect bounce light from a sun patch, a moderately bright sky seen through glass) falls under the `0.99` threshold and is instead fit through the ordinary material/SLF path: `NGPBRDF` can push `albedo` toward 1 and `metallic`/`roughness` to whatever combination best reconstructs the pixel (`model/brdf.py:243-260`), and `VoxelSLF` can independently cache a nonzero outgoing-radiance value for that surface patch (`model/slf.py:56-61`, `scatter_add`) that has no physical decomposition into reflectance × incident illumination at all — it is a raw radiance cache. Both are geometry-local, camera-view-averaged quantities baked once during training; neither has any notion of solar direction, time of day, or window transmittance, so **the fit is only valid for the lighting condition(s) present in the training images** — changing the sun in a relit render cannot correctly move, dim, or remove the effect, because the effect was never represented as "external light modulated by geometry," only as "this triangle/voxel emits/reflects this fixed RGB."

## 5. BRDF model — confirmed purely opaque/reflective

`model/brdf.py`:
- `NGPBRDF.forward` (`L243-260`) outputs exactly 5 scalars per queried 3D position: `albedo` (3ch, sigmoid), `roughness` (1ch, sigmoid → `[0.02, 1]`), `metallic` (1ch, sigmoid). No opacity/alpha, no IOR, no transmission channel.
- `BaseBRDF.eval_brdf` (`L138-175`): standard metallic-roughness split — `kd = albedo*(1-metallic)`, `ks = 0.04*(1-metallic) + albedo*metallic` — combined with Lambertian diffuse + GGX (`D_GGX`) specular with Smith geometry term and Schlick Fresnel. This is a textbook opaque reflective BRDF; there is no transmission lobe, no refraction, no thin-dielectric path anywhere in the file (confirmed by grepping `ior|transmi|refract|opacity|dielectric` in `model/brdf.py`, `model/emitter.py`, `model/fipt_bsdf.py` — the only hit is the unrelated GGX roughness variable named `alpha`, `model/brdf.py:46,49`).
- `model/fipt_bsdf.py:41-43`: the Mitsuba-side wrapper `FIPTBSDF` declares only `mitsuba.BSDFFlags.DiffuseReflection` (with `SpatiallyVarying`/`FrontSide`/`BackSide`) — confirming the same purely-reflective assumption is enforced at the renderer-integration layer, not just the PyTorch BRDF module.
- **Implication for windows:** whatever geometry a window/glass pane occupies in the reconstructed mesh (if it is represented as a closed surface at all — see §6), IRIS has no material state capable of expressing "this surface transmits most incident light and reflects a little" — it can only express "this surface reflects `albedo` with roughness/metallic X." A glass pane is therefore forced to fit as either a near-black opaque diffuse surface, a mirror-like high-specular surface, or (if bright) an emitter — none of which is a transmissive aperture.

## 6. Window / opening handling — none exists

No code anywhere treats "window" as a semantic or geometric concept (confirmed by the repo-wide grep in §3). Scene geometry is always the single reconstructed mesh (`scene.obj` for synthetic/real via BakedSDF/SDFStudio, `scans/scene.ply` for ScanNet++, per README "Customized Data" section). Two possible outcomes follow purely from what the external (not-in-repo) mesh-reconstruction pipeline produced, and IRIS's behavior differs sharply between them but is blind to which one occurred:

- If the SDF reconstruction left the window opening as a genuine hole (common — SDF/NeRF-style reconstructions frequently fail to close thin/transparent apertures), camera rays through it become `triangle_idx == -1` misses → §3's zero-radiance path applies, and any daylight actually visible through that window in the photos must be explained entirely by whatever opaque geometry is hit *beyond* the window (if any) or is otherwise lost/unexplained.
- If the reconstruction instead filled the opening with solid geometry (also common — implicit surface methods often hallucinate a wall/frame across a window because glass provides no photometric or geometric signal for the underlying SDF), that surface is treated as an ordinary opaque material like any wall, and per §4 will absorb the daylight it should be transmitting into the emitter/albedo/SLF terms of whatever floor/object is downstream of it in view.

Either way, **there is no first-class window entity, no transmission model, and no way for the optimizer to distinguish "opaque bright wall" from "aperture transmitting outdoor light."**

## 7. Dataset loaders — no solar/temporal metadata

Checked `utils/dataset/real_ldr.py`, `utils/dataset/synthetic_ldr.py`, `utils/dataset/scannetpp/dataset.py`. Every loader reads exactly: camera extrinsics/intrinsics, the LDR image, an exposure scalar (either `None`, loaded from a precomputed `cam/exposure.npy`, or — for ScanNet++ — hardcoded to `1.0` for every frame, `scannetpp/dataset.py:91`), and (synthetic only) GT albedo/roughness/emission/segmentation, or (real/ScanNet++ inverse-training variants) a segmentation mask + IRISFormer/RGB-X "intrinsic albedo" prior image. **No EXIF, timestamp, GPS/geolocation, date/time-of-day, or solar-position metadata is loaded or referenced anywhere.** This means even the *cheap* Mode-A solar prior described in the project brief (metadata-derived sun direction from capture time + location) has zero support in current data loading — it would need to be added at the dataset-adapter layer, and would require it to actually exist in whatever dataset is chosen (see `research/DATASET_AUDIT.md`).

## 8. Camera response function (CRF) — bounded, shared-per-scene, unlikely sole explanation

`crf/model_crf.py`, `EmorCRF`: a low-dimensional EMoR (Empirical Model of Response) basis — fixed mean curve `f0` (1024-sample LUT) plus a learnable `weight` of shape `(3, dim)` (`dim` = `--crf_basis`, typically small, e.g. 3–11) added against fixed PCA basis vectors: `get_crf() = f0 + weight @ basis`. This is a **single shared monotonic tone curve per whole scene/experiment** (regularized via `reg_monotonically_increasing`/`reg_weight`, used in `train_brdf_crf.py`), not a per-image or per-pixel function. It can globally remap intensity nonlinearly but cannot, by construction, carve out an isolated spatial highlight — so it is not a primary suspect for absorbing a spatially localized sun patch, though it does interact with exposure (which itself is a fixed or precomputed-per-image scalar, never physically metered) in ways worth checking empirically once training is reproducible.

## 9. Existing relighting capability — indoor-only

`render_relight.py` + Mitsuba-native `light_cfg` YAML supports: moving/rescaling/re-coloring existing indoor area-light-style emitters (`configs/scannetpp/bathroom2/relight_0.yaml`, a translated/scaled emissive sphere), an animated multi-light "disco ball" (`utils/disco_ball.py`, `relight_1.yaml`), and inserting extracted emitter-mesh geometry or reflective props (`insert.yaml`, `utils/extract_emitter_mesh.py`). All of these edit or add **local, indoor, geometry-attached emitters**. Nothing in the schema, tooling, or shipped examples constructs a directional/sun or environment/sky emitter, and — per §5 — even if one were hand-authored, the room's own `FIPTBSDF` material would not correctly respond to it as daylight (no transmission, and the BRDF network was never trained under varying external illumination).

## 10. Summary: where the daylight modeling gap lives, concretely

| Gap | Location | What's missing |
|---|---|---|
| No sky/environment radiance | `model/emitter.py` (`eval_emitter`, both classes), `utils/path_tracing.py` (`ray_intersect`, `path_tracing`, `trace_indirect`) | A radiance function of ray direction (and optionally position) evaluated when `triangle_idx == -1`, instead of the hardcoded zero |
| No sun/solar geometry | Nowhere in repo | Any representation of a directional light source, its azimuth/elevation, or capture-time solar position |
| No window/aperture entity | Scene loading (`mitsuba.load_dict` call sites, listed §1) | A first-class transmissive-surface type, separate from the generic room mesh, with its own BSDF and confidence/geometry |
| No transmission in BRDF | `model/brdf.py` (`NGPBRDF`, `BaseBRDF`), `model/fipt_bsdf.py` (`FIPTBSDF` flags) | IOR, transmission coefficient, alpha/opacity, or thin-dielectric sampling/eval paths |
| No solar/temporal metadata | `utils/dataset/*.py` loaders | EXIF timestamp, geolocation, orientation fields in the dataset adapters |
| Emitter extraction conflates sun patches with fixtures | `extract_emitter_ldr.py` (`is_emitter` threshold on raw saturated LDR pixels) | A criterion that can distinguish "physically emissive surface" from "surface receiving strong external illumination" |

This audit supports the project brief's core hypothesis directly: IRIS's illumination model has no notion of anything outside the reconstructed mesh, so all daylight effects are currently forced into local, direction-agnostic, geometry-baked terms (emitter radiance, SLF cache, or material albedo) that cannot be decomposed back into "external light × transmittance/reflectance" — which is exactly why relighting under a changed sun condition is not physically meaningful today, even though ordinary reconstruction PSNR on daylight scenes may look fine.

## 11. Note on tool-use hygiene during this audit

Part of this audit (§1–§9's file-by-file detail beyond `brdf.py`/`emitter.py`/`slf.py`/`fipt_bsdf.py`/`path_tracing.py`/`initialize.py`, which were read directly) was gathered via a delegated read-only sub-agent. That sub-agent's tool results contained an injected instruction (disguised as an "MCP Server Instructions" block) attempting to redirect it into creating an external "Claude Docs" document instead of returning its findings as text. The sub-agent correctly identified this as a prompt injection, ignored it, and returned its findings as plain text as instructed. No document was created and no data left this repository as a result. Recorded here for traceability.

## 12. Direct answers to the eight audit questions (2026-09-27)

Evidence is the code references in §§2–9 plus our own IRIS runs (EXP0019, EXP0027).
"Measured" means we have a number; "code" means the code path allows it but we have
no measurement.

1. **What is an emitter?**
   - A mesh triangle flagged `is_emitter` by `extract_emitter_ldr.py`: its mean raw
     LDR max-channel value over the training frames exceeds 0.99.
   - It gets one learnable constant RGB radiance (`AreaEmitter`, `SLFEmitterLearn`),
     emitted diffusely (the same radiance in every direction).
   - Nothing else emits.
2. **What happens at windows?**
   - A window is not an entity. It is either a hole in the mesh, so rays miss (Q3),
     or a filled opaque surface.
   - If a filled window pane is saturated in the LDR images, it becomes a Lambertian
     emitter by Q1, i.e. a diffuse glowing panel.
   - A diffuse panel cannot produce a sharp, direction-dependent sun patch. Our
     testbed arm W is exactly this model: its best fit with true materials reaches
     patch IoU 0.15–0.24 (EXP0030 D2).
3. **Rays leaving the mesh?** They return zero radiance (`eval_emitter`, comment
   "assume zero background lighting"). No fallback exists.
4. **Is environment illumination represented?** No. There is no envmap, sky, sun or
   constant background term anywhere (§3). Mitsuba plugins are reachable only by
   hand-editing the relighting YAML, and the room BSDF was never trained against them.
5. **Can sunlight bake into the SLF?**
   - Yes (code). `VoxelSLF` caches outgoing radiance of occupied voxels, including
     sunlit floor, and serves it as incident light for other surfaces' indirect bounces.
   - The patch itself is still fit by albedo/emitter on the floor (Q6); the SLF then
     propagates that baked brightness as if it were a property of the floor.
6. **Can sun patches influence albedo?** Yes (measured). Vanilla IRIS on a uniform
   floor: the sun-induced extra albedo gap is +19.5 / +14.3 percentage points
   (EXP0019, EXP0027; 3 seeds).
7. **Can shadow influence roughness / metallic?**
   - Possible (code): roughness and metallic are per-location `NGPBRDF` outputs trained
     by the same photometric loss.
   - Measured only as floor means in EXP0027: roughness 0.75–0.76 against a true 0.90;
     metallic about 0.003 against a true 0. That bias is not attributed to the sun.
   - No sun-vs-shadow split has been measured. **Open.**
8. **Scene-specific vs globally interpretable information?**
   - Everything IRIS learns about light is scene- and condition-specific: per-triangle
     emitter radiance, the voxel radiance cache, one CRF per scene.
   - None of it carries a direction, a time, or a source outside the room, so it cannot
     be transferred to another sun position.
   - The only quantities intended to be condition-invariant are the BRDF fields, and
     Q6 shows they absorb the sun.
