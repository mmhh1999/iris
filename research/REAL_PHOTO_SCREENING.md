# Real-Photo Sun-Patch Screening (EXP0005)

Status: **first pass, mixed and informative result — not a full validation.** Code: `utils/sun_patch.py::screen_image_for_sun_patches`, `experiments/real_photo_screening.py`.

## Why this experiment, and what it is not

The user asked: can we find a real-world dataset, automatically screen it for images containing sun patches, and compare those against our method? Investigated the accessible-dataset landscape first (see `DECISIONS.md` D0009): the best content match (IVGM, explicitly documents "sunlight shines through large glass windows" in two office scenes, with posed panoramic imagery + point cloud + mesh) is gated behind a 282GB unsegmented Baidu Pan bulk download not practical to fetch from this sandboxed environment; ScanNet++ (the large real corpus IRIS itself uses) requires an account/application process only the user can complete. Neither blocks progress entirely, but both mean **the full geometry-grounded comparison the user asked for is not yet possible** — that needs a real reconstructed mesh (floor plane, window aperture) to run `utils.window_geometry`/`utils.sun_patch.search_sun_direction` against, which these uncurated single photos don't have.

What **is** possible without that: build and test the *first stage* of a real-data pipeline — a 2D, geometry-free screening pass that flags candidate sun-patch regions in arbitrary photos, meant to prioritize which images/scenes are worth the geometric analysis once a real posed-multiview+mesh dataset is available (whether IVGM, ScanNet++, or a Phase 0 capture). This experiment builds and honestly evaluates that screening step on real, freely-licensed photographs — not synthetic renders, and not a stand-in for the full method comparison.

## Method

`utils.sun_patch.screen_image_for_sun_patches`: adaptive brightness thresholding (percentile-based, not an absolute cutoff, since real photos vary widely in exposure) → connected components → per-component shape scoring combining **rectangularity** (contour area / minimum-area-rectangle area) and **polygon simplicity** (vertex count after `approxPolyDP` simplification, rewarding near-quadrilateral shapes) and area, with a soft penalty for regions touching the image border. This is deliberately a different test than IRIS's own `extract_emitter_ldr.py` criterion (brightness alone, confirmed in `IRIS_ARCHITECTURE_AUDIT.md` §4 to misclassify sun patches as emitters) — the shape features specifically target the fact that a window-cast patch is a quadrilateral with comparatively straight edges, which most other bright-region causes (specular highlights, light fixtures, a blown-out window itself) are not.

## Test images

6 real, freely-licensed photographs (sourced via Openverse, an open-license image search aggregator; original files hosted on Wikimedia Commons / Flickr — see `experiments/real_photo_screening.py` for exact source URLs), manually labeled by visual inspection (label used only for reporting here, not seen by the algorithm):
- `real1.jpg` — a window-mullion grid patch cast on a tile floor (hard edges), real EXIF timestamp `2025:08:01 09:37:04` present (no usable GPS — stripped/absent, so no ephemeris cross-check possible from this image alone).
- `neg_candidate_2.jpg` — two clean rectangular patches on a reflective epoxy kitchen floor.
- `neg_candidate_3.jpg` — a softer, diagonal patch cast through mall entrance glass doors (harder case: less axis-aligned, softer edges).
- `neg_final.jpg` — a clean rectangular patch on carpet, source window visible but dim (mostly blocked by blinds).
- `neg_final2.jpg` — a bright window (venetian blinds) with **no distinct cast floor patch** — the intended hard negative.
- `neg_candidate_1.jpg` — an outdoor photo (not applicable / sanity check that the tool doesn't crash on a non-matching scene).

## Results (honest, including what didn't work)

| Image | Top score | Top rectangularity | Outcome |
|---|---|---|---|
| `real1.jpg` | 0.68 | 0.75 | **Partial success, known limitation**: correctly finds patch fragments, but the window-mullion shadow lines split the true patch into several disconnected components (connected-component analysis has no notion that adjacent bright cells belong to one grid pattern) — no single top candidate captures the whole patch. |
| `neg_candidate_2.jpg` | 0.73 | 0.85 | **Clean success**: top candidates correctly identify the two floor patches; the two windows visible in the same frame do not out-score them. |
| `neg_candidate_3.jpg` | 0.61 | 0.81 | **Success on a harder case**: correctly finds the diagonal patch despite non-axis-aligned, softer edges. |
| `neg_final.jpg` | 0.66 | 0.70 | **Clean success**: correctly identifies the floor patch over the dim window. |
| `neg_final2.jpg` (hard negative) | 0.66 | 0.71 | **Informative failure**: the top-scoring "candidate" is the window aperture itself (bright venetian blinds, itself fairly rectangular), not a floor patch — because there genuinely isn't a strong cast patch in this image, and the tool has no notion of "this region is a light source, not a receiving surface" without knowing which mesh triangles are the window vs. the floor. |
| `neg_candidate_1.jpg` (outdoor, N/A) | 0.56 | 0.61 | Lower score than most indoor positives, as expected, though the tool has no explicit indoor/outdoor gate — runs without crashing, flags *something*, appropriately low-confidence. |

Raw output and annotated visualizations: `experiments/out/real_photo_test/` (`screening_results.json`, `screened_*.jpg`).

## Interpretation

**Real, honest signal that the shape-based approach is meaningfully better than a brightness-only criterion**, on real photographs: 3 of 4 true positives are cleanly identified as the top candidate over other bright regions (including visible windows in the same frame) using only rectangularity/shape evidence — something IRIS's own brightness-only emitter criterion cannot do by construction. This is qualitative, real-world support for the audit's diagnosed failure mode being addressable.

**Two genuine, specific limitations found, not swept under the rug:**
1. **Mullion/grid fragmentation**: a gridded window casts a *set* of small rectangles, not one contiguous patch, and per-component connected-component analysis doesn't recognize them as one coherent group. Fixable in principle (morphological closing before component analysis, or detecting a *cluster* of aligned small rectangles as a single higher-level "gridded patch" object) but not yet implemented — flagged as a concrete follow-up, not treated as solved.
2. **2D screening alone cannot always distinguish a light source (window aperture) from a receiving surface (floor/wall) it illuminates** — both can be bright and roughly rectangular. This is not a bug to fix within the 2D screening step; it is the precise reason the full method needs real 3D geometry (a floor plane distinct from a window plane, from an actual reconstructed mesh) rather than relying on appearance cues alone, which is exactly what `utils.window_geometry`/`utils.sun_patch.search_sun_direction`'s Mode B analysis is for. This screening step's honest role is to **prioritize candidate images for that geometric analysis**, not to replace it — worth stating plainly since it would be easy to overclaim otherwise.

## What this experiment does NOT show

- Does not test the geometry-grounded Mode B comparison the user actually asked for (predicted patch vs. observed patch, sun direction recovery) — that needs a real posed-multiview scene with reconstructed geometry, still blocked on dataset access (see `DECISIONS.md` D0009).
- Small, informally-curated sample (6 images, hand-picked via search, not a systematic benchmark) — not a precision/recall claim at scale.
- No comparison "against our method" in the sense of running IRIS's own emitter-extraction pipeline on these images, since that requires multi-view + mesh, not single photos.

## Next steps

1. Get access to a real posed-multiview + geometry dataset — either the user pursuing ScanNet++ registration, help fetching IVGM's Baidu Pan archive, or the Phase 0 capture (`DATASET_AUDIT.md`) — any one of these unblocks the actual comparison.
2. Once available: run this screening tool across the full scene corpus to prioritize candidates, then run the real Mode B geometric analysis (`search_sun_direction`) on flagged scenes using their actual reconstructed floor/window geometry — this is the experiment that would actually answer the user's question.
3. Fix the mullion-fragmentation limitation (a clustering step over nearby small rectangular components) before relying on this screener for gridded-window scenes specifically.
