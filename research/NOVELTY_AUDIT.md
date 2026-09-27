# Novelty audit — calibrated solar state for indoor multi-view inverse rendering

2026-09-27. Supersedes the novelty statements in `NOVELTY_GAP.md` and extends
`LITERATURE_AUDIT_2026-09-27_ZH.md`. Every row was checked this session against the
link given. The "Verified" column says how much of the paper was read.
Nothing here is cited from memory alone.

**Verified levels:** `full` = full text read; `abs` = abstract / project page /
proceedings page; `snip` = search-result text only; `ref` = known only as a
reference inside another verified paper.

**Column key:** Sun = explicit sun term. Eph = sun direction computed from time +
place, or taken from metadata. Win = window aperture modelled. BRDF = materials
recovered. MV = multi-view. MT = multiple illumination times used jointly.
XT = evaluated against real images at other times. Y = yes, N = no, P = partial,
? = not determinable from what was read.

## A. Indoor inverse rendering (the family we would extend)

| Paper | Link | Ver. | Task / input | Sun | Eph | Win | BRDF | MV | MT | XT | Closest overlap | Remaining difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IRIS (Lin et al., CVPR 2025) | arXiv [2401.12977](https://arxiv.org/abs/2401.12977) | full (code) | Indoor IR from multi-view LDR + mesh; emitters on mesh + voxel SLF + CRF | N | N | N (bright windows become emitters) | Y | Y | N | N | Our base system | No outside light at all; misses return 0 (`IRIS_ARCHITECTURE_AUDIT.md`) |
| FIPT (Wu et al., ICCV 2023) | arXiv [2304.05669](https://arxiv.org/abs/2304.05669) | abs | Indoor IR from multi-view HDR + geometry; factorized path tracing; error-driven emitter detection | N | N | N | Y | Y | N | N | IRIS's predecessor | Same as IRIS |
| TexIR (Li et al., CVPR 2023) | arXiv [2211.10206](https://arxiv.org/abs/2211.10206) | abs | Large real indoor scenes; "texture-based lighting" (HDR textures on the mesh) | N | N | P (emissive textures) | Y | Y | N | N | Real multi-view indoor IR with windows in view | Window light is an HDR texture, not directional sun; one condition |
| SIR (2024) | arXiv [2402.06136](https://arxiv.org/abs/2402.06136) | abs | Indoor multi-view HDR; decomposable shadows under intense indoor light | N | N | N | Y | Y | N | N | "Shadows contaminate BRDF, model them explicitly" | Lamps, not sun; one condition |
| SGS-Intrinsic (CVPR 2026) | arXiv [2603.27516](https://arxiv.org/abs/2603.27516) | abs | Sparse-view indoor IR, 3DGS; deshadowing + illumination-invariant material constraint | N | N | N | Y | Y | N | N | Sun-patch contamination handled by a learned deshadow model (strong on our EXP0024) | No physical sun; one condition |
| IR-HGP (CVPR 2026) | [papernotes summary](https://en.papernotes.org/CVPR2026/3d_vision/ir-hgp_physically-aware_gaussian_inverse_rendering_for_high-illumination_scenes_/) (arXiv ID not located) | snip | 3DGS IR for high-illumination scenes; hybrid visibility decomposition + generative illumination prior | ? | N | ? | Y | Y | N | N | Baked-in shadows/highlights | Generative prior, not a physical solar state |
| AEGIR (2026) | arXiv [2606.28635](https://arxiv.org/abs/2606.28635) | abs | Indoor IR, explicit area emitters in 3DGS | N | N | N | Y | Y | N | N | "Explicit physical light sources help indoor IR" | Interior fixtures, no sun |
| GLOW (2025) | arXiv [2511.22857](https://arxiv.org/abs/2511.22857) | abs | Indoor multi-object IR with co-located moving light + camera | N | calibrated point light | N | Y | Y | Y (moving flash) | N | **Calibrated, varying illumination disambiguates indoor materials** | Active flash, not the sun; no windows |
| LuxRemix (CVPR 2026) | arXiv [2601.15283](https://arxiv.org/abs/2601.15283) | abs | Indoor multi-view, generative one-light-at-a-time decomposition, relightable 3DGS | N | N | N | N (appearance) | Y | N | N | Indoor relighting | Generative, no physical sun |
| MAIR (CVPR 2023) | arXiv [2303.12368](https://arxiv.org/abs/2303.12368) | abs | Multi-view indoor IR with 3D lighting volume | N | N | N | Y | Y | N | N | — | — |

## B. Indoor scenes with explicit sun / windows

| Paper | Link | Ver. | Task / input | Sun | Eph | Win | BRDF | MV | MT | XT | Closest overlap | Remaining difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Krawez et al., RA-L 2021**, "Real-Time Outdoor Illumination Estimation for Camera Tracking in Indoor Environments" | DOI [10.1109/LRA.2021.3090455](https://doi.org/10.1109/LRA.2021.3090455) | full | Indoor RGB-D tracking. Scene appearance predicted from a diffuse reflectance map + outdoor light. Sun/sky/ground components. Radiosity precomputed; per-frame brightness scales. 4.5x4.8 m room, one window, 12 sequences across times of day, seasons and weather | **Y** (area sun) | **Y** (geolocation + time + *given* scene orientation) | Y (window CAD model) | diffuse only, recovered **separately under lamps after sunset** (Krawez IROS 2018) | RGB-D map | **Y** (many times, forward only) | P (tracking accuracy, not image error) | **Calibrated ephemeris sun through a window predicting indoor appearance at any time: already done** | The sun is never used to *recover* materials. Orientation is assumed, not calibrated. Diffuse radiosity, no BRDF. Evaluated by tracking, not held-out image prediction |
| Krawez et al., IROS 2018, "Building Dense Reflectance Maps of Indoor Environments using an RGB-D Camera" | [PDF](http://ais.informatik.uni-freiburg.de/publications/papers/krawez18iros.pdf) | abs | Diffuse reflectance from RGB-D via radiosity / irradiance | N | N | N | diffuse | Y | N | N | Reflectance map used by the 2021 paper | Lamps only |
| Li et al., ECCV 2022, "Physically-Based Editing of Indoor Scene Lighting from a Single Image" | arXiv [2205.09343](https://arxiv.org/abs/2205.09343) | full (grep) | Single-image indoor lighting editing | Y (window = 3 SGs: sun / sky / ground) | N. To fit window SGs they "render a panorama facing outside the window and then select the brightest direction ... as the sunlight direction and keep it fixed" | Y (3D window) | Y | N (single image) | N | N | Explicit window sun + sky indoors; same lab as FIPT/IRIS | Single image; image-derived sun; one time |
| ProjectiveShading (Luo et al., CGF 2026) | DOI [10.1111/cgf.70320](https://doi.org/10.1111/cgf.70320); [project](https://jundanluo.github.io/publications/projectiveshading/) | abs | Single-view indoor object insertion; "sunlight map"; estimates sunlight direction | Y | N (estimated) | P (sunlight map encodes occlusion) | P | N | N | N | Indoor sun-direction estimation from images | Single view; no calibration; insertion task |
| Ji, Sawyer, Narasimhan, ISVC 2023, "Virtual Home Staging" (introduces **Cali-HDR**) | arXiv [2311.12265](https://arxiv.org/abs/2311.12265) | abs | Indoor HDR panorama + simultaneous outdoor hemispherical HDR photo, luminance-calibrated; re-furnishing and relighting | P (outdoor photo is the light) | N (measured outdoor photo) | Y | P (diffuse) | N (one panorama) | N | N | Indoor inverse rendering with *measured* outdoor light; our Cali-HDR source | Single panorama, single time |
| Ji et al., MVA 2024, "Virtual home staging and relighting from a single panorama under natural illumination" | DOI [10.1007/s00138-024-01559-7](https://doi.org/10.1007/s00138-024-01559-7); [project](https://gzhji.github.io/virtual_home_animation/) | abs / snip | Extension. Project page: "editing the direct illumination of the sun in the outdoor image, allowing the indoor scene to be rendered under varying sun positions" | Y | ? (a search snippet mentions GPS + time; not confirmed in full text) | Y | P | N | forward only | ? | **Forward relighting of a real room across sun positions from one panorama** | One capture; sun used for forward rendering, not supervision |
| Ji et al., ISVC 2025, "Indoor Light and Heat Estimation from a Single Panorama" | arXiv [2502.06973](https://arxiv.org/abs/2502.06973) | abs | Indoor + outdoor panorama; outdoor panorama as envmap for spatially varying light + materials | P | N | Y | P | N | N | N | Same group, same data type | Single capture |

## C. Outdoor / remote sensing with a calibrated sun over time

| Paper | Link | Ver. | Task / input | Sun | Eph | Win | BRDF | MV | MT | XT | Closest overlap | Remaining difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-NeRF (Derksen & Izzo, CVPRW 2021) | arXiv [2104.09877](https://arxiv.org/abs/2104.09877) | full (grep) | Satellite multi-view; sun visibility field + sky colour as a function of sun position; albedo | Y | Y (per-image "view and solar angles" are inputs, not estimated) | N | albedo | Y | Y (multi-date) | ? | **Known sun direction + multi-date to separate shadow from albedo** | Outdoor, top-down; no interreflection, no windows |
| Sat-NeRF (Marí et al., CVPRW 2022) | arXiv [2203.08896](https://arxiv.org/abs/2203.08896) | abs | Satellite; shadow-aware irradiance model; transients = what "cannot be explained by the position of the sun" | Y | Y (sun position given) | N | albedo | Y | Y | ? | Same | Same |
| EO-NeRF (Marí et al., CVPRW 2023), "Multi-Date Earth Observation NeRF: The Detail Is in the Shadows" | [CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Mari_Multi-Date_Earth_Observation_NeRF_The_Detail_Is_in_the_Shadows_CVPRW_2023_paper.html) | abs | Multi-date satellite; shadows rendered strictly from geometry + sun | Y | Y | N | albedo | Y | Y | ? | Geometry-consistent sun shadows across dates improve reconstruction | Outdoor |
| SUNDIAL (CVPRW 2024) | arXiv [2312.16215](https://arxiv.org/abs/2312.16215) | abs (earlier audit) | Satellite; direct sun + sky; sun direction solved jointly (capture time recovered from shadows) | Y | solved | N | albedo | Y | Y | ? | Inverse direction: shadows → sun / time | Outdoor |
| Heliometric stereo (Abrams, Hawley, Pless, ECCV 2012) | DOI [10.1007/978-3-642-33709-3_26](https://doi.org/10.1007/978-3-642-33709-3_26) | abs | Outdoor webcam time-lapse; sun direction from GPS + timestamps; normals via photometric stereo | Y | **Y** | N | P (normals, albedo) | N | **Y** | ? | **"Use the time-stamped, calibrated sun as structured supervision"** | Outdoor, single view, geometry target |
| Outdoor PS (Hold-Geoffroy et al., ICCP 2015), "What is a good day for outdoor photometric stereo?" | [project](http://vision.gel.ulaval.ca/~jflalonde/publications/projects/outdoorPS/index.html) | abs / snip | Conditioning of outdoor PS over a day | Y | Y | N | normals | N | Y | — | **Negative result: on clear days the sun path is near-planar, so normals are ambiguous from photometric cues alone** | Bears on our H4; our target is albedo with known geometry |
| Factored time-lapse video (Sunkavalli et al., SIGGRAPH 2007) | DOI [10.1145/1275808.1276504](https://doi.org/10.1145/1275808.1276504) | abs | Outdoor time-lapse → sun, sky, shadow, reflectance | Y | N | N | reflectance | N | Y | N | Sun/sky/shadow/reflectance factorization over a day | Outdoor, single view |
| SOL-NeRF (SIGGRAPH Asia 2023) | DOI [10.1145/3610548.3618143](https://doi.org/10.1145/3610548.3618143) | abs (earlier audit) | Outdoor; SG sun + SH sky | Y | N | N | albedo | Y | P | ? | Explicit sun + low-order sky | Outdoor; sun from images |
| SR-TensoRF (WACV 2024) | arXiv [2311.03965](https://arxiv.org/abs/2311.03965) | abs (earlier audit) | Outdoor; sun direction drives a shadow branch | Y | P | N | albedo | Y | Y | Y (NeRF-OSR) | Sun-aligned relighting | Outdoor |
| NeRF-OSR (ECCV 2022) | arXiv [2112.05140](https://arxiv.org/abs/2112.05140) | abs (earlier audit) | Outdoor multi-session relighting benchmark; SH light | P | N | N | albedo | Y | Y | **Y** | Cross-session real ground truth | Outdoor |
| UrbanIR | arXiv [2306.09349](https://arxiv.org/abs/2306.09349) | abs | Outdoor single video; sun + sky, visibility, albedo | Y | ? | N | albedo | Y | N | N | Sun + sky + visibility | Outdoor |
| GaRe (ICCV 2025) | arXiv [2507.20512](https://arxiv.org/abs/2507.20512) | abs | Outdoor photo collections; sunlight + sky + indirect | Y | N | N | albedo | Y | Y | ? | Physically split sun / sky | Outdoor |
| OSDR-GS (IJCAI 2025) | [IJCAI](https://www.ijcai.org/proceedings/2025/111) | abs | Outdoor, changing lighting; lighting groups + visibility shadows | P | N | N | Y | Y | Y | ? | Multi-condition decomposition | Outdoor |
| ROS-GS | arXiv [2509.11275](https://arxiv.org/abs/2509.11275) | abs | Outdoor; SG sun + SH-PRT sky | Y | N | N | P | Y | Y | ? | Same sun/sky split we use | Outdoor |
| SkyLume | arXiv [2512.14200](https://arxiv.org/abs/2512.14200) | abs (earlier audit) | UAV, 10 urban areas × 3 times of day; TCC cross-time albedo-stability metric | ? | ? | N | albedo | Y | Y | Y | Cross-time albedo stability as a metric | Outdoor aerial |

## D. Multi-illumination disambiguation (the general principle)

| Paper | Link | Ver. | Point |
|---|---|---|---|
| Eclipse (Verbin et al., CVPR 2024) | arXiv [2305.16321](https://arxiv.org/abs/2305.16321) | abs | Unintended shadows (occluders) disambiguate illumination and materials; objects |
| Dynamic Inverse Rendering (Yunus et al., ECCV 2026) | arXiv [2607.09329](https://arxiv.org/abs/2607.09329) | abs (earlier audit) | States that multiple lighting conditions generally reduce ambiguity; exploits object motion |
| GLOW | see A | abs | Calibrated moving light disambiguates indoor reflectance |
| Daylight coefficients (Tregenza & Waters, Lighting Res. & Tech. 1983) | — | ref (Krawez [13]) | Forward precomputed sun/sky/ground transport for interiors; standard in building daylighting simulation |

## What is left (honest statement)

Each of these is **taken**:
- a calibrated ephemeris sun entering a room through a window, used to predict
  indoor appearance at arbitrary times (Krawez 2021; building daylighting simulation);
- an explicit window sun/sky model indoors (Li 2022);
- indoor sun-direction estimation (ProjectiveShading);
- a known sun direction across multiple dates to separate shadow from albedo
  (S-NeRF / Sat-NeRF / EO-NeRF);
- a time-stamped sun as supervision for scene properties (heliometric stereo);
- calibrated varying illumination indoors (GLOW, active flash);
- sun-patch contamination of materials, and deshadowing (SIR, SGS-Intrinsic, IR-HGP).

Not found (a failed search does not prove absence):
**multi-view indoor inverse rendering whose supervision is the calibrated sun's motion
through the window over several times of day, validated by held-out-time photos.**

This is the transfer of the satellite / heliometric-stereo idea to rooms. The genuinely
new technical content would be:
1. window-aperture transport with interreflection (satellite work has none);
2. calibrating the one yaw DOF from patches, where Krawez assumes it is given;
3. showing that calibration adds something beyond an explicit sun estimated from images.

Known risk from the outdoor literature: within one clear day the sun path is close to
a plane (Hold-Geoffroy 2015). The extra information from more times may therefore be
smaller than expected.

**Novelty confidence: low–moderate.** A reviewer who knows Krawez 2021 and
S-NeRF / EO-NeRF will call it a combination unless the empirical gain is large and the
real-data validation is convincing.
