Nucleus4D 1406-C-int: native Gaussian daylight trial (2026-10-04)

What runs
----------
The complete source point_cloud.ply (6,100,978 Gaussians) is read directly from
the existing ZIP on E: through the workspace symlink. Position, covariance,
rotation and opacity are retained. No scene mesh replaces the Gaussians.
Mitsuba analytic ellipsoids and the official PTIR-Mitsuba integrators perform
Gaussian radiance compositing and PBR path tracing. The source archive is read
only; no duplicate full PLY is extracted.

Upstream versions
-----------------
https://github.com/junkzhu/PTIR-Mitsuba
  b981fb5f704d639c65dcfec511a9e0e68e16fafd (renderer actually used)
https://github.com/junkzhu/PTIR-GS
  8a5a5051639edf8feea486d55e183cedf490556c (reviewed; not the runtime backend)
Runtime: Python 3.10, Mitsuba 3.9.1, Dr.Jit 1.5.0, NumPy 2.2.6, RTX 5070 Ti.
The existing environment is reused. No new Torch/CUDA installation is needed.

Apply experiments/ptir_mitsuba_compat.patch inside the pinned PTIR-Mitsuba repo.
It adapts the quaternion API, fixes finite segment lengths while marching rays,
and assigns MIS weight 1 to delta light samples. The experiment loads the
integrators without importing the upstream training CLI.

Material initialization
-----------------------
Twelve existing posed, undistorted photos and their cached RGB-X predictions
provide approximate material priors. Original Gaussian visibility/depth, not a
mesh, gates projection. A smooth delighting ratio multiplies each Gaussian's
own DC color, retaining its fine texture. Local normal-compatible interpolation
fills incomplete coverage. Surface normals are initialized by local PCA of
the source Gaussians. Representatives are ONLY used for normal estimation; the
renderer still includes every original Gaussian.

This is NOT a completed PTIR inverse-optimization run. Albedo, roughness and
metalness are uncertain initial estimates. Original baked lighting remains in
unobserved regions and may remain elsewhere. See material_report.json and
material_fill_report.json for actual coverage. Old guessed proxy-mesh material
values are not imported into this pipeline.

Illumination and interaction
----------------------------
Five daytime samples (09, 11, 13, 15, 17), two synthetic HDR skies and separate
direct/indirect outputs are precomputed from one camera. No interpolated shadow
positions are presented as physically evaluated times. Controls combine linear
radiance bases before tone mapping. The two daylight windows are explicitly
assumed clear apertures for illumination rays. Camera rays retain the complete
original scene. Glass refraction/transmission and exterior occluders beyond
these openings are not recovered. The sample location/date/orientation are not
measured metadata. Artificial fixtures visible in the original capture are not
recovered light sources.

The browser includes original/relit/split views and a link to full original
Gaussian navigation. Relighting itself is a fixed-view precomputed experiment,
not a free-camera real-time Enscape replacement. OptiX denoising may soften
detail; original linear bases remain available. Wall mottling and illumination
residuals must be assessed rather than hidden as successful material recovery.

Reproduce from the workspace root
---------------------------------
.venv/bin/python experiments/nucleus4d_ptir.py --width 480 --spp 2
.venv/bin/python experiments/nucleus4d_ptir_materials.py
.venv/bin/python experiments/nucleus4d_ptir_normals.py
.venv/bin/python experiments/nucleus4d_ptir_material_fill.py
.venv/bin/python experiments/nucleus4d_ptir_validate.py
.venv/bin/python experiments/nucleus4d_ptir_demo.py
.venv/bin/python experiments/nucleus4d_ptir_ui.py
.venv/bin/python -m http.server 8769 --bind 127.0.0.1 --directory experiments/out

Open http://localhost:8769/nucleus4d_ptir/
GPU work needs the host GPU accessible, not the restricted sandbox device view.
Run material initialization/fill in order; material_fill is not an iterative
optimizer and should not be repeatedly applied to its own output.

Validation artifacts in experiments/out/nucleus4d_ptir
-----------------------------------------------------
adapter_validation.json: blockers before a window still occlude illumination,
  blockers beyond the assumed portal do not; primary rays still hit all cases.
manifest.json: full Gaussian count, source identity, material hash, actual spp,
  assumed-window/original-occlusion comparison, all-lights-off control. Restoring
  original Gaussian occlusion is not equivalent to adding opaque blackout panels;
  nonzero residual sunlight is possible through other openings or capture gaps.
browser_validation.json: slider/mode/GI/blackout checks and browser errors.
demo.png: browser screenshot of the completed trial.

Further work required for reliable relighting
---------------------------------------------
Joint material/light optimization against posed original photos, stronger
surface-normal and geometry regularization, explicit window/glass/exterior
transport, and validation against actual differently lit captures. Preserve
an original-quality reconstruction reference throughout that work.
