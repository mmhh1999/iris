Full-house lighting transport v3 (experimental)

Current incremental revision: v3_opt1, based on the restored pre-spec v3.
Viewer: http://localhost:8769/nucleus4d_transport_v3/
Sun time/movement, power, playback, soft shadows and presets are required
preserved features. Explicit sun on/off and effective-sun status are added.
4096 remains the default shadow size; 8192 is an optional GPU allocation,
validated before replacing the working map. This does not certify scan geometry.
Unchanged float32 light weights reuse the incident atlas; exposure-only and
inactive-sun time updates skip redundant work. No source field is recomputed.
Before/after evidence and code backups: out/nucleus4d_transport_v3/optimization_review/.
Current service: iris-viewer.service; launch with gs_relight_launch.py --no-window.
The service/bake notes below describe earlier experiments, not active jobs.

Original scene: experiments/out/nucleus4d_3dgs/full.splat, all 6,100,978
Gaussians, geometry/scales/rotation/opacity/order remain unchanged. Previously
approved sunlight and weather-controls entries are preserved.

Inputs and approximation:
- Complete supplied scan for visibility, including other rooms/furniture.
- Two already-reviewed living-room apertures repaired for lighting queries.
- 97.6% of indoor splats have a receiver within 16cm; remaining interior
  splats use explicitly flagged local volume estimates. Undefined reference
  irradiance is separately rebound and reported in reference_repair.json.
- Source RGB is multiplied by current/reference incident lighting. Exact
  original capture uses a shader bypass. This is relative diffuse relighting,
  not a recovered measured BRDF or measured absolute illuminance solution.
- 25 independent finite lamps: 4,194,304 emitted rays per source for
  area-averaged coarse flux, 4096 finite-area samples for fine direct light,
  16384 cosine samples near emitters, eight diffuse reflection terms.
- Area-reciprocal conservative transport; bounded captured-color reflectance
  prior used only for secondary bounces, never as the visible GS texture.
- Direct/overcast 8K HDR environment maps, 1024 fine cosine sky samples.
- A single moving geometric sun. The illustrative 12:30-18:15 path is not
  calibrated to a surveyed building/site/date. Runtime 4096 default shadow map with
  finite-sun filtering. Only indirect sunlight uses adaptive time keys, with
  a 2.5% normalized maximum midpoint validation threshold.
- Float32 resident light bases, original full-count renderer, no field
  transfers on slider drag. Deferred shadowing uses composited surface
  positions and is separately compared to finite-sun mesh visibility rays.
- Glass/specular transport, measured lamp powers and material reflectance
  remain unverified. Exterior uses approximate daylight scaling.

Persistent user services (local machine must remain awake with WSL running):
  systemctl --user status iris-transport-v3.service
  systemctl --user status iris-transport-finish.service
Logs: experiments/out/nucleus4d_transport_v3/{compute,finish}.log
State: bake_status.json and finish_status.json in the same output directory.
Additional independent candidate: iris-transport-emitters.service recomputes
the original-quality 25 source fields while excluding each source's own
photographed housing volume from its own shadow/emission rays. This addresses
misalignment of scanned fixture surfaces and the triangulated emitters. The
model is an explicit uncertain alignment estimate; intervening walls, other
lamps and all sky/solar geometry remain opaque. Logs: emitters.log; progress:
emitter_bases/bake_status.json; native image output: review_emitters/.

Raw v3, spatial diffuse filtering, and unbiased window-MIS sky candidates all
show wall mottling under lamp removal. None has passed manual visual review.
The root experimental viewer is for debugging; the approved old entry remains
the recommended browsing version until the fixture correction is reviewed.
The progress dashboard is /nucleus4d_transport_v3/progress.html on port 8769.
Stopping/resuming the bake preserves completed source fields and solar keys.
No cloud deployment, upload or paid GPU allocation is used.

Workflow:
  source experiments/nucleus4d_ptir_env.sh
  python experiments/nucleus4d_transport_bake.py --fine
  python experiments/nucleus4d_transport_audit.py
  python experiments/nucleus4d_transport_review.py
  python experiments/nucleus4d_transport_audit.py
  python experiments/nucleus4d_transport_viewer.py
  python experiments/nucleus4d_transport_check.py
  DRJIT_LIBCUDA_PATH=/dev/null python experiments/nucleus4d_transport_shadow_check.py

The finish service carries out the postcompute workflow independently of chat.
Numerical/browser checks never substitute for manual native-image review.
The new experimental entry is /nucleus4d_transport_v3/ on local port 8769;
do not advertise it as validated until checks and visual review pass.
