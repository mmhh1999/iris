Nucleus4D full-house transport v3_opt1

This is the tested pre-spec v3 renderer with incremental sun-control and
GPU scheduling improvements. It preserves all 6,100,978 original Gaussians,
moving sunlight, time/weather controls, 25 lamps and free camera movement.
Default shadows remain 4K; 8K is optional. Physical accuracy is not certified.

Contents

viewer/: exact published HTML, JavaScript, metadata and license.
build_inputs/: small upstream templates needed by the Python publisher.
verification/: GPU browser results, six-state pixel comparisons and UI capture.
manifest.json: SHA256 identities of the snapshot and required local assets.
requirements.txt: Python package versions observed for this version.

The generated model, light fields and geometry stay outside Git under
experiments/out/, following the repository's existing ignore rule. They total
about 1.59 GB for this snapshot. This release is a source/viewer checkpoint,
not a cloud backup of those assets. A fresh clone needs those files copied
separately from the original workspace, at the paths in manifest.json.
The current workspace already contains them; do not retrain or rebake.
Historical comparison links in viewer/verification.html also refer to older
local image artifacts; the current version's checks are in verification/.

Restore the saved viewer with the existing local assets (repository root):

  mkdir -p experiments/out/nucleus4d_transport_v3
  cp -R experiments/releases/nucleus4d_transport_v3_opt1/viewer/. experiments/out/nucleus4d_transport_v3/
  cp -R experiments/releases/nucleus4d_transport_v3_opt1/build_inputs/. experiments/out/
  .venv/bin/python experiments/gs_relight_launch.py --no-window

Open http://localhost:8769/nucleus4d_transport_v3/ and reload with Ctrl+F5.
Click "开启太阳（晴空）" to enable direct sun, then drag time. The startup
original-capture reference intentionally has added sunlight disabled.

To rebuild the page after source changes, with the same saved assets:

  .venv/bin/python experiments/nucleus4d_transport_viewer.py

Launching v3 no longer depends on the inactive staged app's scene.json.
The launcher creates its log directory if missing. Old staged/world page
URLs redirect to v3; world rendering APIs remain disabled by the selected
baseline. Inactive world-framework implementations are outside this release.

The original full.splat renderer code is MIT licensed; see viewer/LICENSE.
Browser verification uses Node 24 and an isolated Windows Chrome CDP port
9367. Rendering itself requires WebGL2 float render targets and float blend.
Python dependency versions are listed for publishing/baking/verification;
the saved viewer uses the existing Python static server and browser GPU.
