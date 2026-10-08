1406-C-int: complete PTIR training job (2026-10-04)

Status is in experiments/out/nucleus4d_ptir_full/pipeline_status.json.
Training status: http://localhost:8769/nucleus4d_ptir_full/
Relit free-view viewer: http://localhost:8770/
The existing daylight demo is NOT replaced by component-test images.

Actual method
-------------
Official PTIR-GS commit 8a5a5051639edf8feea486d55e183cedf490556c.
This run uses the official CUDA/OptiX forward AND backward code, RGB-X priors,
geometry/normal training, densification/pruning, photometric/SSIM/normal losses,
and material/environment inverse optimization. This is distinct from the earlier
Mitsuba forward-only demo. It is a local full-pipeline adaptation, not a claim
that the paper's benchmark performance or true measured materials are reproduced.

The archive supplies 7,950 undistorted perspective views with exact PINHOLE
intrinsics. 7,446 train, 504 held out by capture second; all derived directions
from one capture stay in the same split. Source images remain in the TAR and are
read by byte offset. White source masks identify the capture operator and are
excluded. The 4,334 x 3,078 high-resolution views are not resized for supervision;
random 512 x 512 native-pixel regions control VRAM. The smaller fisheye-derived
perspective images retain their supplied native resolution as well.
Completely masked/black training views are skipped and logged; 7,446 is the
nominal split size. Poorly covered crops use a deterministic valid-region search.
Validation rejects fully empty views rather than reporting a perfect masked score.

Quality settings
----------------
Initial model: all 6,100,978 original Gaussians, float32; no initial decimation.
Stage 1: 30,000 geometry/normal refinement steps; learned normal + depth-normal
supervision; learned SH detail; splitting/cloning and opacity pruning. Geometry
is allowed to change. A memory budget caps growth at 8M points, prioritizing the
largest gradients and limiting each clone/split update to 100K net new points.
No blanket opacity reset is applied to the already-reconstructed input model.
Stage 2: 16,000 material/SG-environment steps, 64 SPP, 4 bounces; optimized albedo,
roughness, normals and illumination. Metallic optimization follows upstream's
real-scene default (disabled), rather than claiming unsupported metal recovery.
RGB-X: every training view, 50 diffusion steps, 1024-pixel long edge, aspect ratio
preserved; float16 compressed priors. These are weak learned priors, not ground
truth materials. Actual image losses and model parameters remain float32.
Relit free-view output: 3840 x 2496, 1024 SPP, 8 bounces; float32 EXR,
no denoising. The browser receives the full-resolution float32 frame and displays
sRGB with local exposure control. No resolution reduction for this viewer.
Optional fixed-camera comparisons retain their separate 1920-wide preview.

Memory implementation and evidence
----------------------------------
experiments/ptir_full_runtime.patch records local backend changes:
- Python 3.10 typing backport, reusing existing Torch 2.11/CUDA 12.8.
- Locate CUDA headers in the existing conda-style toolkit layout.
- Reuse activated Gaussian tensors across SPP chunks instead of duplicating a
  full 6.1M-point SH array for every chunk.
- Checkpoint each SPP chunk, replaying its exact seeded forward operation during
  backward. No sample reduction or approximate replacement shader is used.
- Bound densification as described above.
Slang 2026.5.2 regenerates the tracked generated .cuh during compilation; that
large generated diff is deliberately not duplicated in the compatibility patch.

The whole-model gradient probes ran actual optimizer steps. The replay check
compares losses/gradients with/without checkpointing at identical seeds. Its
separate visible-environment control verifies nonzero illumination gradients;
upstream otherwise deliberately hides the primary environment background.
See geometry_gradient_check.json, inverse_gradient_check.json,
replay_validation.json, and the renderer/checkpoint probe logs in the status dir.
These are execution/correctness checks, NOT proof of final image quality.

Validation and stopping
-----------------------
Before/after and periodic validation evaluate 24 fixed native 256-pixel crops
spread across the held-out set, without exposure/color fitting to the targets.
The remaining held-out views are available for broader evaluation; 504 is the
split size, not a claim that all 504 full-resolution images were evaluated.
Geometry may lose at most 0.5 dB PSNR / 0.02 SSIM against its initial baseline.
Inverse appearance may lose at most 2 dB / 0.05 SSIM against refined geometry.
Failure stops daylight publication and writes needs_attention, with scores and
logs retained. Passing this appearance gate does NOT establish physically correct
relighting, which requires captures under different known illumination.

The scene has no measured location/north/HDR. Exported daylight uses explicitly
labelled example solar geometry and synthetic HDR skies. The renderer models
opaque GS light transport; refractive glazing and original artificial fixture
emitters are not reconstructed. Those are material limitations even after the
full optimizer finishes. Increasing iterations/SPP cannot guarantee their recovery.

Storage and duration
--------------------
Large products live at E:\Datasets\Nucleus4D\20260929\1406-C-int-ptir-full.
Estimated additional storage about 50-65 GB, mostly per-view learned priors.
No duplicate photo tree or LiDAR extraction. One rolling resume checkpoint and
one inference checkpoint per stage; temporary atomic-save files need extra room.
The source PLY has a temporary 396 MiB Linux cache to avoid slow WSL mmap reads.
Initial timing: ~9.4 seconds per prior view; inverse probes ~5-12 seconds per step
at 64 SPP / 512-pixel regions. Expect roughly 2-4 days for the full job; scene
complexity, E: I/O, densification and other GPU work can change that estimate.

Run/restart (from project root, with host GPU and E: write access)
----------------------------------------------------------------
source experiments/nucleus4d_ptir_env.sh
python experiments/nucleus4d_ptir_full_pipeline.py

A process/file lock prevents duplicate jobs. Existing prior files are reused;
completed stages are skipped and an interrupted optimizer resumes from resume.pt.
The resumed sampler is deterministic from its seed but does not replay an exact
mid-epoch data-loader state. Keep the machine/WSL running for the background job.
All inference is local; external experiment tracking is disabled.

2026-10-07 recovery
-------------------
The geometry process exited near step 7500 after OptiX logged a null output
buffer. Native CUDA/OptiX checks previously built error strings without throwing;
ptir_allocator_recovery.patch makes these failures stop at their original call.
Destructor cleanup logs errors without throwing while another exception unwinds.
The exact first failure in the old run cannot be recovered from that log.
The environment now selects CUDA's cudaMallocAsync allocator to reduce competing
allocation pools/fragmentation. Numerical precision, optimizer settings, point
budget, sample counts, bounces and output resolution are unchanged.
Both stages save rolling checkpoints every 1000 steps. Recovery starts from the
last complete checkpoint (step 5000); later unsaved steps must be recomputed.
Incident and checkpoint integrity records are in the local status directory.

2026-10-07 final deliverable: relit free-view scene
-------------------------------------------------
The requested target is rotation, walking and arbitrary views INSIDE the relit
scene. The original-GS walkthrough is a separate entry with captured illumination.
The earlier fixed-camera comparison alone does not satisfy that target.

nucleus4d_ptir_freeview.py serves a browser camera/time controller and serial
GPU worker. Mouse drag turns the camera; an orbit mode rotates around the room
centre. WASD moves in the horizontal plane, Q/E changes elevation, Shift speeds
movement, arrow keys turn. This is unrestricted camera movement, without a
collision/physics system. Full pose and FOV produce new native world-space rays.
Date/local time (including seconds), IANA timezone, location, north bearing and
cloud amount regenerate the HDR environment and importance-sampling alias table.
Lighting remains synthetic/uncalibrated as described above.

Every completed view uses the unchanged full-quality settings. Finished tiles
are streamed centre-first at the full SPP/bounce count; unrendered areas remain
blank rather than mixing old and new cameras. Before new tiles arrive, the
previous completed view remains visibly labelled as a previous view. New
requests cancel older work at tile boundaries; after movement stops, the latest
view is rendered. This is arbitrary-camera on-demand relighting, not a claim of
30/60 FPS interaction at 4K/1024 SPP. Exposure changes affect display only.
Three recent float32 frames/EXRs are cached; EXR downloads preserve HDR values.

The waiting service uses no CUDA model while training is active. It accepts final
render requests only after the inverse stage completes, the two appearance gates
pass, and a native free-camera/time check passes. That integration check samples
four 128x128 centre tiles using full 4K intrinsics, 1024 SPP and 8 bounces; it is
not a claim that final 4K views have already been evaluated. Browser tests use an
explicitly labelled synthetic fixture and do not establish final model quality.

The pipeline now publishes the free-view service after training and checks.
Old fixed-camera exports are optional: pass --static-exports to the pipeline,
or run nucleus4d_ptir_full_render.py manually. The previously running geometry
worker was adopted by the updated manager without resetting its optimizer or
restarting its training. Adoption requires matching the recorded PID/stage and
process identity, and completion requires final checkpoints plus held-out scores.

Start the waiting/viewer service independently:
  source experiments/nucleus4d_ptir_env.sh
  python experiments/nucleus4d_ptir_freeview.py
For a remote GPU host, NUCLEUS_PTIR_STORAGE/NUCLEUS_PTIR_ARCHIVE can override E:
paths; pass --checkpoint/--output as needed. No cloud instance has been provisioned.

The second recovery exposed a real OOM at split densification (step 5400).
ptir_densification_memory.patch removes a complete gathered SH temporary by
indexing directly into the new tensor and releases the old final Adam moment
before allocating a parameter replacement. CPU/CUDA tests prove bitwise-equivalent
copies; no point budget, precision or optimizer/quality setting was lowered.
This recovery crossed the failed step, reached the existing 8M cap and saved a
new step-6000 checkpoint. This is evidence of recovery, not a guarantee against
every possible later GPU/device failure.

2026-10-08 completion validation recovery
-----------------------------------------
Geometry reached all 30,000 steps and saved its final resume/inference checkpoints.
The following evaluation failed because upstream on_training_end deletes the
datasets and loaders. FullTrainer now recreates the identical fixed held-out
validation loader after this cleanup. A completed checkpoint skips run_training
and runs only its missing final evaluation, leaving model/optimizer checkpoints
unchanged. Regression tests cover both normal completion after loader teardown
and recovery at the target step without another optimizer step/checkpoint save.
Evaluation-only recovery also rebuilds the native acceleration structure from
loaded parameters; normal resume otherwise relies on training iterations to do
this. A --revalidate-stage option reruns the completed checkpoint's evaluation
while preserving its tensors, optimizer state and checkpoint files.
The pipeline continues to inverse training only after the real evaluation and
the existing appearance gate pass. No quality/training settings were changed.
