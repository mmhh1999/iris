# Daylight-aware IRIS extension (research/daylight-aware-iris branch).
# Not part of the original IRIS paper/pipeline.
"""Bake a per-pixel, purely-geometric external-sun visibility term.

For each training-view surface point, computes
    sun_vis = max(0, dot(normal, sun_direction)) * visibility(point -> sun)
where visibility is a hard shadow-ray test against the SAME reconstructed
mesh `bake_shading.py`/`train_brdf_crf.py` already use (`utils.path_tracing
.ray_intersect`), mirroring exactly how `experiments/generate_sunpatch_
benchmark.py`'s ground-truth generator computes its `sun_visible` mask.

This is geometry-only: no radiance, no emitter, no material. It does not
depend on `sun_toward_world` being exactly correct (Mode B / ephemeris could
supply it instead of the oracle direction used in the P1 controlled test),
and it never touches `bake_shading.py`'s own diffuse/specular bake, so a run
that skips this stage sees byte-identical existing behavior. See
`research/SUNPATCH_DAYLIGHT_METHOD_ZH.md` for why this term exists and how it
is consumed by `train_brdf_crf.py`.
"""
import torch
import torch.nn.functional as NF
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
from utils.dataset import SyntheticDatasetLDR, RealDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.path_tracing import ray_intersect
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from const import set_random_seed
set_random_seed()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_root', type=str, help='dataset root')
    parser.add_argument('--scene', type=str, required=True, help='dataset folder')
    parser.add_argument('--output', type=str, required=True, help='shading cache dir (same one bake_shading.py writes into)')
    parser.add_argument('--dataset', type=str, required=True, help='dataset type')
    parser.add_argument('--ldr_img_dir', type=str, default=None)
    parser.add_argument('--res_scale', type=float, default=1.0)
    parser.add_argument('--sun-toward-world', type=str, required=True,
                         help='comma-separated x,y,z unit vector pointing FROM the scene TOWARD the sun '
                              '(same convention as generate_sunpatch_benchmark.py\'s manifest "sun_toward_world")')
    args = parser.parse_args()

    device = torch.device(0)
    sun_dir = torch.tensor([float(v) for v in args.sun_toward_world.split(',')], device=device, dtype=torch.float32)
    sun_dir = NF.normalize(sun_dir, dim=-1)

    DATASET_PATH = args.scene
    OUTPUT_PATH = args.output
    if args.dataset in ['synthetic', 'real']:
        mesh_path = os.path.join(DATASET_PATH, 'scene.obj')
        mesh_type = 'obj'
    elif args.dataset == 'scannetpp':
        mesh_path = os.path.join(args.dataset_root, 'data', args.scene, 'scans', 'scene.ply')
        mesh_type = 'ply'
    assert Path(mesh_path).exists(), 'mesh not found: ' + mesh_path

    scene = mitsuba.load_dict({
        'type': 'scene',
        'shape_id': {'type': mesh_type, 'filename': mesh_path},
    })

    if args.dataset == 'synthetic':
        dataset = SyntheticDatasetLDR(DATASET_PATH, img_dir=args.ldr_img_dir, split='train', pixel=False)
    elif args.dataset == 'real':
        dataset = RealDatasetLDR(DATASET_PATH, img_dir=args.ldr_img_dir, split='train', pixel=False)
    elif args.dataset == 'scannetpp':
        dataset = Scannetpp(args.dataset_root, args.scene, split='train', pixel=False, res_scale=args.res_scale)
    img_hw = dataset.img_hw

    out_dir = os.path.join(OUTPUT_PATH, 'sun_vis')
    os.makedirs(out_dir, exist_ok=True)

    im_id = 0
    for batch in tqdm(dataset):
        rays = batch['rays']
        xs = rays[..., :3].to(device)
        ds = rays[..., 3:6].to(device)

        positions, normals, _, _, valid = ray_intersect(scene, xs, ds)

        cos_theta = (normals * sun_dir[None]).sum(-1, keepdim=True)
        shadow_xs = positions + 1e-3 * sun_dir[None]
        shadow_ds = sun_dir[None].expand_as(positions)
        _, _, _, _, occluded = ray_intersect(scene, shadow_xs, shadow_ds)
        sun_visible = (~occluded).float().unsqueeze(-1)

        zeros = torch.zeros_like(cos_theta)
        sun_vis = torch.where(valid.unsqueeze(-1), cos_theta.clamp_min(0.0) * sun_visible, zeros)

        sun_vis_img = sun_vis.reshape(*img_hw, 1).cpu().numpy()
        sun_vis_rgb = np.repeat(sun_vis_img, 3, axis=-1).astype(np.float32)
        # BGR order + reuse of utils.dataset's open_exr (expects the same layout bake_shading.py writes)
        cv2.imwrite(os.path.join(out_dir, '{:03d}.exr'.format(im_id)), sun_vis_rgb[:, :, [2, 1, 0]])
        im_id += 1

    print('[bake_sun_term] wrote', im_id, 'views to', out_dir, 'sun_dir', sun_dir.tolist())
