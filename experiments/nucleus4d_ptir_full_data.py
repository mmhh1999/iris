"""Archive-backed, native-resolution PTIR inputs. No duplicate photo extraction."""
from pathlib import Path
import io
import os
import json
import struct
import hashlib
import tarfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from threedgrut.datasets.protocols import Batch, BatchPrior

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'experiments/out/nucleus4d_ptir_full'
ARCHIVE = Path(os.environ.get('NUCLEUS_PTIR_ARCHIVE','/mnt/e/Datasets/Nucleus4D/20260929/1406-C-int.tar'))
STORAGE = Path(os.environ.get('NUCLEUS_PTIR_STORAGE','/mnt/e/Datasets/Nucleus4D/20260929/1406-C-int-ptir-full'))
PREFIX = 'developer_data/perspective/'


def read_member(index, name):
    entry = index[name]
    with ARCHIVE.open('rb') as f:
        f.seek(entry['offset'])
        data = f.read(entry['size'])
    if len(data) != entry['size']:
        raise IOError(f'Truncated archive member: {name}')
    return data


def prepare_manifest():
    from scipy.spatial.transform import Rotation
    LOCAL.mkdir(exist_ok=True,parents=True)
    if not (LOCAL/'archive_index.json').exists():
        index={}
        with tarfile.open(ARCHIVE,'r:') as archive:
            for member in archive:
                if member.isfile() and member.name.startswith(PREFIX):
                    index[member.name]=dict(offset=member.offset_data,size=member.size)
        (LOCAL/'archive_index.json').write_text(json.dumps(index))
    index = json.loads((LOCAL/'archive_index.json').read_text())
    f = io.BytesIO(read_member(index, PREFIX+'sparse/cameras.bin'))
    cameras = {}
    for _ in range(struct.unpack('<Q', f.read(8))[0]):
        cid, model, w, h = struct.unpack('<iiQQ', f.read(24))
        if model != 1:
            raise ValueError('Expected supplied undistorted PINHOLE cameras')
        cameras[str(cid)] = dict(width=w, height=h, K=list(struct.unpack('<4d', f.read(32))))
    f = io.BytesIO(read_member(index, PREFIX+'sparse/images.bin'))
    rows = []
    for _ in range(struct.unpack('<Q', f.read(8))[0]):
        iid, *v = struct.unpack('<i7di', f.read(64))
        name = bytearray()
        while (ch := f.read(1)) != b'\0':
            if not ch:
                raise IOError('Truncated COLMAP image name')
            name += ch
        n = struct.unpack('<Q', f.read(8))[0]
        f.seek(24*n, 1)
        name = name.decode()
        q, t, cid = np.array(v[:4]), np.array(v[4:7]), int(v[7])
        r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        pose = np.eye(4); pose[:3, :3] = r.T; pose[:3, 3] = -r.T @ t
        # All derived directions, and all cameras within a capture second, stay together.
        group = name.split('/')[0]+'/'+Path(name).stem.split('.')[0]
        holdout = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20 == 0
        image_key = PREFIX+'images/'+name
        mask_key = PREFIX+'masks/'+str(Path(name).with_suffix('.png'))
        if image_key not in index or mask_key not in index:
            raise FileNotFoundError(name)
        rows.append(dict(id=iid, name=name, camera_id=cid, pose=pose.tolist(),
                         split='val' if holdout else 'train', capture_group=group,
                         image=index[image_key], mask=index[mask_key]))
    rows.sort(key=lambda r: r['name'])
    result = dict(archive=str(ARCHIVE), cameras=cameras, frames=rows,
                  split_policy='SHA256(map/capture_second) mod 20; all derived views kept together',
                  resolution_policy='Supplied perspective images at native resolution; no further downsampling',
                  mask_policy='Source white=exclude (capture operator); black=valid. Also exclude exact-black undistortion padding.',
                  archive_bytes=ARCHIVE.stat().st_size)
    LOCAL.mkdir(exist_ok=True, parents=True)
    (LOCAL/'dataset.json').write_text(json.dumps(result))
    print('DATASET', len(rows), 'train', sum(r['split']=='train' for r in rows), flush=True)
    return result


def load_image(row, kind='image'):
    entry = row[kind]
    with ARCHIVE.open('rb') as f:
        f.seek(entry['offset']); data = f.read(entry['size'])
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert('RGB' if kind=='image' else 'L')).copy()


def prior_path(row):
    return STORAGE/'priors'/Path(row['name']).with_suffix('.npz')


class NucleusDataset(Dataset):
    def __init__(self, split, crop=512, require_priors=True, limit=0):
        self.info = json.loads((LOCAL/'dataset.json').read_text())
        self.rows = [r for r in self.info['frames'] if r['split']==split]
        if limit:
            ids = np.linspace(0, len(self.rows)-1, limit).astype(int)
            self.rows = [self.rows[i] for i in ids]
        self.split, self.crop, self.require_priors = split, crop, require_priors
        self.poses = np.asarray([r['pose'] for r in self.rows], dtype=np.float32)
        self.camera_centers = self.poses[:, :3, 3]
        all_centers = np.array([r['pose'] for r in self.info['frames']])[:, :3, 3]
        self.extent = float(np.linalg.norm(all_centers-all_centers.mean(0), axis=1).max()*1.1)
        self.bbox = tuple(torch.tensor(x, dtype=torch.float32) for x in
                          (all_centers.min(0), all_centers.max(0)))

    def __len__(self): return len(self.rows)
    def get_scene_extent(self): return self.extent
    def get_scene_bbox(self): return self.bbox
    def get_observer_points(self): return self.camera_centers
    def get_poses(self): return self.poses
    def get_frames_per_camera(self):
        return [sum(r['camera_id']==i for r in self.rows) for i in range(8)]

    def __getitem__(self, idx):
        import cv2
        for offset in range(len(self.rows)):
            actual_idx = (idx+offset) % len(self.rows)
            row = self.rows[actual_idx]
            cam = self.info['cameras'][str(row['camera_id'])]
            rgb, mask = load_image(row), load_image(row, 'mask')
            h, w = rgb.shape[:2]
            assert (w, h) == (cam['width'], cam['height'])
            valid = (mask < 128) & (rgb.max(axis=-1) > 0)
            if valid.any():
                idx = actual_idx
                break
            if self.split != 'train':
                raise ValueError(f'No valid pixels in validation view {row["name"]}')
            skipped = getattr(self, '_skipped_empty_views', set())
            if row['name'] not in skipped:
                print('SKIP_EMPTY_TRAIN_VIEW', row['name'], flush=True)
                skipped.add(row['name'])
            self._skipped_empty_views = skipped
        else:
            raise ValueError('No training views contain valid pixels')
        ch, cw = min(self.crop, h), min(self.crop, w)
        if self.split=='train':
            # Rejection samples avoid undistortion padding / sensor masking.
            for _ in range(32):
                x = np.random.randint(w-cw+1); y = np.random.randint(h-ch+1)
                if valid[y:y+ch, x:x+cw].mean() > .8: break
        else:
            x, y = (w-cw)//2, (h-ch)//2
        if valid[y:y+ch, x:x+cw].mean() < .8:
            # Deterministic fallback prevents masked/empty crops from inflating
            # validation scores and contributing no photometric supervision.
            integral = cv2.integral(valid.astype(np.uint8))
            xs = np.unique(np.append(np.arange(0, w-cw+1, 8), w-cw))
            ys = np.unique(np.append(np.arange(0, h-ch+1, 8), h-ch))
            xx, yy = np.meshgrid(xs, ys)
            counts = (integral[yy+ch, xx+cw]-integral[yy, xx+cw]
                      -integral[yy+ch, xx]+integral[yy, xx])
            best = np.unravel_index(np.argmax(counts), counts.shape)
            x, y = int(xx[best]), int(yy[best])
            if counts[best] == 0:
                raise ValueError(f'No valid pixels in {row["name"]}')
        result = dict(rgb=rgb[y:y+ch, x:x+cw].copy(),
                      mask=valid[y:y+ch, x:x+cw].astype(np.float32)[..., None],
                      pose=self.poses[idx], K=np.array(cam['K'], dtype=np.float32),
                      origin=np.array([x, y]), camera_id=row['camera_id'], frame_idx=idx)
        path = prior_path(row)
        if self.require_priors and not path.exists():
            raise FileNotFoundError(f'Missing learned prior: {path}')
        if self.require_priors:
            with np.load(path) as p:
                for key in ('normal', 'albedo', 'roughness'):
                    v = p[key].astype(np.float32)
                    ph, pw = v.shape[:2]
                    xx, yy = np.meshgrid(np.arange(x, x+cw), np.arange(y, y+ch))
                    mx = ((xx+.5)*pw/w-.5).astype(np.float32)
                    my = ((yy+.5)*ph/h-.5).astype(np.float32)
                    v = cv2.remap(v, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    result[key] = v if v.ndim==3 else v[..., None]
        return result

    def get_gpu_batch_with_intrinsics(self, d):
        device='cuda'
        rgb = d['rgb'].to(device).float()/255
        b, h, w, _ = rgb.shape
        x, y = d['origin'][0].tolist()
        xx, yy = torch.meshgrid(torch.arange(w, device=device)+x+.5,
                                torch.arange(h, device=device)+y+.5, indexing='xy')
        coords = torch.stack((xx, yy), -1)[None]
        k = d['K'][0].tolist(); fx, fy, cx, cy = k
        rays = torch.stack(((xx-cx)/fx, (yy-cy)/fy, torch.ones_like(xx)), -1)
        rays = torch.nn.functional.normalize(rays, dim=-1)[None]
        mask = d['mask'].to(device)
        prior = None
        if 'normal' in d:
            prior = BatchPrior(normal=torch.nn.functional.normalize(d['normal'].to(device), dim=-1),
                               albedo=d['albedo'].to(device), roughness=d['roughness'].to(device))
        return Batch(rays_ori=torch.zeros_like(rays), rays_dir=rays,
                     T_to_world=d['pose'].to(device), rgb_gt=rgb, gradient_mask=mask,
                     intrinsics=k, pixel_coords=coords, prior=prior,
                     camera_idx=int(d['camera_id'][0]), frame_idx=int(d['frame_idx'][0]))


if __name__=='__main__':
    prepare_manifest()
