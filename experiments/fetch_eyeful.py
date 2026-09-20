"""Acquire a deterministic real HDR pilot, preserving official calibrations/splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

BASE = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/EyefulTower/'


def fetch(scene, relative, root):
    url = BASE + scene + '/' + relative
    path = root / scene / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                size = int(r.headers['Content-Length'])
                if not path.exists() or path.stat().st_size != size:
                    tmp = path.with_suffix(path.suffix + '.part')
                    with tmp.open('wb') as f:
                        for chunk in r.iter_content(1024 * 1024):
                            f.write(chunk)
                    if tmp.stat().st_size != size:
                        raise IOError('Incomplete transfer')
                    tmp.replace(path)
            h = hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(b)
            return dict(path=str(path), url=url, bytes=size, sha256=h.hexdigest())
        except (requests.RequestException, OSError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['riverview', 'apartment'])
    p.add_argument('--root', type=Path, default=Path('data_download/eyefultower'))
    p.add_argument('--per-camera', type=int, default=8)
    args = p.parse_args()
    for scene in args.scenes:
        records = [fetch(scene, n, args.root) for n in ['cameras.json', 'splits.json']]
        cameras = json.loads((args.root / scene / 'cameras.json').read_text())['KRT']
        selected = []
        # Same-height cameras per official recommendation; 17 is official heldout.
        for rig_camera in ['17', '19', '20', '21']:
            available = sorted([c for c in cameras if c['cameraId'].split('/')[0] == rig_camera],
                               key=lambda c: c['cameraId'])
            if not available:
                raise ValueError(f'Missing requested camera {rig_camera}')
            n = min(args.per_camera, len(available))
            selected.extend(available[round(i * (len(available)-1) / max(1, n-1))] for i in range(n))
        selection = dict(scene=scene, cameras=selected,
                         selection='evenly spaced frames of cameras 17,19,20,21; no label-based selection',
                         original_color_space='linear DCI-P3',
                         photo_only=False, measured_brdf_ground_truth=False,
                         solar_ground_truth=False)
        (args.root / scene / 'pilot_selection.json').write_text(json.dumps(selection, indent=2))
        files = ['mesh.obj', 'mesh.mtl', 'mesh.jpg']
        files += ['images-1k/' + c['cameraId'] + '.exr' for c in selected]
        files += ['images-jpeg-1k/' + c['cameraId'] + '.jpg' for c in selected]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for record in pool.map(lambda n: fetch(scene, n, args.root), files):
                records.append(record)
                (args.root / scene / 'download_manifest.json').write_text(json.dumps(records, indent=2))
                print(scene, Path(record['path']).name, record['bytes'], flush=True)
        print('COMPLETE', scene, len(selected), 'HDR views', flush=True)


if __name__ == '__main__':
    main()
