"""Native camera/light integration check on 128px tiles of the final 4K frame."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nucleus4d_ptir_freeview_scene import View, QUALITY
from nucleus4d_ptir_freeview_backend import PTIRBackend


def validate(checkpoint, local):
    import torch
    from nucleus4d_ptir_full_render import camera
    report_path = local/'freeview_validation.json'
    started = time.monotonic()
    backend = PTIRBackend(checkpoint, local/'freeview_probe')
    old = json.loads(report_path.read_text()) if report_path.exists() else {}
    if old.get('passed') and old.get('checkpoint_identity') == backend.identity:
        return old
    base = View()
    rotated = replace(base, position=(1.7, 1.5, .55), yaw=base.yaw+20., pitch=base.pitch+5.)
    views = [base, rotated, replace(base, seconds=base.seconds+17*60+12),
             replace(rotated, seconds=base.seconds+17*60+12)]
    records = []; arrays = []
    with torch.no_grad():
        for view in views:
            sun = backend.set_light(view)
            # Same focal length, absolute pixel coordinates, SPP and bounces as
            # the final frame. Only the centre tile is sampled for this check.
            x, y = (QUALITY['width']-128)//2, (QUALITY['height']-128)//2
            batch = camera(QUALITY['width'], QUALITY['height'], x, y, 128, 128,
                           pose=view.pose(), fov=view.fov)
            outputs = backend.model(batch, train=False, frame_id=y*QUALITY['width']+x+42)
            image = outputs['pred_pbr'][0].cpu().numpy()
            if not np.isfinite(image).all():
                raise RuntimeError('Free-view native integration produced nonfinite pixels')
            arrays.append(image.copy())
            records.append(dict(view=asdict(view), sun=sun, mean=float(image.mean()),
                                sha256=hashlib.sha256(image.tobytes()).hexdigest()))
            del outputs, batch
    if records[0]['sha256'] == records[1]['sha256']:
        raise RuntimeError('Changing the camera did not change the native rendered tile')
    if records[0]['sha256'] == records[2]['sha256']:
        raise RuntimeError('Changing continuous sunlight did not change the native rendered tile')
    np.savez_compressed(local/'freeview_probe/centre_tiles.npz', *arrays)
    report = dict(passed=True, checkpoint_identity=backend.identity, quality=QUALITY,
                  tile_window=[x,y,128,128], cases=records, seconds=time.monotonic()-started,
                  gaussians=backend.model.num_gaussians,
                  scope='Native camera/time integration at final SPP/bounces; centre tiles only, not a final 4K quality assessment')
    temporary = report_path.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_path)
    print('FREEVIEW_VALIDATION', json.dumps(report), flush=True)
    return report
