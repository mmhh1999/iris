"""Resident full-quality PTIR model; every frame uses the requested world camera."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from nucleus4d_ptir_freeview_scene import QUALITY, daylight_rgba


class Superseded(Exception):
    pass


class PTIRBackend:
    def __init__(self, checkpoint, output, quality=None):
        # Imports and CUDA initialization occur only in the serialized GPU worker.
        import torch
        from omegaconf import OmegaConf
        from threedgrut.model.model import MixtureOfGaussians
        from threedgrut.model.light import create_environment
        from threedgrut.render import Renderer
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.quality = dict(quality or QUALITY)
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        stat = checkpoint.stat()
        code = b''.join(Path(__file__).with_name(name).read_bytes() for name in (
            'nucleus4d_ptir_freeview_backend.py', 'nucleus4d_ptir_freeview_scene.py',
            'nucleus4d_ptir_full_render.py'))
        self.identity = f'{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}:{hashlib.sha256(code).hexdigest()}'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        conf = saved['config']; OmegaConf.set_struct(conf, False)
        if conf.render.method != '3dgptir' or 'environment_state' not in saved:
            raise ValueError('Free-view relighting requires an inverse PTIR checkpoint')
        conf.render.render_spp = self.quality['spp']
        conf.render.render_bounces = self.quality['bounces']
        conf.render.spp_chunk = self.quality['spp_chunk']
        conf.render.filter_type = 'none'; conf.model.optimize_environment = False
        self.model = MixtureOfGaussians(conf)
        self.model.init_from_checkpoint(saved, setup_optimizer=False)
        self.model.requires_grad_(False); self.model.eval()
        original = create_environment(device='cuda', environment_type='spherical_gaussian',
                                      optimize_environment=False)
        original.load_state_dict(saved['environment_state']); original.configure_optimization(False)
        self.scale = max(.001, float(original.get_environment()[...,:3].mean().detach().cpu()))/.4
        Renderer._set_model_environment(self.model, original.get_environment_parameter())
        self.model.build_acc()
        del saved, original
        self.light_key = None

    def set_light(self, view):
        import torch
        from threedgrut.model.light import create_environment
        from threedgrut.render import Renderer
        light_key = (view.date, view.seconds, view.timezone, view.latitude,
                     view.longitude, view.north, view.cloud)
        if light_key != self.light_key:
            rgba, sun = daylight_rgba(view, self.scale)
            light = create_environment(device='cuda', environment_type='2d', optimize_environment=False)
            light._set_environment_tensor(torch.from_numpy(rgba).to('cuda'))
            self.model.renderer.environment_type = '2d'; self.model.conf.environment.type = '2d'
            Renderer._set_model_environment(self.model, light.get_environment_parameter())
            self.model.environment_alias_table = light.build_alias_table()
            self.light_key = light_key; self.sun = sun
        return self.sun

    def render(self, view, publish, is_current):
        import cv2
        import torch
        from nucleus4d_ptir_full_render import camera
        quality = self.quality
        width, height, tile = quality['width'], quality['height'], quality['tile']
        key = view.key(self.identity, quality)
        data_path, record_path = self.output / (key+'.npy'), self.output / (key+'.json')
        if not is_current():
            raise Superseded()
        if data_path.exists() and record_path.exists() and (self.output/(key+'.exr')).exists():
            record = json.loads(record_path.read_text())
            return np.load(data_path, allow_pickle=False), record
        started = time.monotonic()
        with torch.no_grad():
            self.set_light(view)
            pose = view.pose()
            image = np.empty((height, width, 3), dtype=np.float32)
            total = math.ceil(width/tile)*math.ceil(height/tile)
            completed = 0
            windows = [(x,y) for y in range(0,height,tile) for x in range(0,width,tile)]
            windows.sort(key=lambda xy:(xy[0]+tile/2-width/2)**2+(xy[1]+tile/2-height/2)**2)
            for x,y in windows:
                if not is_current():
                    raise Superseded()
                tw, th = min(tile, width-x), min(tile, height-y)
                batch = camera(width, height, x, y, tw, th, pose=pose, fov=view.fov)
                outputs = self.model(batch, train=False, frame_id=y*width+x+42)
                pixels = outputs['pred_pbr'][0].detach().cpu().numpy()
                if not np.isfinite(pixels).all():
                    raise RuntimeError('Nonfinite pixels from native PTIR renderer')
                image[y:y+th, x:x+tw] = pixels
                del outputs, batch
                completed += 1
                elapsed = time.monotonic()-started
                publish(dict(type='tile', x=x, y=y, width=tw, height=th, view=asdict(view),
                             quality=quality, binary=pixels.astype('<f4',copy=False).tobytes()))
                publish(dict(type='progress', tiles=completed, total=total, seconds=elapsed,
                             remaining_seconds=elapsed*(total-completed)/completed))
        if not is_current():
            raise Superseded()
        # Full float32 data are returned and archived; the browser displays sRGB.
        exr = self.output / (key+'.exr')
        if not cv2.imwrite(str(exr), image[...,::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
            raise IOError(f'Cannot save {exr}')
        np.save(data_path, image, allow_pickle=False)
        record = dict(key=key, view=asdict(view), quality=quality, sun=self.sun,
                      gaussians=self.model.num_gaussians, seconds=time.monotonic()-started,
                      checkpoint_identity=self.identity, renderer='official PTIR-GS CUDA/OptiX',
                      exr=f'/frames/{key}.exr')
        record_path.write_text(json.dumps(record, indent=2))
        # Keep the three most recent complete frames; never evict the current one.
        records = sorted(self.output.glob('*.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        for old in records[3:]:
            if old.stem != key:
                for suffix in ('.json', '.exr', '.npy'):
                    old.with_suffix(suffix).unlink(missing_ok=True)
        return image, record
