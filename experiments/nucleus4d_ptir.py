"""Original Nucleus4D Gaussians through the official PTIR Mitsuba integrators.

No mesh conversion, Gaussian decimation, or changes to the source archive.
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time
import types
import zipfile

import numpy as np
import mitsuba as mi
import drjit as dr
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/out/nucleus4d_ptir'
UPSTREAM = ROOT / 'third_party/ptir_mitsuba'


def load_integrators():
    # Load the official renderer without its training CLI/global dataset imports.
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    models = types.ModuleType('models')
    models.DisneyBSDF = module('ptir_disney', UPSTREAM / 'models/disney_bsdf.py').DisneyBSDF
    previous = sys.modules.get('models')
    sys.modules['models'] = models
    package = types.ModuleType('ptir_integrators')
    package.__path__ = [str(UPSTREAM / 'integrators')]
    sys.modules['ptir_integrators'] = package
    for name in ('common', 'reparam', 'volprim_rf', 'gsprim_prb'):
        module('ptir_integrators.' + name, UPSTREAM / 'integrators' / (name + '.py'))
    if previous is not None:
        sys.modules['models'] = previous
    else:
        del sys.modules['models']
    from ptir_integrators.gsprim_prb import GaussianPrimitivePrbIntegrator

    class WindowPTIR(GaussianPrimitivePrbIntegrator):
        """Two assumed transparent apertures for lighting rays only.

        Primary rays still see every original Gaussian. This explicitly models
        unknown glazing as clear openings; it is not recovered glass transport.
        """
        def __init__(self, props):
            self.portals = props.get('window_portals', True)
            super().__init__(props)

        def portal_distance(self, ray):
            t = (5.76 - ray.o.x) / dr.maximum(ray.d.x, 1e-8)
            p = ray(t)
            inside_y = ((p.y > -2.10) & (p.y < -1.38)) | ((p.y > .78) & (p.y < 1.50))
            inside_z = (p.z > -.55) & (p.z < .60) & ~((p.z > .07) & (p.z < .11))
            valid = self.portals & (ray.d.x > 1e-6) & (t > 0) & inside_y & inside_z
            return dr.select(valid, t, dr.inf)

        def shadow_ray_test(self, scene, sampler, pos, ray, active):
            ray = mi.Ray3f(ray)
            ray.maxt = dr.minimum(ray.maxt, self.portal_distance(ray))
            return super().shadow_ray_test(scene, sampler, pos, ray, active)

        def next_ray(self, scene, si, direction, offset, active):
            ray = super().next_ray(scene, si, direction, offset, active)
            ray.maxt = dr.minimum(ray.maxt, self.portal_distance(ray))
            return ray

    mi.register_integrator('nucleus_ptir', lambda p: WindowPTIR(p))


def source_arrays():
    with zipfile.ZipFile(ROOT / 'data_download/nucleus4d_20260929/1406-int.zip') as z:
        member = next(n for n in z.namelist() if n.endswith('/point_cloud.ply'))
        with z.open(member) as f:
            header = []
            while True:
                line = f.readline().decode().strip()
                header.append(line)
                if line == 'end_header':
                    break
            assert len([l for l in header if l.startswith('property float ')]) == 17
            data = np.frombuffer(f.read(), '<f4').reshape(-1, 17)
    assert len(data) == 6100978
    return data, member


def sensor(width, view=-1):
    if view < 0:
        height = round(width * 416 / 640)
        transform = mi.ScalarTransform4f.look_at(origin=[1.4, 1.7, .45], target=[5.65, -.35, -.45], up=[0, 0, 1])
        fov = 77.
    else:
        v = json.loads((ROOT / 'experiments/out/nucleus4d_material/views.json').read_text())[view]
        height = width
        r = np.array(v['rotation'])
        transform = mi.ScalarTransform4f.look_at(origin=v['position'], target=np.array(v['position']) + r[:, 2], up=-r[:, 1])
        fov = np.degrees(2 * np.arctan(v['width'] / (2 * v['K'][0][0])))
    return mi.load_dict({'type': 'perspective', 'fov': float(fov), 'fov_axis': 'x', 'to_world': transform,
                         'near_clip': .01, 'far_clip': 200.,
                         'sampler': {'type': 'independent', 'sample_count': 1},
                         'film': {'type': 'hdrfilm', 'width': width, 'height': height, 'pixel_format': 'rgb',
                                  'rfilter': {'type': 'box'}}})


def shape(data, material=False):
    xyz = np.ascontiguousarray(data[:, :3])
    scales = np.exp(data[:, 10:13])
    q = np.ascontiguousarray(data[:, 13:17][:, [1, 2, 3, 0]])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    opacity = 1 / (1 + np.exp(-np.clip(data[:, 9:10], -80, 80)))
    result = {'type': 'ellipsoids', 'centers': mi.TensorXf(xyz), 'scales': mi.TensorXf(scales),
              'quaternions': mi.TensorXf(q), 'opacities': mi.TensorXf(opacity),
              'sh_coeffs': mi.TensorXf(np.ascontiguousarray(data[:, 6:9]))}
    if material:
        # Smallest covariance axis, oriented toward the nearest capture camera.
        axis = np.eye(3, dtype=np.float32)[scales.argmin(1)]
        normals = Rotation.from_quat(q).apply(axis).astype('float32')
        cameras = np.array([v['position'] for v in json.loads((ROOT / 'experiments/out/nucleus4d_material/raw_images.json').read_text())])
        _, ci = cKDTree(cameras).query(xyz, workers=8)
        normals *= np.where(np.sum(normals * (cameras[ci] - xyz), axis=1) < 0, -1, 1)[:, None]
        rgb = np.clip(.5 + .28209479177387814 * data[:, 6:9], 0, 1)
        albedo = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4).astype('float32')
        # Explicitly a baked-appearance initialization, not recovered material.
        material_file = OUT / 'materials.npz'
        roughness = np.full((len(data), 1), .65, 'float32')
        metallic = np.zeros((len(data), 1), 'float32')
        if material_file.exists():
            with np.load(material_file) as m:
                assert m['albedo'].shape == albedo.shape
                albedo = m['albedo'].astype('float32')
                normals = m['normals'].astype('float32')
                roughness = m['roughness'].astype('float32').reshape(-1, 1)
                metallic = m['metallic'].astype('float32').reshape(-1, 1)
        result.update(normals=mi.TensorXf(normals), albedos=mi.TensorXf(albedo),
                      roughnesses=mi.TensorXf(roughness), metallics=mi.TensorXf(metallic))
    return result


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    mi.set_variant('cuda_ad_rgb')
    load_integrators()
    start = time.time()
    data, member = source_arrays()
    print('Loaded original Gaussians:', len(data), flush=True)
    config = {'type': 'scene', 'shape': shape(data, material=args.mode != 'original')}
    if args.mode == 'original':
        config['integrator'] = {'type': 'volprim_rf', 'max_depth': 512, 'srgb_primitives': True}
    else:
        config['integrator'] = {'type': 'nucleus_ptir', 'max_depth': args.bounces, 'gaussian_max_depth': 256,
                                'use_mis': True, 'geometry_threshold': .3, 'selfocc_offset_max': .05,
                                'separate_direct_indirect': True, 'window_portals':not args.closed}
        config['sky'] = {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [.5, .65, 1.]}}
        if args.sun:
            config['sun']={'type':'directional','direction':[-.65,.2,-.73], 'irradiance':{'type':'rgb','value':[5.,4.4,3.5]}}
    scene = mi.load_dict(config)
    print('Built full Gaussian BVH in', round(time.time()-start, 2), 'seconds', flush=True)
    camera = sensor(args.width, args.view)
    total = None
    for i in range(args.spp):
        tic = time.time()
        # Small wavefronts limit working memory; every Gaussian remains present.
        if args.mode == 'original':
            frame = mi.render(scene, sensor=camera, seed=i, spp=1)
        else:
            frame = scene.integrator().render(scene, sensor=camera, seed=i, spp=1)
        a = np.array(frame)
        total = a if total is None else total + a
        print('sample', i+1, 'seconds', round(time.time()-tic, 2), 'shape', a.shape, 'finite', bool(np.isfinite(a).all()), flush=True)
    result = total / args.spp
    stem = f'{args.mode}_v{args.view}_{args.width}'
    np.save(OUT / (stem + '.npy'), result)
    mi.util.write_bitmap(str(OUT / (stem + '.png')), result[:, :, :3])
    report = dict(source=member, gaussian_count=len(data), geometry_changed=False, decimated=False,
                  renderer='PTIR-Mitsuba with local compatibility/window adapter', shape='analytic ellipsoids', mode=args.mode,
                  spp=args.spp, seconds=time.time()-start, finite=bool(np.isfinite(result).all()),
                  material_status=('partial photo-prior initialization, not joint inverse optimization' if (OUT/'materials.npz').exists() else 'baked-color initialization') if args.mode != 'original' else 'original SH0',
                  window_portals=(not args.closed) if args.mode!='original' else False)
    (OUT / (stem + '.json')).write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['original', 'relight'], default='original')
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--view', type=int, default=-1)
    p.add_argument('--spp', type=int, default=1)
    p.add_argument('--bounces', type=int, default=3)
    p.add_argument('--sun', action='store_true')
    p.add_argument('--closed', action='store_true')
    run(p.parse_args())
