"""Backend-neutral geometric evidence, never a semantic sun-patch classifier.

Callers select a Mitsuba variant before loading a scene. Cameras are pinhole,
OpenCV c2w; pixel indices refer to centers at (x+.5, y+.5).
"""
import numpy as np


def pinhole_rays(pixels_xy, K, c2w):
    pixels = np.asarray(pixels_xy, dtype=float)
    K = np.asarray(K, dtype=float)
    pose = np.asarray(c2w, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError('pixels must be finite Nx2 indices')
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('invalid camera intrinsics')
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError('only zero-skew pinhole intrinsics supported')
    if pose.shape not in ((3, 4), (4, 4)) or not np.isfinite(pose).all():
        raise ValueError('invalid camera pose')
    R = pose[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1, atol=1e-4):
        raise ValueError('camera rotation must be proper orthonormal')
    dirs = np.column_stack(((pixels[:, 0] + .5 - K[0, 2]) / K[0, 0],
                            (pixels[:, 1] + .5 - K[1, 2]) / K[1, 1], np.ones(len(pixels)))) @ R.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.broadcast_to(pose[:3, 3], dirs.shape).copy(), dirs


def candidate_geometry(mask, scene, K, c2w, n_samples=256, seed=0):
    import mitsuba as mi
    mask = np.asarray(mask)
    if mask.ndim != 2 or n_samples < 1:
        raise ValueError('expected 2D mask and positive sample count')
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return dict(n_samples=0, valid_fraction=0., median_distance=None, median_hit_pos=None)
    idx = np.random.default_rng(seed).choice(len(xs), min(n_samples, len(xs)), replace=False)
    origins, dirs = pinhole_rays(np.column_stack((xs[idx], ys[idx])), K, c2w)
    ray = mi.Ray3f(mi.Point3f(*[mi.Float(origins[:, i]) for i in range(3)]),
                   mi.Vector3f(*[mi.Float(dirs[:, i]) for i in range(3)]))
    si = scene.ray_intersect(ray)
    valid = np.asarray(si.is_valid(), dtype=bool)
    distances = np.asarray(si.t)[valid]
    # Read scalar components explicitly; avoid version-dependent vector layout.
    points = np.column_stack([np.asarray(si.p[i]) for i in range(3)])[valid]
    return dict(n_samples=len(idx), valid_fraction=float(valid.mean()),
                median_distance=float(np.median(distances)) if len(distances) else None,
                median_hit_pos=np.median(points, axis=0).tolist() if len(points) else None)


def geometry_decision(evidence, max_invalid_frac=.5, aabb_diag=None, max_depth_factor=1.5):
    if not 0 <= max_invalid_frac <= 1 or max_depth_factor <= 0:
        raise ValueError('invalid geometry thresholds')
    if aabb_diag is not None and (not np.isfinite(aabb_diag) or aabb_diag <= 0):
        raise ValueError('invalid scene extent')
    if not evidence['n_samples'] or evidence['median_distance'] is None:
        return 'no_surface'
    if 1 - evidence['valid_fraction'] > max_invalid_frac:
        return 'insufficient_surface_support'
    if aabb_diag is not None and evidence['median_distance'] > aabb_diag * max_depth_factor:
        return 'distant_surface'
    return 'surface_supported_unverified'


def scannetpp_pose_in_mesh_frame(nerfstudio_pose):
    """Invert ScanNet++ common/utils/nerfstudio.py convert_frames.

    World-axis swap is distinct from the OpenGL -> OpenCV camera-axis flip.
    Applies only to official ScanNet++ exported transforms, not generic NeRF.
    """
    pose = np.asarray(nerfstudio_pose, dtype=float)
    if pose.shape != (4, 4) or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError('expected homogeneous ScanNet++ c2w')
    world = np.array([[0,1,0,0], [1,0,0,0], [0,0,-1,0], [0,0,0,1]])
    return world @ pose @ np.diag([1., -1., -1., 1.])


def validate_scannetpp_bounds(nerfstudio_bounds, mesh_bounds):
    bounds = np.asarray(nerfstudio_bounds)
    expected = bounds[:, [1,0,2]].copy()
    expected[:,2] = -bounds[::-1,2]
    if not np.allclose(expected, mesh_bounds, atol=1e-4, rtol=1e-5):
        raise ValueError('mesh and ScanNet++ camera world bounds disagree')
