# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Window aperture geometry and sun-patch projection.

World-frame convention matches `utils/solar_geometry.py`: +X East,
+Y scene-north, +Z up. A "sun direction" `s` here always means the
unit vector pointing FROM the scene TOWARD the sun (light propagates
along `-s`).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WindowSurface:
    """A planar, axis-aligned-in-its-own-plane window aperture.

    corners_3d: 4x3 array, the four corners of the aperture in world
        space, in order (so consecutive corners share an edge — i.e.
        a simple, non-self-intersecting quadrilateral).
    confidence: placeholder for later (semantic/geometric) confidence
        scoring once window *detection* (as opposed to this module's
        known-geometry projection) is implemented; unused in Phase A.
    """
    corners_3d: np.ndarray  # (4, 3)
    confidence: float = 1.0

    def __post_init__(self):
        self.corners_3d = np.asarray(self.corners_3d, dtype=np.float64)
        assert self.corners_3d.shape == (4, 3), self.corners_3d.shape


@dataclass
class Plane:
    """A receiving plane (e.g. a floor), defined by a point on the
    plane and a unit normal."""
    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)


def project_point_along_direction(p: np.ndarray, propagation_dir: np.ndarray,
                                   plane: Plane) -> np.ndarray | None:
    """Project a single 3D point `p` along `propagation_dir` (unit
    vector, the direction light travels) onto `plane`.

    Returns None if the ray from `p` along `propagation_dir` is
    parallel to the plane (grazing/never-intersecting) or intersects
    it only "behind" `p` (t < 0, i.e. the plane is not in the light's
    forward path from this point).
    """
    denom = np.dot(propagation_dir, plane.normal)
    if abs(denom) < 1e-9:
        return None
    t = np.dot(plane.point - p, plane.normal) / denom
    if t < 0:
        return None
    return p + t * propagation_dir


def project_window_to_plane(window: WindowSurface, sun_direction_to_source: np.ndarray,
                             plane: Plane) -> np.ndarray | None:
    """Predict the sun-patch polygon cast by `window` onto `plane` for
    a given sun direction.

    Args:
        window: the aperture.
        sun_direction_to_source: unit vector pointing FROM the scene
            TOWARD the sun (as returned by
            `utils.solar_geometry.sun_vector_world`). Light propagates
            along the opposite direction.
        plane: the receiving surface (e.g. the floor).
    Returns:
        4x3 array of predicted patch corners (in the same corner
        order as `window.corners_3d`), or None if the sun is below
        the horizon relative to this window/plane pair (no patch is
        cast — e.g. window facing away from the sun, or the plane is
        behind the window from the light's perspective).
    """
    s = sun_direction_to_source / np.linalg.norm(sun_direction_to_source)
    propagation_dir = -s
    pts = []
    for corner in window.corners_3d:
        p = project_point_along_direction(corner, propagation_dir, plane)
        if p is None:
            return None
        pts.append(p)
    return np.stack(pts, axis=0)


def plane_to_2d(points_3d: np.ndarray, plane: Plane) -> np.ndarray:
    """Express points known to lie on `plane` in a local 2D basis of
    that plane (for rasterization/IoU in `utils/sun_patch.py`)."""
    normal = plane.normal
    # Build an arbitrary orthonormal basis (u, v) spanning the plane.
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    rel = points_3d - plane.point
    return np.stack([rel @ u, rel @ v], axis=-1)
