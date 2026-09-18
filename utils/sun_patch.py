# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sun-patch geometric matching: rasterizing predicted/observed patches
and searching over candidate sun directions to maximize their overlap
(Mode B, project brief section 7). Deliberately a plain geometric grid
search, not a differentiable/neural approach, per the brief's guidance
to prefer the simpler method while it remains sufficient.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from utils.solar_geometry import SolarPosition, sun_vector_world
from utils.window_geometry import Plane, WindowSurface, plane_to_2d, project_window_to_plane


@dataclass
class RasterGrid:
    """A fixed 2D rasterization grid on a plane, shared between the
    predicted and observed patches so their masks are comparable."""
    origin: np.ndarray  # (2,) plane-local 2D coords of the grid's corner
    extent: float        # grid side length, plane-local units
    resolution: int      # pixels per side

    def polygon_to_mask(self, polygon_2d: np.ndarray) -> np.ndarray:
        px = (polygon_2d - self.origin) / self.extent * self.resolution
        mask = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        cv2.fillPoly(mask, [px.astype(np.int32)], 1)
        return mask.astype(bool)

    def points_to_mask(self, points_2d: np.ndarray, dilate_px: int = 1) -> np.ndarray:
        """Rasterize a scattered point cloud (e.g. thresholded bright
        floor pixels back-projected onto the plane) into a mask,
        dilated slightly so a sparse point sample still yields a
        filled-in region comparable to a polygon mask."""
        mask = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        px = ((points_2d - self.origin) / self.extent * self.resolution).astype(np.int32)
        valid = (px[:, 0] >= 0) & (px[:, 0] < self.resolution) & \
                (px[:, 1] >= 0) & (px[:, 1] < self.resolution)
        mask[px[valid, 1], px[valid, 0]] = 1
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask.astype(bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def polygon_iou(poly_a_2d: np.ndarray, poly_b_2d: np.ndarray, grid: RasterGrid) -> float:
    return mask_iou(grid.polygon_to_mask(poly_a_2d), grid.polygon_to_mask(poly_b_2d))


def _predicted_patch_2d(window: WindowSurface, plane: Plane, az_deg: float, el_deg: float,
                         scene_north_bearing_deg: float):
    s = sun_vector_world(az_deg, el_deg, scene_north_bearing_deg)
    patch_3d = project_window_to_plane(window, s, plane)
    if patch_3d is None:
        return None
    return plane_to_2d(patch_3d, plane)


def search_sun_direction(window: WindowSurface, plane: Plane, observed_mask: np.ndarray,
                          grid: RasterGrid, scene_north_bearing_deg: float = 0.0,
                          coarse_step_deg: float = 5.0, refine_range_deg: float = 3.0,
                          refine_step_deg: float = 0.25):
    """Coarse-to-fine grid search over (azimuth, elevation) maximizing
    IoU between the predicted window->plane patch and a precomputed
    `observed_mask` (from `RasterGrid.polygon_to_mask` for a synthetic
    ground-truth patch, or `RasterGrid.points_to_mask` for thresholded
    real/rendered floor pixels back-projected onto the plane).

    Returns: (best_az_deg, best_el_deg, best_iou).
    """
    best = (-1.0, None, None)
    az_range = np.arange(0.0, 360.0, coarse_step_deg)
    el_range = np.arange(5.0, 90.0, coarse_step_deg)
    for az in az_range:
        for el in el_range:
            pred = _predicted_patch_2d(window, plane, az, el, scene_north_bearing_deg)
            if pred is None:
                continue
            iou = mask_iou(grid.polygon_to_mask(pred), observed_mask)
            if iou > best[0]:
                best = (iou, az, el)

    if best[1] is None:
        return None, None, 0.0

    # Local refinement around the coarse optimum.
    center_az, center_el = best[1], best[2]
    az_fine = np.arange(center_az - refine_range_deg, center_az + refine_range_deg + 1e-9, refine_step_deg)
    el_fine = np.arange(max(0.1, center_el - refine_range_deg),
                         min(89.9, center_el + refine_range_deg) + 1e-9, refine_step_deg)
    for az in az_fine:
        for el in el_fine:
            pred = _predicted_patch_2d(window, plane, az % 360.0, el, scene_north_bearing_deg)
            if pred is None:
                continue
            iou = mask_iou(grid.polygon_to_mask(pred), observed_mask)
            if iou > best[0]:
                best = (iou, az % 360.0, el)

    return best[1], best[2], best[0]
