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
class PatchCandidate:
    """A single candidate sun-patch region found by `screen_image_for_sun_patches`."""
    mask: np.ndarray            # boolean, image-shaped
    contour: np.ndarray         # cv2 contour (pixel coords)
    area_px: int
    rectangularity: float       # contour_area / min_area_rect_area, in [0,1], 1=perfectly rectangular
    polygon_vertices: int       # vertex count of the simplified polygon approximation
    edge_touch_frac: float      # fraction of the region's bounding box perimeter touching the image border
    score: float                # combined confidence in [0,1]


def screen_image_for_sun_patches(image_rgb: np.ndarray, bright_percentile: float = 97.0,
                                  min_area_frac: float = 0.001, max_area_frac: float = 0.5,
                                  top_k: int = 5):
    """2D, geometry-free screening: find candidate cast sun-patch regions in an
    arbitrary photo, ranked by a combination of brightness, size, and shape
    (rectangularity + low polygon-vertex-count) evidence.

    This is deliberately NOT the same test as IRIS's own emitter-extraction
    criterion (`extract_emitter_ldr.py`'s `mean_LDR_max_channel > threshold`,
    see IRIS_ARCHITECTURE_AUDIT.md Sec.4) -- that criterion only looks at
    brightness and would equally flag a light fixture, a reflective highlight,
    or a genuine sun patch. The `rectangularity`/`polygon_vertices` features
    here specifically target the fact that a sun patch cast through a
    (typically rectangular) window aperture projects to a quadrilateral with
    comparatively straight edges, which most other bright-region causes
    (specular highlights, light fixtures, blown-out sky seen through a
    window) do not share. This is a *screening* heuristic to prioritize
    candidate images/regions for the geometry-grounded Mode B analysis
    (`search_sun_direction`) once real window/floor geometry is available for
    a scene -- not a replacement for that analysis.

    Args:
        image_rgb: HxWx3 float array, arbitrary scale (will be normalized).
        bright_percentile: pixels above this percentile of image brightness
            are candidates (adaptive, not an absolute threshold, since real
            photos vary widely in exposure).
        min_area_frac/max_area_frac: reject components too small (noise/
            specular highlights) or too large (the whole image blown out)
            relative to total image area.
        top_k: return at most this many candidates, ranked by score.
    Returns:
        list of PatchCandidate, sorted by descending score.
    """
    img = image_rgb.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    brightness = img.max(axis=-1)

    thresh = np.percentile(brightness, bright_percentile)
    bright_mask = (brightness >= thresh).astype(np.uint8)

    h, w = bright_mask.shape
    total_area = h * w
    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    candidates = []
    for label in range(1, n_components):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_frac * total_area or area > max_area_frac * total_area:
            continue
        comp_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        if contour_area <= 0:
            continue

        rect = cv2.minAreaRect(contour)
        rect_area = max(rect[1][0] * rect[1][1], 1e-6)
        rectangularity = float(np.clip(contour_area / rect_area, 0.0, 1.0))

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        n_vertices = len(approx)
        # reward near-quadrilateral shapes (4-8 vertices after simplification);
        # penalize highly irregular contours (many vertices) or degenerate ones (<3)
        shape_score = 1.0 if 4 <= n_vertices <= 8 else max(0.0, 1.0 - 0.1 * abs(n_vertices - 6))

        x, y, bw, bh = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
            stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        touches = [x <= 0, y <= 0, x + bw >= w, y + bh >= h]
        edge_touch_frac = sum(touches) / 4.0

        size_score = float(np.clip(area / total_area, 0, max_area_frac) / max_area_frac)
        score = 0.5 * rectangularity + 0.3 * shape_score + 0.2 * size_score
        score *= (1.0 - 0.3 * edge_touch_frac)  # soft penalty, not a hard filter

        candidates.append(PatchCandidate(
            mask=(labels == label), contour=contour, area_px=int(area),
            rectangularity=rectangularity, polygon_vertices=n_vertices,
            edge_touch_frac=edge_touch_frac, score=float(score)))

    candidates.sort(key=lambda c: -c.score)
    return candidates[:top_k]


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
