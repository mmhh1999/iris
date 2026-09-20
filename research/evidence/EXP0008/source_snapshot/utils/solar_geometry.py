# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Solar geometry utilities for daylight-aware IRIS.

World-frame convention used throughout this module and the rest of the
daylight-aware extension: +X = East, +Y = "scene north" (a reference
direction fixed to the room, which may differ from true north by
`scene_north_bearing_deg`), +Z = up. This is independent of whatever
world frame a given reconstructed mesh uses; Phase D/E is responsible
for calibrating `scene_north_bearing_deg` (and, if needed, a full
scene-to-ENU rotation) against the actual capture's coordinate system.

Two modes, per the project brief:
  - Mode A (metadata-derived): `solar_position` + `sun_vector_world`,
    using pvlib's NREL-based solar position algorithm rather than a
    hand-derived formula, to avoid introducing solar-geometry bugs.
  - Mode B (geometry-inferred): see `utils/sun_patch.py` /
    `utils/window_geometry.py` (built once Phase A validates that this
    module's forward geometry is correct).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pvlib


@dataclass
class SolarPosition:
    """Sun position in the standard meteorological convention.

    azimuth_deg: clockwise from true north, in [0, 360).
    elevation_deg: angle above the horizon, in [-90, 90]. Negative
        means the sun is below the horizon (night).
    """
    azimuth_deg: float
    elevation_deg: float


def solar_position(latitude_deg: float, longitude_deg: float, when: datetime,
                    altitude_m: float = 0.0) -> SolarPosition:
    """Compute true (geographic) solar position via pvlib.

    Args:
        latitude_deg, longitude_deg: capture location, WGS84 degrees.
        when: timezone-aware datetime of capture. A naive datetime is
            assumed to already be in the desired local/UTC frame and
            is used as-is (pvlib will treat it as UTC-naive); callers
            should pass timezone-aware datetimes to avoid ambiguity.
        altitude_m: capture site altitude above sea level, meters.
    Returns:
        SolarPosition with true (north-referenced) azimuth/elevation.
    """
    times = pd.DatetimeIndex([when])
    result = pvlib.solarposition.get_solarposition(
        times, latitude_deg, longitude_deg, altitude=altitude_m)
    row = result.iloc[0]
    # pvlib's apparent_elevation includes atmospheric refraction, which
    # is the physically-observed direction of the sun disc and is what
    # we want for matching against an observed sun patch.
    return SolarPosition(azimuth_deg=float(row['azimuth']),
                          elevation_deg=float(row['apparent_elevation']))


def sun_vector_world(azimuth_deg: float, elevation_deg: float,
                      scene_north_bearing_deg: float = 0.0) -> np.ndarray:
    """Convert (true) solar azimuth/elevation into a world-frame unit
    vector pointing FROM the scene TOWARD the sun.

    Args:
        azimuth_deg, elevation_deg: true solar position (see
            `solar_position`), or a candidate/hypothesized position in
            Mode B search.
        scene_north_bearing_deg: compass bearing (clockwise from true
            north) of the scene's world-frame +Y axis. E.g. if the
            room's +Y axis points 30 degrees east of true north,
            pass 30.0. Default 0.0 assumes scene +Y == true north.
    Returns:
        3-vector (x=East, y=scene-north, z=up), unit length, pointing
        toward the sun. For a Mitsuba `directional` emitter (whose
        `direction` parameter is the direction light *travels*), use
        `-sun_vector_world(...)`.
    """
    scene_azimuth_deg = azimuth_deg - scene_north_bearing_deg
    az = np.deg2rad(scene_azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x = np.cos(el) * np.sin(az)  # East component
    y = np.cos(el) * np.cos(az)  # scene-north component
    z = np.sin(el)               # up component
    v = np.array([x, y, z], dtype=np.float64)
    return v / np.linalg.norm(v)


def vector_to_azimuth_elevation(v: np.ndarray,
                                 scene_north_bearing_deg: float = 0.0) -> SolarPosition:
    """Inverse of `sun_vector_world`: world-frame unit vector (pointing
    toward the sun) -> (true) azimuth/elevation. Used to report Mode B
    search results and to compute angular error against ground truth.
    """
    v = v / np.linalg.norm(v)
    elevation_deg = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    scene_azimuth_deg = np.rad2deg(np.arctan2(v[0], v[1]))  # atan2(East, North)
    azimuth_deg = (scene_azimuth_deg + scene_north_bearing_deg) % 360.0
    return SolarPosition(azimuth_deg=float(azimuth_deg), elevation_deg=float(elevation_deg))


def angular_error_deg(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Angular separation in degrees between two direction vectors."""
    v_a = v_a / np.linalg.norm(v_a)
    v_b = v_b / np.linalg.norm(v_b)
    cos_angle = np.clip(np.dot(v_a, v_b), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def _self_test():
    """Sanity checks against well-known astronomical facts (not a
    substitute for pvlib's own validation, just a smoke test that this
    module's wrapper/convention code isn't obviously broken)."""
    from datetime import timezone

    # Summer solstice, solar noon, at the Tropic of Cancer (23.44N):
    # the sun should be within a fraction of a degree of directly
    # overhead (elevation ~90).
    when = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    # Tropic of Cancer longitude 0 has solar noon at ~12:00 UTC.
    pos = solar_position(23.44, 0.0, when)
    assert pos.elevation_deg > 85.0, f"expected near-overhead sun, got {pos}"

    # Round-trip: azimuth/elevation -> vector -> azimuth/elevation
    # should be the identity (up to floating point) for any bearing.
    for bearing in [0.0, 30.0, 200.0]:
        for az in [10.0, 90.0, 180.0, 270.0, 355.0]:
            for el in [-30.0, 0.0, 15.0, 60.0, 89.0]:
                v = sun_vector_world(az, el, bearing)
                back = vector_to_azimuth_elevation(v, bearing)
                assert abs(back.azimuth_deg - az) < 1e-3 or abs(back.azimuth_deg - az - 360) < 1e-3 \
                    or abs(back.azimuth_deg - az + 360) < 1e-3, (az, el, bearing, back)
                assert abs(back.elevation_deg - el) < 1e-3, (az, el, bearing, back)

    # Straight overhead (elevation=90) should map to world +Z regardless of azimuth/bearing.
    v = sun_vector_world(azimuth_deg=123.0, elevation_deg=90.0, scene_north_bearing_deg=45.0)
    assert np.allclose(v, [0, 0, 1], atol=1e-6), v

    # angular_error_deg sanity: identical vectors -> 0, opposite -> 180, orthogonal -> 90.
    assert angular_error_deg(np.array([1, 0, 0]), np.array([1, 0, 0])) < 1e-9
    assert abs(angular_error_deg(np.array([1, 0, 0]), np.array([-1, 0, 0])) - 180.0) < 1e-6
    assert abs(angular_error_deg(np.array([1, 0, 0]), np.array([0, 1, 0])) - 90.0) < 1e-6

    print("utils/solar_geometry.py self-test: all checks passed")


if __name__ == '__main__':
    _self_test()
