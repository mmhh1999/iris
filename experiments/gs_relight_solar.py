"""Solar-position service: pvlib NREL SPA, explicit civil time and scene north.

No irradiance, lux calibration, material decomposition, or scene training here.
"""
from functools import lru_cache
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib


def scene_direction(azimuth, elevation, north=0):
    """Surface->sun. North is clockwise from scene +Y toward +X, Z is up."""
    a, h, n = np.radians([azimuth, elevation, north])
    return [float(np.cos(h)*np.sin(a+n)), float(np.cos(h)*np.cos(a+n)), float(np.sin(h))]


def validated_inputs(query):
    def number(key, default, lo, hi):
        value = float(query.get(key, [default])[0])
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be within {lo}..{hi}')
        return value
    latitude = number('latitude', 39.742476, -90, 90)
    longitude = number('longitude', -105.1786, -180, 180)
    altitude = number('altitude', 0, -500, 9000)
    date = query.get('date', ['2026-06-21'])[0]
    parsed = pd.Timestamp(date)
    if parsed.strftime('%Y-%m-%d') != date or not 1900 <= parsed.year <= 2100:
        raise ValueError('date must be YYYY-MM-DD within 1900..2100')
    timezone = query.get('timezone', ['America/Denver'])[0]
    ZoneInfo(timezone)  # validate IANA zone without assuming midnight exists
    return latitude, longitude, altitude, date, timezone


@lru_cache(maxsize=32)
def solar_day(latitude, longitude, altitude, date, timezone):
    naive = pd.date_range(date, periods=1441, freq='min')
    # At a fall-back repeat use the first occurrence. Spring-gap minutes are
    # invalid, NOT silently shifted into a different civil time.
    localized = naive.tz_localize(timezone, ambiguous=True, nonexistent='NaT')
    valid = ~localized.isna()
    moments=localized[valid]
    positions = pvlib.solarposition.spa_python(moments.append(moments+pd.Timedelta(minutes=1)), latitude, longitude,
                                              altitude=altitude, delta_t=None)
    following=positions[['azimuth','apparent_elevation']].to_numpy()[len(moments):]
    rows = [None] * len(naive)
    for index, (_, row), moment, next_angle in zip(np.flatnonzero(valid), positions.iloc[:len(moments)].iterrows(), moments,following):
        rows[int(index)] = [float(row.azimuth), float(row.apparent_elevation),
                            int(moment.utcoffset().total_seconds()/60),float(next_angle[0]),float(next_angle[1])]
    return dict(algorithm='pvlib NREL SPA', pvlib_version=pvlib.__version__, latitude=latitude,
                longitude=longitude, altitude=altitude, date=date, timezone=timezone,
                columns=['azimuth_deg_clockwise_from_north', 'apparent_elevation_deg', 'UTC_offset_minutes','azimuth_one_absolute_minute_later','elevation_one_absolute_minute_later'],
                minute_step=1, samples=rows, invalid_minutes=np.flatnonzero(~valid).tolist(),
                repeated_time_policy='first occurrence (DST side); offset shown',
                scene_georeferencing='USER INPUT / EXAMPLE ONLY; capture location and north unmeasured')
