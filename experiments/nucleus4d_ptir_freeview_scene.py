"""Camera and daylight inputs shared by the arbitrary-view PTIR service and tests."""
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUALITY = dict(width=3840, height=2496, spp=1024, bounces=8,
               tile=128, spp_chunk=4, dtype='float32', denoising=False)


def bounded(value, low, high, name):
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


@dataclass(frozen=True)
class View:
    position: tuple = (1.4, 1.7, .45)
    yaw: float = math.degrees(math.atan2(-2.05, 4.25))
    pitch: float = math.degrees(math.atan2(-.9, math.hypot(4.25, 2.05)))
    fov: float = 77.
    date: str = '2026-03-21'
    seconds: float = 13 * 3600.
    timezone: str = 'America/Los_Angeles'
    latitude: float = 37.7749
    longitude: float = -122.4194
    north: float = 90.
    cloud: float = 0.

    @classmethod
    def parse(cls, values):
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError('Unknown camera/daylight fields')
        data = asdict(cls()); data.update(values)
        if not isinstance(data['position'], (tuple, list)) or len(data['position']) != 3:
            raise ValueError('position must contain three coordinates')
        data['position'] = tuple(bounded(x, -10000, 10000, 'position') for x in data['position'])
        limits = dict(yaw=(-1e6, 1e6), pitch=(-89, 89), fov=(20, 120),
                      seconds=(0, 86399.999), latitude=(-89.9, 89.9),
                      longitude=(-180, 180), north=(-360, 360), cloud=(0, 1))
        for key, (lo, hi) in limits.items():
            data[key] = bounded(data[key], lo, hi, key)
        data['yaw'] = (data['yaw'] + 180) % 360 - 180
        day = dt.date.fromisoformat(data['date'])
        if not 2000 <= day.year <= 2100:
            raise ValueError('date must be between 2000 and 2100')
        ZoneInfo(data['timezone'])
        return cls(**data)

    def pose(self):
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        forward = np.array([math.cos(pitch)*math.cos(yaw),
                            math.cos(pitch)*math.sin(yaw), math.sin(pitch)])
        right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.stack([right, down, forward], axis=1)
        pose[:3, 3] = self.position
        return pose

    def when(self):
        day = dt.date.fromisoformat(self.date)
        # Local civil time is interpreted in the explicitly selected timezone.
        midnight = dt.datetime.combine(day, dt.time())
        return (midnight + dt.timedelta(seconds=self.seconds)).replace(tzinfo=ZoneInfo(self.timezone))

    def key(self, checkpoint_identity, quality=QUALITY):
        payload = dict(view=asdict(self), checkpoint=checkpoint_identity, quality=quality)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def daylight_rgba(view, scale=1., shape=(1024, 2048)):
    """Synthetic clear/overcast HDR sky + finite solar disk at arbitrary time."""
    sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    from utils.solar_geometry import solar_position, sun_vector_world
    h, w = shape
    phi = ((np.arange(h, dtype=np.float32)+.5)/h-.5)*np.pi
    theta = ((np.arange(w, dtype=np.float32)+.5)/w*2-1)*np.pi
    z = -np.sin(phi)[:, None]
    clear = np.broadcast_to(np.array([.4, .6, 1.], np.float32) *
                            np.maximum(.15, .55+.45*z)[..., None], (h,w,3)).copy()
    overcast = np.broadcast_to(np.array([.7, .72, .75], np.float32) *
                               np.maximum(.1, (1+2*z)/3)[..., None], (h,w,3)).copy()
    clear[z[:,0] < 0] = [.06, .055, .05]
    overcast[z[:,0] < 0] = [.06, .055, .05]
    sun = solar_position(view.latitude, view.longitude, view.when())
    direction = np.asarray(sun_vector_world(sun.azimuth_deg, sun.elevation_deg, view.north), np.float32)
    dot = (np.cos(phi)[:,None] * (np.sin(theta)[None]*direction[0] +
           np.cos(theta)[None]*direction[1]) + z*direction[2])
    sigma = math.radians(.266)/2
    disk = np.exp(-np.maximum(0, 1-dot)/(sigma*sigma))
    solid_angle = float(disk.sum(axis=1) @ (np.cos(phi)*(2*np.pi/w)*(np.pi/h)))
    power = 5*max(0., min(1., math.sin(math.radians(sun.elevation_deg))/.65))
    warmth = max(0., min(1., sun.elevation_deg/35))
    color = np.array([1, .72+.24*warmth, .42+.44*warmth], np.float32)
    rgb = clear*(1-view.cloud) + overcast*view.cloud
    rgb += disk[...,None] * (power/max(solid_angle, 1e-10)) * color * (1-view.cloud)**2
    # Fade synthetic daylight out below the horizon; this is not a night-light model.
    sky_fade = min(1., max(0., (sun.elevation_deg+12)/12))
    rgb *= float(scale)*sky_fade
    rgba = np.concatenate([rgb, np.ones((h,w,1), np.float32)], axis=-1)
    return rgba, dict(azimuth=sun.azimuth_deg, elevation=sun.elevation_deg,
                      when=view.when().isoformat(), world_direction=direction.tolist())
