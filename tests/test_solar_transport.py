"""Unit tests for the SolarIR testbed's solar geometry and window transport (EXP0030).
Run: .venv/bin/python -m unittest tests.test_solar_transport"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import mitsuba as mi
import drjit as dr

if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

from experiments import solarir_scene as S
from utils.solar_geometry import sun_vector_world, vector_to_azimuth_elevation


class SolarConversionTests(unittest.TestCase):
    def test_cardinal_directions(self):
        np.testing.assert_allclose(sun_vector_world(180, 0), [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(sun_vector_world(90, 0), [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(sun_vector_world(123, 90), [0, 0, 1], atol=1e-12)

    def test_bearing_rotates_about_vertical(self):
        # scene +Y pointing 90 deg east of north: the true-south sun appears along scene +X
        np.testing.assert_allclose(sun_vector_world(180, 0, 90), [1, 0, 0], atol=1e-12)
        v = sun_vector_world(200, 30, 17)
        p = vector_to_azimuth_elevation(v, 17)
        self.assertAlmostEqual(p.azimuth_deg, 200, places=9)
        self.assertAlmostEqual(p.elevation_deg, 30, places=9)

    def test_ephemeris_times_enter_south_window(self):
        for t in S.TRAIN_TIMES + S.HELDOUT_TIMES:
            d = S.sun_dir(t)
            self.assertLess(d[1], 0, t)
            self.assertGreater(d[2], 0, t)


class WindowTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.obj = Path(cls.tmp.name) / 'room.obj'
        S.room_shell_obj(cls.obj)
        cls.geom = S.make_scene(cls.obj, np.full((4, 4, 3), 0.5, np.float32), S.WALL_GT, 'gt', with_emitters=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ray_traced_floor_patch_matches_projection(self):
        p, n = S.floor_texel_points()
        for t in S.TRAIN_TIMES + S.HELDOUT_TIMES:
            d = S.sun_dir(t)
            traced = S.sunlit(self.geom, p, n, d)[0].reshape(S.MODEL_TEX, S.MODEL_TEX)
            analytic = S.analytic_floor_sunlit(d)
            iou = (traced & analytic).sum() / max((traced | analytic).sum(), 1)
            self.assertGreater(analytic.sum(), 0, t)
            self.assertGreater(iou, 0.98, t)

    def test_window_centre_projects_into_patch(self):
        d = S.sun_dir('11:30')
        c = np.array([0.0, 0.0, sum(S.WIN_Z) / 2])
        q = c - (c[2] / d[2]) * d                       # follow the sunbeam back to the floor
        self.assertTrue(S.sunlit(self.geom, q[None], np.array([[0, 0, 1.0]]), d)[0, 0])
        far = np.array([[1.9, 3.9, 0.0]])
        self.assertFalse(S.sunlit(self.geom, far, np.array([[0, 0, 1.0]]), d)[0, 0])

    def test_sun_below_horizon_lights_nothing(self):
        p, n = S.floor_texel_points(16)
        self.assertFalse(S.sunlit(self.geom, p, n, [0, -0.9, -0.1]).any())
        self.assertFalse(S.analytic_floor_sunlit([0, -0.9, -0.1]).any())


class EmitterConventionTests(unittest.TestCase):
    def test_envmap_top_row_is_up(self):
        img = np.zeros((8, 16, 3), np.float32)
        img[:4] = 1.0
        em = mi.load_dict({'type': 'envmap', 'bitmap': mi.Bitmap(img), 'to_world': S.ENV_TO_WORLD})
        si = dr.zeros(mi.SurfaceInteraction3f, 2)
        si.wi = mi.Vector3f([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0])   # looking up, looking down
        v = np.array(em.eval(si))[0]
        self.assertGreater(v[0], 0.5)
        self.assertLess(v[1], 0.5)

    def test_env_padding_matches_mitsuba(self):
        h, w = 4, 6
        img = np.random.default_rng(0).uniform(0, 1, (h, w, 3)).astype(np.float32)
        em = mi.load_dict({'type': 'envmap', 'bitmap': mi.Bitmap(img)})
        ours = np.array(S.padded_env(mi.Float(img.ravel()), h, w, S.env_pad_index(h, w)))
        np.testing.assert_allclose(ours, np.array(mi.traverse(em)['data']), atol=1e-6)

    def test_upsampled_env_padding_matches_mitsuba(self):
        h, w, up = 3, 5, 4
        img = np.random.default_rng(1).uniform(0, 1, (h, w, 3)).astype(np.float32)
        em = mi.load_dict({'type': 'envmap', 'bitmap': mi.Bitmap(S.upsample(img, up))})
        ours = np.array(S.padded_env(mi.Float(img.ravel()), h, w, S.env_pad_index(h, w, up), up))
        np.testing.assert_allclose(ours, np.array(mi.traverse(em)['data']), atol=1e-6)

    def test_window_emitter_lights_the_room(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / 'room.obj'
            S.room_shell_obj(obj)
            sc = S.make_scene(obj, np.full((4, 4, 3), 0.5, np.float32), S.WALL_GT, 'W', S.init_state('W'))
            self.assertGreater(float(np.array(mi.render(sc, sensor=0, spp=4)).mean()), 0.01)


class PatchEstimationTests(unittest.TestCase):
    def test_true_direction_explains_rendered_patch(self):
        """The patch in a physically rendered image comes from transport, not a mask: the ephemeris
        direction must explain it better than directions 10 degrees away."""
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / 'room.obj'
            S.room_shell_obj(obj)
            tex = S.floor_texture_gt(0)
            d = S.sun_dir('11:30')
            sc = S.make_scene(obj, tex, S.WALL_GT, 'gt', d=d)
            imgs = [np.array(mi.render(sc, sensor=i, spp=32, seed=i)) for i in range(len(S.CAMERAS))]
            geom = S.make_scene(obj, tex, S.WALL_GT, 'gt', with_emitters=False)
            m = S.PatchMatcher(geom, imgs)
            az, el = S.az_el(d)
            s = m.iou(np.stack([d, S.dir_from_az_el(az + 10, el), S.dir_from_az_el(az, el + 10)]))
            self.assertGreater(s[0], 0.85)
            self.assertGreater(s[0], s[1] + 0.1)
            self.assertGreater(s[0], s[2] + 0.1)


if __name__ == '__main__':
    unittest.main()
