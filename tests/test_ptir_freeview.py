"""Free-camera geometry, continuous sunlight, quality gates and WS frame routing."""
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from nucleus4d_ptir_freeview_scene import View, QUALITY, daylight_rgba
from nucleus4d_ptir_freeview import Service, create_app
from nucleus4d_ptir_freeview_backend import Superseded


class SceneTest(unittest.TestCase):
    def test_default_camera_and_free_translation_rotation(self):
        view = View.parse(asdict(View()))
        pose = view.pose()
        expected = np.array([5.65, -.35, -.45])-np.array(view.position)
        np.testing.assert_allclose(pose[:3,2], expected/np.linalg.norm(expected), atol=1e-7)
        np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(pose[:3,:3])), 1., places=6)
        moved = replace(view, position=(3., 1., .9), yaw=70., pitch=22.)
        self.assertFalse(np.array_equal(pose, moved.pose()))
        self.assertNotEqual(view.key('same-model'), moved.key('same-model'))

    def test_continuous_time_and_cloud_control_change_light(self):
        view = replace(View(), seconds=13*3600+17*60+12)
        a, sun_a = daylight_rgba(view)
        b, sun_b = daylight_rgba(replace(view, seconds=view.seconds+60))
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(b).all())
        self.assertFalse(np.array_equal(a, b))
        self.assertNotEqual(sun_a['world_direction'], sun_b['world_direction'])
        self.assertIn('13:17:12', sun_a['when'])
        overcast, _ = daylight_rgba(replace(view, cloud=1.))
        self.assertFalse(np.array_equal(a, overcast))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(QUALITY['spp'], 1024)

    def test_reject_invalid_camera_or_quality_override(self):
        for values in ({'pitch':90}, {'position':[1,2,float('nan')]},
                       {'fov':0}, {'spp':4}, {'date':'1999-01-01'},
                       {'seconds':86400}, {'cloud':-1}, {'position':[1,2]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                View.parse(values)


class FakeBackend:
    loaded = 0
    rendered = []

    def __init__(self, checkpoint, output, quality):
        type(self).loaded += 1
        self.quality = quality

    def render(self, view, publish, is_current):
        for index in range(5):
            time.sleep(.02)
            if not is_current():
                raise Superseded()
            publish(dict(type='progress', tiles=index+1, total=5, seconds=.1, remaining_seconds=.1))
            if index == 0:
                pixels = np.full((1,2,3), view.position[0]+view.seconds/86400, np.float32)
                publish(dict(type='tile', x=0, y=0, width=2, height=1, view=asdict(view),
                             quality=self.quality, binary=pixels.astype('<f4').tobytes()))
        type(self).rendered.append(view)
        # Each requested pose and minute produce different float32 pixels.
        image = np.full((2, 3, 3), view.position[0]+view.seconds/86400, np.float32)
        return image, dict(key=view.key('test', self.quality), view=asdict(view), quality=self.quality,
                          sun={'elevation':30}, gaussians=6100978, seconds=.1, exr='/frames/test.exr')


class ServiceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.local = Path(self.tmp.name)
        self.checkpoint = self.local/'weights.pt'; self.checkpoint.touch()
        FakeBackend.loaded = 0; FakeBackend.rendered = []
        self.quality = dict(QUALITY, width=3, height=2)
        self.service = Service(self.checkpoint, self.local/'frames', self.local,
                               backend_factory=FakeBackend, quality=self.quality)
        self.client = TestClient(TestServer(create_app(self.service)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close(); self.tmp.cleanup()

    def finish_gates(self):
        (self.local/'pipeline_status.json').write_text(json.dumps(dict(status='completed', completed_stages=['priors','geometry','inverse','viewer'])))
        for name in ('geometry_quality','inverse_quality','freeview_validation'):
            (self.local/(name+'.json')).write_text('{"passed":true}')

    async def receive_until(self, socket, kind):
        for _ in range(40):
            raw = await socket.receive(timeout=3)
            if raw.type == WSMsgType.BINARY:
                continue
            message = json.loads(raw.data)
            if message['type'] == kind:
                return message
        self.fail(f'No {kind} message received')

    async def test_model_is_not_loaded_before_both_gates_pass(self):
        socket = await self.client.ws_connect('/ws')
        self.assertFalse((await socket.receive_json())['ready'])
        await socket.send_json(dict(type='render', view=asdict(View())))
        self.assertFalse((await self.receive_until(socket,'waiting'))['ready'])
        self.assertEqual(FakeBackend.loaded, 0)
        self.finish_gates()
        (self.local/'inverse_quality.json').write_text('{"passed":false}')
        self.assertFalse(self.service.status()['ready'])

    async def test_arbitrary_view_and_time_return_full_float32_frame(self):
        self.finish_gates()
        socket = await self.client.ws_connect('/ws'); await socket.receive_json()
        first = replace(View(), position=(3., 2., .7), yaw=15, seconds=13*3600+17*60)
        await socket.send_json(dict(type='render', view=asdict(first)))
        tile = await self.receive_until(socket,'tile')
        tile_binary = await socket.receive_bytes(timeout=3)
        self.assertEqual(tile['quality']['spp'], 1024)
        self.assertEqual(len(tile_binary), 2*1*3*4)
        np.testing.assert_array_equal(np.frombuffer(tile_binary,dtype='<f4'),
                                      np.full(6,first.position[0]+first.seconds/86400,np.float32))
        metadata = await self.receive_until(socket,'frame')
        binary = await socket.receive_bytes(timeout=3)
        self.assertEqual(metadata['view']['position'], [3.,2.,.7])
        self.assertEqual(metadata['view']['seconds'], first.seconds)
        self.assertEqual(metadata['quality']['spp'], 1024)
        self.assertEqual(len(binary), 3*2*3*4)
        second = replace(first, position=(4., 2., .7), seconds=first.seconds+60)
        await socket.send_json(dict(type='render', view=asdict(second)))
        await self.receive_until(socket,'frame')
        new_binary = await socket.receive_bytes(timeout=3)
        self.assertNotEqual(binary, new_binary)
        self.assertEqual(FakeBackend.loaded, 1)

    async def test_new_view_supersedes_render_and_cancel_is_effective(self):
        self.finish_gates()
        socket = await self.client.ws_connect('/ws'); await socket.receive_json()
        await socket.send_json(dict(type='render', view=asdict(View())))
        await self.receive_until(socket,'progress')
        latest = replace(View(), yaw=45, seconds=45001)
        await socket.send_json(dict(type='render', view=asdict(latest)))
        metadata = await self.receive_until(socket,'frame')
        await socket.receive_bytes(timeout=3)
        self.assertEqual(metadata['view']['yaw'],45)
        self.assertEqual(len(FakeBackend.rendered),1)
        await socket.send_json(dict(type='render', view=asdict(View())))
        await self.receive_until(socket,'progress')
        await socket.send_json(dict(type='cancel'))
        await self.receive_until(socket,'cancelled')
        await asyncio.sleep(.08)
        self.assertEqual(len(FakeBackend.rendered),1)


if __name__ == '__main__':
    unittest.main()
