"""Local arbitrary-camera/daylight viewer; CUDA is idle until final gates pass."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import re

from aiohttp import web, WSMsgType
from nucleus4d_ptir_freeview_scene import ROOT, QUALITY, View

LOCAL = ROOT / 'experiments/out/nucleus4d_ptir_full'
STORAGE = Path(os.environ.get('NUCLEUS_PTIR_STORAGE', '/mnt/e/Datasets/Nucleus4D/20260929/1406-C-int-ptir-full'))


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class Service:
    def __init__(self, checkpoint, output, status_dir=LOCAL, probe=False, backend_factory=None, quality=None):
        self.checkpoint, self.output = Path(checkpoint), Path(output)
        self.status_dir, self.probe = Path(status_dir), probe
        self.quality = dict(quality or QUALITY)
        self.backend_factory = backend_factory
        self.backend = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptir-gpu')
        self.queue = asyncio.Queue(maxsize=1)
        self.sequence = 0; self.current = None; self.phase = 'waiting'; self.error = None
        self.task = None; self.closed = False; self.sockets = set(); self.send_locks = {}

    def status(self):
        pipeline = read_json(self.status_dir/'pipeline_status.json')
        training = read_json(self.status_dir/'training_status.json')
        ready = self.checkpoint.is_file()
        reason = '等待完整模型训练与外观检查通过。'
        if not self.probe:
            ready = ready and 'inverse' in pipeline.get('completed_stages', [])
            ready = ready and pipeline.get('status') == 'completed'
            ready = ready and all(read_json(self.status_dir/(name+'.json')).get('passed') is True
                                  for name in ('geometry_quality', 'inverse_quality', 'freeview_validation'))
        if ready:
            reason = '完整模型已就绪；首次请求会载入 GPU。' if self.backend is None else '模型已载入。'
        if self.error:
            reason = self.error
        return dict(ready=bool(ready), reason=reason, phase=self.phase, error=self.error,
                    probe=self.probe, quality=self.quality, default_view=asdict(View()),
                    pipeline=pipeline, training=training,
                    lighting_assumptions='示例旧金山地点与朝向；模拟 HDR 天空；不透明高斯光传输。')

    async def send(self, socket, value):
        if not socket.closed:
            try:
                async with self.send_locks.setdefault(socket, asyncio.Lock()):
                    if socket.closed:return
                    metadata = dict(value)
                    binary = metadata.pop('binary', None)
                    await socket.send_json(metadata)
                    if binary is not None:await socket.send_bytes(binary)
            except ConnectionError:
                pass

    async def submit(self, socket, raw):
        view = View.parse(raw)
        if not self.status()['ready']:
            await self.send(socket, dict(type='waiting', **self.status()))
            return
        self.sequence += 1
        if self.current:
            await self.send(self.current[1], dict(type='superseded', id=self.current[0]))
        job = (self.sequence, socket, view)
        self.current = job
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(job)
        await self.send(socket, dict(type='queued', id=job[0], view=asdict(view), quality=self.quality))

    def load_backend(self):
        if self.backend is None:
            if self.backend_factory is None:
                from nucleus4d_ptir_freeview_backend import PTIRBackend
                factory = PTIRBackend
            else:
                factory = self.backend_factory
            self.backend = factory(self.checkpoint, self.output, self.quality)
        return self.backend

    async def run(self):
        from nucleus4d_ptir_freeview_backend import Superseded
        loop = asyncio.get_running_loop()
        while not self.closed:
            job = await self.queue.get()
            number, socket, view = job
            def current():
                return not self.closed and not socket.closed and self.current is job
            if not current():
                continue
            def publish(event):
                if current():
                    event['id'] = number
                    loop.call_soon_threadsafe(lambda ws=socket, ev=event: asyncio.create_task(self.send(ws, ev)))
            try:
                self.phase = 'loading' if self.backend is None else 'rendering'
                await self.send(socket, dict(type=self.phase, id=number))
                backend = await loop.run_in_executor(self.executor, self.load_backend)
                if not current():
                    continue
                self.phase = 'rendering'
                await self.send(socket, dict(type='rendering', id=number))
                pixels, record = await loop.run_in_executor(self.executor, backend.render, view, publish, current)
                if current():
                    await self.send(socket, dict(type='frame', id=number, **record,
                                                binary=pixels.astype('<f4',copy=False).tobytes()))
                    self.error = None
            except Superseded:
                pass
            except Exception as exc:
                self.error = f'渲染失败：{type(exc).__name__}: {exc}'
                await self.send(socket, dict(type='error', id=number, message=self.error))
                # A CUDA device fault must not trigger endless automatic retries.
                self.phase = 'error'
                continue
            finally:
                if self.phase != 'error':
                    self.phase = 'idle'


def create_app(service):
    app = web.Application(client_max_size=65536)
    async def index(request):
        return web.FileResponse(Path(__file__).with_name('nucleus4d_ptir_freeview.html'))
    async def status(request):
        return web.json_response(service.status(), headers={'Cache-Control':'no-store'})
    async def frame(request):
        key = request.match_info['key']
        if not re.fullmatch(r'[a-f0-9]{24}', key):
            raise web.HTTPNotFound()
        path = service.output/(key+'.exr')
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Content-Disposition':f'attachment; filename="{key}.exr"'})
    async def websocket(request):
        origin = request.headers.get('Origin')
        if origin and origin != f'{request.scheme}://{request.host}':
            raise web.HTTPForbidden(text='Use the viewer on this same origin')
        socket = web.WebSocketResponse(heartbeat=30, max_msg_size=65536)
        await socket.prepare(request); service.sockets.add(socket)
        await service.send(socket, dict(type='status', **service.status()))
        try:
            async for message in socket:
                if message.type == WSMsgType.TEXT:
                    try:
                        value = json.loads(message.data)
                        if value.get('type') == 'render':
                            await service.submit(socket, value.get('view'))
                        elif value.get('type') == 'cancel':
                            if service.current and service.current[1] is socket:
                                service.current = None
                                if service.queue.full():
                                    service.queue.get_nowait()
                            await service.send(socket, dict(type='cancelled'))
                        else:
                            raise ValueError('Expected a render request')
                    except (ValueError, TypeError, KeyError) as exc:
                        await service.send(socket, dict(type='error', message=str(exc)))
        finally:
            service.sockets.discard(socket)
        return socket
    async def start(app):
        service.task = asyncio.create_task(service.run())
    async def stop(app):
        service.closed = True; service.current = None
        for socket in list(service.sockets):
            await socket.close()
        service.task.cancel()
        await asyncio.gather(service.task, return_exceptions=True)
        service.executor.shutdown(wait=True, cancel_futures=True)
    app.add_routes([web.get('/', index), web.get('/api/status', status),
                    web.get('/ws', websocket), web.get('/frames/{key}.exr', frame)])
    app.on_startup.append(start); app.on_cleanup.append(stop)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--checkpoint', type=Path, default=STORAGE/'inverse/weights.pt')
    parser.add_argument('--output', type=Path, default=STORAGE/'freeview_frames')
    parser.add_argument('--status-dir', type=Path, default=LOCAL)
    parser.add_argument('--probe', action='store_true', help='Explicit development checkpoint; never publishes final status')
    args = parser.parse_args()
    service = Service(args.checkpoint, args.output, args.status_dir, args.probe)
    web.run_app(create_app(service), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
