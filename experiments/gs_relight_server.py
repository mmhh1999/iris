"""Loopback staged viewer, SPA API and bounded local world-sun worker."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from gs_relight_solar import solar_day, validated_inputs
from gs_relight_baseline import selected_baseline

ROOT = Path(__file__).resolve().parent/'out'


class Handler(SimpleHTTPRequestHandler):
    def json_response(self, response, status=200):
        body = json.dumps(response, allow_nan=False, separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        origin = self.headers.get('Origin', '')
        if origin in ('http://localhost:8769', 'http://127.0.0.1:8769',
                      'http://localhost:8771', 'http://127.0.0.1:8771'):
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        super().end_headers()

    def do_GET(self):
        url = urlsplit(self.path)
        baseline = selected_baseline()
        if baseline:
            if url.path.rstrip('/') in ('', '/index.html', '/gs_relight_stages',
                    '/gs_relight_stages/app', '/gs_relight_stages/app/index.html',
                    '/gs_relight_stages/app_world', '/gs_relight_stages/app_world/index.html'):
                self.send_response(302)
                self.send_header('Location', baseline['viewer'])
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if url.path.startswith('/api/world/'):
                self.json_response({'error': '后续实验已退出当前基线，请使用已恢复的旧版。',
                                    'viewer': baseline['viewer']}, 410)
                return
        if url.path.startswith('/api/world/sun')or url.path.startswith('/api/world/environment'):
            origin = self.headers.get('Origin', '')
            if origin and origin not in ('http://localhost:8771', 'http://127.0.0.1:8771'):
                self.json_response({'error': 'Local viewer origin required'}, 403)
                return
            from gs_relight_sun_runtime import get_runtime, validated
            try:
                runtime = get_runtime()
                if url.path in ('/api/world/sun','/api/world/environment'):
                    query=parse_qs(url.query);args=list(validated(query));environment=None
                    if url.path=='/api/world/environment':
                        from gs_relight_environment_models import validate, hdr_direction
                        environment=validate(query)
                        if environment['source']!='physical':args[:2]=hdr_direction(environment)[:2]
                    status, response = runtime.submit(*args,environment=environment)
                    self.json_response(response, status)
                elif url.path.startswith('/api/world/sun/cache/'):
                    body = runtime.cached(url.path.removeprefix('/api/world/sun/cache/'))
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Cache-Control', 'private, max-age=31536000, immutable')
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.json_response({'error': 'Unknown API route'}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                self.json_response({'error': str(exc)}, 400)
            except FileNotFoundError:
                self.json_response({'error': '太阳缓存已过期，请重新计算'}, 404)
            return
        if url.path == '/api/solar/day':
            try:
                response = solar_day(*validated_inputs(parse_qs(url.query)))
                status = 200
            except (ValueError, KeyError, TypeError) as exc:
                response, status = {'error': str(exc)}, 400
            self.json_response(response, status)
        else:
            super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8771)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT)))
    baseline = selected_baseline()
    print('VIEWER '+(baseline['viewer'] if baseline else f'http://localhost:{args.port}/gs_relight_stages/app/'), flush=True)
    server.serve_forever()
