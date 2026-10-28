"""Exercise the real full-count browser viewer without occupying the CUDA GPU."""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import time

import aiohttp

ROOT=Path(__file__).resolve().parents[1]


class CDP:
    def __init__(self,socket):self.socket=socket;self.sequence=0;self.errors=[]

    async def call(self,method,params=None):
        self.sequence+=1;identifier=self.sequence
        await self.socket.send_json(dict(id=identifier,method=method,params=params or {}))
        while True:
            message=await self.socket.receive_json(timeout=120)
            if message.get('method')=='Runtime.exceptionThrown':self.errors.append(message['params'])
            if message.get('id')!=identifier:continue
            if 'error' in message:raise RuntimeError(message['error'])
            return message.get('result',{})

    async def evaluate(self,expression):
        result=await self.call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
        if result.get('exceptionDetails'):raise RuntimeError(result['exceptionDetails'])
        return result.get('result',{}).get('value')


async def run(args):
    output=args.output.resolve();output.mkdir(exist_ok=True,parents=True)
    executable=Path('/home/minghao/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell')
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']='/tmp/iris-baked-browser-libs/root/usr/lib/x86_64-linux-gnu:'+env.get('LD_LIBRARY_PATH','')
    log=(output/'browser.log').open('w')
    browser=subprocess.Popen([str(executable),'--no-sandbox','--disable-dev-shm-usage',
        '--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader',
        '--remote-debugging-port=9342','--remote-allow-origins=*',
        '--user-data-dir=/tmp/iris-baked-chrome-navigation','about:blank'],stdout=log,stderr=subprocess.STDOUT,env=env,
        start_new_session=True)
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                try:
                    async with session.get('http://127.0.0.1:9342/json/list') as response:pages=await response.json()
                    if pages:break
                except (aiohttp.ClientError,OSError):pass
                await asyncio.sleep(.1)
            else:raise RuntimeError('Headless browser did not start')
            async with session.ws_connect(pages[0]['webSocketDebuggerUrl']) as socket:
                cdp=CDP(socket)
                await cdp.call('Runtime.enable');await cdp.call('Page.enable')
                # Step real frames individually: software WebGL must not queue
                # hundreds of expensive full-scene draws during screenshot capture.
                await cdp.call('Page.addScriptToEvaluateOnNewDocument',dict(source='''
window.requestAnimationFrame = callback => { window.__pendingFrame=callback; return 1; };
window.__stepFrame = () => { const callback=window.__pendingFrame; if(callback){window.__pendingFrame=null;callback(performance.now());} };
'''))
                # A small diagnostic viewport limits CPU fill rate. All 6.1M
                # Gaussians still load and draw; the actual viewer has no such cap.
                await cdp.call('Emulation.setDeviceMetricsOverride',dict(width=160,height=104,deviceScaleFactor=1,mobile=False))
                screenshots=[];states=[]
                # Separate browser invocations keep software WebGL from waiting
                # on disposal of another full 6.1M-point context.
                for mode in [args.mode]:
                    url='http://127.0.0.1:8769/nucleus4d_baked_relight/?mode='+mode
                    await cdp.call('Page.navigate',dict(url=url))
                    deadline=time.monotonic()+180
                    while time.monotonic()<deadline:
                        state=await cdp.evaluate('({ready:!!window.bakedReady,count:window.bakedCount||0,text:document.getElementById("message")?.textContent})')
                        if state and state['ready']:break
                        await cdp.evaluate('window.__stepFrame?.()')
                        await asyncio.sleep(.5)
                    else:raise RuntimeError(f'Full scene did not load: {mode} {state}')
                    asset=await cdp.evaluate('window.activeSceneAsset')
                    expected='/relit.splat' if mode=='relit' else '/nucleus4d_3dgs/full.splat'
                    if not asset or not asset.endswith(expected):
                        raise RuntimeError(f'Requested {mode} is unavailable; quality gate selected {asset}')
                    await cdp.evaluate('document.getElementById("info").style.display="none"')
                    for view in range(3):
                        await cdp.evaluate(f'window.selectView({view});')
                        await cdp.evaluate('window.__stepFrame()')
                        await asyncio.sleep(.5)
                        await cdp.evaluate('window.__stepFrame()')
                        state=await cdp.evaluate('window.bakedView()');states.append(dict(mode=mode,view=view,**state))
                        if args.screenshots:
                            frame=await cdp.call('Page.captureScreenshot',dict(format='png'))
                            name=f'{mode}_{view:02}.png';(output/name).write_bytes(base64.b64decode(frame['data']));screenshots.append(name)
                        print('BROWSER_VIEW',mode,view,state['count'],flush=True)
                before=await cdp.evaluate('window.bakedView().viewMatrix')
                await cdp.call('Input.dispatchKeyEvent',dict(type='keyDown',key='ArrowUp',code='ArrowUp',windowsVirtualKeyCode=38))
                await cdp.evaluate('window.__stepFrame()')
                await asyncio.sleep(.2)
                await cdp.call('Input.dispatchKeyEvent',dict(type='keyUp',key='ArrowUp',code='ArrowUp',windowsVirtualKeyCode=38))
                after=await cdp.evaluate('window.bakedView().viewMatrix')
                if before==after:raise RuntimeError('Navigation did not change the world camera')
                if args.screenshots:
                    frame=await cdp.call('Page.captureScreenshot',dict(format='png'))
                    (output/'relit_walk.png').write_bytes(base64.b64decode(frame['data']))
                if cdp.errors:raise RuntimeError(cdp.errors)
                report=dict(passed=True,scope='browser loading, three world cameras and actual keyboard movement only',
                    gaussians=6100978,renderer='same full-count renderer as original; SwiftShader software test',
                    diagnostic_viewport=[160,104],
                    screenshots=screenshots,states=states,keyboard_camera_changed=True,
                    pixel_capture_performed=args.screenshots,
                    active_scene_asset=asset,
                    original_quality_verified=False,physical_accuracy_verified=False,completed_at=time.time())
                (output/f'browser_validation_{args.mode}.json').write_text(json.dumps(report,indent=2))
                if args.mode=='relit':(output/'browser_validation.json').write_text(json.dumps(report,indent=2))
                print('BROWSER_VALIDATION',json.dumps(report),flush=True)
    finally:
        import signal
        browser.terminate()
        try:browser.wait(timeout=10)
        except subprocess.TimeoutExpired:browser.kill();browser.wait()
        try:os.killpg(browser.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/out/nucleus4d_baked_relight')
    parser.add_argument('--screenshots',action='store_true',help='Slow on software WebGL; structural navigation checks run by default')
    parser.add_argument('--mode',choices=['original','relit'],default='relit')
    asyncio.run(run(parser.parse_args()))
