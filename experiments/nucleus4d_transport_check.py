"""Real full-count browser rendering, slider isolation, and moving-shadow checks."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import aiohttp
import numpy as np
from PIL import Image

from nucleus4d_baked_relight_check import CDP
from nucleus4d_transport_geometry import OUT


async def main():
    executable=Path('/home/minghao/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell')
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(OUT.parent/'nucleus4d_weather_presets/browser_runtime')+':'+env.get('LD_LIBRARY_PATH','')
    with (OUT/'browser.log').open('w') as log:
        browser=subprocess.Popen([str(executable),'--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--remote-debugging-port=9358','--remote-allow-origins=*','--user-data-dir=/tmp/iris-transport-check','about:blank'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                try:
                    async with session.get('http://127.0.0.1:9358/json/list') as r:pages=await r.json()
                    if pages:break
                except (aiohttp.ClientError,OSError):pass
                await asyncio.sleep(.1)
            else:raise RuntimeError('Browser did not start')
            async with session.ws_connect(pages[0]['webSocketDebuggerUrl']) as ws:
                c=CDP(ws);await c.call('Runtime.enable');await c.call('Page.enable')
                await c.call('Page.addScriptToEvaluateOnNewDocument',dict(source='''window.requestAnimationFrame=f=>{window.__pendingFrame=f;return 1};window.__stepFrame=(now=performance.now())=>{const f=window.__pendingFrame;if(f){window.__pendingFrame=null;f(now)}};'''))
                await c.call('Emulation.setDeviceMetricsOverride',dict(width=160,height=104,deviceScaleFactor=1,mobile=False))
                await c.call('Page.navigate',dict(url='http://127.0.0.1:8769/nucleus4d_transport_v3/'))
                for _ in range(600):
                    state=await c.evaluate('({s:window.transportState?.(),error:window.transportError,message:document.getElementById("message")?.textContent})')
                    if state.get('error'):raise RuntimeError(state)
                    if state.get('s',{}).get('ready') and state['s']['gaussians']==6100978:break
                    if c.errors:raise RuntimeError(c.errors)
                    # The original renderer starts its sorting worker from a
                    # view frame. Stepping is required while the model loads.
                    await c.evaluate('window.__stepFrame?.()')
                    await asyncio.sleep(.3)
                else:raise RuntimeError(f'Full model and fields failed to load: {state}')
                if state['s']['glError']:raise RuntimeError(state)
                async def frame(name,debug=False):
                    result=await c.evaluate('''(()=>{window.__stepFrame();const g=document.getElementById('canvas').getContext('webgl2');return {pixels:transportReadPixels(),world:transportReadWorld(),sunMask:''' + ('transportReadSunMask()' if debug else 'null') + ''',visibility:''' + ('transportReadPixels(1)' if debug else 'null')+''',width:g.drawingBufferWidth,height:g.drawingBufferHeight,state:transportState()};})()''')
                    pixels=np.array(result.pop('pixels'),np.uint8).reshape(result['height'],result['width'],4);Image.fromarray(pixels[::-1]).save(OUT/f'browser_{name}.png')
                    if result['state']['glError']:raise RuntimeError(result['state'])
                    if result['world'] is not None:np.save(OUT/f'browser_{name}_world.npy',np.array(result.pop('world'),np.float32).reshape(result['height'],result['width'],4))
                    if result['visibility'] is not None:np.save(OUT/f'browser_{name}_visibility.npy',np.array(result.pop('visibility'),np.uint8).reshape(result['height'],result['width'],4))
                    if result['sunMask'] is not None:np.save(OUT/f'browser_{name}_sun_mask.npy',np.array(result.pop('sunMask'),np.float32).reshape(result['height'],result['width']))
                    result.pop('world',None);result.pop('visibility',None);result.pop('sunMask',None);print('BROWSER_FRAME',name,float(pixels[:,:,:3].mean()),flush=True);return pixels,result
                original,o=await frame('original')
                camera=await c.evaluate('window.incrementView().viewMatrix')
                drag=await c.evaluate('''(()=>{const a=transportState().stats,start=performance.now();for(let i=0;i<120;i++){const t=document.getElementById('time-slider'),cloud=document.getElementById('cloud-slider');t.value=750+i*2;cloud.value=i%101;t.dispatchEvent(new Event('input',{bubbles:true}));cloud.dispatchEvent(new Event('input',{bubbles:true}));}return {before:a,after:transportState().stats,milliseconds:performance.now()-start};})()''')
                if drag['before']['uploads']!=drag['after']['uploads']:raise RuntimeError('Dragging uploaded large light fields')
                if camera!=await c.evaluate('window.incrementView().viewMatrix'):raise RuntimeError('Slider moved camera')
                await c.evaluate('transportPreset("original");setLampGroup("all",0)');off,f=await frame('lamps_off')
                if np.array_equal(original,off):raise RuntimeError('Lamp removal did not change room pixels')
                if off[:,:,:3].mean()>=original[:,:,:3].mean():raise RuntimeError('Turning off lamps failed to reduce light')
                independent=await c.evaluate('''(()=>{setLampGroup('all',0);setLamp(0,.4);setLamp(24,1.6);return transportState().lamps})()''')
                if independent[0]!=.4 or independent[24]!=1.6 or any(v!=0 for v in independent[1:24]):raise RuntimeError('Source controls coupled')
                await c.evaluate('transportPreset("sunny")');a,sa=await frame('sun_1230',True)
                await c.evaluate('setTransport({minute:960})');b,sb=await frame('sun_1600',True)
                if np.array_equal(a,b) or sa['state']['solar']['direction']==sb['state']['solar']['direction'] or sa['state']['lightMatrix']==sb['state']['lightMatrix']:raise RuntimeError('Sun direction / geometric shadow did not move')
                await c.evaluate('transportPreset("black")');black,bc=await frame('black')
                if black[:,:,:3].max()!=0:raise RuntimeError('Zero source power left residual illumination')
                await c.evaluate('transportPreset("original")');restored,rs=await frame('restored')
                if not np.array_equal(original,restored):raise RuntimeError('Original reference is not pixel-exact')
                await c.evaluate('document.activeElement?.blur()');before=await c.evaluate('window.incrementView().viewMatrix')
                await c.call('Input.dispatchKeyEvent',dict(type='keyDown',key='ArrowUp',code='ArrowUp',windowsVirtualKeyCode=38));await c.evaluate('window.__stepFrame()');await c.call('Input.dispatchKeyEvent',dict(type='keyUp',key='ArrowUp',code='ArrowUp',windowsVirtualKeyCode=38))
                if before==await c.evaluate('window.incrementView().viewMatrix'):raise RuntimeError('World walking failed')
                if c.errors:raise RuntimeError(c.errors)
                report=dict(passed=True,gaussians=6100978,diagnostic_viewport=[160,104],source_asset=state['s']['asset'],no_uploads_during_drag=True,drag=drag,slider_camera_unchanged=True,lamps_independent=True,room_pixels_respond_to_lamp_removal=True,source_zero_black=True,original_restore_pixel_exact=True,sun_direction_and_shadow_matrix_move=True,keyboard_walk=True,frames=[o,f,sa,sb,bc,rs],physical_ground_truth_verified=False,completed_at=time.time())
                (OUT/'browser_validation.json').write_text(json.dumps(report,indent=2));print('TRANSPORT_BROWSER_PASS',flush=True)
    finally:
        browser.terminate()
        try:browser.wait(timeout=10)
        except subprocess.TimeoutExpired:browser.kill();browser.wait()
        try:os.killpg(browser.pid,signal.SIGTERM)
        except ProcessLookupError:pass


if __name__=='__main__':asyncio.run(main())
