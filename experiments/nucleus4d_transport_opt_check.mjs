import fs from 'node:fs/promises';
import {newPage,writeJSON}from './gs_relight_cdp.mjs';
const output=process.argv[2],mode=process.argv[3]||'before';
const {cdp,target}=await newPage('about:blank');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function evaluate(s){return cdp.evaluate(s);}
async function waitReady(){for(let i=0;i<600;i++){const r=await evaluate('({ready:window.transportState?.().ready,count:window.incrementCount,error:window.transportError})');if(r.error)throw Error(r.error);if(r.ready&&r.count===6100978)return;await delay(250);}throw Error('v3 did not load');}
async function frame(){return evaluate('new Promise(requestAnimationFrame).then(()=>{const gl=document.getElementById("canvas").getContext("webgl2");gl.finish();return transportState();})');}
async function capture(name){await frame();await delay(150);const r=await cdp.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(output+'/'+mode+'_'+name+'.png',Buffer.from(r.data,'base64'));return evaluate('transportState()');}
const report={mode,states:{}};
try{
 await cdp.call('Network.enable');await cdp.call('Network.setCacheDisabled',{cacheDisabled:true});
 await cdp.call('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
 await cdp.call('Page.navigate',{url:'http://localhost:8769/nucleus4d_transport_v3/?review='+Date.now()});await waitReady();
 await evaluate('document.getElementById("info").style.display="none";document.getElementById("fps").style.display="none";selectView(0)');
 for(const[name,script]of[['original','transportPreset("original")'],['sun750','transportPreset("sunny")'],['sun960','setTransport({minute:960})'],['sun1095','setTransport({minute:1095})'],['cloud','setTransport({minute:960,cloud:.4})'],['night','transportPreset("night")']]){await evaluate(script);report.states[name]=await capture(name);}
 report.redundantWork=await evaluate(`(async()=>{transportPreset('night');await new Promise(requestAnimationFrame);let initial=transportState().stats;for(let i=0;i<12;i++){setTransport({ev:-1+i/10});await new Promise(requestAnimationFrame);}let exposure=transportState().stats;for(let i=0;i<12;i++){setTransport({minute:760+i*20});await new Promise(requestAnimationFrame);}let time=transportState().stats;return {exposureAtlasPasses:exposure.atlasPasses-initial.atlasPasses,inactiveSunTimeAtlasPasses:time.atlasPasses-exposure.atlasPasses,inactiveSunShadowPasses:time.shadowPasses-exposure.shadowPasses};})()`);
 if(mode!=='before'){
  report.sunRecovery=await evaluate(`(async()=>{transportPreset('original');setDirectSunEnabled(true);await new Promise(requestAnimationFrame);const enabled=transportState();setDirectSunEnabled(false);setTransport({minute:1000});await new Promise(requestAnimationFrame);setDirectSunEnabled(true);await new Promise(requestAnimationFrame);return {enabled,recovered:transportState()};})()`);
  await evaluate('transportPreset("sunny");setSunShadowResolution(8192)');report.states.sun8k=await capture('sun8k');
  await evaluate('setTransport({minute:960})');report.states.sun8k960=await capture('sun8k960');
  await evaluate('setSunShadowResolution(4096)');report.states.back4k=await capture('back4k');
  report.playback=await evaluate(`(async()=>{transportPreset('overcast');setTransport({sun:1});toggleTimePlayback();const a=transportState().minute;await new Promise(r=>setTimeout(r,500));const b=transportState();toggleTimePlayback();return {advanced:b.minute>a,effectiveSun:b.sun*(1-b.cloud)**2};})()`);
  const requests=cdp.events.filter(e=>e.method==='Network.requestWillBeSent').length;
  report.drag=await evaluate(`(()=>{const view=transportState().viewMatrix,uploads=transportState().stats.uploads;for(let i=0;i<60;i++){const e=document.getElementById('time-slider');e.value=750+i*5;e.dispatchEvent(new Event('input',{bubbles:true}));}const s=transportState();return {viewUnchanged:JSON.stringify(view)===JSON.stringify(s.viewMatrix),uploadsUnchanged:uploads===s.stats.uploads,gaussians:s.gaussians,glError:s.glError};})()`);
  await frame();report.networkDuringDrag=cdp.events.filter(e=>e.method==='Network.requestWillBeSent').length-requests;
  await evaluate('transportPreset("night")');await frame();
  const lampBefore=await evaluate('transportState()');
  await evaluate('setLamp(0,.4)');await frame();
  report.lamp=await evaluate('transportState()');report.lamp.atlasUpdated=report.lamp.stats.atlasPasses>lampBefore.stats.atlasPasses;
  await evaluate('document.activeElement?.blur()');const viewBeforeWalk=await evaluate('transportState().viewMatrix');
  await cdp.call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});await delay(400);
  await cdp.call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});
  report.keyboardWalk=JSON.stringify(viewBeforeWalk)!==JSON.stringify(await evaluate('transportState().viewMatrix'));
  await evaluate('transportPreset("original");selectView(0);document.getElementById("info").style.display=""');await capture('ui');
 }
 report.errors=cdp.events.filter(e=>e.method==='Runtime.exceptionThrown'||e.method==='Runtime.consoleAPICalled'&&e.params.type==='error');
 report.pass=Object.values(report.states).every(s=>s.gaussians===6100978&&s.glError===0)&&!report.errors.length;
 if(mode!=='before')report.pass&&=report.redundantWork.exposureAtlasPasses===0&&report.redundantWork.inactiveSunTimeAtlasPasses===0&&report.redundantWork.inactiveSunShadowPasses===0&&report.sunRecovery.enabled.sun>0&&report.sunRecovery.enabled.cloud<1&&report.sunRecovery.recovered.minute===1000&&report.states.sun8k.shadowResolution===8192&&report.states.back4k.shadowResolution===4096&&report.playback.advanced&&report.playback.effectiveSun>0&&report.drag.viewUnchanged&&report.drag.uploadsUnchanged&&report.networkDuringDrag===0&&report.lamp.lamps[0]===.4&&report.lamp.lamps.slice(1).every(p=>p===1)&&report.lamp.atlasUpdated&&report.keyboardWalk;
 await writeJSON(output+'/'+mode+'.json',report);console.log(JSON.stringify({pass:report.pass,redundantWork:report.redundantWork,errors:report.errors}));if(!report.pass)process.exitCode=1;
}catch(error){await writeJSON(output+'/'+mode+'_failure.json',{message:String(error),report,errors:cdp.events.filter(e=>e.method==='Runtime.exceptionThrown')});throw error;}
finally{cdp.close();await fetch('http://127.0.0.1:9367/json/close/'+target.id);}
