import fs from 'node:fs/promises';
import {newPage,writeJSON}from './gs_relight_cdp.mjs';
const output=process.argv[2],{cdp,target}=await newPage('http://localhost:8771/gs_relight_stages/app_world/');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
try{
 await cdp.call('Network.enable');await cdp.call('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
 let state;for(let i=0;i<400;i++){state=await cdp.evaluate('({url:location.href,state:window.transportState?.(),error:window.transportError})');if(state.error)throw Error(state.error);if(state.state?.ready&&state.state.gaussians===6100978)break;await delay(300);}
 if(!state.state?.ready)throw Error('Legacy viewer did not finish loading');
 const before=await cdp.evaluate('transportState()');
 const controls=await cdp.evaluate(`(()=>{const camera=transportState().viewMatrix,uploads=transportState().stats.uploads;const t=document.getElementById('time-slider');t.value=960;t.dispatchEvent(new Event('input',{bubbles:true}));const c=document.getElementById('cloud-slider');c.value=40;c.dispatchEvent(new Event('input',{bubbles:true}));setLamp(0,.4);const state=transportState();return {minute:state.minute,cloud:state.cloud,lamps:state.lamps,uploadsBefore:uploads,uploadsAfter:state.stats.uploads,cameraUnchanged:JSON.stringify(camera)===JSON.stringify(state.viewMatrix)};})()`);
 await cdp.evaluate('transportPreset("original");document.activeElement?.blur()');
 const start=await cdp.evaluate('transportState().viewMatrix');
 await cdp.call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});await delay(400);await cdp.call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});
 const end=await cdp.evaluate('transportState().viewMatrix');
 await cdp.evaluate('selectView(0);transportPreset("original")');await delay(400);
 const shot=await cdp.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(output+'/restored_v3.png',Buffer.from(shot.data,'base64'));
 const final=await cdp.evaluate('transportState()'),errors=cdp.events.filter(e=>e.method==='Runtime.exceptionThrown');
 const report={url:state.url,gaussians:final.gaussians,initialReferenceMode:before.referenceMode,finalReferenceMode:final.referenceMode,controls,keyboardWalk:JSON.stringify(start)!==JSON.stringify(end),glError:final.glError,errors};
 report.pass=state.url==='http://localhost:8769/nucleus4d_transport_v3/'&&final.gaussians===6100978&&before.referenceMode&&final.referenceMode&&controls.minute===960&&Math.abs(controls.cloud-.4)<1e-6&&controls.lamps[0]===.4&&controls.lamps.slice(1).every(p=>p===1)&&controls.cameraUnchanged&&controls.uploadsBefore===controls.uploadsAfter&&report.keyboardWalk&&final.glError===0&&!errors.length;
 await writeJSON(output+'/browser.json',report);console.log(JSON.stringify(report));if(!report.pass)process.exitCode=1;
}finally{cdp.close();await fetch('http://127.0.0.1:9367/json/close/'+target.id);}
