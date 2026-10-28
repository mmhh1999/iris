// Local Chrome diagnostics. Node 24 supplies WebSocket without dependencies.
import fs from 'node:fs/promises';

export class CDP {
    constructor(socket) {
        this.socket=socket;this.next=1;this.pending=new Map();this.events=[];
        socket.addEventListener('message',event=>{
            const message=JSON.parse(event.data);
            if(message.id){const p=this.pending.get(message.id);if(!p)return;clearTimeout(p.timer);this.pending.delete(message.id);if(message.error)p.reject(Error(JSON.stringify(message.error)));else p.resolve(message.result);}
            else this.events.push(message);
        });
    }
    static async connect(url){const socket=new WebSocket(url);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});return new CDP(socket);}
    call(method,params={},timeout=120000){return new Promise((resolve,reject)=>{const id=this.next++;const timer=setTimeout(()=>{this.pending.delete(id);reject(Error('CDP timeout: '+method));},timeout);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params}));});}
    async evaluate(expression){const r=await this.call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
    close(){this.socket.close();}
}

export async function newPage(url,port=9367){
    const r=await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`,{method:'PUT'});if(!r.ok)throw Error('Chrome target failed '+r.status);const target=await r.json();const cdp=await CDP.connect(target.webSocketDebuggerUrl);await cdp.call('Runtime.enable');await cdp.call('Page.enable');return {cdp,target};
}

export async function writeJSON(path,value){await fs.writeFile(path,JSON.stringify(value,null,2));}

if(process.argv.includes('--probe')){
    const output=process.argv[process.argv.indexOf('--output')+1];
    const version=await (await fetch('http://127.0.0.1:9367/json/version')).json();const browser=await CDP.connect(version.webSocketDebuggerUrl);
    const info=await browser.call('SystemInfo.getInfo');browser.close();
    const {cdp,target}=await newPage('http://localhost:8769/gs_relight_stages/stage0/probe.html');
    try{
        await new Promise(resolve=>setTimeout(resolve,700));
        const probe=await cdp.evaluate(`(async()=>{
            const canvas=document.createElement('canvas');canvas.width=16;canvas.height=16;document.body.appendChild(canvas);
            const gl=canvas.getContext('webgl2');if(!gl)throw Error('No WebGL2');
            const debug=gl.getExtension('WEBGL_debug_renderer_info'),floatColor=!!gl.getExtension('EXT_color_buffer_float'),floatBlend=!!gl.getExtension('EXT_float_blend');
            const f=gl.createFramebuffer(),t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,16,16,0,gl.RGBA,gl.FLOAT,null);gl.bindFramebuffer(gl.FRAMEBUFFER,f);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,t,0);
            const framebufferComplete=gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE;gl.clearColor(.125,.25,.5,1);gl.clear(gl.COLOR_BUFFER_BIT);const pixel=new Float32Array(4);gl.readPixels(0,0,1,1,gl.RGBA,gl.FLOAT,pixel);
            let webgpu=null;if(navigator.gpu){const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(a){const device=await a.requestDevice();const buffer=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(buffer,0,new Uint32Array([1,2,3,4]));await device.queue.onSubmittedWorkDone();webgpu={adapter:{vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description,isFallbackAdapter:a.info.isFallbackAdapter},maxStorageBufferBindingSize:a.limits.maxStorageBufferBindingSize,maxBufferSize:a.limits.maxBufferSize,features:[...a.features],deviceCreated:true};buffer.destroy();device.destroy();}}
            return {secureContext:isSecureContext,webglVersion:gl.getParameter(gl.VERSION),renderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),vendor:debug?gl.getParameter(debug.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR),maxTextureSize:gl.getParameter(gl.MAX_TEXTURE_SIZE),maxDrawBuffers:gl.getParameter(gl.MAX_DRAW_BUFFERS),floatColor,floatBlend,framebufferComplete,readback:Array.from(pixel),glError:gl.getError(),webgpu};
        })()`);
        const report={date:new Date().toISOString(),browser:version.Browser,probe,systemInfo:info,scope:'Executed hardware/API capability audit; no Gaussian feature or FPS claim'};
        await writeJSON(output,report);console.log(JSON.stringify({browser:report.browser,probe}));
    }finally{cdp.close();await fetch(`http://127.0.0.1:9367/json/close/${target.id}`);}
}
