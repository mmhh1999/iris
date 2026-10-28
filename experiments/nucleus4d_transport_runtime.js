    const meta=/* BANK_META */;
    const solarKeys=/* SUN_KEYS */;
    const fixtures=/* LIGHTS */;
    window.activeSceneAsset=url.pathname;
    if(!gl.getExtension('EXT_color_buffer_float')||!gl.getExtension('EXT_float_blend'))throw Error('此版本需要浮点颜色缓冲及浮点混合支持');
    const state={ready:false,minute:750,cloud:1,sky:1,sun:0,ev:0,lamps:fixtures.map(()=>1),playing:false};
    const stats={uploads:0,atlasPasses:0,shadowPasses:0,frames:0};
    let dirty=true,sunDirty=true,referenceMode=true,lastClock=0,playDirection=1,cachedSourcePowers=null,lastSunPower=1;
    const count=6100978,bindingHeight=Math.ceil(count/1024),textures=new Map();
    function restore(){gl.useProgram(program);gl.bindVertexArray(null);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);}
    function texture2(unit,internal,width,height,format,type,data){
        gl.activeTexture(gl.TEXTURE0+unit);const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
        for(const key of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,key,gl.NEAREST);
        for(const key of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,key,gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D,0,internal,width,height,0,format,type,data);const old=textures.get(unit);if(old)gl.deleteTexture(old);textures.set(unit,t);stats.uploads++;restore();return t;
    }
    function pad(data,length,Type=Float32Array){if(data.length===length)return data;const a=new Type(length);a.set(data);return a;}
    async function asset(name,Type,length){const r=await fetch(name);if(!r.ok)throw Error(name+' 加载失败');const a=new Type(await r.arrayBuffer());if(length!==undefined&&a.length!==length)throw Error(name+' 数据长度不一致');return a;}
    texture2(1,gl.RGBA32UI,1,1,gl.RGBA_INTEGER,gl.UNSIGNED_INT,new Uint32Array([0xffffffff,0,0,0]));
    for(const unit of [2,3,4,5,8])texture2(unit,gl.RGBA32F,1,1,gl.RGBA,gl.FLOAT,new Float32Array([1,1,1,1]));
    // Small resident arrays avoid browser per-texture allocation limits while
    // keeping every time node and every float32 coefficient at full precision.
    const bankCapacity=32,bankTextures=[];
    for(let start=0;start<meta.layers;start+=bankCapacity){
        gl.activeTexture(gl.TEXTURE6);const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D_ARRAY,t);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.texStorage3D(gl.TEXTURE_2D_ARRAY,1,gl.RGB32F,meta.width,meta.height,Math.min(bankCapacity,meta.layers-start));bankTextures.push(t);
        const error=gl.getError();if(error)throw Error('光照分块缓冲创建失败，WebGL '+error);
    }restore();
    const bankUniforms=bankTextures.map(()=>({powers:new Float32Array(29),layers:new Int32Array(29)}));
    const uniforms={};
    for(const name of ['u_sunDirection','u_sunIrradiance','u_facePower[0]','u_exterior','u_referenceMode'])uniforms[name]=gl.getUniformLocation(program,name);
    for(const [name,unit]of [['u_binding',1],['u_bindingNormal',2],['u_patchPosition',3],['u_incident',4],['u_reference',5],['u_fixtureMask',8]])gl.uniform1i(gl.getUniformLocation(program,name),unit);
    gl.uniform1i(uniforms.u_referenceMode,1);
    /* PROGRAMS */
    const incidentFBO=gl.createFramebuffer();
    texture2(4,gl.RGBA32F,meta.width,meta.height,gl.RGBA,gl.FLOAT,null);
    gl.bindFramebuffer(gl.FRAMEBUFFER,incidentFBO);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,textures.get(4),0);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('光照缓冲创建失败');
    let shadowSize=4096;const shadowFBO=gl.createFramebuffer();
    const shadowLimit=Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE),...gl.getParameter(gl.MAX_VIEWPORT_DIMS));
    texture2(7,gl.DEPTH_COMPONENT32F,shadowSize,shadowSize,gl.DEPTH_COMPONENT,gl.FLOAT,null);
    gl.bindFramebuffer(gl.FRAMEBUFFER,shadowFBO);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,textures.get(7),0);gl.drawBuffers([gl.NONE]);gl.readBuffer(gl.NONE);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('太阳阴影缓冲创建失败');
    gl.clearDepth(1);gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);restore();
    let mrt=null,mrtSize='',casterCount=0,lightMatrix=new Float32Array(16),depthSpan=1,uvPerMeter=[1,1];
    function ensureMRT(){
        const key=canvas.width+','+canvas.height;if(key===mrtSize)return;mrtSize=key;
        if(mrt)gl.deleteFramebuffer(mrt);mrt=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,mrt);
        for(let k=0;k<3;k++){const old=textures.get(9+k);if(old)gl.deleteTexture(old);texture2(9+k,gl.RGBA32F,canvas.width,canvas.height,gl.RGBA,gl.FLOAT,null);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+k,gl.TEXTURE_2D,textures.get(9+k),0);}
        gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1,gl.COLOR_ATTACHMENT2]);
        if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('漫游浮点缓冲创建失败');
    }
    const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>{const l=Math.hypot(...a);return a.map(v=>v/l);};
    function sunState(minute){
        const a=solarKeys.states[0],b=solarKeys.states.at(-1),t=Math.max(0,Math.min(1,(minute-750)/345));
        const omega=Math.acos(Math.max(-1,Math.min(1,dot(a.direction,b.direction))));
        return {direction:norm(a.direction.map((v,i)=>(Math.sin((1-t)*omega)*v+Math.sin(t*omega)*b.direction[i])/Math.sin(omega))),irradiance:a.irradiance.map((v,i)=>v*(1-t)+b.irradiance[i]*t)};
    }
    function updateMatrix(d){
        const bounds=meta.indoor_bounds,center=bounds[0].map((v,i)=>(v+bounds[1][i])/2),eye=center.map((v,i)=>v+40*d[i]);
        const right=norm(cross([0,0,1],d)),up=cross(d,right);
        const view=new Float32Array([right[0],up[0],d[0],0,right[1],up[1],d[1],0,right[2],up[2],d[2],0,-dot(right,eye),-dot(up,eye),-dot(d,eye),1]);
        function limits(box,axis){const p=[];for(let x=0;x<2;x++)for(let y=0;y<2;y++)for(let z=0;z<2;z++)p.push(dot(axis,[box[x][0]-eye[0],box[y][1]-eye[1],box[z][2]-eye[2]]));return [Math.min(...p),Math.max(...p)];}
        const x=limits(bounds,right),y=limits(bounds,up),z=limits(meta.caster_bounds,d);x[0]-=.25;x[1]+=.25;y[0]-=.25;y[1]+=.25;
        const near=-z[1]-1,far=-z[0]+1;depthSpan=far-near;uvPerMeter=[1/(x[1]-x[0]),1/(y[1]-y[0])];
        const p=new Float32Array([2/(x[1]-x[0]),0,0,0,0,2/(y[1]-y[0]),0,0,0,0,-2/(far-near),0,-(x[1]+x[0])/(x[1]-x[0]),-(y[1]+y[0])/(y[1]-y[0]),-(far+near)/(far-near),1]);
        lightMatrix=new Float32Array(multiply4(p,view));
    }
    function sourcePowers(){
        const powers=new Float32Array(meta.layers);powers[0]=(1-state.cloud)*state.sky;powers[1]=state.cloud*state.sky;
        state.lamps.forEach((p,i)=>powers[meta.lamp_layers[i]]=p);
        const minutes=solarKeys.minutes;let i=0;while(i<minutes.length-2&&minutes[i+1]<state.minute)i++;
        const t=Math.max(0,Math.min(1,(state.minute-minutes[i])/(minutes[i+1]-minutes[i]))),strength=state.sun*(1-state.cloud)**2;
        powers[meta.sun_gi_layers[i]]=strength*(1-t);powers[meta.sun_gi_layers[i+1]]=strength*t;return powers;
    }
    function refresh(){
        referenceMode=state.sun*(1-state.cloud)**2===0&&state.cloud===1&&state.sky===1&&state.ev===0&&state.lamps.every(v=>v===1);
        const powers=sourcePowers();if(!cachedSourcePowers||powers.some((p,i)=>p!==cachedSourcePowers[i])){cachedSourcePowers=powers;dirty=true;}
        if(state.sun>0)lastSunPower=state.sun;
        restore();gl.uniform1i(uniforms.u_referenceMode,referenceMode?1:0);
        for(const [key,value]of [['time',state.minute],['cloud',state.cloud*100],['sky',state.sky*100],['sun',state.sun*100],['exposure',state.ev]]){const e=document.getElementById(key+'-slider');if(e)e.value=value;}
        document.getElementById('time-value').textContent=String(Math.floor(state.minute/60)).padStart(2,'0')+':'+String(Math.floor(state.minute%60)).padStart(2,'0');
        document.getElementById('cloud-value').textContent=Math.round(state.cloud*100)+'%';
        document.getElementById('sun-state').textContent=referenceMode?'原拍摄光照 · 精确对照':'全屋光照估计 · 单一移动太阳';
        state.lamps.forEach((p,i)=>{const e=document.getElementById('lamp-'+i),v=document.getElementById('lamp-value-'+i);if(e)e.value=p*100;if(v)v.textContent=Math.round(p*100)+'%';});
        document.getElementById('play-time').textContent=state.playing?'暂停时间':'播放时间';
        const status=document.getElementById('direct-sun-status'),s=sunState(state.minute),d=s.direction;
        if(status){const angle=(Math.atan2(d[0],d[1])*180/Math.PI+360)%360,height=Math.asin(Math.max(-1,Math.min(1,d[2])))*180/Math.PI;
            status.textContent=state.sun===0?'新增太阳已关闭；可点“开启太阳（晴空）”后拖动时间。':state.cloud===1?'云量 100% 已屏蔽直射；可点“开启太阳（晴空）”。':`直射 ${Math.round(100*state.sun*(1-state.cloud)**2)}% · 场景方位 ${angle.toFixed(1)}° · 高度 ${height.toFixed(1)}°`;
        }
        const quality=document.getElementById('sun-shadow-resolution');if(quality)quality.value=shadowSize;
    }
    window.setTransport=(values)=>{
        if(!state.ready)return;const old=state.minute;
        for(const key of ['minute','cloud','sky','sun','ev'])if(values[key]!==undefined){const v=Number(values[key]);if(!Number.isFinite(v))throw Error('无效光照参数');const range={minute:[750,1095],cloud:[0,1],sky:[0,2],sun:[0,2],ev:[-3,3]}[key];state[key]=Math.max(range[0],Math.min(range[1],v));}
        sunDirty||=old!==state.minute;refresh();
    };
    window.setLamp=(id,power)=>{if(!state.ready)return;if(!Number.isInteger(id)||!fixtures[id])throw Error('无效灯具');state.lamps[id]=Math.max(0,Math.min(2,Number(power)));if(!Number.isFinite(state.lamps[id]))throw Error('无效亮度');refresh();};
    window.setLampGroup=(group,power)=>{if(!state.ready)return;fixtures.forEach((f,i)=>{if(group==='all'||f.group===group)state.lamps[i]=Math.max(0,Math.min(2,Number(power)));});refresh();};
    window.setDirectSunEnabled=(enabled)=>{if(!state.ready)return;const values={sun:enabled?lastSunPower:0};if(enabled&&state.cloud===1)values.cloud=0;window.setTransport(values);};
    window.setSunShadowResolution=(value)=>{
        if(!state.ready)return false;const size=Number(value);
        if(![4096,8192].includes(size)||size>shadowLimit)throw Error('此设备不支持所选太阳阴影尺寸');
        if(size===shadowSize)return true;
        // Allocate and validate before retiring the working shadow texture.
        // A failed upgrade must retain the current, functioning sun map.
        const old=textures.get(7),next=gl.createTexture();gl.activeTexture(gl.TEXTURE7);gl.bindTexture(gl.TEXTURE_2D,next);
        for(const key of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,key,gl.NEAREST);
        for(const key of[gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,key,gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.DEPTH_COMPONENT32F,size,size,0,gl.DEPTH_COMPONENT,gl.FLOAT,null);
        gl.bindFramebuffer(gl.FRAMEBUFFER,shadowFBO);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,next,0);
        const complete=gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE,error=gl.getError();
        if(!complete||error){gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,old,0);gl.deleteTexture(next);gl.bindTexture(gl.TEXTURE_2D,old);gl.bindFramebuffer(gl.FRAMEBUFFER,null);restore();return false;}
        gl.depthMask(true);gl.clearDepth(1);gl.clear(gl.DEPTH_BUFFER_BIT);textures.set(7,next);gl.deleteTexture(old);shadowSize=size;sunDirty=true;stats.uploads++;
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);restore();refresh();return true;
    };
    window.transportPreset=(name)=>{
        if(!state.ready)return;state.playing=false;state.lamps.fill(1);state.ev=0;state.sky=1;state.minute=750;state.sun=1;state.cloud=0;
        if(name==='original'){state.cloud=1;state.sun=0;}if(name==='overcast'){state.cloud=1;state.sun=0;}if(name==='dusk')state.minute=1095;
        if(name==='night'){state.sky=0;state.sun=0;}if(name==='black'){state.sky=0;state.sun=0;state.lamps.fill(0);}
        sunDirty=true;refresh();
    };
    window.toggleTimePlayback=()=>{if(!state.ready)return;state.playing=!state.playing;if(state.playing&&state.sun*(1-state.cloud)**2===0){state.sun=lastSunPower;state.cloud=0;}lastClock=0;refresh();};
    window.transportState=()=>({...state,lamps:[...state.lamps],stats:{...stats},referenceMode,gaussians:window.incrementCount||0,asset:url.pathname,viewMatrix:Array.from(viewMatrix),solar:sunState(state.minute),lightMatrix:Array.from(lightMatrix),depthSpan,uvPerMeter,shadowResolution:shadowSize,effectiveSun:state.sun*(1-state.cloud)**2,glError:gl.getError()});
    window.lampState=()=>state.lamps.map((power,i)=>({id:i,label:fixtures[i].label,power}));
    window.incrementGLError=()=>gl.getError();
    function transportClock(now){if(state.ready&&state.playing&&lastClock){let m=state.minute+playDirection*Math.min((now-lastClock)/1000,.1)*12;if(m>=1095){m=1095;playDirection=-1;}if(m<=750){m=750;playDirection=1;}window.setTransport({minute:m});}lastClock=now;}
    function beginTransportFrame(){
        if(!state.ready||referenceMode){gl.bindFramebuffer(gl.FRAMEBUFFER,null);restore();return;}
        const sun=sunState(state.minute),strength=state.sun*(1-state.cloud)**2;
        if(dirty){
            gl.bindFramebuffer(gl.FRAMEBUFFER,incidentFBO);gl.viewport(0,0,meta.width,meta.height);gl.disable(gl.BLEND);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(lightProgram);gl.bindVertexArray(quadVAO);
            gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.ONE,gl.ONE,gl.ONE,gl.ONE);
            const all=cachedSourcePowers;
            for(let bank=0;bank<bankTextures.length;bank++){
                const{powers,layers}=bankUniforms[bank];powers.fill(0);layers.fill(0);let slot=0;
                for(let i=bank*bankCapacity;i<Math.min((bank+1)*bankCapacity,all.length);i++)if(all[i]!==0){if(slot>=29)throw Error('同时激活的光源数超出范围');powers[slot]=all[i];layers[slot++]=i%bankCapacity;}
                if(!slot)continue;gl.activeTexture(gl.TEXTURE6);gl.bindTexture(gl.TEXTURE_2D_ARRAY,bankTextures[bank]);gl.uniform1fv(powerLocation,powers);gl.uniform1iv(layerLocation,layers);gl.drawArrays(gl.TRIANGLES,0,3);stats.atlasPasses++;
            }
            gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA,gl.ONE,gl.ONE_MINUS_DST_ALPHA,gl.ONE);dirty=false;
        }
        if(sunDirty&&strength>0){
            updateMatrix(sun.direction);gl.bindFramebuffer(gl.FRAMEBUFFER,shadowFBO);gl.viewport(0,0,shadowSize,shadowSize);gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LESS);gl.depthMask(true);
            gl.useProgram(shadowProgram);gl.uniformMatrix4fv(shadowMatrixLocation,false,lightMatrix);gl.bindVertexArray(casterVAO);gl.clearDepth(1);gl.clear(gl.DEPTH_BUFFER_BIT);gl.drawElements(gl.TRIANGLES,casterCount,gl.UNSIGNED_INT,0);stats.shadowPasses++;sunDirty=false;
        }
        ensureMRT();gl.bindFramebuffer(gl.FRAMEBUFFER,mrt);gl.viewport(0,0,canvas.width,canvas.height);gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);restore();
        gl.uniform3fv(uniforms.u_sunDirection,sun.direction);gl.uniform3fv(uniforms.u_sunIrradiance,sun.irradiance.map(v=>v*strength));
        const face=new Float32Array(27);face.fill(1);fixtures.forEach((f,i)=>face[f.source_id]=state.lamps[i]);gl.uniform1fv(uniforms['u_facePower[0]'],face);
        gl.uniform1f(uniforms.u_exterior,state.sky+.15*strength*dot(sun.irradiance,[.2126,.7152,.0722]));
    }
    function post(debug=0){
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,canvas.width,canvas.height);gl.disable(gl.BLEND);gl.disable(gl.DEPTH_TEST);gl.useProgram(toneProgram);gl.bindVertexArray(quadVAO);
        gl.uniformMatrix4fv(tone.matrix,false,lightMatrix);gl.uniform2f(tone.texel,1/shadowSize,1/shadowSize);gl.uniform2fv(tone.uvPerMeter,uvPerMeter);gl.uniform1f(tone.depthSpan,depthSpan);gl.uniform1f(tone.exposure,state.ev);gl.uniform1i(tone.debug,debug);gl.drawArrays(gl.TRIANGLES,0,3);
        gl.enable(gl.BLEND);restore();
    }
    function finishTransportFrame(){stats.frames++;if(state.ready&&!referenceMode)post();}
    window.transportReadPixels=(debug=0)=>{if(debug&&state.ready&&!referenceMode)post(debug);const pixels=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);return Array.from(pixels);};
    window.transportReadWorld=()=>{if(!mrt)return null;gl.bindFramebuffer(gl.READ_FRAMEBUFFER,mrt);gl.readBuffer(gl.COLOR_ATTACHMENT2);const a=new Float32Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.FLOAT,a);gl.bindFramebuffer(gl.READ_FRAMEBUFFER,null);return Array.from(a);};
    window.transportReadSunMask=()=>{if(!mrt)return null;const a=new Float32Array(canvas.width*canvas.height*4),b=new Float32Array(a.length);gl.bindFramebuffer(gl.READ_FRAMEBUFFER,mrt);for(const [i,data]of [[0,a],[1,b]]){gl.readBuffer(gl.COLOR_ATTACHMENT0+i);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.FLOAT,data);}gl.bindFramebuffer(gl.READ_FRAMEBUFFER,null);return Array.from({length:a.length/4},(_,i)=>Math.max(...[0,1,2].map(k=>Math.abs(a[4*i+k]-b[4*i+k]))));};
    const panel=document.getElementById('info');for(const event of ['pointerdown','mousedown','mousemove','wheel','touchstart','touchmove','keydown'])panel.addEventListener(event,e=>e.stopPropagation(),{passive:true});
    for(const [id,key,scale]of [['time','minute',1],['cloud','cloud',.01],['sky','sky',.01],['sun','sun',.01],['exposure','ev',1]])document.getElementById(id+'-slider').oninput=e=>window.setTransport({[key]:Number(e.target.value)*scale});
    const quality=document.getElementById('sun-shadow-resolution');if(quality){quality.querySelector('[value="8192"]').disabled=shadowLimit<8192;quality.onchange=e=>{try{if(!window.setSunShadowResolution(e.target.value))throw Error('阴影精度调整失败，已保留原值');}catch(error){e.target.value=shadowSize;document.getElementById('direct-sun-status').textContent=error.message;}};}
    refresh();
    // Fetches and uploads happen once. Slider changes only update uniforms and
    // a small resident incident-light atlas; the original model is untouched.
    window.transportLoading=(async()=>{
        const status=document.getElementById('loading');status.textContent='加载完整模型及全屋光照数据…';
        const binding=await asset('binding_compact.u32',Uint32Array,count*4);texture2(1,gl.RGBA32UI,1024,bindingHeight,gl.RGBA_INTEGER,gl.UNSIGNED_INT,pad(binding,1024*bindingHeight*4,Uint32Array));
        const normals=await asset(meta.normal_asset,Float32Array,count*4);texture2(2,gl.RGBA32F,1024,bindingHeight,gl.RGBA,gl.FLOAT,pad(normals,1024*bindingHeight*4));
        const positions=await asset('patch_positions.f32',Float32Array,meta.nodes*3);texture2(3,gl.RGB32F,meta.width,meta.height,gl.RGB,gl.FLOAT,pad(positions,meta.width*meta.height*3));
        const reference=await asset('reference_irradiance.f32',Float32Array,meta.nodes*3);texture2(5,gl.RGB32F,meta.width,meta.height,gl.RGB,gl.FLOAT,pad(reference,meta.width*meta.height*3));
        const mask=await asset('../nucleus4d_weather_controls/lamps/fixture_mask.f32',Float32Array,count);texture2(8,gl.R32F,1024,bindingHeight,gl.RED,gl.FLOAT,pad(mask,1024*bindingHeight));
        const bases=await asset('light_bank.rgb.f32',Float32Array,meta.layers*meta.width*meta.height*3),layerSize=meta.width*meta.height*3;
        for(let bank=0;bank<bankTextures.length;bank++){
            const start=bank*bankCapacity,depth=Math.min(bankCapacity,meta.layers-start);gl.activeTexture(gl.TEXTURE6);gl.bindTexture(gl.TEXTURE_2D_ARRAY,bankTextures[bank]);gl.texSubImage3D(gl.TEXTURE_2D_ARRAY,0,0,0,0,meta.width,meta.height,depth,gl.RGB,gl.FLOAT,bases.subarray(start*layerSize,(start+depth)*layerSize));stats.uploads++;
        }restore();
        const vertices=await asset('caster_positions.f32',Float32Array),indices=await asset('caster_indices.u32',Uint32Array);casterCount=indices.length;
        gl.bindVertexArray(casterVAO);const vertex=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vertex);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
        const index=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,index);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW);restore();
        const error=gl.getError();if(error!==gl.NO_ERROR)throw Error('光照数据上传失败，WebGL '+error);
        state.ready=true;refresh();document.querySelectorAll('#info input,#info button,#info select').forEach(e=>e.disabled=false);status.textContent='全屋光照已加载 · 610 万原始高斯';
    })().catch(error=>{document.getElementById('loading').textContent=error.message;console.error(error);window.transportError=error.message;});
