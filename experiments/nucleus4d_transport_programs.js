    function compile(type,source){const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));return shader;}
    function link(vertex,fragment){const p=gl.createProgram();gl.attachShader(p,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
    const fullscreen=`#version 300 es
precision highp float;out vec2 uv;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.-1.,0.,1.);}`;
    const lightProgram=link(fullscreen,`#version 300 es
precision highp float;uniform highp sampler2DArray bases;uniform float powers[29];uniform int layers[29];out vec4 color;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec3 sum=vec3(0);for(int k=0;k<29;k++)if(powers[k]!=0.)sum+=texelFetch(bases,ivec3(p,layers[k]),0).rgb*powers[k];color=vec4(sum,1.);}`);
    const shadowProgram=link(`#version 300 es
precision highp float;layout(location=0)in vec3 position;uniform mat4 lightMatrix;void main(){gl_Position=lightMatrix*vec4(position,1.);}`,`#version 300 es
precision highp float;void main(){}`);
    const toneProgram=link(fullscreen,`#version 300 es
precision highp float;precision highp sampler2D;uniform sampler2D noSun;uniform sampler2D withSun;uniform sampler2D world;uniform sampler2D sunDepth;
uniform mat4 lightMatrix;uniform vec2 shadowTexel;uniform vec2 uvPerMeter;uniform float depthSpan;uniform float exposure;uniform int debugVisibility;
in vec2 uv;out vec4 color;
const vec2 disc[16]=vec2[16](vec2(.13,.08),vec2(-.27,.21),vec2(.25,-.36),vec2(-.08,-.52),vec2(.51,.16),vec2(-.54,-.16),vec2(.37,.54),vec2(-.39,.57),vec2(.71,-.32),vec2(-.71,.29),vec2(.08,.81),vec2(-.18,-.82),vec2(.77,.52),vec2(-.74,-.55),vec2(.48,-.78),vec2(-.48,.79));
vec3 decode(vec3 c){return mix(pow(max((c+.055)/1.055,vec3(0)),vec3(2.4)),c/12.92,lessThanEqual(c,vec3(.04045)));}
vec3 encode(vec3 c){return mix(1.055*pow(max(c,vec3(0)),vec3(1./2.4))-.055,12.92*c,lessThanEqual(c,vec3(.0031308)));}
float visibility(vec3 p){vec4 h=lightMatrix*vec4(p,1);vec3 s=h.xyz/h.w*.5+.5;if(any(lessThan(s,vec3(0)))||any(greaterThan(s,vec3(1))))return 1.;
float z=s.z-.000025;vec2 search=uvPerMeter*(.004625*depthSpan*s.z);float blocker=0.,count=0.;
for(int i=0;i<8;i++){float d=texture(sunDepth,s.xy+disc[i]*search).r;if(d<z){blocker+=d;count+=1.;}}
if(count==0.)return 1.;blocker/=count;vec2 radius=max(shadowTexel*.75,uvPerMeter*(.004625*depthSpan*max(z-blocker,0.)));float sum=0.;
for(int i=0;i<16;i++)sum+=texture(sunDepth,s.xy+disc[i]*radius).r>=z?1.:0.;return sum/16.;}
void main(){vec4 a=texture(noSun,uv),b=texture(withSun,uv),p=texture(world,uv);float v=1.;if(p.a>.001&&any(greaterThan(abs(b.rgb-a.rgb),vec3(1e-6))))v=visibility(p.rgb/p.a);
if(debugVisibility!=0){color=vec4(vec3(v),1);return;}vec3 linear=mix(decode(a.rgb),decode(b.rgb),v)*exp2(exposure);float peak=max(linear.r,max(linear.g,linear.b));
if(peak>.8){float shoulder=.8+(peak-.8)/(1.+(peak-.8)/.2);linear*=shoulder/peak;}color=vec4(encode(max(linear,vec3(0))),a.a);}`);
    const quadVAO=gl.createVertexArray(),casterVAO=gl.createVertexArray();
    const powerLocation=gl.getUniformLocation(lightProgram,'powers[0]'),layerLocation=gl.getUniformLocation(lightProgram,'layers[0]'),shadowMatrixLocation=gl.getUniformLocation(shadowProgram,'lightMatrix');
    const tone={matrix:gl.getUniformLocation(toneProgram,'lightMatrix'),texel:gl.getUniformLocation(toneProgram,'shadowTexel'),uvPerMeter:gl.getUniformLocation(toneProgram,'uvPerMeter'),depthSpan:gl.getUniformLocation(toneProgram,'depthSpan'),exposure:gl.getUniformLocation(toneProgram,'exposure'),debug:gl.getUniformLocation(toneProgram,'debugVisibility')};
    gl.useProgram(lightProgram);gl.uniform1i(gl.getUniformLocation(lightProgram,'bases'),6);
    gl.useProgram(toneProgram);for(const [name,unit]of [['noSun',9],['withSun',10],['world',11],['sunDepth',7]])gl.uniform1i(gl.getUniformLocation(toneProgram,name),unit);
    gl.useProgram(program);
