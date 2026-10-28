uniform highp usampler2D u_binding;
uniform highp sampler2D u_bindingNormal;
uniform highp sampler2D u_patchPosition;
uniform highp sampler2D u_incident;
uniform highp sampler2D u_reference;
uniform highp sampler2D u_fixtureMask;
uniform vec3 u_sunDirection;
uniform vec3 u_sunIrradiance;
uniform float u_facePower[27];
uniform float u_exterior;
uniform int u_referenceMode;
out vec4 vSunColor;
out vec3 vWorld;

vec3 nodeValue(sampler2D field, uint index) {
    return texelFetch(field,ivec2(index&0x3ffu,index>>10),0).rgb;
}

void applyTransport(uint index,vec3 sourcePosition) {
    vSunColor=vColor;vWorld=sourcePosition;
    if(u_referenceMode!=0)return;
    ivec2 uv=ivec2(index&0x3ffu,index>>10);
    uvec4 bind=texelFetch(u_binding,uv,0);
    vec3 base=decodeColor(vColor.rgb);
    vec3 color,solar;
    if(bind.x==0xffffffffu){
        color=base*u_exterior;solar=color;
    }else{
        bool volume=(bind.x&0x80000000u)!=0u;bind.x&=0x7fffffffu;
        vec4 normalWeight=texelFetch(u_bindingNormal,uv,0);
        float w0=uintBitsToFloat(bind.w),w1=normalWeight.r,w2=max(0.,1.-w0-w1);
        vec3 incident=nodeValue(u_incident,bind.x)*w0+nodeValue(u_incident,bind.y)*w1+nodeValue(u_incident,bind.z)*w2;
        vec3 reference=nodeValue(u_reference,bind.x)*w0+nodeValue(u_reference,bind.y)*w1+nodeValue(u_reference,bind.z)*w2;
        vec3 n=normalize(normalWeight.gba);
        vec3 world=nodeValue(u_patchPosition,bind.x)*w0+nodeValue(u_patchPosition,bind.y)*w1+nodeValue(u_patchPosition,bind.z)*w2;
        vWorld=(volume?sourcePosition:world)+n*.002;
        vec3 ratio=incident/max(reference,vec3(1e-7));
        vec3 direct=u_sunIrradiance*max(dot(n,u_sunDirection),0.)/3.141592653589793;
        color=base*ratio;solar=base*(incident+direct)/max(reference,vec3(1e-7));
    }
    float packedLight=texelFetch(u_fixtureMask,uv,0).r;
    int fixture=int(floor(packedLight));float mask=fract(packedLight)*2.;
    if(fixture>0){
        float power=u_facePower[fixture];
        // Emission follows the lamp power. A small diffuse term remains when
        // an extinguished bulb receives illumination from the other sources.
        color=mix(color,(base*power+.08*color)/1.08,mask);
        solar=mix(solar,(base*power+.08*solar)/1.08,mask);
    }
    vColor.rgb=encodeColor(max(color,vec3(0)));
    vSunColor=vec4(encodeColor(max(solar,vec3(0))),vColor.a);
}
