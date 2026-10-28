let cameras = [{"id": 0, "img_name": "living", "width": 1600, "height": 1040, "fx": 1006, "fy": 1006, "position": [1.4, 1.7, 0.45], "rotation": [[-0.4344527107420301, -0.1687521189946857, 0.8847449149126009], [-0.9006946442212821, 0.08139808092684839, -0.4267593118990192], [0.0, -0.982291746252726, -0.18735774668737432]]}, {"id": 1, "img_name": "living", "width": 1600, "height": 1600, "fx": 1093.75, "fy": 1093.75, "position": [1.8750053217816791, -1.3752638726529205, 0.06838344721824047], "rotation": [[-0.01196836334839052, -0.23986808218871486, 0.9707317144432179], [-0.9978111625944277, -0.060276304609690634, -0.027196523758680202], [0.06503569851026098, -0.9689324384340262, -0.23862164124315025]]}, {"id": 2, "img_name": "living", "width": 1600, "height": 1600, "fx": 1093.75, "fy": 1093.75, "position": [5.198755765484285, 1.5513189755958736, 0.05658371531520823], "rotation": [[-0.036345783738277126, 0.3185386029292932, -0.947212828485924], [0.9991443406030684, 0.03030331484261617, -0.02814774858508512], [0.019737544054675926, -0.9474253889111288, -0.3193674401082339]]}];

let camera = cameras[0];

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = camera.position;
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1,
        ],
    ].flat();
    return camToWorld;
}
// function translate4(a, x, y, z) {
//     return [
//         ...a.slice(0, 12),
//         a[0] * x + a[4] * y + a[8] * z + a[12],
//         a[1] * x + a[5] * y + a[9] * z + a[13],
//         a[2] * x + a[6] * y + a[10] * z + a[14],
//         a[3] * x + a[7] * y + a[11] * z + a[15],
//     ];
// }

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

function createWorker(self) {
    let buffer;
    let vertexCount = 0;
    let viewProj;
    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2; // Set to your desired width
        var texheight = Math.ceil((2 * vertexCount) / texwidth); // Set to your desired height
        var texdata = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA)
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        for (let i = 0; i < vertexCount; i++) {
            // x, y, z
            texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
            texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
            texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

            // r, g, b, a
            texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
            texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
            texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
            texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

            // quaternions
            let scale = [
                f_buffer[8 * i + 3 + 0],
                f_buffer[8 * i + 3 + 1],
                f_buffer[8 * i + 3 + 2],
            ];
            let rot = [
                (u_buffer[32 * i + 28 + 0] - 128) / 128,
                (u_buffer[32 * i + 28 + 1] - 128) / 128,
                (u_buffer[32 * i + 28 + 2] - 128) / 128,
                (u_buffer[32 * i + 28 + 3] - 128) / 128,
            ];

            // Compute the matrix product of S and R (M = S * R)
            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }

        self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
    }

    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        if (lastVertexCount == vertexCount) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = vertexCount;
        }

        console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * f_buffer[8 * i + 0] +
                    viewProj[6] * f_buffer[8 * i + 1] +
                    viewProj[10] * f_buffer[8 * i + 2]) *
                    4096) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // This is a 16 bit single-pass counting sort
        let depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        lastProj = viewProj;
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
    }

    function processPlyBuffer(inputBuffer) {
        const ubuf = new Uint8Array(inputBuffer);
        // 10KB ought to be enough for a header...
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const header_end = "end_header\n";
        const header_end_index = header.indexOf(header_end);
        if (header_end_index < 0)
            throw new Error("Unable to read .ply file header");
        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        console.log("Vertex Count", vertexCount);
        let row_offset = 0,
            offsets = {},
            types = {};
        const TYPE_MAP = {
            double: "getFloat64",
            int: "getInt32",
            uint: "getUint32",
            float: "getFloat32",
            short: "getInt16",
            ushort: "getUint16",
            uchar: "getUint8",
        };
        for (let prop of header
            .slice(0, header_end_index)
            .split("\n")
            .filter((k) => k.startsWith("property "))) {
            const [p, type, name] = prop.split(" ");
            const arrayType = TYPE_MAP[type] || "getInt8";
            types[name] = arrayType;
            offsets[name] = row_offset;
            row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
        }
        console.log("Bytes per row", row_offset, types, offsets);

        let dataView = new DataView(
            inputBuffer,
            header_end_index + header_end.length,
        );
        let row = 0;
        const attrs = new Proxy(
            {},
            {
                get(target, prop) {
                    if (!types[prop]) throw new Error(prop + " not found");
                    return dataView[types[prop]](
                        row * row_offset + offsets[prop],
                        true,
                    );
                },
            },
        );

        console.time("calculate importance");
        let sizeList = new Float32Array(vertexCount);
        let sizeIndex = new Uint32Array(vertexCount);
        for (row = 0; row < vertexCount; row++) {
            sizeIndex[row] = row;
            if (!types["scale_0"]) continue;
            const size =
                Math.exp(attrs.scale_0) *
                Math.exp(attrs.scale_1) *
                Math.exp(attrs.scale_2);
            const opacity = 1 / (1 + Math.exp(-attrs.opacity));
            sizeList[row] = size * opacity;
        }
        console.timeEnd("calculate importance");

        console.time("sort");
        sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
        console.timeEnd("sort");

        // 6*4 + 4 + 4 = 8*4
        // XYZ - Position (Float32)
        // XYZ - Scale (Float32)
        // RGBA - colors (uint8)
        // IJKL - quaternion/rot (uint8)
        const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
        const buffer = new ArrayBuffer(rowLength * vertexCount);

        console.time("build buffer");
        for (let j = 0; j < vertexCount; j++) {
            row = sizeIndex[j];

            const position = new Float32Array(buffer, j * rowLength, 3);
            const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
            const rgba = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3,
                4,
            );
            const rot = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3 + 4,
                4,
            );

            if (types["scale_0"]) {
                const qlen = Math.sqrt(
                    attrs.rot_0 ** 2 +
                        attrs.rot_1 ** 2 +
                        attrs.rot_2 ** 2 +
                        attrs.rot_3 ** 2,
                );

                rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
                rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
                rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
                rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

                scales[0] = Math.exp(attrs.scale_0);
                scales[1] = Math.exp(attrs.scale_1);
                scales[2] = Math.exp(attrs.scale_2);
            } else {
                scales[0] = 0.01;
                scales[1] = 0.01;
                scales[2] = 0.01;

                rot[0] = 255;
                rot[1] = 0;
                rot[2] = 0;
                rot[3] = 0;
            }

            position[0] = attrs.x;
            position[1] = attrs.y;
            position[2] = attrs.z;

            if (types["f_dc_0"]) {
                const SH_C0 = 0.28209479177387814;
                rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
                rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
                rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
            } else {
                rgba[0] = attrs.red;
                rgba[1] = attrs.green;
                rgba[2] = attrs.blue;
            }
            if (types["opacity"]) {
                rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
            } else {
                rgba[3] = 255;
            }
        }
        console.timeEnd("build buffer");
        return buffer;
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.ply) {
            vertexCount = 0;
            runSort(viewProj);
            buffer = processPlyBuffer(e.data.ply);
            vertexCount = Math.floor(buffer.byteLength / rowLength);
            postMessage({ buffer: buffer, save: !!e.data.save });
        } else if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
        }
    };
}

const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform highp sampler2D u_gain;
uniform float u_addedSun;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

vec3 decodeColor(vec3 c) {
    return mix(pow((c+0.055)/1.055,vec3(2.4)),c/12.92,lessThanEqual(c,vec3(0.04045)));
}
vec3 encodeColor(vec3 c) {
    return mix(1.055*pow(c,vec3(1.0/2.4))-0.055,12.92*c,lessThanEqual(c,vec3(0.0031308)));
}

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

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;

    applyTransport(uint(index),uintBitsToFloat(cen.xyz));
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter
        + position.x * majorAxis / viewport
        + position.y * minorAxis / viewport, 0.0, 1.0);

}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec4 vSunColor;
in vec3 vWorld;
in vec2 vPosition;

layout(location=0) out vec4 fragColor;
layout(location=1) out vec4 sunColor;
layout(location=2) out vec4 worldColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
    sunColor = vec4(B*vSunColor.rgb,B);
    worldColor = vec4(B*vWorld,B);
}

`.trim();

let defaultViewMatrix = getViewMatrix(cameras[0]);
let viewMatrix = defaultViewMatrix;
window.selectView = (n) => { viewMatrix = getViewMatrix(cameras[n]); };
window.incrementView=()=>({viewMatrix,count:window.incrementCount||0,amount:window.addedSunAmount});
async function main() {
    let carousel = false;
    const params = new URLSearchParams(location.search);
    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        carousel = false;
    } catch (err) {}
    const url = new URL(
        // "nike.splat",
        // location.href,
        "../nucleus4d_3dgs/full.splat",
        location.href,
    );
    const req = await fetch(url, {
        mode: "cors", // no-cors, *cors, same-origin
        credentials: "omit", // include, *same-origin, omit
    });
    console.log(req);
    if (req.status != 200)
        throw new Error(req.status + " Unable to load " + req.url);

    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    const reader = req.body.getReader();
    let splatData = new Uint8Array(req.headers.get("content-length"));

    const downsample =
        splatData.length / rowLength > 500000 ? 1 : 1 / devicePixelRatio;
    console.log(splatData.length / rowLength, downsample);

    const worker = new Worker(
        URL.createObjectURL(
            new Blob(["(", createWorker.toString(), ")(self)"], {
                type: "application/javascript",
            }),
        ),
    );

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");
    const camid = document.getElementById("camid");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");

    // positions
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);


    const meta={"nodes": 317440, "width": 1024, "height": 310, "layers": 304, "sky_layers": [0, 1], "lamp_layers": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], "sun_gi_layers": [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303], "baseline": {"sky_clear": 0, "sky_overcast": 1, "lamps": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "direct_sun": 0}, "direct_sun": "runtime geometric shadow map", "all_lamps_independent": true, "reference_appearance": "original capture, exact bypass", "reference_min_rgb": [0.0, 0.0, 0.0], "reference_max_rgb": [214.5994110107422, 183.53016662597656, 159.07315063476562], "estimated_geometry_and_bounce_reflectance": true, "indoor_bounds": [[-4.050000190734863, -2.700000047683716, -1.2799999713897705], [6.199999809265137, 11.5, 1.649999976158142]], "caster_bounds": [[-14.279096603393555, -5.968130111694336, -1.225000023841858], [14.091584205627441, 16.69512939453125, 6.0835700035095215]], "normal_asset": "normal_weights_full.f32"};
    const solarKeys={"minutes": [750.0, 751.875, 753.75, 755.625, 756.09375, 756.5625, 757.5, 757.96875, 758.4375, 759.375, 759.84375, 760.3125, 761.25, 762.1875, 763.125, 765.0, 765.9375, 766.875, 768.75, 769.21875, 769.6875, 770.625, 772.5, 773.4375, 774.375, 776.25, 778.125, 779.0625, 780.0, 781.875, 783.75, 785.625, 787.5, 789.375, 790.3125, 791.25, 793.125, 795.0, 795.9375, 796.875, 797.8125, 798.75, 799.6875, 800.625, 801.5625, 802.5, 803.4375, 804.375, 805.3125, 806.25, 807.1875, 808.125, 810.0, 811.875, 813.75, 814.6875, 815.625, 816.5625, 817.5, 818.4375, 819.375, 821.25, 822.1875, 823.125, 824.0625, 825.0, 825.9375, 826.875, 827.8125, 828.75, 829.6875, 830.625, 832.5, 833.4375, 834.375, 835.3125, 836.25, 837.1875, 838.125, 839.0625, 840.0, 841.875, 842.8125, 843.75, 844.6875, 845.625, 846.5625, 847.5, 848.4375, 849.375, 850.3125, 851.25, 853.125, 855.0, 856.875, 857.8125, 858.75, 859.6875, 860.625, 861.5625, 862.5, 864.375, 865.3125, 866.25, 867.1875, 868.125, 870.0, 871.875, 873.75, 874.6875, 875.625, 877.5, 878.4375, 879.375, 880.3125, 881.25, 882.1875, 883.125, 885.0, 885.9375, 886.875, 887.8125, 888.75, 889.6875, 890.625, 891.5625, 892.5, 893.4375, 894.375, 895.3125, 896.25, 897.1875, 898.125, 899.0625, 900.0, 900.9375, 901.875, 903.75, 904.6875, 905.625, 906.5625, 907.5, 908.4375, 909.375, 910.3125, 911.25, 913.125, 914.0625, 915.0, 915.9375, 916.875, 917.8125, 918.75, 919.6875, 920.625, 922.5, 924.375, 925.3125, 926.25, 928.125, 930.0, 931.875, 933.75, 935.625, 937.5, 939.375, 940.3125, 941.25, 942.1875, 943.125, 945.0, 945.9375, 946.875, 948.75, 949.6875, 950.625, 952.5, 953.4375, 954.375, 956.25, 958.125, 959.0625, 960.0, 961.875, 963.75, 964.6875, 965.625, 967.5, 968.4375, 969.375, 970.3125, 971.25, 972.1875, 972.65625, 973.125, 974.0625, 975.0, 975.9375, 976.875, 977.8125, 978.75, 980.625, 982.5, 984.375, 986.25, 988.125, 990.0, 991.875, 993.75, 994.6875, 995.625, 997.5, 999.375, 1001.25, 1002.1875, 1003.125, 1005.0, 1005.9375, 1006.875, 1008.75, 1009.6875, 1010.625, 1012.5, 1014.375, 1016.25, 1018.125, 1019.0625, 1020.0, 1021.875, 1023.75, 1025.625, 1027.5, 1029.375, 1031.25, 1033.125, 1034.0625, 1035.0, 1036.875, 1038.75, 1040.625, 1041.5625, 1042.5, 1044.375, 1045.3125, 1046.25, 1047.1875, 1048.125, 1050.0, 1050.9375, 1051.875, 1052.8125, 1053.75, 1054.6875, 1055.625, 1057.5, 1059.375, 1061.25, 1063.125, 1065.0, 1066.875, 1068.75, 1070.625, 1072.5, 1074.375, 1076.25, 1077.1875, 1078.125, 1080.0, 1081.875, 1083.75, 1085.625, 1086.5625, 1087.5, 1089.375, 1091.25, 1093.125, 1095.0], "states": [{"minute": 750.0, "direction": [0.6301792118544377, 0.22936647536835056, 0.7417986121068303], "irradiance": [8.002760485171857, 8.067180422611607, 7.326394495853758]}, {"minute": 751.875, "direction": [0.633829122968109, 0.2275967822079031, 0.7392295635363119], "irradiance": [7.976889144276697, 8.026008626027949, 7.286577134463248]}, {"minute": 753.75, "direction": [0.6374644219908079, 0.2258218421050761, 0.7366434730072584], "irradiance": [7.951017803381539, 7.984836829444289, 7.246759773072738]}, {"minute": 755.625, "direction": [0.6410850251155298, 0.22404169597876292, 0.7340404001385621], "irradiance": [7.9251464624863805, 7.94366503286063, 7.206942411682229]}, {"minute": 756.09375, "direction": [0.6419878698737489, 0.22359585081263866, 0.7333869854547044], "irradiance": [7.918678627262592, 7.9333720837147155, 7.1969880713346015]}, {"minute": 756.5625, "direction": [0.6428897896190471, 0.2231496834767964, 0.7327325140648415], "irradiance": [7.912210792038802, 7.9230791345688, 7.187033730986974]}, {"minute": 757.5, "direction": [0.6446908488740651, 0.22225638486787477, 0.7314204049406199], "irradiance": [7.899275121591223, 7.902493236276972, 7.16712505029172]}, {"minute": 757.96875, "direction": [0.6455899857887156, 0.22180925488191125, 0.7307627690968235], "irradiance": [7.892807286367432, 7.892200287131057, 7.157170709944092]}, {"minute": 758.4375, "direction": [0.6464881925003101, 0.22136180530045954, 0.7301040803281431], "irradiance": [7.886339451143642, 7.881907337985142, 7.147216369596464]}, {"minute": 759.375, "direction": [0.6482818101389228, 0.2204659499303949, 0.7287835478139516], "irradiance": [7.873403780696063, 7.861321439693312, 7.12730768890121]}, {"minute": 759.84375, "direction": [0.6491772184815938, 0.22001754543258173, 0.7281217059711398], "irradiance": [7.866935945472274, 7.851028490547398, 7.117353348553582]}, {"minute": 760.3125, "direction": [0.6500716914525185, 0.219568823920878, 0.7274588150088388], "irradiance": [7.860468110248485, 7.840735541401483, 7.107399008205955]}, {"minute": 761.25, "direction": [0.6518578261252467, 0.2186704324424294, 0.7261298895478058], "irradiance": [7.847532439800905, 7.820149643109653, 7.0874903275107]}, {"minute": 762.1875, "direction": [0.6536402038628502, 0.2177707806728629, 0.7247967790900176], "irradiance": [7.834596769353325, 7.799563744817825, 7.067581646815445]}, {"minute": 763.125, "direction": [0.6554188143927244, 0.21686987379725597, 0.7234594913187592], "irradiance": [7.821661098905746, 7.778977846525995, 7.0476729661201905]}, {"minute": 765.0, "direction": [0.6589646928474878, 0.21506431550436952, 0.7207724146893061], "irradiance": [7.795789758010588, 7.7378060499423365, 7.007855604729682]}, {"minute": 765.9375, "direction": [0.6607319403359645, 0.21415967449329373, 0.7194226413178821], "irradiance": [7.782854087563008, 7.717220151650507, 6.987946924034426]}, {"minute": 766.875, "direction": [0.6624953797440052, 0.21325379918852508, 0.7180687216064398], "irradiance": [7.769918417115429, 7.696634253358678, 6.968038243339172]}, {"minute": 768.75, "direction": [0.6660107936869663, 0.21143836658877851, 0.7153484744002231], "irradiance": [7.74404707622027, 7.655462456775019, 6.928220881948662]}, {"minute": 769.21875, "direction": [0.6668872512686987, 0.21098374517158305, 0.7146658333575574], "irradiance": [7.737579240996482, 7.645169507629104, 6.918266541601035]}, {"minute": 769.6875, "direction": [0.6677627479610545, 0.2105288197569142, 0.7139821625833954], "irradiance": [7.731111405772691, 7.63487655848319, 6.908312201253407]}, {"minute": 770.625, "direction": [0.6695108536331587, 0.20961805955752397, 0.7126117357823526], "irradiance": [7.7181757353251115, 7.614290660191359, 6.888403520558152]}, {"minute": 772.5, "direction": [0.6729954788933358, 0.2077929200595292, 0.7098585688447128], "irradiance": [7.692304394429954, 7.573118863607702, 6.848586159167644]}, {"minute": 773.4375, "direction": [0.6747319783980247, 0.20687855127998328, 0.7084758445757929], "irradiance": [7.679368723982374, 7.552532965315872, 6.828677478472388]}, {"minute": 774.375, "direction": [0.6764645891340774, 0.20596299017096809, 0.7070890370579208], "irradiance": [7.666433053534795, 7.531947067024042, 6.808768797777134]}, {"minute": 776.25, "direction": [0.6799181043796414, 0.20412831207845056, 0.704303204269864], "irradiance": [7.640561712639636, 7.490775270440383, 6.768951436386624]}, {"minute": 778.125, "direction": [0.6833559450138076, 0.20228892807805018, 0.7015011347042279], "irradiance": [7.6146903717444765, 7.449603473856724, 6.729134074996114]}, {"minute": 779.0625, "direction": [0.6850689625703065, 0.2013674846096085, 0.7000940313021125], "irradiance": [7.601754701296898, 7.429017575564895, 6.7092253943008595]}, {"minute": 780.0, "direction": [0.6867780317817137, 0.20044488057432902, 0.6986828929590151], "irradiance": [7.588819030849319, 7.408431677273066, 6.6893167136056055]}, {"minute": 781.875, "direction": [0.6901842857916812, 0.1985962120793599, 0.6958485440050565], "irradiance": [7.56294768995416, 7.367259880689407, 6.649499352215096]}, {"minute": 783.75, "direction": [0.6935746285170354, 0.19674296521174656, 0.6929981531845131], "irradiance": [7.537076349059001, 7.3260880841057485, 6.609681990824586]}, {"minute": 785.625, "direction": [0.696948981797915, 0.1948851826956411, 0.6901317862093704], "irradiance": [7.511205008163842, 7.284916287522089, 6.569864629434076]}, {"minute": 787.5, "direction": [0.7003072678430741, 0.19302290735975886, 0.6872495091599229], "irradiance": [7.485333667268684, 7.243744490938431, 6.530047268043567]}, {"minute": 789.375, "direction": [0.7036494092316757, 0.19115618213639138, 0.6843513884832503], "irradiance": [7.459462326373526, 7.202572694354772, 6.490229906653058]}, {"minute": 790.3125, "direction": [0.705314401587479, 0.19022116426131971, 0.6828964076493332], "irradiance": [7.446526655925946, 7.181986796062942, 6.470321225957802]}, {"minute": 791.25, "direction": [0.7069753289150768, 0.18928505006041627, 0.6814374909916866], "irradiance": [7.433590985478367, 7.161400897771114, 6.450412545262548]}, {"minute": 793.125, "direction": [0.7102849502186035, 0.18740955426830538, 0.6785078838612791], "irradiance": [7.407719644583208, 7.120229101187454, 6.410595183872038]}, {"minute": 795.0, "direction": [0.7135781968433204, 0.18552973799713024, 0.6755626346302405], "irradiance": [7.381848303688049, 7.079057304603795, 6.370777822481529]}, {"minute": 795.9375, "direction": [0.7152186559108816, 0.18458822322074092, 0.6740841654313221], "irradiance": [7.36891263324047, 7.058471406311967, 6.350869141786274]}, {"minute": 796.875, "direction": [0.7168549928677881, 0.18364564458356536, 0.6726018111973906], "irradiance": [7.355976962792891, 7.037885508020136, 6.330960461091019]}, {"minute": 797.8125, "direction": [0.7184871982831315, 0.18270200751808688, 0.6711155804718866], "irradiance": [7.343041292345312, 7.017299609728307, 6.311051780395764]}, {"minute": 798.75, "direction": [0.7201152627498141, 0.18175731746288903, 0.6696254818205927], "irradiance": [7.330105621897732, 6.996713711436478, 6.29114309970051]}, {"minute": 799.6875, "direction": [0.7217391768846054, 0.18081157986262417, 0.6681315238315839], "irradiance": [7.317169951450152, 6.9761278131446485, 6.271234419005254]}, {"minute": 800.625, "direction": [0.7233589313281942, 0.17986480016798206, 0.6666337151151783], "irradiance": [7.304234281002574, 6.9555419148528195, 6.25132573831]}, {"minute": 801.5625, "direction": [0.7249745167452439, 0.1789169838356586, 0.6651320643038875], "irradiance": [7.291298610554994, 6.9349560165609905, 6.231417057614745]}, {"minute": 802.5, "direction": [0.7265859238244454, 0.17796813632832417, 0.6636265800523666], "irradiance": [7.278362940107415, 6.914370118269161, 6.2115083769194905]}, {"minute": 803.4375, "direction": [0.7281931432785714, 0.1770182631145924, 0.6621172710373646], "irradiance": [7.265427269659836, 6.893784219977332, 6.191599696224236]}, {"minute": 804.375, "direction": [0.7297961658445296, 0.1760673696689883, 0.6606041459576739], "irradiance": [7.252491599212257, 6.873198321685502, 6.171691015528981]}, {"minute": 805.3125, "direction": [0.731394982283416, 0.1751154614719171, 0.659087213534081], "irradiance": [7.239555928764677, 6.852612423393673, 6.151782334833726]}, {"minute": 806.25, "direction": [0.7329895833805686, 0.17416254400963227, 0.6575664825093152], "irradiance": [7.226620258317097, 6.832026525101843, 6.131873654138471]}, {"minute": 807.1875, "direction": [0.7345799599456201, 0.17320862277420435, 0.656041961647999], "irradiance": [7.213684587869518, 6.811440626810014, 6.111964973443216]}, {"minute": 808.125, "direction": [0.7361661028125507, 0.17225370326348885, 0.6545136597365978], "irradiance": [7.200748917421938, 6.790854728518184, 6.092056292747961]}, {"minute": 810.0, "direction": [0.7393256509100274, 0.17034089143635345, 0.6514457480183079], "irradiance": [7.1748775765267805, 6.749682931934526, 6.052238931357452]}, {"minute": 811.875, "direction": [0.7424681548338019, 0.16842415262556926, 0.648362818081087], "irradiance": [7.149006235631621, 6.708511135350866, 6.012421569966942]}, {"minute": 813.75, "direction": [0.7455935421376082, 0.16650353101901097, 0.6452649409978013], "irradiance": [7.123134894736464, 6.667339338767208, 5.972604208576433]}, {"minute": 814.6875, "direction": [0.7471497945254184, 0.16554177800053144, 0.643710419580923], "irradiance": [7.110199224288884, 6.646753440475379, 5.952695527881178]}, {"minute": 815.625, "direction": [0.7487017407697815, 0.16457907089406607, 0.6421521881859044], "irradiance": [7.097263553841303, 6.626167542183548, 5.932786847185922]}, {"minute": 816.5625, "direction": [0.7502493719261679, 0.1636154152481072, 0.6405902557934987], "irradiance": [7.084327883393725, 6.605581643891719, 5.912878166490668]}, {"minute": 817.5, "direction": [0.7517926790749185, 0.16265081661661393, 0.6390246314057896], "irradiance": [7.071392212946146, 6.58499574559989, 5.892969485795414]}, {"minute": 818.4375, "direction": [0.7533316533212948, 0.16168528055898035, 0.6374553240461392], "irradiance": [7.058456542498566, 6.5644098473080605, 5.873060805100159]}, {"minute": 819.375, "direction": [0.7548662857955307, 0.1607188126400031, 0.6358823427591364], "irradiance": [7.045520872050987, 6.543823949016231, 5.853152124404904]}, {"minute": 821.25, "direction": [0.7579224900736856, 0.1587831035040265, 0.6327253946872478], "irradiance": [7.019649531155828, 6.502652152432573, 5.813334763014394]}, {"minute": 822.1875, "direction": [0.7594440442633941, 0.15781387344334574, 0.6311414460972024], "irradiance": [7.006713860708249, 6.482066254140744, 5.79342608231914]}, {"minute": 823.125, "direction": [0.7609612214526418, 0.1568437338338945, 0.6295538599693803], "irradiance": [6.99377819026067, 6.461480355848914, 5.773517401623885]}, {"minute": 824.0625, "direction": [0.7624740128972887, 0.1558726902670017, 0.6279626454537187], "irradiance": [6.98084251981309, 6.440894457557084, 5.75360872092863]}, {"minute": 825.0, "direction": [0.7639824098784713, 0.1549007483392063, 0.6263678117210666], "irradiance": [6.967906849365511, 6.420308559265256, 5.7337000402333755]}, {"minute": 825.9375, "direction": [0.7654864037026536, 0.15392791365222486, 0.624769367963132], "irradiance": [6.954971178917932, 6.399722660973426, 5.713791359538121]}, {"minute": 826.875, "direction": [0.7669859857016765, 0.15295419181291922, 0.6231673233924293], "irradiance": [6.942035508470353, 6.379136762681596, 5.693882678842866]}, {"minute": 827.8125, "direction": [0.7684811472328086, 0.1519795884332643, 0.6215616872422254], "irradiance": [6.929099838022772, 6.358550864389766, 5.673973998147611]}, {"minute": 828.75, "direction": [0.769971879678795, 0.15100410913031587, 0.6199524687664879], "irradiance": [6.916164167575194, 6.337964966097937, 5.654065317452356]}, {"minute": 829.6875, "direction": [0.7714581744479079, 0.15002775952617767, 0.6183396772398299], "irradiance": [6.903228497127615, 6.317379067806109, 5.634156636757102]}, {"minute": 830.625, "direction": [0.7729400229739954, 0.14905054524796965, 0.616723321957458], "irradiance": [6.890292826680035, 6.296793169514278, 5.614247956061846]}, {"minute": 832.5, "direction": [0.7758903471606654, 0.14709354520270876, 0.6134799574090427], "irradiance": [6.864421485784877, 6.25562137293062, 5.5744305946713375]}, {"minute": 833.4375, "direction": [0.7773588058172691, 0.14611377071468307, 0.6118529668358947], "irradiance": [6.8514858153372975, 6.23503547463879, 5.554521913976083]}, {"minute": 834.375, "direction": [0.7788227842229882, 0.14513315411057717, 0.610222449892716], "irradiance": [6.838550144889719, 6.214449576346961, 5.534613233280828]}, {"minute": 835.3125, "direction": [0.7802822739402898, 0.14415170104210343, 0.6085884159768724], "irradiance": [6.825614474442139, 6.193863678055132, 5.514704552585574]}, {"minute": 836.25, "direction": [0.7817372665575112, 0.14316941716579515, 0.6069508745059994], "irradiance": [6.81267880399456, 6.1732777797633025, 5.494795871890318]}, {"minute": 837.1875, "direction": [0.7831877536889081, 0.142186308142974, 0.605309834917948], "irradiance": [6.79974313354698, 6.152691881471473, 5.474887191195063]}, {"minute": 838.125, "direction": [0.784633726974704, 0.14120237963971718, 0.6036653066707304], "irradiance": [6.7868074630994, 6.132105983179644, 5.454978510499808]}, {"minute": 839.0625, "direction": [0.7860751780811367, 0.14021763732682507, 0.6020172992424655], "irradiance": [6.773871792651821, 6.111520084887815, 5.435069829804554]}, {"minute": 840.0, "direction": [0.787512098700508, 0.13923208687978822, 0.6003658221313241], "irradiance": [6.7609361222042414, 6.090934186595985, 5.415161149109299]}, {"minute": 841.875, "direction": [0.7903723153778752, 0.13725858430849838, 0.5970524969530264], "irradiance": [6.735064781309084, 6.049762390012326, 5.37534378771879]}, {"minute": 842.8125, "direction": [0.7917955949512205, 0.13627064355838378, 0.5953906679819787], "irradiance": [6.722129110861504, 6.029176491720496, 5.355435107023535]}, {"minute": 843.75, "direction": [0.7932143110682982, 0.13528191742233567, 0.5937254075201612], "irradiance": [6.709193440413925, 6.008590593428668, 5.33552642632828]}, {"minute": 844.6875, "direction": [0.7946284555524411, 0.13429241159880517, 0.5920567251651815], "irradiance": [6.696257769966346, 5.988004695136839, 5.315617745633026]}, {"minute": 845.625, "direction": [0.7960380202533305, 0.13330213179073716, 0.590384630534369], "irradiance": [6.683322099518766, 5.967418796845008, 5.29570906493777]}, {"minute": 846.5625, "direction": [0.7974429970470427, 0.13231108370553712, 0.5887091332647191], "irradiance": [6.670386429071186, 5.946832898553178, 5.275800384242515]}, {"minute": 847.5, "direction": [0.7988433778360963, 0.13131927305503854, 0.5870302430128383], "irradiance": [6.657450758623607, 5.92624700026135, 5.255891703547261]}, {"minute": 848.4375, "direction": [0.8002391545494987, 0.13032670555547002, 0.5853479694548889], "irradiance": [6.644515088176028, 5.905661101969521, 5.235983022852007]}, {"minute": 849.375, "direction": [0.8016303191427924, 0.1293333869274219, 0.5836623222865314], "irradiance": [6.631579417728449, 5.88507520367769, 5.216074342156751]}, {"minute": 850.3125, "direction": [0.8030168635981014, 0.1283393228958139, 0.5819733112228708], "irradiance": [6.61864374728087, 5.864489305385861, 5.196165661461497]}, {"minute": 851.25, "direction": [0.8043987799241777, 0.12734451918986162, 0.5802809459983993], "irradiance": [6.605708076833291, 5.843903407094032, 5.176256980766242]}, {"minute": 853.125, "direction": [0.8071486963570559, 0.12535271569306944, 0.5768861921015945], "irradiance": [6.579836735938132, 5.8027316105103735, 5.136439619375732]}, {"minute": 855.0, "direction": [0.8098800050457454, 0.12335802235543686, 0.5734781388576718], "irradiance": [6.553965395042972, 5.761559813926715, 5.0966222579852225]}, {"minute": 856.875, "direction": [0.8125926430235417, 0.12136048516197703, 0.5700568648347852], "irradiance": [6.528094054147815, 5.720388017343056, 5.056804896594714]}, {"minute": 857.8125, "direction": [0.8139419409364158, 0.12036066450758443, 0.5683412946671683], "irradiance": [6.515158383700236, 5.699802119051226, 5.036896215899459]}, {"minute": 858.75, "direction": [0.8152865477541693, 0.11936015016326433, 0.5666224489058758], "irradiance": [6.502222713252655, 5.679216220759397, 5.016987535204203]}, {"minute": 859.6875, "direction": [0.8166264557272587, 0.11835894789540839, 0.5649003374573498], "irradiance": [6.489287042805076, 5.658630322467568, 4.997078854508949]}, {"minute": 860.625, "direction": [0.817961657133222, 0.11735706347437294, 0.5631749702468545], "irradiance": [6.476351372357497, 5.638044424175738, 4.977170173813694]}, {"minute": 861.5625, "direction": [0.8192921442767233, 0.11635450267444582, 0.5614463572184178], "irradiance": [6.463415701909917, 5.617458525883908, 4.957261493118439]}, {"minute": 862.5, "direction": [0.8206179094895966, 0.11535127127381341, 0.5597145083347757], "irradiance": [6.450480031462338, 5.596872627592079, 4.9373528124231845]}, {"minute": 864.375, "direction": [0.8232552435869125, 0.1133428198024685, 0.5562411429460057], "irradiance": [6.4246086905671795, 5.55570083100842, 4.897535451032675]}, {"minute": 865.3125, "direction": [0.8245667972712721, 0.11233761130731983, 0.5544996464593663], "irradiance": [6.411673020119601, 5.535114932716591, 4.87762677033742]}, {"minute": 866.25, "direction": [0.8258735986249253, 0.11133175536252685, 0.5527549541543835], "irradiance": [6.398737349672022, 5.514529034424762, 4.857718089642166]}, {"minute": 867.1875, "direction": [0.8271756401162175, 0.1103252577652668, 0.551007076086463], "irradiance": [6.385801679224442, 5.493943136132932, 4.83780940894691]}, {"minute": 868.125, "direction": [0.8284729142409274, 0.10931812431641554, 0.549256022329374], "irradiance": [6.372866008776862, 5.473357237841103, 4.817900728251655]}, {"minute": 870.0, "direction": [0.8310531305111397, 0.10730197308573156, 0.5457444281342171], "irradiance": [6.346994667881704, 5.432185441257444, 4.778083366861146]}, {"minute": 871.875, "direction": [0.8336141879520937, 0.10528334815017114, 0.5422202525240654], "irradiance": [6.3211233269865446, 5.3910136446737855, 4.738266005470637]}, {"minute": 873.75, "direction": [0.8361560275220018, 0.10326229604645887, 0.5386835767441207], "irradiance": [6.295251986091387, 5.349841848090127, 4.698448644080127]}, {"minute": 874.6875, "direction": [0.8374197222758625, 0.10225087436460166, 0.5369105767584504], "irradiance": [6.282316315643808, 5.329255949798298, 4.678539963384872]}, {"minute": 875.625, "direction": [0.8386785906221191, 0.1012388633672741, 0.5351344823277588], "irradiance": [6.269380645196228, 5.308670051506468, 4.658631282689618]}, {"minute": 877.5, "direction": [0.8411818190980944, 0.09921309676017742, 0.531573051094651], "irradiance": [6.243509304301069, 5.26749825492281, 4.618813921299108]}, {"minute": 878.4375, "direction": [0.8424261648006381, 0.09819935282576668, 0.5297877348182843], "irradiance": [6.23057363385349, 5.246912356630981, 4.5989052406038535]}, {"minute": 879.375, "direction": [0.8436656552413108, 0.09718504292653496, 0.5279993651488774], "irradiance": [6.2176379634059105, 5.22632645833915, 4.578996559908599]}, {"minute": 880.3125, "direction": [0.8449002832763993, 0.09617017290838357, 0.5262079523935684], "irradiance": [6.204702292958331, 5.20574056004732, 4.559087879213343]}, {"minute": 881.25, "direction": [0.8461300417902151, 0.09515474862044193, 0.5244135068770345], "irradiance": [6.191766622510752, 5.185154661755491, 4.539179198518089]}, {"minute": 882.1875, "direction": [0.8473549236951343, 0.09413877591503401, 0.5226160389414317], "irradiance": [6.178830952063173, 5.164568763463663, 4.519270517822834]}, {"minute": 883.125, "direction": [0.8485749219316396, 0.09312226064764452, 0.5208155589463356], "irradiance": [6.165895281615593, 5.143982865171832, 4.499361837127579]}, {"minute": 885.0, "direction": [0.8510002393021108, 0.09108762586446112, 0.5172056043027051], "irradiance": [6.140023940720435, 5.102811068588174, 4.45954447573707]}, {"minute": 885.9375, "direction": [0.8522055444579365, 0.09006951807513665, 0.5153961504598815], "irradiance": [6.127088270272855, 5.082225170296344, 4.439635795041815]}, {"minute": 886.875, "direction": [0.8534059379891485, 0.0890508911767019, 0.5135837261688662], "irradiance": [6.114152599825276, 5.061639272004515, 4.419727114346561]}, {"minute": 887.8125, "direction": [0.8546014129773659, 0.0880317510399389, 0.5117683418754347], "irradiance": [6.101216929377697, 5.041053373712686, 4.399818433651306]}, {"minute": 888.75, "direction": [0.8557919625325561, 0.08701210353858761, 0.5099500080424222], "irradiance": [6.088281258930118, 5.0204674754208565, 4.379909752956051]}, {"minute": 889.6875, "direction": [0.8569775797930728, 0.08599195454931216, 0.5081287351496635], "irradiance": [6.075345588482538, 4.999881577129027, 4.360001072260795]}, {"minute": 890.625, "direction": [0.8581582579256974, 0.08497130995166703, 0.506304533693932], "irradiance": [6.062409918034959, 4.979295678837198, 4.340092391565542]}, {"minute": 891.5625, "direction": [0.8593339901256777, 0.083950175628063, 0.5044774141888805], "irradiance": [6.049474247587379, 4.958709780545369, 4.320183710870286]}, {"minute": 892.5, "direction": [0.8605047696167664, 0.0829285574637334, 0.5026473871649787], "irradiance": [6.0365385771398, 4.938123882253539, 4.300275030175031]}, {"minute": 893.4375, "direction": [0.8616705896512615, 0.08190646134670017, 0.5008144631694545], "irradiance": [6.023602906692221, 4.91753798396171, 4.280366349479777]}, {"minute": 894.375, "direction": [0.8628314435100439, 0.0808838931677399, 0.4989786527662317], "irradiance": [6.010667236244641, 4.896952085669881, 4.260457668784523]}, {"minute": 895.3125, "direction": [0.8639873245026168, 0.0798608588203498, 0.49713996653587006], "irradiance": [5.997731565797062, 4.87636618737805, 4.240548988089267]}, {"minute": 896.25, "direction": [0.8651382259671445, 0.07883736420071388, 0.4952984150755039], "irradiance": [5.984795895349483, 4.855780289086222, 4.220640307394013]}, {"minute": 897.1875, "direction": [0.86628414127049, 0.07781341520766884, 0.49345400899878056], "irradiance": [5.971860224901905, 4.835194390794393, 4.200731626698758]}, {"minute": 898.125, "direction": [0.867425063808254, 0.07678901774267015, 0.49160675893580064], "irradiance": [5.958924554454324, 4.814608492502562, 4.180822946003503]}, {"minute": 899.0625, "direction": [0.8685609870048123, 0.07576417770975812, 0.48975667553305535], "irradiance": [5.945988884006745, 4.794022594210733, 4.160914265308248]}, {"minute": 900.0, "direction": [0.8696919043133546, 0.07473890101552354, 0.48790376945336544], "irradiance": [5.933053213559166, 4.773436695918904, 4.141005584612994]}, {"minute": 900.9375, "direction": [0.8708178092159211, 0.07371319356907403, 0.48604805137582024], "irradiance": [5.920117543111587, 4.752850797627075, 4.1210969039177385]}, {"minute": 901.875, "direction": [0.8719386952234408, 0.07268706128199982, 0.48418953199571596], "irradiance": [5.907181872664006, 4.732264899335244, 4.101188223222484]}, {"minute": 903.75, "direction": [0.8741653847417239, 0.07063354584454669, 0.48046413218967793], "irradiance": [5.8813105317688485, 4.691093102751586, 4.061370861831975]}, {"minute": 904.6875, "direction": [0.8752711754191242, 0.06960617452945443, 0.47859727323481477], "irradiance": [5.868374861321269, 4.670507204459756, 4.041462181136719]}, {"minute": 905.625, "direction": [0.8763719215348259, 0.06857840204424273, 0.47672765591940985], "irradiance": [5.85543919087369, 4.6499213061679265, 4.021553500441465]}, {"minute": 906.5625, "direction": [0.8774676167447589, 0.06755023431240341, 0.47485529101886625], "irradiance": [5.842503520426111, 4.629335407876098, 4.00164481974621]}, {"minute": 907.5, "direction": [0.8785582547339638, 0.06652167725970641, 0.4729801893244232], "irradiance": [5.829567849978531, 4.6087495095842685, 3.981736139050955]}, {"minute": 908.4375, "direction": [0.8796438292166275, 0.06549273681416538, 0.47110236164309255], "irradiance": [5.816632179530951, 4.5881636112924395, 3.9618274583557005]}, {"minute": 909.375, "direction": [0.8807243339361209, 0.06446341890600367, 0.46922181879759733], "irradiance": [5.803696509083373, 4.56757771300061, 3.941918777660446]}, {"minute": 910.3125, "direction": [0.8817997626650337, 0.06343372946762013, 0.4673385716263096], "irradiance": [5.790760838635793, 4.54699181470878, 3.9220100969651908]}, {"minute": 911.25, "direction": [0.8828701092052104, 0.06240367443355484, 0.465452630983187], "irradiance": [5.777825168188214, 4.526405916416951, 3.9021014162699355]}, {"minute": 913.125, "direction": [0.884995531073225, 0.060342491327040816, 0.46167271277482536], "irradiance": [5.751953827293056, 4.485234119833292, 3.8622840548794266]}, {"minute": 914.0625, "direction": [0.8860505941513488, 0.059311375134070816, 0.4597787569948696], "irradiance": [5.739018156845475, 4.464648221541462, 3.8423753741841713]}, {"minute": 915.0, "direction": [0.8871005505413788, 0.058279917104308195, 0.45788215131352034], "irradiance": [5.726082486397896, 4.444062323249633, 3.822466693488917]}, {"minute": 915.9375, "direction": [0.8881453941919679, 0.05724812318248619, 0.4559829066617265], "irradiance": [5.713146815950317, 4.423476424957805, 3.8025580127936625]}, {"minute": 916.875, "direction": [0.8891851190812358, 0.05621599931527395, 0.4540810339856462], "irradiance": [5.700211145502738, 4.402890526665974, 3.782649332098407]}, {"minute": 917.8125, "direction": [0.8902197192168037, 0.05518355145124228, 0.4521765442465844], "irradiance": [5.687275475055159, 4.382304628374145, 3.7627406514031527]}, {"minute": 918.75, "direction": [0.8912491886358291, 0.054150785540829235, 0.4502694484209288], "irradiance": [5.6743398046075795, 4.361718730082316, 3.7428319707078983]}, {"minute": 919.6875, "direction": [0.89227352140504, 0.05311770753630599, 0.44835975750008744], "irradiance": [5.66140413416, 4.341132831790486, 3.722923290012643]}, {"minute": 920.625, "direction": [0.893292711620769, 0.05208432339174239, 0.446447482490425], "irradiance": [5.648468463712421, 4.320546933498657, 3.7030146093173886]}, {"minute": 922.5, "direction": [0.8953156409253392, 0.05001666050756141, 0.44261522430449735], "irradiance": [5.622597122817262, 4.279375136914998, 3.663197247926879]}, {"minute": 924.375, "direction": [0.8973179299135842, 0.04794784455551559, 0.4387727622107829], "irradiance": [5.596725781922103, 4.2382033403313395, 3.623379886536369]}, {"minute": 925.3125, "direction": [0.8983113198454304, 0.04691301908235125, 0.4368477323715208], "irradiance": [5.583790111474524, 4.2176174420395105, 3.6034712058411147]}, {"minute": 926.25, "direction": [0.899299532425383, 0.04587792322941675, 0.4349201847921562], "irradiance": [5.5708544410269445, 4.197031543747681, 3.5835625251458594]}, {"minute": 928.125, "direction": [0.9012604027775143, 0.04380694424855964, 0.4310575808646873], "irradiance": [5.544983100131787, 4.155859747164022, 3.5437451637553505]}, {"minute": 930.0, "direction": [0.9032004957647091, 0.04173495535662175, 0.4271850394755938], "irradiance": [5.519111759236628, 4.114687950580363, 3.503927802364841]}, {"minute": 931.875, "direction": [0.905119766660693, 0.039662004320562855, 0.4233026499011887], "irradiance": [5.493240418341469, 4.073516153996705, 3.464110440974331]}, {"minute": 933.75, "direction": [0.9070181712192176, 0.037588138929523555, 0.41941050164482163], "irradiance": [5.46736907744631, 4.032344357413045, 3.4242930795838213]}, {"minute": 935.625, "direction": [0.90889566567508, 0.035513406993723844, 0.41550868443481587], "irradiance": [5.441497736551152, 3.991172560829387, 3.3844757181933125]}, {"minute": 937.5, "direction": [0.910752206745132, 0.03343785634336064, 0.41159728822239955], "irradiance": [5.415626395655993, 3.950000764245728, 3.3446583568028028]}, {"minute": 939.375, "direction": [0.9125877516292783, 0.03136153482750526, 0.4076764031796319], "irradiance": [5.389755054760834, 3.9088289676620693, 3.304840995412293]}, {"minute": 940.3125, "direction": [0.9134976372589346, 0.03032309995274238, 0.4057124305855071], "irradiance": [5.376819384313255, 3.88824306937024, 3.2849323147170386]}, {"minute": 941.25, "direction": [0.9144022580114625, 0.029284490313000382, 0.4037461196973248], "irradiance": [5.363883713865675, 3.8676571710784104, 3.2650236340217833]}, {"minute": 942.1875, "direction": [0.915301608673146, 0.02824571189423052, 0.40177748184777506], "irradiance": [5.3509480434180965, 3.847071272786581, 3.245114953326529]}, {"minute": 943.125, "direction": [0.9161956840606432, 0.027206770683356366, 0.3998065283829585], "irradiance": [5.338012372970517, 3.8264853744947516, 3.2252062726312736]}, {"minute": 945.0, "direction": [0.917967988431758, 0.025128423837647566, 0.39585772005859315], "irradiance": [5.312141032075358, 3.785313577911093, 3.1853889112407647]}, {"minute": 945.9375, "direction": [0.9188462072008284, 0.02408903018121351, 0.3938798879577134], "irradiance": [5.299205361627779, 3.7647276796192637, 3.1654802305455094]}, {"minute": 946.875, "direction": [0.9197191302666768, 0.023049497689407776, 0.39189978575877465], "irradiance": [5.2862696911802, 3.7441417813274342, 3.145571549850255]}, {"minute": 948.75, "direction": [0.9214490691951442, 0.020970040165526033, 0.38793281672843627], "irradiance": [5.26039835028504, 3.7029699847437754, 3.1057541884597453]}, {"minute": 949.6875, "direction": [0.922306075087386, 0.019930127118252098, 0.3859459727603741], "irradiance": [5.247462679837462, 3.682384086451946, 3.0858455077644904]}, {"minute": 950.625, "direction": [0.9231577653357091, 0.01889009920514147, 0.3839569044207953], "irradiance": [5.234527009389883, 3.6617981881601165, 3.0659368270692355]}, {"minute": 952.5, "direction": [0.9248451792966451, 0.016809722758538197, 0.3799721404952439], "irradiance": [5.208655668494724, 3.6206263915764576, 3.026119465678726]}, {"minute": 953.4375, "direction": [0.9256808932839705, 0.015769386215143767, 0.3779764678751683], "irradiance": [5.195719998047144, 3.600040493284628, 3.006210784983471]}, {"minute": 954.375, "direction": [0.9265112721768582, 0.01472895878603994, 0.3759786168152373], "irradiance": [5.182784327599565, 3.579454594992799, 2.9863021042882165]}, {"minute": 956.25, "direction": [0.9281560055667836, 0.012647855256904187, 0.3719764254461744], "irradiance": [5.156912986704406, 3.5382827984091403, 2.946484742897707]}, {"minute": 958.125, "direction": [0.9297793415492713, 0.010566460148216604, 0.36796565865327674], "irradiance": [5.131041645809248, 3.4971110018254814, 2.906667381507197]}, {"minute": 959.0625, "direction": [0.9305829737984767, 0.009525668246248436, 0.365957088359887], "irradiance": [5.118105975361669, 3.476525103533652, 2.8867587008119426]}, {"minute": 960.0, "direction": [0.9313812427004602, 0.008484821443784759, 0.36394640889946084], "irradiance": [5.105170304914089, 3.4559392052418225, 2.8668500201166873]}, {"minute": 961.875, "direction": [0.932961672090641, 0.006402987133032079, 0.35991876884320667], "irradiance": [5.079298964018931, 3.414767408658164, 2.8270326587261785]}, {"minute": 963.75, "direction": [0.9345205932851071, 0.004321005209891199, 0.3558828313364211], "irradiance": [5.053427623123772, 3.373595612074505, 2.7872152973356688]}, {"minute": 964.6875, "direction": [0.9352919770586607, 0.0032799738927399755, 0.3538617801076638], "irradiance": [5.040491952676192, 3.3530097137826758, 2.7673066166404134]}, {"minute": 965.625, "direction": [0.9360579703449952, 0.002238923671698127, 0.35183868942229857], "irradiance": [5.0275562822286135, 3.3324238154908463, 2.747397935945159]}, {"minute": 967.5, "direction": [0.9375737678281131, 0.000156790518084999, 0.34778643633317424], "irradiance": [5.001684941333455, 3.2912520189071874, 2.7075805745546493]}, {"minute": 968.4375, "direction": [0.9383235632887081, -0.0008842804142632741, 0.3457572972842811], "irradiance": [4.988749270885876, 3.270666120615358, 2.687671893859395]}, {"minute": 969.375, "direction": [0.9390679507897575, -0.0019253462501258203, 0.3437261654883765], "irradiance": [4.975813600438295, 3.2500802223235286, 2.6677632131641396]}, {"minute": 970.3125, "direction": [0.9398069260410378, -0.0029664009893953557, 0.341693052651741], "irradiance": [4.962877929990716, 3.229494324031699, 2.647854532468885]}, {"minute": 971.25, "direction": [0.9405404847835191, -0.004007438632028767, 0.3396579704920722], "irradiance": [4.949942259543137, 3.20890842573987, 2.6279458517736307]}, {"minute": 972.1875, "direction": [0.941268622789389, -0.005048453178081418, 0.337620930738418], "irradiance": [4.937006589095558, 3.1883225274480402, 2.6080371710783754]}, {"minute": 972.65625, "direction": [0.9416306577034453, -0.005568949914942637, 0.33660168043226074], "irradiance": [4.930538753871769, 3.178029578302126, 2.598082830730749]}, {"minute": 973.125, "direction": [0.9419913358620772, -0.006089438627741631, 0.3355819451311088], "irradiance": [4.9240709186479785, 3.1677366291562112, 2.588128490383121]}, {"minute": 974.0625, "direction": [0.9427086198362792, -0.007130388981365722, 0.3335410254216895], "irradiance": [4.911135248200399, 3.1471507308643814, 2.5682198096878657]}, {"minute": 975.0, "direction": [0.9434204705779813, -0.008171298239511949, 0.3314981833728526], "irradiance": [4.89819957775282, 3.1265648325725524, 2.5483111289926113]}, {"minute": 975.9375, "direction": [0.9441268839844834, -0.009212160402975655, 0.3294534307583693], "irradiance": [4.885263907305241, 3.105978934280723, 2.528402448297357]}, {"minute": 976.875, "direction": [0.9448278559844235, -0.010252969472823585, 0.3274067793630224], "irradiance": [4.872328236857661, 3.0853930359888935, 2.5084937676021015]}, {"minute": 977.8125, "direction": [0.9455233825378003, -0.011293719450428338, 0.32535824098253846], "irradiance": [4.859392566410083, 3.064807137697064, 2.488585086906847]}, {"minute": 978.75, "direction": [0.9462134596359973, -0.012334404337503239, 0.323307827423519], "irradiance": [4.846456895962503, 3.044221239405235, 2.4686764062115927]}, {"minute": 980.625, "direction": [0.9475772495894457, -0.014415554848826926, 0.3192014220502504], "irradiance": [4.820585555067344, 3.003049442821576, 2.428859044821083]}, {"minute": 982.5, "direction": [0.9489191944043972, -0.016496373028625853, 0.315087657910954], "irradiance": [4.794714214172186, 2.9618776462379173, 2.389041683430573]}, {"minute": 984.375, "direction": [0.9502392631440918, -0.018576810906392776, 0.31096662984301354], "irradiance": [4.768842873277027, 2.9207058496542584, 2.3492243220400635]}, {"minute": 986.25, "direction": [0.9515374253760928, -0.020656820520387763, 0.3068384328512734], "irradiance": [4.742971532381868, 2.8795340530705995, 2.3094069606495538]}, {"minute": 988.125, "direction": [0.9528136511729892, -0.022736353918743984, 0.3027031621058476], "irradiance": [4.7171001914867094, 2.838362256486941, 2.269589599259045]}, {"minute": 990.0, "direction": [0.9540679111130856, -0.024815363160573194, 0.29856091293992626], "irradiance": [4.691228850591552, 2.797190459903282, 2.229772237868535]}, {"minute": 991.875, "direction": [0.9553001762810804, -0.02689380031707078, 0.2944117808475778], "irradiance": [4.665357509696392, 2.7560186633196233, 2.1899548764780254]}, {"minute": 993.75, "direction": [0.9565104182687321, -0.028971617472620767, 0.2902558614815476], "irradiance": [4.639486168801234, 2.7148468667359644, 2.1501375150875157]}, {"minute": 994.6875, "direction": [0.9571072718311646, -0.030010278580288596, 0.28817538650523017], "irradiance": [4.626550498353654, 2.694260968444135, 2.1302288343922613]}, {"minute": 995.625, "direction": [0.9576986091755145, -0.031048766725900637, 0.2860932506510522], "irradiance": [4.6136148279060745, 2.6736750701523055, 2.110320153697006]}, {"minute": 997.5, "direction": [0.95886472160926, -0.03312520019098525, 0.2819240443195715], "irradiance": [4.587743487010917, 2.632503273568647, 2.070502792306497]}, {"minute": 999.375, "direction": [0.9600087286867907, -0.03520086999845126, 0.2777483386026355], "irradiance": [4.561872146115758, 2.5913314769849882, 2.0306854309159874]}, {"minute": 1001.25, "direction": [0.9611306040345385, -0.03727572829648028, 0.2735662297656092], "irradiance": [4.536000805220599, 2.5501596804013293, 1.9908680695254775]}, {"minute": 1002.1875, "direction": [0.9616832342075002, -0.03831283818084808, 0.27147280430041304], "irradiance": [4.52306513477302, 2.5295737821095, 1.970959388830223]}, {"minute": 1003.125, "direction": [0.9622303217891536, -0.03934972725196227, 0.2693778142214729], "irradiance": [4.51012946432544, 2.5089878838176705, 1.9510507081349677]}, {"minute": 1005.0, "direction": [0.963307856598099, -0.04142281905159806, 0.26518318852859973], "irradiance": [4.484258123430282, 2.467816087234012, 1.9112333467444589]}, {"minute": 1005.9375, "direction": [0.963838297615098, -0.0424590098320061, 0.2630835770900865], "irradiance": [4.471322452982703, 2.447230188942182, 1.8913246660492036]}, {"minute": 1006.875, "direction": [0.9643631836202373, -0.04349495590300196, 0.2609824493885296], "irradiance": [4.458386782535124, 2.426644290650353, 1.8714159853539492]}, {"minute": 1008.75, "direction": [0.9653962785264015, -0.04556609003580323, 0.2567756936437398], "irradiance": [4.432515441639964, 2.3854724940666943, 1.8315986239634394]}, {"minute": 1009.6875, "direction": [0.965904481473259, -0.04660126616077777, 0.2546700898458376], "irradiance": [4.419579771192385, 2.3648865957748653, 1.811689943268185]}, {"minute": 1010.625, "direction": [0.9664071174999568, -0.047636173702747475, 0.25256301827541267], "irradiance": [4.406644100744806, 2.3443006974830354, 1.7917812625729297]}, {"minute": 1012.5, "direction": [0.9673956772373494, -0.049705159180797424, 0.24834452040119923], "irradiance": [4.380772759849647, 2.3031289008993765, 1.75196390118242]}, {"minute": 1014.375, "direction": [0.968361934948644, -0.05177299877223305, 0.2441202972729812], "irradiance": [4.354901418954489, 2.261957104315718, 1.7121465397919111]}, {"minute": 1016.25, "direction": [0.9693058683580486, -0.05383964480575136, 0.23989044627462774], "irradiance": [4.32903007805933, 2.220785307732059, 1.6723291784014014]}, {"minute": 1018.125, "direction": [0.9702274557044286, -0.05590504963756508, 0.23565506491975174], "irradiance": [4.303158737164171, 2.1796135111484003, 1.6325118170108917]}, {"minute": 1019.0625, "direction": [0.9706798629444802, -0.05693727172194342, 0.23353533086654532], "irradiance": [4.290223066716592, 2.1590276128565713, 1.6126031363156372]}, {"minute": 1020.0, "direction": [0.9711266757418086, -0.05796916565250134, 0.23141425084946093], "irradiance": [4.2772873962690126, 2.138441714564741, 1.592694455620382]}, {"minute": 1021.875, "direction": [0.9720035077398612, -0.06003194526509907, 0.22716810183010716], "irradiance": [4.251416055373855, 2.097269917981083, 1.5528770942298729]}, {"minute": 1023.75, "direction": [0.9728579314843862, -0.06209334092070646, 0.22291671575103209], "irradiance": [4.225544714478696, 2.056098121397424, 1.5130597328393631]}, {"minute": 1025.625, "direction": [0.9736899272777757, -0.06415330509657668, 0.2186601906223112], "irradiance": [4.199673373583537, 2.014926324813765, 1.4732423714488534]}, {"minute": 1027.5, "direction": [0.9744994759394686, -0.06621179030296402, 0.21439862457249412], "irradiance": [4.1738020326883785, 1.9737545282301063, 1.4334250100583437]}, {"minute": 1029.375, "direction": [0.9752865588063929, -0.06826874908421827, 0.21013211584634184], "irradiance": [4.14793069179322, 1.9325827316464472, 1.393607648667834]}, {"minute": 1031.25, "direction": [0.9760511577333953, -0.07032413401987907, 0.2058607628025625], "irradiance": [4.122059350898061, 1.8914109350627892, 1.353790287277325]}, {"minute": 1033.125, "direction": [0.9767932550936602, -0.07237789772576891, 0.20158466391154323], "irradiance": [4.096188010002902, 1.8502391384791301, 1.3139729258868154]}, {"minute": 1034.0625, "direction": [0.9771558603197195, -0.07340415682018424, 0.19944486557529342], "irradiance": [4.083252339555322, 1.8296532401873002, 1.29406424519156]}, {"minute": 1035.0, "direction": [0.9775128337791159, -0.0744299928550856, 0.1973039177530808], "irradiance": [4.0703166691077435, 1.8090673418954712, 1.2741555644963056]}, {"minute": 1036.875, "direction": [0.9782098772008289, -0.07648037209949379, 0.19301862301410827], "irradiance": [4.044445328212585, 1.7678955453118121, 1.234338203105796]}, {"minute": 1038.75, "direction": [0.9788843692893859, -0.07852898819021552, 0.1887288784864204], "irradiance": [4.018573987317426, 1.7267237487281533, 1.1945208417152862]}, {"minute": 1040.625, "direction": [0.9795362944952657, -0.08057579389912006, 0.18443478306439587], "irradiance": [3.992702646422268, 1.6855519521444953, 1.1547034803247773]}, {"minute": 1041.5625, "direction": [0.9798537898002221, -0.0815985031129825, 0.18228613469998983], "irradiance": [3.9797669759746883, 1.6649660538526652, 1.134794799629522]}, {"minute": 1042.5, "direction": [0.9801656377891956, -0.08262074203981268, 0.18013643574271726], "irradiance": [3.9668313055271094, 1.6443801555608362, 1.1148861189342676]}, {"minute": 1044.375, "direction": [0.9807723846624998, -0.08466378546872234, 0.1758339356140894], "irradiance": [3.9409599646319506, 1.6032083589771773, 1.0750687575437579]}, {"minute": 1045.3125, "direction": [0.981067280049889, -0.0856845781958697, 0.17368115923988042], "irradiance": [3.9280242941843713, 1.5826224606853483, 1.0551600768485034]}, {"minute": 1046.25, "direction": [0.9813565211274327, -0.08670487708618867, 0.1715273818669543], "irradiance": [3.915088623736792, 1.5620365623935182, 1.0352513961532481]}, {"minute": 1047.1875, "direction": [0.9816401062281107, -0.08772467625926067, 0.16937261590845032], "irradiance": [3.9021529532892125, 1.5414506641016892, 1.0153427154579937]}, {"minute": 1048.125, "direction": [0.9819180337175014, -0.08874396983754758, 0.16721687378320513], "irradiance": [3.889217282841633, 1.5208647658098602, 0.9954340347627392]}, {"minute": 1050.0, "direction": [0.9824569094877771, -0.09078101671421646, 0.16290251073589654], "irradiance": [3.8633459419464744, 1.479692969226201, 0.9556166733722294]}, {"minute": 1050.9375, "direction": [0.982717854662891, -0.09179875827222707, 0.16074391467935054], "irradiance": [3.850410271498895, 1.4591070709343712, 0.9357079926769741]}, {"minute": 1051.875, "direction": [0.9829731360151927, -0.09281597075477739, 0.1585843921869547], "irradiance": [3.8374746010513157, 1.4385211726425422, 0.9157993119817197]}, {"minute": 1052.8125, "direction": [0.9832227520733867, -0.09383264829923743, 0.15642395570495968], "irradiance": [3.8245389306037367, 1.4179352743507132, 0.8958906312864653]}, {"minute": 1053.75, "direction": [0.9834667013988291, -0.09484878504606016, 0.15426261768488372], "irradiance": [3.811603260156157, 1.397349376058883, 0.87598195059121]}, {"minute": 1054.6875, "direction": [0.9837049825855357, -0.09586437513881538, 0.15210039058344138], "irradiance": [3.798667589708578, 1.376763477767054, 0.8560732698959556]}, {"minute": 1055.625, "direction": [0.9839375942601902, -0.09687941272422355, 0.1499372868624709], "irradiance": [3.785731919260998, 1.3561775794752242, 0.8361645892007002]}, {"minute": 1057.5, "direction": [0.9843858037434647, -0.0989078069758358, 0.1456084994344899], "irradiance": [3.75986057836584, 1.315005782891566, 0.7963472278101913]}, {"minute": 1059.375, "direction": [0.9848113195157766, -0.10093392103895399, 0.14127635519540147], "irradiance": [3.733989237470681, 1.2738339863079071, 0.7565298664196816]}, {"minute": 1061.25, "direction": [0.9852141317674232, -0.10295770820420178, 0.1369409540170536], "irradiance": [3.7081178965755224, 1.2326621897242482, 0.7167125050291718]}, {"minute": 1063.125, "direction": [0.9855942312121019, -0.10497912181584644, 0.1326023958463786], "irradiance": [3.6822465556803636, 1.1914903931405891, 0.6768951436386621]}, {"minute": 1065.0, "direction": [0.985951609087123, -0.10699811527287433, 0.12826078070308897], "irradiance": [3.6563752147852058, 1.1503185965569311, 0.6370777822481533]}, {"minute": 1066.875, "direction": [0.9862862571536131, -0.10901464203006529, 0.12391620867737164], "irradiance": [3.630503873890047, 1.109146799973272, 0.5972604208576435]}, {"minute": 1068.75, "direction": [0.9865981676967042, -0.1110286555990657, 0.11956877992758064], "irradiance": [3.6046325329948883, 1.0679750033896132, 0.5574430594671338]}, {"minute": 1070.625, "direction": [0.9868873335257118, -0.11304010954946014, 0.11521859467792797], "irradiance": [3.5787611920997295, 1.026803206805954, 0.5176256980766241]}, {"minute": 1072.5, "direction": [0.9871537479743008, -0.11504895750984177, 0.11086575321617315], "irradiance": [3.552889851204571, 0.9856314102222952, 0.47780833668611433]}, {"minute": 1074.375, "direction": [0.9873974049006388, -0.11705515316888145, 0.10651035589131114], "irradiance": [3.5270185103094125, 0.9444596136386371, 0.4379909752956054]}, {"minute": 1076.25, "direction": [0.9876182986875383, -0.11905865027639527, 0.10215250311125862], "irradiance": [3.5011471694142537, 0.9032878170549781, 0.39817361390509565]}, {"minute": 1077.1875, "direction": [0.9877202077918168, -0.1200593724371294, 0.09997268731855183], "irradiance": [3.488211498966674, 0.8827019187631482, 0.3782649332098404]}, {"minute": 1078.125, "direction": [0.9878164242425852, -0.12105940264441094, 0.09779229534053994], "irradiance": [3.475275828519095, 0.8621160204713192, 0.3583562525145859]}, {"minute": 1080.0, "direction": [0.9879917769982576, -0.12305736414823243, 0.09342983309797052], "irradiance": [3.4494044876239363, 0.8209442238876601, 0.3185388911240762]}, {"minute": 1081.875, "direction": [0.9881443529120298, -0.12505248872750346, 0.08906521695433961], "irradiance": [3.4235331467287775, 0.7797724273040012, 0.27872152973356645]}, {"minute": 1083.75, "direction": [0.9882741484664662, -0.12704473038726913, 0.08469854753009186], "irradiance": [3.397661805833619, 0.738600630720343, 0.23890416834305755]}, {"minute": 1085.625, "direction": [0.9883811606693023, -0.12903404319903666, 0.08032992549300737], "irradiance": [3.3717904649384605, 0.6974288341366841, 0.19908680695254782]}, {"minute": 1086.5625, "direction": [0.9884261222224061, -0.13002758695270084, 0.078144913715705], "irradiance": [3.3588547944908815, 0.6768429358448551, 0.17917812625729337]}, {"minute": 1087.5, "direction": [0.9884653870535133, -0.13102038130183383, 0.07595945155588141], "irradiance": [3.3459191240433017, 0.6562570375530251, 0.1592694455620381]}, {"minute": 1089.375, "direction": [0.9885268256773716, -0.13300369890326644, 0.07158722647420225], "irradiance": [3.320047783148143, 0.6150852409693661, 0.11945208417152837]}, {"minute": 1091.25, "direction": [0.988565475124491, -0.1349839502805738, 0.06721335104382851], "irradiance": [3.2941764422529847, 0.573913444385708, 0.07963472278101946]}, {"minute": 1093.125, "direction": [0.9885813345038599, -0.1369610897816834, 0.06283792609866527], "irradiance": [3.268305101357826, 0.532741647802049, 0.03981736139050973]}, {"minute": 1095.0, "direction": [0.9885744034498609, -0.13893507182626233, 0.05846105250833977], "irradiance": [3.242433760462667, 0.4915698512183901, 0.0]}]};
    const fixtures=[{"id": 0, "label": "客厅筒灯 01", "group": 0, "center": [1.409951535033376, -1.4074108125483609, 1.3032983178957573], "radius": 0.055277418941546475, "radiance": [150.19803414420554, 134.28313044834798, 126.33018322321246], "views": 33, "ray_error": 0.00578769465122191, "evidence": "multiview photographic triangulation", "source_id": 1, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 1, "label": "客厅筒灯 02", "group": 0, "center": [3.1802854936120206, -1.4808946221876087, 1.2804342265711064], "radius": 0.051750055564715264, "radiance": [171.37131151028996, 153.21289862253852, 144.13883181394726], "views": 30, "ray_error": 0.008439845763900795, "evidence": "multiview photographic triangulation", "source_id": 2, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 2, "label": "客厅筒灯 03", "group": 0, "center": [0.19828964548925535, -1.3713109527617642, 1.3059933839784708], "radius": 0.0594327472576767, "radiance": [129.9296144168863, 116.16234167948592, 109.2826020588754], "views": 27, "ray_error": 0.009135932658765675, "evidence": "multiview photographic triangulation", "source_id": 3, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 3, "label": "客厅筒灯 04", "group": 0, "center": [5.116747834982646, 0.8089859780523732, 1.2767479749121198], "radius": 0.06187155959437592, "radiance": [119.8885298073127, 107.18520504682056, 100.83713827025265], "views": 24, "ray_error": 0.006435724270869463, "evidence": "multiview photographic triangulation", "source_id": 4, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 4, "label": "客厅筒灯 05", "group": 0, "center": [5.025945700266967, -1.54465940317233, 1.258091965758752], "radius": 0.05839435435687772, "radiance": [134.59162583387217, 120.3303688499198, 113.20377692545907], "views": 22, "ray_error": 0.004788790875434096, "evidence": "multiview photographic triangulation", "source_id": 5, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 5, "label": "客厅筒灯 06", "group": 0, "center": [-0.6700382233899392, 0.3500743434031984, 1.3253542990810576], "radius": 0.06608770420577696, "radiance": [105.0796047593048, 93.9454258090164, 88.38148779977122], "views": 21, "ray_error": 0.004867051549571459, "evidence": "multiview photographic triangulation", "source_id": 6, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 6, "label": "客厅筒灯 07", "group": 0, "center": [-0.6423835693148122, 1.171665650502846, 1.330158178678859], "radius": 0.05793465886448622, "radiance": [136.7359943418317, 122.24752121296066, 115.00738552827555], "views": 20, "ray_error": 0.005295744362564711, "evidence": "multiview photographic triangulation", "source_id": 7, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 7, "label": "客厅筒灯 08", "group": 0, "center": [-0.7204598262933004, -1.346762315352812, 1.3150848422534769], "radius": 0.06077850751753389, "radiance": [124.2395010794422, 111.0751497204734, 104.49670013561851], "views": 20, "ray_error": 0.005990388564940898, "evidence": "multiview photographic triangulation", "source_id": 8, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 8, "label": "客厅筒灯 09", "group": 0, "center": [3.2555262999996972, 0.8813396290781141, 1.287480987244906], "radius": 0.06344570979543457, "radiance": [114.01322765663339, 101.93244678258354, 95.89547574173756], "views": 20, "ray_error": 0.005825305848293155, "evidence": "multiview photographic triangulation", "source_id": 9, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 9, "label": "客厅筒灯 10", "group": 0, "center": [0.8754203998064711, 1.1157130843899634, 1.297637754680928], "radius": 0.06524528194683211, "radiance": [107.81061833328941, 96.38706263935924, 90.67851816473134], "views": 19, "ray_error": 0.006170074686491397, "evidence": "multiview photographic triangulation", "source_id": 10, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 10, "label": "其它灯具 01", "group": 1, "center": [-0.18912313132311134, 3.8369310312118, 1.0041230824535883], "radius": 0.0967421416498986, "radiance": [164.9670129024613, 141.12786580964053, 122.3646864791275], "views": 19, "ray_error": 0.014899426091453586, "evidence": "multiview photographic triangulation", "source_id": 11, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 11, "label": "其它灯具 02", "group": 1, "center": [1.8486645860346496, 3.29414865792757, 1.3310862075317815], "radius": 0.06434115507745625, "radiance": [110.86183294549286, 99.11497217646438, 93.24486667394633], "views": 17, "ray_error": 0.005077781917068214, "evidence": "multiview photographic triangulation", "source_id": 12, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 12, "label": "其它灯具 03", "group": 1, "center": [3.490060843344431, 5.655466800275069, 1.3093107972714728], "radius": 0.05194505906284193, "radiance": [170.08705927659267, 152.06472507147473, 143.05865908790457], "views": 14, "ray_error": 0.016610794092907184, "evidence": "multiview photographic triangulation", "source_id": 13, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 13, "label": "其它灯具 04", "group": 1, "center": [5.317244564600869, 7.520314353809452, 1.2837696660613092], "radius": 0.063152342714941, "radiance": [115.07495918126386, 102.88167780038896, 96.78848834875777], "views": 13, "ray_error": 0.009738554974557214, "evidence": "multiview photographic triangulation", "source_id": 14, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 14, "label": "其它灯具 05", "group": 1, "center": [5.269873782533136, 5.588709583314067, 1.2778135636028851], "radius": 0.05829777234316025, "radiance": [135.03795157050644, 120.72940214923295, 113.57917739196655], "views": 13, "ray_error": 0.008185213761166698, "evidence": "multiview photographic triangulation", "source_id": 15, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 15, "label": "其它灯具 06", "group": 1, "center": [1.9255672311768255, 5.441109176177112, 1.3296704037904643], "radius": 0.06454986658006676, "radiance": [110.14608450934692, 98.47506406337345, 92.64285725621167], "views": 12, "ray_error": 0.010284716499965244, "evidence": "multiview photographic triangulation", "source_id": 16, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 16, "label": "其它灯具 07", "group": 1, "center": [1.9983953226666533, 7.55352337883041, 1.3290771688776284], "radius": 0.06292292008262773, "radiance": [115.91563631007642, 103.63327722841406, 97.4955741393247], "views": 11, "ray_error": 0.011461968005565875, "evidence": "multiview photographic triangulation", "source_id": 17, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 17, "label": "其它灯具 08", "group": 1, "center": [5.3025777612274645, 9.346804640145384, 1.2808616338076066], "radius": 0.05868333287231156, "radiance": [133.2693313471769, 119.14818398272268, 112.09160721078173], "views": 9, "ray_error": 0.020100359476097392, "evidence": "multiview photographic triangulation", "source_id": 18, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 18, "label": "其它灯具 09", "group": 1, "center": [2.8920180884973488, 9.884774642928177, 1.2644299475814922], "radius": 0.06809413552084614, "radiance": [98.97837965469735, 88.49068326671298, 83.24980355541483], "views": 9, "ray_error": 0.011244629618000802, "evidence": "multiview photographic triangulation", "source_id": 19, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 19, "label": "其它灯具 10", "group": 1, "center": [4.4002058428952795, 6.581691120193081, 0.9366775384076527], "radius": 0.07879639421079944, "radiance": [248.66566134462155, 212.73134227422062, 184.44836427116715], "views": 8, "ray_error": 0.013695911627937533, "evidence": "multiview photographic triangulation", "source_id": 20, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 20, "label": "其它灯具 11", "group": 1, "center": [3.552190419962272, 7.588066515630468, 1.3076946240769873], "radius": 0.06632665915528714, "radiance": [104.32382838321536, 93.26973109517773, 87.74581125039984], "views": 8, "ray_error": 0.012528818580207272, "evidence": "multiview photographic triangulation", "source_id": 21, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 21, "label": "其它灯具 12", "group": 1, "center": [-0.07308643498435788, 6.530220787189412, 1.347632531694947], "radius": 0.06831463009874535, "radiance": [98.34047893679252, 87.92037416910344, 82.71327113653001], "views": 7, "ray_error": 0.011933880993800775, "evidence": "multiview photographic triangulation", "source_id": 22, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 22, "label": "其它灯具 14", "group": 1, "center": [-0.020476953557617757, 9.276795877716479, 1.0121848534500448], "radius": 0.08328536891368725, "radiance": [222.58253172127638, 190.41744840772148, 165.10113889198072], "views": 5, "ray_error": 0.003829078842841065, "evidence": "multiview photographic triangulation", "source_id": 24, "type": "disk", "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 23, "label": "厨房吊灯 01", "group": 2, "center": [0.061, -0.32, 0.669], "radius": 0.04, "radii": [0.065, 0.065, 0.165], "radiance": [45.0, 40.5, 36.0], "views": null, "ray_error": null, "evidence": "reviewed native GS projection; pendant_candidates.json components 28 and 19", "source_id": 25, "type": "cylinder", "height": 0.26, "power_calibration": "estimated relative radiance; not measured lumens"}, {"id": 24, "label": "厨房吊灯 02", "group": 2, "center": [-0.527, -0.297, 0.669], "radius": 0.04, "radii": [0.065, 0.065, 0.165], "radiance": [45.0, 40.5, 36.0], "views": null, "ray_error": null, "evidence": "reviewed native GS projection; pendant_candidates.json components 28 and 19", "source_id": 26, "type": "cylinder", "height": 0.26, "power_calibration": "estimated relative radiance; not measured lumens"}];
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

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const resize = () => {
        const focalScale = innerWidth / camera.width;
        gl.uniform2fv(u_focal, new Float32Array([camera.fx * focalScale, camera.fy * focalScale]));

        projectionMatrix = getProjectionMatrix(
            camera.fx * focalScale,
            camera.fy * focalScale,
            innerWidth,
            innerHeight,
        );

        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));

        gl.canvas.width = Math.round(innerWidth / downsample);
        gl.canvas.height = Math.round(innerHeight / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();

    worker.onmessage = (e) => {
        if (e.data.buffer) {
            splatData = new Uint8Array(e.data.buffer);
            if (e.data.save) {
                const blob = new Blob([splatData.buffer], {
                    type: "application/octet-stream",
                });
                const link = document.createElement("a");
                link.download = "model.splat";
                link.href = URL.createObjectURL(blob);
                document.body.appendChild(link);
                link.click();
            }
        } else if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            // console.log(texdata)
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_S,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_T,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA32UI,
                texwidth,
                texheight,
                0,
                gl.RGBA_INTEGER,
                gl.UNSIGNED_INT,
                texdata,
            );
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            const { depthIndex, viewProj } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount; window.incrementCount=vertexCount;window.incrementReady=vertexCount===6100978;
        }
    };

    let activeKeys = [];
    let currentCameraIndex = 0;

    window.addEventListener("keydown", (e) => {
        // if (document.activeElement != document.body) return;
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (/\d/.test(e.key)) {
            currentCameraIndex = parseInt(e.key);
            camera = cameras[currentCameraIndex];
            viewMatrix = getViewMatrix(camera);
        }
        if (["-", "_"].includes(e.key)) {
            currentCameraIndex =
                (currentCameraIndex + cameras.length - 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        if (["+", "="].includes(e.key)) {
            currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        camid.innerText = "cam  " + currentCameraIndex;
        if (e.code == "KeyV") {
            location.hash =
                "#" +
                JSON.stringify(
                    viewMatrix.map((k) => Math.round(k * 100) / 100),
                );
            camid.innerText = "";
        } else if (e.code === "KeyP") {
            carousel = true;
            camid.innerText = "";
        }
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener(
        "wheel",
        (e) => {
            carousel = false;
            e.preventDefault();
            const lineHeight = 10;
            const scale =
                e.deltaMode == 1
                    ? lineHeight
                    : e.deltaMode == 2
                      ? innerHeight
                      : 1;
            let inv = invert4(viewMatrix);
            if (e.shiftKey) {
                inv = translate4(
                    inv,
                    (e.deltaX * scale) / innerWidth,
                    (e.deltaY * scale) / innerHeight,
                    0,
                );
            } else if (e.ctrlKey || e.metaKey) {
                // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
                // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
                // let preY = inv[13];
                inv = translate4(
                    inv,
                    0,
                    0,
                    (-10 * (e.deltaY * scale)) / innerHeight,
                );
                // inv[13] = preY;
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }

            viewMatrix = invert4(inv);
        },
        { passive: false },
    );

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / innerWidth;
            let dy = (5 * (e.clientY - startY)) / innerHeight;
            let d = 4;

            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            // let postAngle = Math.atan2(inv[0], inv[10])
            // inv = rotate4(inv, postAngle - preAngle, 0, 0, 1)
            // console.log(postAngle)
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            // inv = rotateY(inv, );
            // let preY = inv[13];
            inv = translate4(
                inv,
                (-10 * (e.clientX - startX)) / innerWidth,
                0,
                (10 * (e.clientY - startY)) / innerHeight,
            );
            // inv[13] = preY;
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    let altX = 0,
        altY = 0;
    canvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                carousel = false;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                // console.log('beep')
                carousel = false;
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                // inv = translate4(inv,  -x, -y, -z);
                // inv = translate4(inv,  x, y, z);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                // alert('beep')
                const dtheta =
                    Math.atan2(startY - altY, startX - altX) -
                    Math.atan2(
                        e.touches[0].clientY - e.touches[1].clientY,
                        e.touches[0].clientX - e.touches[1].clientX,
                    );
                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                    );
                const dx =
                    (e.touches[0].clientX +
                        e.touches[1].clientX -
                        (startX + altX)) /
                    2;
                const dy =
                    (e.touches[0].clientY +
                        e.touches[1].clientY -
                        (startY + altY)) /
                    2;
                let inv = invert4(viewMatrix);
                // inv = translate4(inv,  0, 0, d);
                inv = rotate4(inv, dtheta, 0, 0, 1);

                inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);

                // let preY = inv[13];
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));
                // inv[13] = preY;

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );

    let jumpDelta = 0;
    let vertexCount = 0;

    let lastFrame = 0;
    let avgFps = 0;
    let start = 0;

    window.addEventListener("gamepadconnected", (e) => {
        const gp = navigator.getGamepads()[e.gamepad.index];
        console.log(
            `Gamepad connected at index ${gp.index}: ${gp.id}. It has ${gp.buttons.length} buttons and ${gp.axes.length} axes.`,
        );
    });
    window.addEventListener("gamepaddisconnected", (e) => {
        console.log("Gamepad disconnected");
    });

    let leftGamepadTrigger, rightGamepadTrigger;

    const frame = (now) => {
        transportClock(now);
        let inv = invert4(viewMatrix);
        let shiftKey =
            activeKeys.includes("Shift") ||
            activeKeys.includes("ShiftLeft") ||
            activeKeys.includes("ShiftRight");

        if (activeKeys.includes("ArrowUp")) {
            if (shiftKey) {
                inv = translate4(inv, 0, -0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, 0.1);
            }
        }
        if (activeKeys.includes("ArrowDown")) {
            if (shiftKey) {
                inv = translate4(inv, 0, 0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, -0.1);
            }
        }
        if (activeKeys.includes("ArrowLeft"))
            inv = translate4(inv, -0.03, 0, 0);
        //
        if (activeKeys.includes("ArrowRight"))
            inv = translate4(inv, 0.03, 0, 0);
        // inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
        if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
        if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
        if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
        if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

        const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
        let isJumping = activeKeys.includes("Space");
        for (let gamepad of gamepads) {
            if (!gamepad) continue;

            const axisThreshold = 0.1; // Threshold to detect when the axis is intentionally moved
            const moveSpeed = 0.06;
            const rotateSpeed = 0.02;

            // Assuming the left stick controls translation (axes 0 and 1)
            if (Math.abs(gamepad.axes[0]) > axisThreshold) {
                inv = translate4(inv, moveSpeed * gamepad.axes[0], 0, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[1]) > axisThreshold) {
                inv = translate4(inv, 0, 0, -moveSpeed * gamepad.axes[1]);
                carousel = false;
            }
            if (gamepad.buttons[12].pressed || gamepad.buttons[13].pressed) {
                inv = translate4(
                    inv,
                    0,
                    -moveSpeed *
                        (gamepad.buttons[12].pressed -
                            gamepad.buttons[13].pressed),
                    0,
                );
                carousel = false;
            }

            if (gamepad.buttons[14].pressed || gamepad.buttons[15].pressed) {
                inv = translate4(
                    inv,
                    -moveSpeed *
                        (gamepad.buttons[14].pressed -
                            gamepad.buttons[15].pressed),
                    0,
                    0,
                );
                carousel = false;
            }

            // Assuming the right stick controls rotation (axes 2 and 3)
            if (Math.abs(gamepad.axes[2]) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * gamepad.axes[2], 0, 1, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[3]) > axisThreshold) {
                inv = rotate4(inv, -rotateSpeed * gamepad.axes[3], 1, 0, 0);
                carousel = false;
            }

            let tiltAxis = gamepad.buttons[6].value - gamepad.buttons[7].value;
            if (Math.abs(tiltAxis) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * tiltAxis, 0, 0, 1);
                carousel = false;
            }
            if (gamepad.buttons[4].pressed && !leftGamepadTrigger) {
                camera =
                    cameras[(cameras.indexOf(camera) + 1) % cameras.length];
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            if (gamepad.buttons[5].pressed && !rightGamepadTrigger) {
                camera =
                    cameras[
                        (cameras.indexOf(camera) + cameras.length - 1) %
                            cameras.length
                    ];
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            leftGamepadTrigger = gamepad.buttons[4].pressed;
            rightGamepadTrigger = gamepad.buttons[5].pressed;
            if (gamepad.buttons[0].pressed) {
                isJumping = true;
                carousel = false;
            }
            if (gamepad.buttons[3].pressed) {
                carousel = true;
            }
        }

        if (
            ["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))
        ) {
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(
                inv,
                activeKeys.includes("KeyJ")
                    ? -0.05
                    : activeKeys.includes("KeyL")
                      ? 0.05
                      : 0,
                0,
                1,
                0,
            );
            inv = rotate4(
                inv,
                activeKeys.includes("KeyI")
                    ? 0.05
                    : activeKeys.includes("KeyK")
                      ? -0.05
                      : 0,
                1,
                0,
                0,
            );
            inv = translate4(inv, 0, 0, -d);
        }

        viewMatrix = invert4(inv);

        if (carousel) {
            let inv = invert4(defaultViewMatrix);

            const t = Math.sin((Date.now() - start) / 5000);
            inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
            inv = rotate4(inv, -0.6 * t, 0, 1, 0);

            viewMatrix = invert4(inv);
        }

        if (isJumping) {
            jumpDelta = Math.min(1, jumpDelta + 0.05);
        } else {
            jumpDelta = Math.max(0, jumpDelta - 0.05);
        }

        let inv2 = invert4(viewMatrix);
        inv2 = translate4(inv2, 0, -jumpDelta, 0);
        inv2 = rotate4(inv2, -0.1 * jumpDelta, 1, 0, 0);
        let actualViewMatrix = invert4(inv2);

        const viewProj = multiply4(projectionMatrix, actualViewMatrix);
        worker.postMessage({ view: viewProj });

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
            beginTransportFrame();
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            finishTransportFrame();
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }
        const progress = (100 * vertexCount) / (splatData.length / rowLength);
        if (progress < 100) {
            document.getElementById("progress").style.width = progress + "%";
        } else {
            document.getElementById("progress").style.display = "none";
        }
        fps.innerText = Math.round(avgFps) + " fps";
        if (isNaN(currentCameraIndex)) {
            camid.innerText = "";
        }
        lastFrame = now;
        if (!window.freezeRendering) requestAnimationFrame(frame);
        window.redrawScene = () => { window.freezeRendering = false; requestAnimationFrame(frame); };
    };

    frame();

    const isPly = (splatData) =>
        splatData[0] == 112 &&
        splatData[1] == 108 &&
        splatData[2] == 121 &&
        splatData[3] == 10;

    const selectFile = (file) => {
        const fr = new FileReader();
        if (/\.json$/i.test(file.name)) {
            fr.onload = () => {
                cameras = JSON.parse(fr.result);
                viewMatrix = getViewMatrix(cameras[0]);
                projectionMatrix = getProjectionMatrix(
                    camera.fx / downsample,
                    camera.fy / downsample,
                    canvas.width,
                    canvas.height,
                );
                gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

                console.log("Loaded Cameras");
            };
            fr.readAsText(file);
        } else {
            stopLoading = true;
            fr.onload = () => {
                splatData = new Uint8Array(fr.result);
                console.log("Loaded", Math.floor(splatData.length / rowLength));

                if (isPly(splatData)) {
                    // ply file magic header means it should be handled differently
                    worker.postMessage({ ply: splatData.buffer, save: true });
                } else {
                    worker.postMessage({
                        buffer: splatData.buffer,
                        vertexCount: Math.floor(splatData.length / rowLength),
                    });
                }
            };
            fr.readAsArrayBuffer(file);
        }
    };

    window.addEventListener("hashchange", (e) => {
        try {
            viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
            carousel = false;
        } catch (err) {}
    });

    const preventDefault = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };
    document.addEventListener("dragenter", preventDefault);
    document.addEventListener("dragover", preventDefault);
    document.addEventListener("dragleave", preventDefault);
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectFile(e.dataTransfer.files[0]);
    });

    let bytesRead = 0;
    let lastVertexCount = -1;
    let stopLoading = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done || stopLoading) break;

        splatData.set(value, bytesRead);
        bytesRead += value.length;

        if (vertexCount > lastVertexCount) {
            if (!isPly(splatData)) {
                worker.postMessage({
                    buffer: splatData.buffer,
                    vertexCount: Math.floor(bytesRead / rowLength),
                });
            }
            lastVertexCount = vertexCount;
        }
    }
    if (!stopLoading) {
        if (isPly(splatData)) {
            // ply file magic header means it should be handled differently
            worker.postMessage({ ply: splatData.buffer, save: false });
        } else {
            worker.postMessage({
                buffer: splatData.buffer,
                vertexCount: Math.floor(bytesRead / rowLength),
            });
        }
    }
}

main().catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
});
