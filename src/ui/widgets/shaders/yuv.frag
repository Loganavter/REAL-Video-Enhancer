#version 440

// YUV420 planes -> RGB, in one pass, with the range and the matrix taken from the FILE's own tags.
//
// Why the shader and not swscale: a 4K 10-bit frame is 12.4 MB of planes against 33.2 MB of RGBA8888, and the conversion
// is a matrix multiply the GPU does for free while it samples. The CPU path this replaces paid for `to_ndarray`,
// `tobytes`, a QImage and `convertToFormat` per frame - four copies and a scaler pass.
//
// The arithmetic is deliberately split the way ITU-T H.273 splits it:
//
//   1. NORMALISE. An R16 texture samples `code / 65535` and an R8 texture `code / 255`, but the plane's codes run
//      0..1023 (10-bit) or 0..255 (8-bit). `ubuf.luma.x` folds that scale AND the range span into one multiplier, so the
//      one trap of a 10-bit upload (dividing by 65535 instead of 1023) cannot be repeated here.
//   2. EXPAND THE RANGE. `s * multiplier + offset` maps limited ("tv") luma 64..940 (10-bit) or 16..235 (8-bit) to
//      0..1, and chroma 64..960 / 16..240 to -0.5..0.5. For full ("pc") range the offset is the chroma centre only.
//   3. APPLY THE MATRIX. `ubuf.row0..2` are the FULL-RANGE coefficients of the file's matrix (bt709, smpte170m/bt470bg,
//      bt2020nc), supplied per frame from `PreviewFrame.color_space`. Nothing here assumes BT.709: a bt601 or a
//      bt2020-tagged file gets its own coefficients, and an untagged file of this app's writer is bt709/tv.
//
// Chroma is sampled at the same uv as luma, with a LINEAR sampler even in the nearest view: 4:2:0 chroma is half the
// picture's size, so every draw upsamples it 2x and a nearest tap there would replicate 2x2 colour blocks. Luma keeps
// the view's rule (nearest is pixel-exact at 1:1 and above), the RGBA path's rule.
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform sampler2D yTex;
layout(binding = 2) uniform sampler2D uTex;
layout(binding = 3) uniform sampler2D vTex;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;      // unused here; kept byte-identical with yuv.vert
    vec4 tint;     // rgb = multiply tint, a = extra opacity (dims a stale frame during a seek)
    vec4 luma;     // (multiplier, offset)
    vec4 chroma;   // (multiplier, offset)
    vec4 row0;
    vec4 row1;
    vec4 row2;
} ubuf;

void main()
{
    float ycode = texture(yTex, v_uv).r;
    float ucode = texture(uTex, v_uv).r;
    float vcode = texture(vTex, v_uv).r;

    float y = ycode * ubuf.luma.x + ubuf.luma.y;
    float cb = ucode * ubuf.chroma.x + ubuf.chroma.y;
    float cr = vcode * ubuf.chroma.x + ubuf.chroma.y;

    vec3 yuv = vec3(y, cb, cr);
    vec3 rgb = vec3(dot(ubuf.row0.xyz, yuv),
                    dot(ubuf.row1.xyz, yuv),
                    dot(ubuf.row2.xyz, yuv));
    // Rounding and a slightly out-of-gamut matrix put a few values outside 0..1; clamp rather than let the
    // framebuffer's conversion wrap a channel.
    fragColor = vec4(clamp(rgb, 0.0, 1.0) * ubuf.tint.rgb, ubuf.tint.a);
}
