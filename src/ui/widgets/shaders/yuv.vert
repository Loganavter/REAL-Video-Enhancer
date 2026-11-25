#version 440

// The planes path's vertex stage: the same textured quad as `frame.vert`, and the same placement matrix (the CPU
// side builds fit/zoom/pan because that is where the widget's letterbox and clamping rules live).
//
// The uniform block is WIDER than `frame.vert`'s on purpose: the fragment stage needs the luma/chroma affine and the
// colour matrix, and QRhi links ONE block across both stages, so both declare it. The extra members are unused here.
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 v_uv;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;      // 64: unit quad -> the picture's rect
    vec4 tint;     // 16: rgb multiply, a = opacity
    vec4 luma;     // 16: (sample multiplier, offset) - range expansion, depth folded in
    vec4 chroma;   // 16: (sample multiplier, offset) - the same, about chroma's centre
    vec4 row0;     // 16: (Y, Cb, Cr) -> R
    vec4 row1;     // 16: -> G
    vec4 row2;     // 16: -> B
} ubuf;

void main()
{
    v_uv = uv;
    gl_Position = ubuf.mvp * vec4(pos, 0.0, 1.0);
}
