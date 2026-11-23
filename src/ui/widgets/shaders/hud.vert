#version 440

// HUD geometry: one vertex buffer of coloured triangles in widget pixels. No instancing and no per-item uniform - the
// strip is a handful of rectangles, and one buffer upload per change is cheaper than a draw call per rectangle.
layout(location = 0) in vec2 pos;      // device pixels, origin top-left
layout(location = 1) in vec4 colour;   // straight alpha

layout(location = 0) out vec4 v_colour;

layout(std140, binding = 0) uniform buf {
    vec4 viewport;   // width, height, devicePixelRatio, 0
} ubuf;

void main()
{
    vec2 clip = pos / ubuf.viewport.xy * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
    v_colour = colour;
}
