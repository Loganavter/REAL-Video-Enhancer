#version 440

// Full-resolution frame, sampled straight: 1:1 at zoom 1 means one texel per pixel, so the sampler is nearest there and
// linear above it (the widget switches samplers, not this shader).
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform sampler2D frameTex;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 tint;          // rgb = multiply tint, a = extra opacity (used to dim a stale frame during a seek)
} ubuf;

void main()
{
    vec4 texel = texture(frameTex, v_uv);
    fragColor = vec4(texel.rgb * ubuf.tint.rgb, texel.a * ubuf.tint.a);
}
