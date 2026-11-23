#version 440

// Flat rounded rectangles with one anti-aliased edge: the radius and colour come per instance through a second buffer.
layout(location = 0) in vec2 v_local;
layout(location = 1) flat in vec4 v_rect;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    vec4 viewport;
} ubuf;

layout(std140, binding = 2) uniform item {
    vec4 color;     // straight alpha
    vec4 radius;    // x = corner radius in px
} ibuf;

float rounded_box(vec2 p, vec2 half_size, float r)
{
    vec2 q = abs(p) - half_size + vec2(r);
    return length(max(q, vec2(0.0))) - r;
}

void main()
{
    vec2 px = v_local * v_rect.zw;
    vec2 half_size = v_rect.zw * 0.5;
    float dist = rounded_box(px - half_size, half_size, min(ibuf.radius.x, min(half_size.x, half_size.y)));
    float aa = max(1.0, fwidth(dist));
    float alpha = 1.0 - smoothstep(0.0, aa, dist);
    fragColor = vec4(ibuf.color.rgb, ibuf.color.a * alpha);
}
