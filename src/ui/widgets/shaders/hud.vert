#version 440

// HUD geometry (strip, playhead, handle, badge): flat coloured rectangles in pixel space. The CPU side converts pixels to
// clip space with the widget size as one uniform, so a resize never rebuilds a vertex buffer.
layout(location = 0) in vec4 rect_px;     // x, y, w, h in device pixels

layout(location = 0) out vec2 v_local;    // 0..1 inside the rect
layout(location = 1) flat out vec4 v_rect;

layout(std140, binding = 0) uniform buf {
    vec4 viewport;   // width, height, devicePixelRatio, 0
} ubuf;

void main()
{
    vec2 corner = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));
    vec2 px = rect_px.xy + corner * rect_px.zw;
    vec2 clip = px / ubuf.viewport.xy * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
    v_local = corner;
    v_rect = rect_px;
}
