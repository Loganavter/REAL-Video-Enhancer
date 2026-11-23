#version 440

// One textured quad, already in clip space: the CPU side builds the transform (fit, zoom, pan) because that is where
// the widget's letterbox and clamping rules live. `pos` is a unit quad corner in [-1, 1].
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 v_uv;

// The block must match frame.frag byte for byte: QRhi links one uniform block across both stages, and a size mismatch
// fails the program at link time ("uniform `ubuf' declared as type `buf' and type `buf'"). `tint` is unused here on
// purpose - the vertex stage keeps the block identical.
layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 tint;
} ubuf;

void main()
{
    v_uv = uv;
    gl_Position = ubuf.mvp * vec4(pos, 0.0, 1.0);
}
