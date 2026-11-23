#version 440

layout(location = 0) in vec4 v_colour;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    vec4 viewport;
} ubuf;

void main()
{
    fragColor = v_colour;
}
