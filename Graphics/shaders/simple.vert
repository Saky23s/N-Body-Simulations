#version 410 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec3 norm;

uniform mat4 transformation;
uniform mat3 normal_transform;

out vec4 vert_col;
out vec3 normal;

void main()
{
    gl_Position = vec4(transformation * vec4(position.xyz, 1.0));
    vert_col = vec4(color.rgba);
    normal = vec3(normalize(normal_transform * norm));
    //normal = vec3(norm.xyz);
}