#version 410 core

out vec4 color;
in vec4 vert_col;

void main()
{
    color = vec4(vert_col);
}