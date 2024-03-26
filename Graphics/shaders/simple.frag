#version 410 core

out vec4 color;
in vec4 gl_FragCoord;
in vec4 vert_col;
in vec3 normal;

void main()
{
    //vec3 lightDirection = normalize(vec3(-0.5, 0.5, -0.5));
    //float luminated = max(0.0, dot(normal, - lightDirection));
    //color = vec4((vert_col * luminated).rgb, 1.0);

    color = vec4(vert_col);
}