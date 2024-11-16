#version 410 core

layout (location = 0) in vec3 position;

layout(std140) uniform InstanceMatrices {
    mat4 modelMatrices[1024]; 
};

out vec4 vert_col;

//Colors of the bodies
vec4 colors[3] = vec4[](
    vec4(1.0,0.8274509803921568,0.00392156862745098, 1.0), //Yellow
    vec4(0.36470588235294116, 0.6784313725490196, 0.9215686274509803, 1.0), // Blue
    vec4(0.7607843137254902,0.23137254901960785,0.12941176470588237, 1.0)  // Red
);

void main()
{
    gl_Position = modelMatrices[gl_InstanceID] * vec4(position, 1.0);
    
    int numberOfColors = 3;
    vert_col = colors[gl_InstanceID % numberOfColors];
}