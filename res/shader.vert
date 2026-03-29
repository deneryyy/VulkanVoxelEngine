#version 450
#extension GL_EXT_buffer_reference : require

struct Vertex {
	vec3 position;
	vec3 normal;
	vec3 color;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer { 
	Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants
{	
	VertexBuffer vertexBuffer;
} PushConstants;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec3 fragColor;

void main() {

    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	//output data
	gl_Position =  ubo.proj * ubo.view * ubo.model * vec4(v.position, 1.0F);
    fragColor = v.color;
}