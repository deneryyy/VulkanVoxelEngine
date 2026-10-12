#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

struct Vertex {
	vec3 position;
	float uv1;
	vec3 normal;
	float uv2;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer { 
	Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants
{	
	VertexBuffer vertexBuffer;
	uint materialId;
} PushConstants;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out uint materialId;

void main() {

    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	//output data
	gl_Position =  ubo.proj * ubo.view  * vec4(v.position, 1.0F);
	fragTexCoord = vec2(v.uv1, v.uv2);
	materialId = PushConstants.materialId;
}