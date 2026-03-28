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
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

layout(location = 0) out vec3 fragColor;

void main() {

    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	//output data
	gl_Position = /*PushConstants.render_matrix * */vec4(v.position, 1.0F);
    fragColor = v.color;
}