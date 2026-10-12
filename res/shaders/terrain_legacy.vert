#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in float inUV1;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in float inUV2;

layout( push_constant ) uniform constants
{	
	uint materialId;
} PushConstants;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out uint materialId;

void main() {

	//output data
	gl_Position =  ubo.proj * ubo.view  * vec4(inPos, 1.0F);
	fragTexCoord = vec2(inUV1, inUV2);
	materialId = PushConstants.materialId;
}