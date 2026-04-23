#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout(set = 2, binding = 0) uniform sampler2D texSampler[];

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) flat in uint materialId;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler[nonuniformEXT(materialId)], fragTexCoord);
}