#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in flat uint fragTexIndex;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler[2];

void main() {
    outColor = texture(texSampler[fragTexIndex], fragTexCoord, 1.0);
}
