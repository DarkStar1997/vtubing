"""GLSL shaders: linear-blend skinning + morph targets, simple lit/textured."""

VERTEX_SHADER = """#version 430
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_uv;
layout(location=3) in vec4 a_joints;
layout(location=4) in vec4 a_weights;

layout(std140, binding=0) uniform Joints { mat4 m[256]; } u_joints;
layout(std430, binding=1) readonly buffer MorphData { vec4 d[]; } u_morph;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
uniform int u_morph_count;
uniform int u_vert_count;

out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec3 pos = a_pos;
    for (int t = 0; t < u_morph_count; t++) {
        vec4 entry = u_morph.d[t * u_vert_count + gl_VertexID];
        float w = entry.w;
        if (w > 0.0) {
            pos += entry.xyz * w;
        }
    }
    mat4 skin = a_weights.x * u_joints.m[int(a_joints.x)]
              + a_weights.y * u_joints.m[int(a_joints.y)]
              + a_weights.z * u_joints.m[int(a_joints.z)]
              + a_weights.w * u_joints.m[int(a_joints.w)];
    vec4 world = u_model * skin * vec4(pos, 1.0);
    v_normal = normalize(mat3(u_model) * mat3(skin) * a_normal);
    v_uv = a_uv;
    gl_Position = u_proj * u_view * world;
}
"""

FRAGMENT_SHADER = """#version 430
in vec3 v_normal;
in vec2 v_uv;
out vec4 frag;

uniform sampler2D u_tex;
uniform vec3 u_base_color;
uniform int u_has_tex;

void main() {
    vec3 n = normalize(v_normal);
    vec3 L = normalize(vec3(0.3, 0.8, 0.5));
    float diff = max(dot(n, L), 0.0);
    vec3 albedo = u_base_color;
    if (u_has_tex == 1) albedo *= texture(u_tex, v_uv).rgb;
    vec3 col = albedo * (0.35 + 0.65 * diff);
    frag = vec4(col, 1.0);
}
"""
