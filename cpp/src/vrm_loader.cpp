#include "vrm_loader.h"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>

// --- Minimal JSON helpers for VRM extension parsing ---

// Find a float value following a JSON key, starting from `pos`.
static bool jsonFindFloat(const std::string& j, const std::string& key, size_t pos, float& out) {
    size_t kp = j.find(key, pos);
    if (kp == std::string::npos) return false;
    size_t colon = j.find(':', kp + key.size());
    if (colon == std::string::npos) return false;
    size_t ns = j.find_first_of("-.0123456789", colon + 1);
    if (ns == std::string::npos) return false;
    size_t ne = j.find_first_not_of("-.0123456789", ns + 1);
    if (ne == std::string::npos) ne = j.size();
    try { out = std::stof(j.substr(ns, ne - ns)); } catch (...) { return false; }
    return true;
}

// Find a 3-element float array following a JSON key.
static bool jsonFindVec3(const std::string& j, const std::string& key, size_t pos, float out[3]) {
    size_t kp = j.find(key, pos);
    if (kp == std::string::npos) return false;
    size_t bracket = j.find('[', kp + key.size());
    if (bracket == std::string::npos) return false;
    for (int i = 0; i < 3; i++) {
        size_t ns = j.find_first_of("-.0123456789", bracket + 1);
        if (ns == std::string::npos) return false;
        size_t ne = j.find_first_not_of("-.0123456789", ns + 1);
        if (ne == std::string::npos) return false;
        try { out[i] = std::stof(j.substr(ns, ne - ns)); } catch (...) { return false; }
        bracket = ne;
    }
    return true;
}

// Extract float data from a cgltf accessor into a flat vector.
static void extractFloats(const cgltf_accessor* acc, std::vector<float>& out) {
    if (!acc) return;
    cgltf_size num = cgltf_num_components(acc->type);
    out.resize(acc->count * num);
    cgltf_accessor_unpack_floats(acc, out.data(), out.size());
}

// Extract uint16 joint indices (4 per vertex)
static void extractJoints(const cgltf_accessor* acc, std::vector<uint16_t>& out) {
    if (!acc) return;
    out.resize(acc->count * 4);
    cgltf_size num = cgltf_num_components(acc->type);
    for (cgltf_size i = 0; i < acc->count; i++) {
        float tmp[4];
        cgltf_accessor_read_float(acc, i, tmp, num);
        for (int c = 0; c < 4; c++)
            out[i * 4 + c] = static_cast<uint16_t>(tmp[c]);
    }
}

static glm::mat4 nodeLocalMatrix(const VRMModel::Node& n) {
    glm::mat4 t = glm::translate(glm::mat4(1), n.translation);
    glm::mat4 r = glm::mat4_cast(n.rotation);
    glm::mat4 s = glm::scale(glm::mat4(1), n.scale);
    return t * r * s;
}

int VRMModel::totalTriangles() const {
    int total = 0;
    for (const auto& m : meshes)
        for (const auto& p : m.primitives)
            total += p.triangleCount();
    return total;
}

int VRMModel::totalVertices() const {
    int total = 0;
    for (const auto& m : meshes)
        for (const auto& p : m.primitives)
            total += p.vertexCount();
    return total;
}

VRMModel loadVRM(const std::string& path) {
    VRMModel model;
    model.path = path;

    cgltf_options options = {};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&options, path.c_str(), &data);
    if (res != cgltf_result_success) {
        fprintf(stderr, "[vrm] cgltf_parse_file failed: %d\n", res);
        return model;
    }
    res = cgltf_load_buffers(&options, data, path.c_str());
    if (res != cgltf_result_success) {
        fprintf(stderr, "[vrm] cgltf_load_buffers failed: %d\n", res);
        cgltf_free(data);
        return model;
    }
    // Parse VRM extension: head node + MToon material properties
    struct MtoonProps {
        float shadeColor[3] = {1, 1, 1};
        float shadeShift = 0;
        float shadeToony = 0.9f;
    };
    std::vector<MtoonProps> mtoonMats;

    for (cgltf_size i = 0; i < data->data_extensions_count; i++) {
        const cgltf_extension* ext = &data->data_extensions[i];
        if (!ext->name || !ext->data) continue;
        std::string ename(ext->name);
        if (ename != "VRM" && ename != "VRMC_vrm") continue;
        std::string d(ext->data);

        if (ename == "VRM") {
            // VRM 0.x: humanoid.humanBones is array of {"bone":"head","node":N}
            if (model.headNodeIndex < 0) {
                size_t pos = 0;
                while ((pos = d.find("\"bone\"", pos)) != std::string::npos) {
                    size_t vc = d.find('"', d.find(':', pos) + 1);
                    size_t ve = d.find('"', vc + 1);
                    if (vc == std::string::npos || ve == std::string::npos) break;
                    std::string bn = d.substr(vc + 1, ve - vc - 1);
                    size_t objEnd = d.find('}', ve);
                    if (bn == "head") {
                        size_t np = d.find("\"node\"", ve);
                        if (np != std::string::npos && np < objEnd) {
                            size_t ns = d.find_first_of("0123456789", d.find(':', np));
                            size_t ne = d.find_first_not_of("0123456789", ns);
                            model.headNodeIndex = std::stoi(d.substr(ns, ne - ns));
                            break;
                        }
                    }
                    pos = ve + 1;
                }
            }

            // VRM 0.x materialProperties: array aligned with glTF materials
            size_t mp = d.find("\"materialProperties\"");
            if (mp != std::string::npos) {
                size_t arrStart = d.find('[', mp);
                if (arrStart != std::string::npos) {
                    int depth = 0;
                    size_t objStart = std::string::npos;
                    for (size_t k = arrStart + 1; k < d.size(); k++) {
                        if (d[k] == '{') {
                            if (depth == 0) objStart = k;
                            depth++;
                        } else if (d[k] == '}') {
                            depth--;
                            if (depth == 0 && objStart != std::string::npos) {
                                MtoonProps mp_;
                                jsonFindVec3(d, "_ShadeColor", objStart, mp_.shadeColor);
                                jsonFindFloat(d, "_ShadeShift", objStart, mp_.shadeShift);
                                jsonFindFloat(d, "_ShadeToony", objStart, mp_.shadeToony);
                                mtoonMats.push_back(mp_);
                                objStart = std::string::npos;
                            }
                        } else if (d[k] == ']' && depth == 0) break;
                    }
                }
            }
        } else {
            // VRM 1.0: humanBones is dict {"head":{"node":N}}
            if (model.headNodeIndex < 0) {
                size_t hp = d.find("\"head\"");
                if (hp != std::string::npos) {
                    size_t np = d.find("\"node\"", hp);
                    if (np != std::string::npos) {
                        size_t ns = d.find_first_of("0123456789", d.find(':', np));
                        size_t ne = d.find_first_not_of("0123456789", ns);
                        model.headNodeIndex = std::stoi(d.substr(ns, ne - ns));
                    }
                }
            }
        }
    }

    // --- Nodes ---
    model.nodes.resize(data->nodes_count);
    for (cgltf_size i = 0; i < data->nodes_count; i++) {
        const cgltf_node* n = &data->nodes[i];
        auto& node = model.nodes[i];
        if (n->name) node.name = n->name;

        if (n->has_translation)
            memcpy(&node.translation, n->translation, sizeof(float) * 3);
        if (n->has_rotation) {
            // glTF quaternion = (x,y,z,w); GLM quat = (w,x,y,z)
            node.rotation.x = n->rotation[0];
            node.rotation.y = n->rotation[1];
            node.rotation.z = n->rotation[2];
            node.rotation.w = n->rotation[3];
        }
        if (n->has_scale)
            memcpy(&node.scale, n->scale, sizeof(float) * 3);
        // If matrix is present, decompose (most VRM nodes use TRS, but handle matrix)
        if (n->has_matrix && !n->has_translation) {
            glm::mat4 m;
            memcpy(&m, n->matrix, sizeof(float) * 16);
            node.translation = glm::vec3(m[3]);
            node.rotation = glm::quat_cast(m);
            glm::vec3 s = glm::vec3(
                glm::length(glm::vec3(m[0])),
                glm::length(glm::vec3(m[1])),
                glm::length(glm::vec3(m[2])));
            node.scale = s;
        }

        // Parent: find which node has this as a child
        node.parent = -1;
    }
    for (cgltf_size i = 0; i < data->nodes_count; i++) {
        const cgltf_node* n = &data->nodes[i];
        for (cgltf_size j = 0; j < n->children_count; j++) {
            int childIdx = static_cast<int>(n->children[j] - data->nodes);
            model.nodes[childIdx].parent = static_cast<int>(i);
            model.nodes[i].children.push_back(childIdx);
        }
    }

    // --- Textures ---
    model.textures.resize(data->images_count);
    for (cgltf_size i = 0; i < data->images_count; i++) {
        const cgltf_image* img = &data->images[i];
        if (img->buffer_view) {
            const cgltf_buffer_view* bv = img->buffer_view;
            const uint8_t* bytes = (const uint8_t*)bv->buffer->data + bv->offset;
            int w, h, ch;
            stbi_uc* px = stbi_load_from_memory(bytes, static_cast<int>(bv->size),
                                                &w, &h, &ch, 4);
            if (px) {
                model.textures[i].width = w;
                model.textures[i].height = h;
                model.textures[i].pixels.resize(w * h * 4);
                memcpy(model.textures[i].pixels.data(), px, w * h * 4);
                stbi_image_free(px);
            } else {
                fprintf(stderr, "[vrm] failed to decode texture %zu\n", i);
            }
        }
    }

    // --- Skin ---
    if (data->skins_count > 0) {
        const cgltf_skin* skin = &data->skins[0];
        model.jointNodes.resize(skin->joints_count);
        for (cgltf_size i = 0; i < skin->joints_count; i++) {
            model.jointNodes[i] = static_cast<int>(skin->joints[i] - data->nodes);
        }
        if (skin->inverse_bind_matrices) {
            std::vector<float> raw;
            extractFloats(skin->inverse_bind_matrices, raw);
            int count = static_cast<int>(raw.size() / 16);
            model.inverseBindMatrices.resize(count);
            for (int i = 0; i < count; i++) {
                // cgltf matrices are column-major (OpenGL convention) → glm::mat4 expects column-major
                memcpy(&model.inverseBindMatrices[i], &raw[i * 16], sizeof(float) * 16);
            }
        }
    }

    // --- Meshes ---
    model.meshes.resize(data->meshes_count);
    for (cgltf_size mi = 0; mi < data->meshes_count; mi++) {
        const cgltf_mesh* m = &data->meshes[mi];
        model.meshes[mi].name = m->name ? m->name : "";
        model.meshes[mi].primitives.resize(m->primitives_count);

        for (cgltf_size pi = 0; pi < m->primitives_count; pi++) {
            const cgltf_primitive* prim = &m->primitives[pi];
            auto& p = model.meshes[mi].primitives[pi];

            // Attributes
            for (cgltf_size ai = 0; ai < prim->attributes_count; ai++) {
                const cgltf_attribute* attr = &prim->attributes[ai];
                cgltf_attribute_type type = attr->type;
                int idx = attr->index;

                if (type == cgltf_attribute_type_position && idx == 0)
                    extractFloats(attr->data, p.positions);
                else if (type == cgltf_attribute_type_normal && idx == 0)
                    extractFloats(attr->data, p.normals);
                else if (type == cgltf_attribute_type_texcoord && idx == 0)
                    extractFloats(attr->data, p.uvs);
                else if (type == cgltf_attribute_type_joints && idx == 0)
                    extractJoints(attr->data, p.joints);
                else if (type == cgltf_attribute_type_weights && idx == 0)
                    extractFloats(attr->data, p.weights);
            }

            // Indices
            if (prim->indices) {
                p.indices.resize(prim->indices->count);
                for (cgltf_size i = 0; i < prim->indices->count; i++) {
                    p.indices[i] = static_cast<uint32_t>(
                        cgltf_accessor_read_index(prim->indices, i));
                }
            } else {
                int vc = p.vertexCount();
                p.indices.resize(vc);
                for (int i = 0; i < vc; i++)
                    p.indices[i] = i;
            }

            // Morph targets
            if (prim->targets_count > 0) {
                p.morphCount = static_cast<int>(prim->targets_count);
                int vc = p.vertexCount();
                p.morphDeltas.resize(p.morphCount * vc * 3, 0.0f);
                for (int t = 0; t < p.morphCount; t++) {
                    const cgltf_morph_target* tgt = &prim->targets[t];
                    for (cgltf_size ai = 0; ai < tgt->attributes_count; ai++) {
                        if (tgt->attributes[ai].type == cgltf_attribute_type_position) {
                            std::vector<float> deltas;
                            extractFloats(tgt->attributes[ai].data, deltas);
                            for (int v = 0; v < vc && v * 3 + 2 < (int)deltas.size(); v++) {
                                p.morphDeltas[(t * vc + v) * 3 + 0] = deltas[v * 3 + 0];
                                p.morphDeltas[(t * vc + v) * 3 + 1] = deltas[v * 3 + 1];
                                p.morphDeltas[(t * vc + v) * 3 + 2] = deltas[v * 3 + 2];
                            }
                        }
                    }
                }
            }

            // Material
            if (prim->material) {
                p.baseColor = glm::vec4(
                    prim->material->pbr_metallic_roughness.base_color_factor[0],
                    prim->material->pbr_metallic_roughness.base_color_factor[1],
                    prim->material->pbr_metallic_roughness.base_color_factor[2],
                    prim->material->pbr_metallic_roughness.base_color_factor[3]);
                p.doubleSided = (prim->material->double_sided);
                p.alphaMode = (int)prim->material->alpha_mode;
                if (prim->material->name)
                    p.matName = prim->material->name;
                if (prim->material->pbr_metallic_roughness.base_color_texture.texture) {
                    const cgltf_texture* tex = prim->material->pbr_metallic_roughness.base_color_texture.texture;
                    if (tex->image) {
                        p.textureIndex = static_cast<int>(tex->image - data->images);
                    }
                }

                // MToon properties from VRM extension
                int matIdx = static_cast<int>(prim->material - data->materials);
                if (matIdx >= 0 && matIdx < (int)mtoonMats.size()) {
                    auto& mp = mtoonMats[matIdx];
                    p.mtoonShadeColor = glm::vec3(mp.shadeColor[0], mp.shadeColor[1], mp.shadeColor[2]);
                    // Transform shadeShift/shadeToony per three-vrm convention
                    float toony = mp.shadeToony + (1.0f - mp.shadeToony) * (0.5f + 0.5f * mp.shadeShift);
                    float shift = -mp.shadeShift - (1.0f - toony);
                    float rampWidth = 2.0f * (1.0f - toony);
                    p.mtoonRampScale = (rampWidth > 1e-5f) ? (1.0f / rampWidth) : 100000.0f;
                    p.mtoonRampBias = shift + 1.0f - toony;
                }
            }
        }
    }

    // Find which node owns each mesh
    for (cgltf_size i = 0; i < data->nodes_count; i++) {
        if (data->nodes[i].mesh) {
            int meshIdx = static_cast<int>(data->nodes[i].mesh - data->meshes);
            if (meshIdx < (int)model.meshes.size())
                model.meshes[meshIdx].nodeIndex = static_cast<int>(i);
        }
    }

    // Bounding box from all positions
    glm::vec3 bmin(1e30f), bmax(-1e30f);
    for (const auto& m : model.meshes)
        for (const auto& p : m.primitives) {
            for (int v = 0; v < p.vertexCount(); v++) {
                glm::vec3 pos(p.positions[v * 3], p.positions[v * 3 + 1], p.positions[v * 3 + 2]);
                bmin = glm::min(bmin, pos);
                bmax = glm::max(bmax, pos);
            }
        }
    model.bboxMin = bmin;
    model.bboxMax = bmax;

    cgltf_free(data);

    fprintf(stderr, "[vrm] loaded: %zu meshes, %zu textures, %d verts, %d tris\n",
            model.meshes.size(), model.textures.size(),
            model.totalVertices(), model.totalTriangles());
    fprintf(stderr, "[vrm] nodes: %zu, joints: %zu, morph targets (mesh0 prim0): %d\n",
            model.nodes.size(), model.jointNodes.size(),
            model.meshes.empty() ? 0 : model.meshes[0].primitives[0].morphCount);
    fprintf(stderr, "[vrm] bbox: [%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f]\n",
            bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);

    return model;
}
