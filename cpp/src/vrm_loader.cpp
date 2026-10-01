#include "vrm_loader.h"
#include "logging.h"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <cmath>

// sRGB → linear conversion for material colors (matching Three.js color management)
static inline float sRGBToLinear(float s) {
    return (s <= 0.04045f) ? s / 12.92f
                           : powf((s + 0.055f) / 1.055f, 2.4f);
}

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

// Find a string value following a JSON key, starting from `pos`.
static bool jsonFindString(const std::string& j, const std::string& key, size_t pos, std::string& out) {
    size_t kp = j.find(key, pos);
    if (kp == std::string::npos) return false;
    size_t colon = j.find(':', kp + key.size());
    if (colon == std::string::npos) return false;
    size_t vs = j.find('"', colon + 1);
    if (vs == std::string::npos) return false;
    size_t ve = j.find('"', vs + 1);
    if (ve == std::string::npos) return false;
    out = j.substr(vs + 1, ve - vs - 1);
    return true;
}

// Find an integer value following a JSON key, starting from `pos`.
static bool jsonFindInt(const std::string& j, const std::string& key, size_t pos, int& out) {
    size_t kp = j.find(key, pos);
    if (kp == std::string::npos) return false;
    size_t colon = j.find(':', kp + key.size());
    if (colon == std::string::npos) return false;
    size_t ns = j.find_first_of("-0123456789", colon + 1);
    if (ns == std::string::npos) return false;
    size_t ne = j.find_first_not_of("-0123456789", ns + 1);
    if (ne == std::string::npos) ne = j.size();
    try { out = std::stoi(j.substr(ns, ne - ns)); } catch (...) { return false; }
    return true;
}

// Iterate the top-level JSON objects of the array that follows `key`,
// calling fn(objStart, objEnd) with the object's brace positions.
template <typename Fn>
static void jsonForEachObjectInArray(const std::string& j, const std::string& key, size_t pos, Fn fn) {
    size_t kp = j.find(key, pos);
    if (kp == std::string::npos) return;
    size_t arrS = j.find('[', kp);
    if (arrS == std::string::npos) return;
    int depth = 0;
    size_t objS = std::string::npos;
    for (size_t k = arrS + 1; k < j.size(); k++) {
        char c = j[k];
        if (c == '{') {
            if (depth == 0) objS = k;
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0 && objS != std::string::npos) {
                fn(objS, k);
                objS = std::string::npos;
            }
        } else if (c == ']' && depth == 0) {
            break;
        }
    }
}

// Iterate the top-level JSON numbers of the array that follows `key`.
template <typename Fn>
static void jsonForEachIntInArray(const std::string& j, const std::string& key, size_t pos, Fn fn) {
    size_t kp = j.find(key, pos);
    if (kp == std::string::npos) return;
    size_t arrS = j.find('[', kp);
    if (arrS == std::string::npos) return;
    int depth = 0;
    for (size_t k = arrS + 1; k < j.size(); k++) {
        char c = j[k];
        if (c == '[' || c == '{') depth++;
        else if (c == ']' || c == '}') {
            if (depth == 0 && c == ']') break;
            depth--;
        } else if (depth == 0 && (c == '-' || (c >= '0' && c <= '9'))) {
            size_t ns = k;
            size_t ne = j.find_first_not_of("-0123456789", ns + 1);
            if (ne == std::string::npos) ne = j.size();
            try { fn(std::stoi(j.substr(ns, ne - ns))); } catch (...) {}
            k = ne - 1;
        }
    }
}

// Iterate the top-level JSON strings of the array that follows `key`.
template <typename Fn>
static void jsonForEachStringInArray(const std::string& j, const std::string& key, size_t pos, Fn fn) {
    size_t kp = j.find(key, pos);
    if (kp == std::string::npos) return;
    size_t arrS = j.find('[', kp);
    if (arrS == std::string::npos) return;
    int depth = 0;
    size_t k = arrS + 1;
    while (k < j.size()) {
        char c = j[k];
        if (c == '[' || c == '{') depth++;
        else if (c == ']' || c == '}') {
            if (depth == 0 && c == ']') break;
            depth--;
        } else if (depth == 0 && c == '"') {
            size_t ve = j.find('"', k + 1);
            if (ve == std::string::npos) break;
            fn(j.substr(k + 1, ve - k - 1));
            k = ve;
        }
        k++;
    }
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
        float cullMode = 2;  // VRM _CullMode: 0=off(doubleSided), 2=back
        int renderQueue = 2450;
    };
    std::vector<MtoonProps> mtoonMats;

    // VRM 0.x springbone groups reference bones either by NAME (nested
    // string arrays) or by node index (flat array of chain ROOTS, with the
    // chain continuing down the node hierarchy). Nodes are not loaded yet,
    // so both are stashed and resolved after node loading.
    struct Vrm0xCollider { int node = -1; float offset[3] = {0, 0, 0}; float radius = 0.08f; };
    std::vector<std::vector<Vrm0xCollider>> vrm0xColliderGroups;
    struct Vrm0xBoneGroup {
        float stiffness = 1.0f, gravityPower = 0.0f, dragForce = 0.4f, hitRadius = 0.02f;
        float gravityDir[3] = {0.0f, -1.0f, 0.0f};
        std::vector<std::vector<std::string>> nameChains;  // nested string arrays
        std::vector<std::vector<int>> intChains;           // nested int arrays
        std::vector<int> rootIndices;                      // flat ints: chain roots
        std::vector<int> colliderGroups;
    };
    std::vector<Vrm0xBoneGroup> vrm0xBoneGroups;

    // VRM 1.0 expressions reference morph-target indices that apply to every
    // mesh; meshes are not loaded yet during extension parsing, so binds are
    // stashed and resolved after mesh loading.
    struct Vrm1Expression {
        std::string name;
        bool isCustom = false;
        std::vector<std::pair<int, float>> binds;  // (morph target index, weight 0-1)
    };
    std::vector<Vrm1Expression> vrm1Expressions;

    for (cgltf_size i = 0; i < data->data_extensions_count; i++) {
        const cgltf_extension* ext = &data->data_extensions[i];
        if (!ext->name || !ext->data) continue;
        std::string ename(ext->name);
        if (ename != "VRM" && ename != "VRMC_vrm" &&
            ename != "VRMC_vrmExpressions" && ename != "VRMC_springBone")
            continue;
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
                    // Parse node index for every bone
                    size_t np = d.find("\"node\"", ve);
                    if (np != std::string::npos && np < objEnd) {
                        size_t ns = d.find_first_of("0123456789", d.find(':', np));
                        size_t ne = d.find_first_not_of("0123456789", ns);
                        if (ns != std::string::npos && ns < objEnd) {
                            int nodeIdx = std::stoi(d.substr(ns, ne - ns));
                            model.boneNodes[bn] = nodeIdx;
                            if (bn == "head") model.headNodeIndex = nodeIdx;
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
                                jsonFindFloat(d, "_CullMode", objStart, mp_.cullMode);
                                float rq = 2450;
                                jsonFindFloat(d, "\"renderQueue\"", objStart, rq);
                                mp_.renderQueue = (int)rq;
                                mtoonMats.push_back(mp_);
                                objStart = std::string::npos;
                            }
                        } else if (d[k] == ']' && depth == 0) break;
                    }
                }
            }

            // VRM 0.x blendShapeMaster: blendShapeGroups with preset + binds
            size_t bsm = d.find("\"blendShapeMaster\"");
            if (bsm != std::string::npos) {
                size_t groupsPos = d.find("\"blendShapeGroups\"", bsm);
                if (groupsPos != std::string::npos) {
                    size_t arrS = d.find('[', groupsPos);
                    if (arrS != std::string::npos) {
                        int depth = 0;
                        size_t objS = std::string::npos;
                        for (size_t k = arrS + 1; k < d.size(); k++) {
                            if (d[k] == '{') {
                                if (depth == 0) objS = k;
                                depth++;
                            } else if (d[k] == '}') {
                                depth--;
                                if (depth == 0 && objS != std::string::npos) {
                                    VRMModel::BlendShapeGroup g;
                                    // Extract preset name
                                    size_t pp = d.find("\"presetName\"", objS);
                                    if (pp != std::string::npos && pp < k) {
                                        size_t vc = d.find('"', d.find(':', pp) + 1);
                                        size_t ve = d.find('"', vc + 1);
                                        if (vc != std::string::npos && ve != std::string::npos)
                                            g.presetName = d.substr(vc + 1, ve - vc - 1);
                                    }
                                    // Extract name
                                    size_t np2 = d.find("\"name\"", objS);
                                    if (np2 != std::string::npos && np2 < k) {
                                        size_t vc = d.find('"', d.find(':', np2) + 1);
                                        size_t ve = d.find('"', vc + 1);
                                        if (vc != std::string::npos && ve != std::string::npos)
                                            g.name = d.substr(vc + 1, ve - vc - 1);
                                    }
                                    // Extract binds array
                                    size_t bp = d.find("\"binds\"", objS);
                                    if (bp != std::string::npos && bp < k) {
                                        size_t bArrS = d.find('[', bp);
                                        if (bArrS != std::string::npos) {
                                            size_t bEnd = bArrS + 1;
                                            int bDepth = 0;
                                            for (size_t bk = bArrS + 1; bk < d.size(); bk++) {
                                                if (d[bk] == '{') { bDepth++; }
                                                else if (d[bk] == ']' && bDepth == 0) break;
                                                else if (d[bk] == '}') {
                                                    bDepth--;
                                                    if (bDepth == 0) {
                                                        VRMModel::BlendShapeBind b;
                                                        size_t mp2 = d.find("\"mesh\"", bEnd);
                                                        size_t ip2 = d.find("\"index\"", bEnd);
                                                        size_t wp2 = d.find("\"weight\"", bEnd);
                                                        if (mp2 != std::string::npos && mp2 < bk) {
                                                            auto s = d.find_first_of("0123456789", d.find(':', mp2));
                                                            auto e = d.find_first_not_of("0123456789", s);
                                                            b.mesh = std::stoi(d.substr(s, e - s));
                                                        }
                                                        if (ip2 != std::string::npos && ip2 < bk) {
                                                            auto s = d.find_first_of("0123456789", d.find(':', ip2));
                                                            auto e = d.find_first_not_of("0123456789", s);
                                                            b.index = std::stoi(d.substr(s, e - s));
                                                        }
                                                        if (wp2 != std::string::npos && wp2 < bk) {
                                                            auto s = d.find_first_of("0123456789-", d.find(':', wp2));
                                                            auto e = d.find_first_not_of("0123456789.", s);
                                                            b.weight = std::stof(d.substr(s, e - s));
                                                        }
                                                        g.binds.push_back(b);
                                                        bEnd = bk + 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    model.blendShapeGroups.push_back(g);
                                    objS = std::string::npos;
                                }
                            } else if (d[k] == ']' && depth == 0) break;
                        }
                    }
                }
            }

            // VRM 0.x lookAt: firstPerson.lookAtTypeName + range maps.
            // Range maps map a blendshape input weight to eye degrees:
            // {"xRange": 90, "yRange": 10} (inputMax, outputScale).
            {
                size_t fpp = d.find("\"firstPerson\"");
                if (fpp != std::string::npos) {
                    std::string typeName;
                    if (jsonFindString(d, "\"lookAtTypeName\"", fpp, typeName))
                        model.lookAt.type = (typeName == "Bone") ? "bone" : "expression";
                    size_t rm = d.find("\"lookAtHorizontalInner\"", fpp);
                    float yr = 10.0f;
                    if (rm != std::string::npos && jsonFindFloat(d, "\"yRange\"", rm, yr))
                        model.lookAt.hOut = yr;
                    rm = d.find("\"lookAtVerticalUp\"", fpp);
                    yr = 10.0f;
                    if (rm != std::string::npos && jsonFindFloat(d, "\"yRange\"", rm, yr))
                        model.lookAt.vUpOut = yr;
                    rm = d.find("\"lookAtVerticalDown\"", fpp);
                    yr = 10.0f;
                    if (rm != std::string::npos && jsonFindFloat(d, "\"yRange\"", rm, yr))
                        model.lookAt.vDownOut = yr;
                }
            }

            // VRM 0.x secondaryAnimation (springbones)
            {
                size_t sa = d.find("\"secondaryAnimation\"");
                if (sa != std::string::npos) {
                    // colliderGroups: [{node, colliders: [{offset, radius}]}]
                    jsonForEachObjectInArray(d, "\"colliderGroups\"", sa,
                        [&](size_t os, size_t oe) {
                            std::vector<Vrm0xCollider> group;
                            int node = -1;
                            jsonFindInt(d, "\"node\"", os, node);
                            jsonForEachObjectInArray(d, "\"colliders\"", os,
                                [&](size_t cs, size_t) {
                                    Vrm0xCollider c;
                                    c.node = node;
                                    jsonFindVec3(d, "\"offset\"", cs, c.offset);
                                    jsonFindFloat(d, "\"radius\"", cs, c.radius);
                                    group.push_back(c);
                                });
                            vrm0xColliderGroups.push_back(std::move(group));
                        });

                    // boneGroups: [{stiffiness, gravityPower, gravityDir,
                    //               dragForce, hitRadius, bones, colliderGroups}]
                    jsonForEachObjectInArray(d, "\"boneGroups\"", sa,
                        [&](size_t os, size_t oe) {
                            Vrm0xBoneGroup bg;
                            jsonFindFloat(d, "\"stiffiness\"", os, bg.stiffness);
                            jsonFindFloat(d, "\"gravityPower\"", os, bg.gravityPower);
                            jsonFindVec3(d, "\"gravityDir\"", os, bg.gravityDir);
                            jsonFindFloat(d, "\"dragForce\"", os, bg.dragForce);
                            jsonFindFloat(d, "\"hitRadius\"", os, bg.hitRadius);
                            jsonForEachIntInArray(d, "\"colliderGroups\"", os,
                                [&](int idx) { bg.colliderGroups.push_back(idx); });
                            // bones: nested string arrays (explicit chains by
                            // node name), nested int arrays (explicit chains
                            // by node index), or a flat int array (chain ROOT
                            // indices — chains continue down the hierarchy,
                            // the format VRoid Studio exports).
                            size_t bp = d.find("\"bones\"", os);
                            if (bp != std::string::npos && bp < oe) {
                                size_t outerArrS = d.find('[', bp);
                                if (outerArrS != std::string::npos) {
                                    int depth = 0;
                                    size_t innerS = std::string::npos;
                                    std::vector<std::string> flatNames;
                                    for (size_t k = outerArrS + 1; k < d.size(); k++) {
                                        char c = d[k];
                                        if (c == '[') {
                                            if (depth == 0) innerS = k;
                                            depth++;
                                        } else if (c == ']') {
                                            depth--;
                                            if (depth == 0 && innerS != std::string::npos) {
                                                // Parse one inner array
                                                std::vector<std::string> names;
                                                std::vector<int> idxs;
                                                int d2 = 0;
                                                size_t q = innerS + 1;
                                                while (q < k) {
                                                    char cc = d[q];
                                                    if (cc == '[') d2++;
                                                    else if (cc == ']') d2--;
                                                    else if (d2 == 0 && cc == '"') {
                                                        size_t ve = d.find('"', q + 1);
                                                        if (ve == std::string::npos || ve >= k) break;
                                                        names.push_back(d.substr(q + 1, ve - q - 1));
                                                        q = ve;
                                                    } else if (d2 == 0 &&
                                                               (cc == '-' || (cc >= '0' && cc <= '9'))) {
                                                        size_t ns2 = q;
                                                        size_t ne2 = d.find_first_not_of("-0123456789", ns2 + 1);
                                                        if (ne2 == std::string::npos) ne2 = k;
                                                        try { idxs.push_back(std::stoi(d.substr(ns2, ne2 - ns2))); } catch (...) {}
                                                        q = ne2 - 1;
                                                    }
                                                    q++;
                                                }
                                                if (!names.empty())
                                                    bg.nameChains.push_back(std::move(names));
                                                else if (!idxs.empty())
                                                    bg.intChains.push_back(std::move(idxs));
                                                innerS = std::string::npos;
                                            }
                                        } else if (c == ']' && depth == 0) break;
                                        else if (c == '}' && depth == 0) break;
                                        else if (depth == 0 && c == '"') {
                                            // flat top-level string (chain by name)
                                            size_t ve = d.find('"', k + 1);
                                            if (ve == std::string::npos) break;
                                            flatNames.push_back(d.substr(k + 1, ve - k - 1));
                                            k = ve;
                                        } else if (depth == 0 &&
                                                   (c == '-' || (c >= '0' && c <= '9'))) {
                                            // flat top-level int (chain ROOT index)
                                            size_t ns2 = k;
                                            size_t ne2 = d.find_first_not_of("-0123456789", ns2 + 1);
                                            if (ne2 == std::string::npos) ne2 = d.size();
                                            try { bg.rootIndices.push_back(std::stoi(d.substr(ns2, ne2 - ns2))); } catch (...) {}
                                            k = ne2 - 1;
                                        }
                                    }
                                    if (!flatNames.empty())
                                        bg.nameChains.push_back(std::move(flatNames));
                                }
                            }
                            vrm0xBoneGroups.push_back(std::move(bg));
                        });
                }
            }
        } else if (ename == "VRMC_vrm") {
            // VRM 1.0: humanBones is dict {"head":{"node":N}, "hips":{"node":N}, ...}
            if (model.headNodeIndex < 0) {
                // Find the "humanBones" key in the JSON
                size_t hbp = d.find("\"humanBones\"");
                if (hbp != std::string::npos) {
                    size_t objStart = d.find('{', hbp);
                    if (objStart != std::string::npos) {
                        int depth = 0;
                        size_t k = objStart;
                        while (k < d.size()) {
                            if (d[k] == '{') depth++;
                            else if (d[k] == '}') { depth--; if (depth == 0) break; }
                            else if (d[k] == '"') {
                                // Found a key (bone name)
                                size_t ks = k + 1;
                                size_t ke = d.find('"', ks);
                                if (ke == std::string::npos) break;
                                std::string boneName = d.substr(ks, ke - ks);
                                // Find "node" value in this bone's object
                                size_t innerEnd = d.find('}', ke);
                                size_t np = d.find("\"node\"", ke);
                                if (np != std::string::npos && np < innerEnd) {
                                    size_t ns = d.find_first_of("0123456789", d.find(':', np));
                                    size_t ne = d.find_first_not_of("0123456789", ns);
                                    if (ns != std::string::npos) {
                                        int nodeIdx = std::stoi(d.substr(ns, ne - ns));
                                        model.boneNodes[boneName] = nodeIdx;
                                        if (boneName == "head") model.headNodeIndex = nodeIdx;
                                    }
                                }
                                // Jump past the inner {"node":N} object: count
                                // its opening brace (skipped below) so the
                                // matching '}' at innerEnd keeps depth
                                // balanced when the loop head processes it.
                                // (A plain `continue` here would skip the
                                // loop's k++ and abort after the first bone.)
                                size_t innerStart = d.find('{', ke);
                                if (innerStart != std::string::npos && innerStart < innerEnd)
                                    depth++;
                                k = innerEnd;
                            }
                            k++;
                        }
                    }
                }
            }

            // VRM 1.0 lookAt: type + rangeMap outputScale values
            {
                size_t la = d.find("\"lookAt\"");
                if (la != std::string::npos) {
                    std::string t;
                    if (jsonFindString(d, "\"type\"", la, t))
                        model.lookAt.type = t;
                    float os = 10.0f;
                    size_t rm = d.find("\"rangeMapHorizontalInner\"", la);
                    if (rm != std::string::npos &&
                        jsonFindFloat(d, "\"outputScale\"", rm, os))
                        model.lookAt.hOut = os;
                    rm = d.find("\"rangeMapVerticalUp\"", la);
                    if (rm != std::string::npos &&
                        jsonFindFloat(d, "\"outputScale\"", rm, os))
                        model.lookAt.vUpOut = os;
                    rm = d.find("\"rangeMapVerticalDown\"", la);
                    if (rm != std::string::npos &&
                        jsonFindFloat(d, "\"outputScale\"", rm, os))
                        model.lookAt.vDownOut = os;
                }
            }
        } else if (ename == "VRMC_vrmExpressions") {
            // VRM 1.0 expressions: {expressions: {preset: {...}, custom: {...}}}
            // Morph binds carry a morph-target index + 0-1 weight and apply to
            // every mesh; converted to the 0.x-style group representation.
            size_t ep = d.find("\"expressions\"");
            if (ep != std::string::npos) {
                auto parseExpressionMap = [&](const char* sectionKey, bool isCustom) {
                    size_t sp = d.find(sectionKey, ep);
                    if (sp == std::string::npos) return;
                    // Each key in this object is an expression name; its value
                    // object holds binds[]. Walk top-level key/value pairs.
                    size_t objS = d.find('{', sp);
                    if (objS == std::string::npos) return;
                    int depth = 0;
                    size_t k = objS;
                    while (k < d.size()) {
                        char c = d[k];
                        if (c == '{') { depth++; if (depth == 1) { k++; continue; } }
                        else if (c == '}') { depth--; if (depth == 0) break; }
                        else if (depth == 1 && c == '"') {
                            size_t ve = d.find('"', k + 1);
                            if (ve == std::string::npos) break;
                            std::string exprName = d.substr(k + 1, ve - k - 1);
                            // value object spans until its matching close brace
                            size_t vStart = d.find('{', ve);
                            if (vStart == std::string::npos) break;
                            int vDepth = 0;
                            size_t vEnd = vStart;
                            for (; vEnd < d.size(); vEnd++) {
                                if (d[vEnd] == '{') vDepth++;
                                else if (d[vEnd] == '}') { vDepth--; if (vDepth == 0) break; }
                            }
                            Vrm1Expression raw;
                            raw.name = exprName;
                            raw.isCustom = isCustom;
                            jsonForEachObjectInArray(d, "\"binds\"", vStart,
                                [&](size_t bs, size_t) {
                                    int morphIdx = -1;
                                    float weight = 1.0f;
                                    jsonFindInt(d, "\"expression\"", bs, morphIdx);
                                    jsonFindFloat(d, "\"weight\"", bs, weight);
                                    if (morphIdx >= 0)
                                        raw.binds.push_back({morphIdx, weight});
                                });
                            if (!raw.binds.empty())
                                vrm1Expressions.push_back(std::move(raw));
                            k = vEnd;
                        }
                        k++;
                    }
                };
                parseExpressionMap("\"preset\"", false);
                parseExpressionMap("\"custom\"", true);
            }
        } else if (ename == "VRMC_springBone") {
            // VRM 1.0 springbone: colliders (node + sphere shape),
            // colliderGroups (name + collider indices), springs (joints).
            std::vector<VRMModel::SpringCollider> colliders1;
            size_t cp = d.find("\"colliders\"");
            if (cp != std::string::npos) {
                jsonForEachObjectInArray(d, "\"colliders\"", cp,
                    [&](size_t os, size_t) {
                        VRMModel::SpringCollider c;
                        jsonFindInt(d, "\"node\"", os, c.node);
                        jsonFindVec3(d, "\"offset\"", os, &c.offset[0]);
                        jsonFindFloat(d, "\"radius\"", os, c.radius);
                        colliders1.push_back(c);
                    });
            }
            // colliderGroups: [{name, colliders: [indices]}]
            std::vector<std::vector<int>> colliderGroups1;
            size_t gp = d.find("\"colliderGroups\"");
            if (gp != std::string::npos) {
                jsonForEachObjectInArray(d, "\"colliderGroups\"", gp,
                    [&](size_t os, size_t) {
                        std::vector<int> idxs;
                        jsonForEachIntInArray(d, "\"colliders\"", os,
                            [&](int idx) { idxs.push_back(idx); });
                        colliderGroups1.push_back(std::move(idxs));
                    });
            }
            size_t spp = d.find("\"springs\"");
            if (spp != std::string::npos) {
                jsonForEachObjectInArray(d, "\"springs\"", spp,
                    [&](size_t os, size_t) {
                        VRMModel::SpringChain sc;
                        jsonForEachObjectInArray(d, "\"joints\"", os,
                            [&](size_t js, size_t) {
                                int node = -1;
                                jsonFindInt(d, "\"node\"", js, node);
                                if (node >= 0) sc.joints.push_back(node);
                            });
                        if (sc.joints.empty()) return;
                        // Per-joint params: take them from the first joint
                        // (chains animate uniformly in practice).
                        size_t j0 = d.find("\"joints\"", os);
                        if (j0 != std::string::npos) {
                            float f;
                            if (jsonFindFloat(d, "\"hitRadius\"", j0, f)) sc.hitRadius = f;
                            if (jsonFindFloat(d, "\"stiffness\"", j0, f)) sc.stiffness = f;
                            if (jsonFindFloat(d, "\"gravityPower\"", j0, f)) sc.gravityPower = f;
                            if (jsonFindFloat(d, "\"dragForce\"", j0, f)) sc.dragForce = f;
                            jsonFindVec3(d, "\"gravityDir\"", j0, &sc.gravityDir[0]);
                        }
                        jsonForEachIntInArray(d, "\"colliderGroups\"", os,
                            [&](int gIdx) {
                                if (gIdx >= 0 && gIdx < (int)colliderGroups1.size()) {
                                    for (int cIdx : colliderGroups1[gIdx]) {
                                        if (cIdx >= 0 && cIdx < (int)colliders1.size())
                                            sc.colliders.push_back(colliders1[cIdx]);
                                    }
                                }
                            });
                        if (sc.joints.size() >= 2)
                            model.springChains.push_back(std::move(sc));
                    });
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

    // Resolve VRM 0.x springbone chains → node indices
    if (!vrm0xBoneGroups.empty()) {
        std::unordered_map<std::string, int> nodeByName;
        for (size_t ni = 0; ni < model.nodes.size(); ni++)
            if (!model.nodes[ni].name.empty())
                nodeByName[model.nodes[ni].name] = (int)ni;

        for (const auto& bg : vrm0xBoneGroups) {
            // Gather colliders from referenced collider groups
            std::vector<VRMModel::SpringCollider> chainColliders;
            for (int gi : bg.colliderGroups) {
                if (gi >= 0 && gi < (int)vrm0xColliderGroups.size()) {
                    for (const auto& c : vrm0xColliderGroups[gi]) {
                        VRMModel::SpringCollider sc;
                        sc.node = c.node;
                        sc.offset = glm::vec3(c.offset[0], c.offset[1], c.offset[2]);
                        sc.radius = c.radius;
                        chainColliders.push_back(sc);
                    }
                }
            }

            auto validChain = [&](const std::vector<int>& joints) {
                if (joints.size() < 2) return false;
                for (int j : joints)
                    if (j < 0 || j >= (int)model.nodes.size()) return false;
                return true;
            };
            auto pushChain = [&](std::vector<int>&& joints) {
                if (!validChain(joints)) return;
                VRMModel::SpringChain sc;
                sc.joints = std::move(joints);
                sc.stiffness = bg.stiffness;
                sc.gravityPower = bg.gravityPower;
                sc.gravityDir = glm::vec3(bg.gravityDir[0], bg.gravityDir[1], bg.gravityDir[2]);
                sc.dragForce = bg.dragForce;
                sc.hitRadius = bg.hitRadius;
                sc.colliders = chainColliders;
                model.springChains.push_back(std::move(sc));
            };

            // Chains listed by node name
            for (const auto& chain : bg.nameChains) {
                std::vector<int> joints;
                for (const auto& boneName : chain) {
                    auto it = nodeByName.find(boneName);
                    if (it != nodeByName.end()) joints.push_back(it->second);
                }
                pushChain(std::move(joints));
            }
            // Chains listed explicitly by node index
            for (const auto& chain : bg.intChains)
                pushChain(std::vector<int>(chain));
            // Flat root indices: the chain continues down the hierarchy
            // (single-child walk, matching how the exporter builds them)
            for (int root : bg.rootIndices) {
                if (root < 0 || root >= (int)model.nodes.size()) continue;
                std::vector<int> chain{root};
                int cur = root;
                while (chain.size() < 32) {
                    const auto& n = model.nodes[cur];
                    if (n.children.size() != 1) break;
                    int next = n.children[0];
                    if (next < 0 || next >= (int)model.nodes.size()) break;
                    chain.push_back(next);
                    cur = next;
                }
                pushChain(std::move(chain));
            }
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
                    sRGBToLinear(prim->material->pbr_metallic_roughness.base_color_factor[0]),
                    sRGBToLinear(prim->material->pbr_metallic_roughness.base_color_factor[1]),
                    sRGBToLinear(prim->material->pbr_metallic_roughness.base_color_factor[2]),
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
                    p.mtoonShadeColor = glm::vec3(
                        sRGBToLinear(mp.shadeColor[0]),
                        sRGBToLinear(mp.shadeColor[1]),
                        sRGBToLinear(mp.shadeColor[2]));
                    // Transform shadeShift/shadeToony per three-vrm convention
                    float toony = mp.shadeToony + (1.0f - mp.shadeToony) * (0.5f + 0.5f * mp.shadeShift);
                    float shift = -mp.shadeShift - (1.0f - toony);
                    float rampWidth = 2.0f * (1.0f - toony);
                    p.mtoonRampScale = (rampWidth > 1e-5f) ? (1.0f / rampWidth) : 100000.0f;
                    p.mtoonRampBias = shift + 1.0f - toony;
                    // VRM _CullMode overrides glTF doubleSided: 0=off(doubleSided), 2=back
                    p.doubleSided = (mp.cullMode == 0.0f);
                    p.renderQueue = mp.renderQueue;
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

    // Resolve VRM 1.0 expressions: each morph bind applies to every mesh
    // that has that morph target. (Deferred until here because meshes are
    // loaded after the extension section.)
    for (const auto& e : vrm1Expressions) {
        VRMModel::BlendShapeGroup g;
        g.name = e.name;
        g.presetName = e.isCustom ? "" : e.name;
        for (const auto& [morphIdx, weight] : e.binds) {
            if (morphIdx < 0) continue;
            for (size_t mi = 0; mi < model.meshes.size(); mi++) {
                if (!model.meshes[mi].primitives.empty() &&
                    model.meshes[mi].primitives[0].morphCount > morphIdx) {
                    VRMModel::BlendShapeBind b;
                    b.mesh = (int)mi;
                    b.index = morphIdx;
                    b.weight = weight * 100.0f;  // 0-1 → 0-100
                    g.binds.push_back(b);
                }
            }
        }
        if (!g.binds.empty())
            model.blendShapeGroups.push_back(std::move(g));
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

    VLOG("[vrm] loaded: %zu meshes, %zu textures, %d verts, %d tris\n",
         model.meshes.size(), model.textures.size(),
         model.totalVertices(), model.totalTriangles());
    VLOG("[vrm] nodes: %zu, joints: %zu, morph targets (mesh0 prim0): %d\n",
         model.nodes.size(), model.jointNodes.size(),
         model.meshes.empty() ? 0 : model.meshes[0].primitives[0].morphCount);
    VLOG("[vrm] bbox: [%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f]\n",
            bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);

    return model;
}
