#include "rasterizer.h"
#include "BS_thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <atomic>
#include <cctype>
#include <string>

// Lighting: normalize(vec3(0.3, 0.8, 0.5)) — matches Python fragment shader
static constexpr float LIGHT_INV_LEN = 1.0f / 0.989949373f; // 1/sqrt(0.98)
static constexpr float LX = 0.3f * LIGHT_INV_LEN;
static constexpr float LY = 0.8f * LIGHT_INV_LEN;
static constexpr float LZ = 0.5f * LIGHT_INV_LEN;

// VRM material naming: "{name}_{order}_{TAG}"
// Render groups: 0=body, 1=hair, 2=face (has morphs)
static int computeRenderOrder(const MeshPrimitive& prim) {
    if (prim.morphCount > 0) return 2;
    if (!prim.matName.empty()) {
        size_t pos = prim.matName.rfind('_');
        std::string tag = (pos != std::string::npos)
            ? prim.matName.substr(pos + 1) : prim.matName;
        std::transform(tag.begin(), tag.end(), tag.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        if (tag == "HAIR") return 1;
    }
    return 0;
}

// Face part layering: skin→eyewhite→eyeline→brow→mouth→iris→highlight
static int computeFaceSortKey(const std::string& matName) {
    if (matName.empty()) return 7;
    std::string low;
    low.reserve(matName.size());
    for (char c : matName) low.push_back((char)std::tolower((unsigned char)c));
    if (low.find("skin") != std::string::npos) return 0;
    if (low.find("eyewhite") != std::string::npos) return 1;
    if (low.find("eyeline") != std::string::npos) return 2;
    if (low.find("brow") != std::string::npos) return 3;
    if (low.find("mouth") != std::string::npos) return 4;
    if (low.find("iris") != std::string::npos) return 5;
    if (low.find("highlight") != std::string::npos) return 6;
    return 7;
}

// Persistent thread pool — created once, reused every frame
static BS::thread_pool<>& getThreadPool() {
    static BS::thread_pool<> pool;
    return pool;
}

Framebuffer::Framebuffer(int w, int h)
    : width(w), height(h), color(w * h * 4, 0), depth(w * h, 1.0f) {}

void Framebuffer::clear(float depthClear) {
    // Match Python renderer background: (0.05, 0.05, 0.08)
    for (int i = 0; i < width * height; i++) {
        color[i * 4 + 0] = 13;
        color[i * 4 + 1] = 13;
        color[i * 4 + 2] = 20;
        color[i * 4 + 3] = 255;
    }
    std::fill(depth.begin(), depth.end(), depthClear);
}

std::vector<glm::mat4> computeWorldMatrices(const VRMModel& model) {
    int n = static_cast<int>(model.nodes.size());
    std::vector<glm::mat4> world(n, glm::mat4(std::numeric_limits<float>::quiet_NaN()));

    // Recursive helper
    std::function<void(int)> compute = [&](int i) {
        if (!std::isnan(world[i][0][0])) return; // already computed
        glm::mat4 local = glm::translate(glm::mat4(1), model.nodes[i].translation)
                        * glm::mat4_cast(model.nodes[i].rotation)
                        * glm::scale(glm::mat4(1), model.nodes[i].scale);
        if (model.nodes[i].parent >= 0) {
            compute(model.nodes[i].parent);
            world[i] = world[model.nodes[i].parent] * local;
        } else {
            world[i] = local;
        }
    };

    for (int i = 0; i < n; i++)
        compute(i);

    return world;
}

std::vector<glm::mat4> computeJointMatrices(
    const VRMModel& model,
    const std::vector<glm::mat4>& worldMatrices)
{
    int jc = static_cast<int>(model.jointNodes.size());
    std::vector<glm::mat4> joints(jc);
    for (int i = 0; i < jc; i++) {
        int nodeIdx = model.jointNodes[i];
        glm::mat4 ibm = (i < (int)model.inverseBindMatrices.size())
                        ? model.inverseBindMatrices[i] : glm::mat4(1);
        joints[i] = worldMatrices[nodeIdx] * ibm;
    }
    return joints;
}

std::vector<ProcessedMesh> processVertices(
    const VRMModel& model,
    const std::vector<glm::mat4>& jointMatrices,
    const glm::mat4& viewProj,
    const std::vector<float>& morphWeights,
    int fbWidth, int fbHeight)
{
    std::vector<ProcessedMesh> result(model.meshes.size());

    for (size_t mi = 0; mi < model.meshes.size(); mi++) {
        const auto& mesh = model.meshes[mi];
        result[mi].prims.resize(mesh.primitives.size());

        for (size_t pi = 0; pi < mesh.primitives.size(); pi++) {
            const auto& prim = mesh.primitives[pi];
            int vc = prim.vertexCount();
            auto& pp = result[mi].prims[pi];
            pp.prim = &prim;
            pp.screenX.resize(vc);
            pp.screenY.resize(vc);
            pp.clipZ.resize(vc);
            pp.uvU.resize(vc);
            pp.uvV.resize(vc);
            pp.worldNX.resize(vc);
            pp.worldNY.resize(vc);
            pp.worldNZ.resize(vc);

            for (int v = 0; v < vc; v++) {
                // --- 1. Morph targets ---
                glm::vec3 pos(prim.positions[v * 3], prim.positions[v * 3 + 1], prim.positions[v * 3 + 2]);
                if (prim.morphCount > 0) {
                    int weightBase = 0; // For now, we pass weights for mesh0 only
                    // Find the correct weight offset: sum of all prior meshes' morph counts
                    for (size_t mj = 0; mj < mi; mj++) {
                        if (!model.meshes[mj].primitives.empty())
                            weightBase += model.meshes[mj].primitives[0].morphCount;
                    }
                    for (int t = 0; t < prim.morphCount; t++) {
                        float w = morphWeights[weightBase + t];
                        if (w > 0.0f) {
                            float dx = prim.morphDeltas[(t * vc + v) * 3 + 0];
                            float dy = prim.morphDeltas[(t * vc + v) * 3 + 1];
                            float dz = prim.morphDeltas[(t * vc + v) * 3 + 2];
                            pos.x += dx * w;
                            pos.y += dy * w;
                            pos.z += dz * w;
                        }
                    }
                }

                // --- 2. Skinning (4-bone LBS) ---
                glm::vec4 skinned(0);
                if (!prim.weights.empty() && !jointMatrices.empty()) {
                    uint16_t j0 = prim.joints[v * 4 + 0];
                    uint16_t j1 = prim.joints[v * 4 + 1];
                    uint16_t j2 = prim.joints[v * 4 + 2];
                    uint16_t j3 = prim.joints[v * 4 + 3];
                    float w0 = prim.weights[v * 4 + 0];
                    float w1 = prim.weights[v * 4 + 1];
                    float w2 = prim.weights[v * 4 + 2];
                    float w3 = prim.weights[v * 4 + 3];

                    glm::vec4 hp(pos, 1.0f);
                    skinned = (jointMatrices[j0] * hp) * w0;
                    skinned += (jointMatrices[j1] * hp) * w1;
                    skinned += (jointMatrices[j2] * hp) * w2;
                    skinned += (jointMatrices[j3] * hp) * w3;
                } else {
                    skinned = glm::vec4(pos, 1.0f);
                }

                // --- 3. Model matrix (node world) ---
                // The mesh node's world transform is already baked into jointMatrices for skinned meshes.
                // For non-skinned meshes, we need the node world matrix.
                // For skinned meshes, the skin joint matrices already include the world transform,
                // so the model matrix is identity. For non-skinned, use node world.
                // (In practice, VRM meshes are always skinned, so this is identity.)

                // --- 4. View-Projection ---
                glm::vec4 clip = viewProj * skinned;
                float w = clip.w;
                if (w <= 0.0f) w = 1e-10f; // Avoid division by zero
                float invW = 1.0f / w;

                // NDC
                float ndcX = clip.x * invW;
                float ndcY = clip.y * invW;
                float ndcZ = clip.z * invW;

                // Screen space (Y flipped: top = 0)
                pp.screenX[v] = (ndcX * 0.5f + 0.5f) * fbWidth;
                pp.screenY[v] = (1.0f - (ndcY * 0.5f + 0.5f)) * fbHeight;
                pp.clipZ[v] = ndcZ;

                // UVs (pass through, no transform)
                pp.uvU[v] = prim.uvs.empty() ? 0 : prim.uvs[v * 2];
                pp.uvV[v] = prim.uvs.empty() ? 0 : prim.uvs[v * 2 + 1];

                // Skinned normal
                if (!prim.normals.empty() && !prim.weights.empty() && !jointMatrices.empty()) {
                    uint16_t j0 = prim.joints[v * 4 + 0];
                    uint16_t j1 = prim.joints[v * 4 + 1];
                    uint16_t j2 = prim.joints[v * 4 + 2];
                    uint16_t j3 = prim.joints[v * 4 + 3];
                    float w0 = prim.weights[v * 4 + 0];
                    float w1 = prim.weights[v * 4 + 1];
                    float w2 = prim.weights[v * 4 + 2];
                    float w3 = prim.weights[v * 4 + 3];
                    glm::vec3 n(prim.normals[v * 3], prim.normals[v * 3 + 1], prim.normals[v * 3 + 2]);
                    glm::vec4 hn(n, 0.0f);
                    glm::vec4 sn = (jointMatrices[j0] * hn) * w0 + (jointMatrices[j1] * hn) * w1
                                 + (jointMatrices[j2] * hn) * w2 + (jointMatrices[j3] * hn) * w3;
                    pp.worldNX[v] = sn.x;
                    pp.worldNY[v] = sn.y;
                    pp.worldNZ[v] = sn.z;
                } else if (!prim.normals.empty()) {
                    pp.worldNX[v] = prim.normals[v * 3];
                    pp.worldNY[v] = prim.normals[v * 3 + 1];
                    pp.worldNZ[v] = prim.normals[v * 3 + 2];
                } else {
                    pp.worldNX[v] = 0; pp.worldNY[v] = 1; pp.worldNZ[v] = 0;
                }
            }
        }
    }
    return result;
}

// Rasterize a single triangle. xMin/xMax/yMin/yMax are inclusive clip bounds.
static inline void rasterizeTri(
    const ProcessedPrim& pp,
    const MeshPrimitive& prim,
    int triIdx,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int xMin, int xMax, int yMin, int yMax,
    bool depthTest, bool depthWrite, bool doubleSided)
{
    const uint32_t* indices = prim.indices.data();
    uint32_t i0 = indices[triIdx * 3 + 0];
    uint32_t i1 = indices[triIdx * 3 + 1];
    uint32_t i2 = indices[triIdx * 3 + 2];

    float x0 = pp.screenX[i0], y0 = pp.screenY[i0], z0 = pp.clipZ[i0];
    float x1 = pp.screenX[i1], y1 = pp.screenY[i1], z1 = pp.clipZ[i1];
    float x2 = pp.screenX[i2], y2 = pp.screenY[i2], z2 = pp.clipZ[i2];
    float u0 = pp.uvU[i0], v0uv = pp.uvV[i0];
    float u1 = pp.uvU[i1], v1uv = pp.uvV[i1];
    float u2 = pp.uvU[i2], v2uv = pp.uvV[i2];

    // World-space normals (unnormalized; normalized per-pixel)
    float n0x = pp.worldNX[i0], n0y = pp.worldNY[i0], n0z = pp.worldNZ[i0];
    float n1x = pp.worldNX[i1], n1y = pp.worldNY[i1], n1z = pp.worldNZ[i1];
    float n2x = pp.worldNX[i2], n2y = pp.worldNY[i2], n2z = pp.worldNZ[i2];

    // Backface culling. Python uses flip_y=True (negated proj[1][1]) + front_face='ccw'.
    // With flip_y, CW-world triangles become CCW in NDC → front-facing → rendered.
    // In C++'s Y-down screen space, CW-world → area > 0, CCW-world → area < 0.
    // So: cull area < 0 (CCW-world = back-facing in Python's flip_y setup).
    float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (area < 0) {           // back-facing for single-sided meshes
        if (!doubleSided) return;
        // Double-sided: flip winding to make area positive
        std::swap(x1, x2); std::swap(y1, y2); std::swap(z1, z2);
        std::swap(u1, u2); std::swap(v1uv, v2uv);
        std::swap(n1x, n2x); std::swap(n1y, n2y); std::swap(n1z, n2z);
        area = -area;
    }
    if (area < 0.01f) return;

    // Bounding box: convert to int first, then clip to [xMin,xMax]×[yMin,yMax]
    int ix0 = std::max(xMin, static_cast<int>(std::min({x0, x1, x2})));
    int ix1 = std::min(xMax, static_cast<int>(std::max({x0, x1, x2})));
    int iy0 = std::max(yMin, static_cast<int>(std::min({y0, y1, y2})));
    int iy1 = std::min(yMax, static_cast<int>(std::max({y0, y1, y2})));
    if (iy1 < iy0 || ix1 < ix0) return;

    float invArea = 1.0f / area;
    bool hasTex = prim.textureIndex >= 0 && prim.textureIndex < (int)textures.size();
    const TextureData* tex = hasTex ? &textures[prim.textureIndex] : nullptr;

    const uint8_t* texPixels = nullptr;
    int texW = 0, texH = 0;
    if (tex) {
        texPixels = tex->pixels.data();
        texW = tex->width;
        texH = tex->height;
    }

    // Precompute baseColor * ambient for no-lighting fast path
    float bcr = prim.baseColor.r, bcg = prim.baseColor.g, bcb = prim.baseColor.b;

    // --- Incremental edge functions ---
    const float dE0dx = y1 - y2, dE0dy = x2 - x1;
    const float dE1dx = y2 - y0, dE1dy = x0 - x2;
    const float dE2dx = y0 - y1, dE2dy = x1 - x0;

    // Initialize edge values at top-left pixel center
    float fx0 = ix0 + 0.5f, fy0 = iy0 + 0.5f;
    float e0_row = (x2 - x1) * (fy0 - y1) - (y2 - y1) * (fx0 - x1);
    float e1_row = (x0 - x2) * (fy0 - y2) - (y0 - y2) * (fx0 - x2);
    float e2_row = (x1 - x0) * (fy0 - y0) - (y1 - y0) * (fx0 - x0);

    int rowOffset = iy0 * fb.width;
    for (int py = iy0; py <= iy1; py++) {
        float e0 = e0_row, e1 = e1_row, e2 = e2_row;
        int pixIdx = rowOffset + ix0;
        for (int px = ix0; px <= ix1; px++) {
            if (e0 >= 0 && e1 >= 0 && e2 >= 0) {
                float b0 = e0 * invArea;
                float b1 = e1 * invArea;
                float b2 = e2 * invArea;
                float z = b0 * z0 + b1 * z1 + b2 * z2;

                if (!depthTest || z < fb.depth[pixIdx]) {
                    if (depthWrite) fb.depth[pixIdx] = z;

                    // Lambertian diffuse lighting
                    float nx = b0 * n0x + b1 * n1x + b2 * n2x;
                    float ny = b0 * n0y + b1 * n1y + b2 * n2y;
                    float nz = b0 * n0z + b1 * n1z + b2 * n2z;
                    float nlen2 = nx * nx + ny * ny + nz * nz;
                    float diff = 0.0f;
                    if (nlen2 > 1e-12f) {
                        float dot = (nx * LX + ny * LY + nz * LZ) / sqrtf(nlen2);
                        if (dot > 0.0f) diff = dot;
                    }
                    float lit = 0.35f + 0.65f * diff;
                    float fr = bcr * lit;
                    float fg = bcg * lit;
                    float fb_ = bcb * lit;

                    if (texPixels) {
                        float u = b0 * u0 + b1 * u1 + b2 * u2;
                        float v = b0 * v0uv + b1 * v1uv + b2 * v2uv;
                        // Bilinear filtering (GL_LINEAR / CLAMP_TO_EDGE)
                        float fx = u * texW - 0.5f;
                        float fy = v * texH - 0.5f;
                        int tx0 = (int)floorf(fx);
                        int ty0 = (int)floorf(fy);
                        float wx = fx - tx0;
                        float wy = fy - ty0;
                        int tx1 = tx0 + 1, ty1 = ty0 + 1;
                        tx0 = std::max(0, std::min(tx0, texW - 1));
                        tx1 = std::max(0, std::min(tx1, texW - 1));
                        ty0 = std::max(0, std::min(ty0, texH - 1));
                        ty1 = std::max(0, std::min(ty1, texH - 1));
                        const uint8_t *t00 = &texPixels[(ty0 * texW + tx0) * 4];
                        const uint8_t *t01 = &texPixels[(ty0 * texW + tx1) * 4];
                        const uint8_t *t10 = &texPixels[(ty1 * texW + tx0) * 4];
                        const uint8_t *t11 = &texPixels[(ty1 * texW + tx1) * 4];
                        float w00 = (1 - wx) * (1 - wy);
                        float w01 = wx * (1 - wy);
                        float w10 = (1 - wx) * wy;
                        float w11 = wx * wy;
                        float tr = t00[0] * w00 + t01[0] * w01 + t10[0] * w10 + t11[0] * w11;
                        float tg = t00[1] * w00 + t01[1] * w01 + t10[1] * w10 + t11[1] * w11;
                        float tb = t00[2] * w00 + t01[2] * w01 + t10[2] * w10 + t11[2] * w11;
                        fb.color[pixIdx * 4 + 0] = (uint8_t)std::min(255.0f, tr * fr);
                        fb.color[pixIdx * 4 + 1] = (uint8_t)std::min(255.0f, tg * fg);
                        fb.color[pixIdx * 4 + 2] = (uint8_t)std::min(255.0f, tb * fb_);
                    } else {
                        fb.color[pixIdx * 4 + 0] = (uint8_t)std::min(255.0f, fr * 255.0f);
                        fb.color[pixIdx * 4 + 1] = (uint8_t)std::min(255.0f, fg * 255.0f);
                        fb.color[pixIdx * 4 + 2] = (uint8_t)std::min(255.0f, fb_ * 255.0f);
                    }
                    fb.color[pixIdx * 4 + 3] = 255;
                }
            }
            e0 += dE0dx; e1 += dE1dx; e2 += dE2dx;
            pixIdx++;
        }
        e0_row += dE0dy; e1_row += dE1dy; e2_row += dE2dy;
        rowOffset += fb.width;
    }
}

static void rasterizePrim(
    const ProcessedPrim& pp,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int yMin, int yMax,
    bool depthTest, bool depthWrite, bool doubleSided)
{
    const MeshPrimitive& prim = *pp.prim;
    int numTris = prim.triangleCount();
    for (int t = 0; t < numTris; t++)
        rasterizeTri(pp, prim, t, fb, textures, 0, fb.width - 1, yMin, yMax - 1,
                     depthTest, depthWrite, doubleSided);
}

// ---------------------------------------------------------------------------
// Parallel vertex processing
// ---------------------------------------------------------------------------
std::vector<ProcessedMesh> processVerticesParallel(
    const VRMModel& model,
    const std::vector<glm::mat4>& jointMatrices,
    const glm::mat4& viewProj,
    const std::vector<float>& morphWeights,
    int fbWidth, int fbHeight,
    int numThreads)
{
    std::vector<ProcessedMesh> result(model.meshes.size());

    // --- Phase 1: Pre-allocate + build flat arrays for cache-friendly access ---
    struct PrimInfo {
        const MeshPrimitive* prim;
        ProcessedPrim* pp;
        int weightBase;
    };
    std::vector<PrimInfo> primsFlat;

    for (size_t mi = 0; mi < model.meshes.size(); mi++) {
        const auto& mesh = model.meshes[mi];
        result[mi].prims.resize(mesh.primitives.size());
        int weightBase = 0;
        for (size_t mj = 0; mj < mi; mj++) {
            if (!model.meshes[mj].primitives.empty())
                weightBase += model.meshes[mj].primitives[0].morphCount;
        }
        for (size_t pi = 0; pi < mesh.primitives.size(); pi++) {
            const auto& prim = mesh.primitives[pi];
            int vc = prim.vertexCount();
            auto& pp = result[mi].prims[pi];
            pp.prim = &prim;
            pp.screenX.resize(vc);
            pp.screenY.resize(vc);
            pp.clipZ.resize(vc);
            pp.uvU.resize(vc);
            pp.uvV.resize(vc);
            pp.worldNX.resize(vc);
            pp.worldNY.resize(vc);
            pp.worldNZ.resize(vc);
            primsFlat.push_back({&prim, &pp, weightBase});
        }
    }

    const int numPrims = static_cast<int>(primsFlat.size());
    const glm::mat4* jm = jointMatrices.data();

    // --- Phase 2: Parallel vertex processing with lock-free work queue ---
    auto processPrimVertices = [&](const PrimInfo& info, int v) {
        const MeshPrimitive& prim = *info.prim;
        ProcessedPrim& pp = *info.pp;
        int vc = prim.vertexCount();

        glm::vec3 pos(prim.positions[v * 3], prim.positions[v * 3 + 1], prim.positions[v * 3 + 2]);
        if (prim.morphCount > 0) {
            for (int t = 0; t < prim.morphCount; t++) {
                float w = morphWeights[info.weightBase + t];
                if (w > 0.0f) {
                    const float* dp = &prim.morphDeltas[(t * vc + v) * 3];
                    pos.x += dp[0] * w;
                    pos.y += dp[1] * w;
                    pos.z += dp[2] * w;
                }
            }
        }

        const uint16_t* j = &prim.joints[v * 4];
        const float* wt = &prim.weights[v * 4];
        glm::vec4 hp(pos, 1.0f);
        glm::vec4 skinned = (jm[j[0]] * hp) * wt[0];
        skinned += (jm[j[1]] * hp) * wt[1];
        skinned += (jm[j[2]] * hp) * wt[2];
        skinned += (jm[j[3]] * hp) * wt[3];

        glm::vec4 clip = viewProj * skinned;
        float w = clip.w;
        if (w <= 0.0f) w = 1e-10f;
        float invW = 1.0f / w;

        pp.screenX[v] = (clip.x * invW * 0.5f + 0.5f) * fbWidth;
        pp.screenY[v] = (1.0f - (clip.y * invW * 0.5f + 0.5f)) * fbHeight;
        pp.clipZ[v] = clip.z * invW;
        pp.uvU[v] = prim.uvs.empty() ? 0 : prim.uvs[v * 2];
        pp.uvV[v] = prim.uvs.empty() ? 0 : prim.uvs[v * 2 + 1];

        // Skinned normal (w=0 ignores joint translation)
        if (!prim.normals.empty()) {
            glm::vec3 n(prim.normals[v * 3], prim.normals[v * 3 + 1], prim.normals[v * 3 + 2]);
            glm::vec4 hn(n, 0.0f);
            glm::vec4 sn = (jm[j[0]] * hn) * wt[0] + (jm[j[1]] * hn) * wt[1]
                         + (jm[j[2]] * hn) * wt[2] + (jm[j[3]] * hn) * wt[3];
            pp.worldNX[v] = sn.x;
            pp.worldNY[v] = sn.y;
            pp.worldNZ[v] = sn.z;
        } else {
            pp.worldNX[v] = 0; pp.worldNY[v] = 1; pp.worldNZ[v] = 0;
        }
    };

    if (numThreads <= 1) {
        for (int pi = 0; pi < numPrims; pi++) {
            int vc = primsFlat[pi].prim->vertexCount();
            for (int v = 0; v < vc; v++)
                processPrimVertices(primsFlat[pi], v);
        }
    } else {
        // Build flat vertex work queue: (primIdx, vertexIdx) pairs
        struct VertTask { int16_t primIdx; int32_t vertIdx; };
        std::vector<VertTask> tasks;
        const int CHUNK = 512;
        for (int pi = 0; pi < numPrims; pi++) {
            int vc = primsFlat[pi].prim->vertexCount();
            for (int v = 0; v < vc; v += CHUNK)
                tasks.push_back({(int16_t)pi, v});
        }

        std::atomic<int> next{0};
        int numTasks = static_cast<int>(tasks.size());

        auto worker = [&]() {
            while (true) {
                int ti = next.fetch_add(1, std::memory_order_relaxed);
                if (ti >= numTasks) break;
                auto& task = tasks[ti];
                int vc = primsFlat[task.primIdx].prim->vertexCount();
                int end = std::min(task.vertIdx + CHUNK, vc);
                for (int v = task.vertIdx; v < end; v++)
                    processPrimVertices(primsFlat[task.primIdx], v);
            }
        };

        auto& pool = getThreadPool();
        for (int t = 0; t < numThreads; t++)
            pool.detach_task(worker);
        pool.wait();
    }

    return result;
}

// ---------------------------------------------------------------------------
// Parallel rasterization with VRM render ordering (body → hair → face)
// ---------------------------------------------------------------------------
struct PrimEntry {
    const ProcessedPrim* pp;
    int renderOrder;
    int faceSortKey;
    int origIndex;
};

// Tile-based binning + parallel rasterization for a subset of prims.
static void rasterizePassTiles(
    const PrimEntry* entries, int startIdx, int endIdx,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int numThreads,
    bool depthTest, bool depthWrite, bool doubleSided)
{
    const int TILE = 64;
    int tilesX = (fb.width + TILE - 1) / TILE;
    int tilesY = (fb.height + TILE - 1) / TILE;
    int numTiles = tilesX * tilesY;

    struct TriRef { const ProcessedPrim* pp; int triIdx; };
    std::vector<std::vector<TriRef>> tileBins(numTiles);

    // Binning pass: assign each triangle to overlapping tiles
    for (int k = startIdx; k < endIdx; k++) {
        const auto& pp = *entries[k].pp;
        const auto& prim = *pp.prim;
        const auto& indices = prim.indices;
        int numTris = (int)indices.size() / 3;
        for (int t = 0; t < numTris; t++) {
            uint32_t i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
            float minY = std::min({pp.screenY[i0], pp.screenY[i1], pp.screenY[i2]});
            float maxY = std::max({pp.screenY[i0], pp.screenY[i1], pp.screenY[i2]});
            float minX = std::min({pp.screenX[i0], pp.screenX[i1], pp.screenX[i2]});
            float maxX = std::max({pp.screenX[i0], pp.screenX[i1], pp.screenX[i2]});
            if (maxX < 0 || minX >= fb.width || maxY < 0 || minY >= fb.height) continue;
            int tx0 = std::max(0, (int)minX / TILE);
            int tx1 = std::min(tilesX - 1, (int)maxX / TILE);
            int ty0 = std::max(0, (int)minY / TILE);
            int ty1 = std::min(tilesY - 1, (int)maxY / TILE);
            for (int ty = ty0; ty <= ty1; ty++)
                for (int tx = tx0; tx <= tx1; tx++)
                    tileBins[ty * tilesX + tx].push_back({entries[k].pp, t});
        }
    }

    // Rasterize pass: atomic tile work queue
    std::atomic<int> next{0};
    auto worker = [&]() {
        while (true) {
            int tileIdx = next.fetch_add(1, std::memory_order_relaxed);
            if (tileIdx >= numTiles) break;
            int tx = tileIdx % tilesX;
            int ty = tileIdx / tilesX;
            int xStart = tx * TILE;
            int yStart = ty * TILE;
            int xEnd = std::min(xStart + TILE, fb.width);
            int yEnd = std::min(yStart + TILE, fb.height);
            auto& bin = tileBins[tileIdx];
            for (auto& tr : bin)
                rasterizeTri(*tr.pp, *tr.pp->prim, tr.triIdx, fb, textures,
                             xStart, xEnd - 1, yStart, yEnd - 1,
                             depthTest, depthWrite, doubleSided);
        }
    };

    auto& pool = getThreadPool();
    for (int t = 0; t < numThreads; t++)
        pool.detach_task(worker);
    pool.wait();
}

void rasterizeParallel(
    const std::vector<ProcessedMesh>& processed,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int numThreads)
{
    // 1. Flatten all prims with render-order metadata
    std::vector<PrimEntry> entries;
    int origIdx = 0;
    for (const auto& mesh : processed)
        for (const auto& prim : mesh.prims) {
            int rorder = computeRenderOrder(*prim.prim);
            int fskey = computeFaceSortKey(prim.prim->matName);
            entries.push_back({&prim, rorder, fskey, origIdx++});
        }
    int numEntries = (int)entries.size();
    if (numEntries == 0) return;

    // 2. Sort: (renderOrder, faceSortKey for face parts, original index)
    std::sort(entries.begin(), entries.end(), [](const PrimEntry& a, const PrimEntry& b) {
        if (a.renderOrder != b.renderOrder) return a.renderOrder < b.renderOrder;
        int ka = (a.renderOrder == 2) ? a.faceSortKey : 0;
        int kb = (b.renderOrder == 2) ? b.faceSortKey : 0;
        if (ka != kb) return ka < kb;
        return a.origIndex < b.origIndex;
    });

    // 3. Process contiguous render-order groups as sequential passes
    //    Body (0): depth test ON, no culling
    //    Hair (1): depth test ON, cull backfaces
    //    Face (2): depth test OFF (painter's algorithm), sorted by face part
    int i = 0;
    while (i < numEntries) {
        int currentOrder = entries[i].renderOrder;
        int j = i;
        while (j < numEntries && entries[j].renderOrder == currentOrder)
            j++;

        bool depthTest = (currentOrder != 2);
        bool depthWrite = (currentOrder != 2);
        bool doubleSided = (currentOrder == 0);

        if (numThreads <= 1) {
            for (int k = i; k < j; k++)
                rasterizePrim(*entries[k].pp, fb, textures, 0, 0x7FFFFFFF,
                              depthTest, depthWrite, doubleSided);
        } else {
            rasterizePassTiles(entries.data(), i, j, fb, textures, numThreads,
                               depthTest, depthWrite, doubleSided);
        }

        i = j;
    }
}
