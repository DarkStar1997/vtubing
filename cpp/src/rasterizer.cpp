#include "rasterizer.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <thread>
#include <immintrin.h>

Framebuffer::Framebuffer(int w, int h)
    : width(w), height(h), color(w * h * 4, 0), depth(w * h, 1.0f) {}

void Framebuffer::clear(float depthClear) {
    std::fill(color.begin(), color.end(), 30); // dark background
    // Set alpha to 255
    for (int i = 3; i < width * height * 4; i += 4)
        color[i] = 255;
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
            }
        }
    }
    return result;
}

static inline void sampleTextureNearest(
    const TextureData& tex, float u, float v,
    uint8_t out[4])
{
    // Clamp UVs to [0,1]
    u = std::max(0.0f, std::min(1.0f, u));
    v = std::max(0.0f, std::min(1.0f, v));
    int x = static_cast<int>(u * tex.width);
    int y = static_cast<int>(v * tex.height);
    if (x >= tex.width) x = tex.width - 1;
    if (y >= tex.height) y = tex.height - 1;
    const uint8_t* px = &tex.pixels[(y * tex.width + x) * 4];
    out[0] = px[0]; out[1] = px[1]; out[2] = px[2]; out[3] = px[3];
}

void rasterizePrim(
    const ProcessedPrim& pp,
    Framebuffer& fb,
    const std::vector<TextureData>& textures)
{
    const MeshPrimitive& prim = *pp.prim;
    const uint32_t* indices = prim.indices.data();
    int numTris = prim.triangleCount();
    bool hasTex = prim.textureIndex >= 0 && prim.textureIndex < (int)textures.size();
    const TextureData* tex = hasTex ? &textures[prim.textureIndex] : nullptr;

    for (int t = 0; t < numTris; t++) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        float x0 = pp.screenX[i0], y0 = pp.screenY[i0], z0 = pp.clipZ[i0];
        float x1 = pp.screenX[i1], y1 = pp.screenY[i1], z1 = pp.clipZ[i1];
        float x2 = pp.screenX[i2], y2 = pp.screenY[i2], z2 = pp.clipZ[i2];
        float u0 = pp.uvU[i0], v0uv = pp.uvV[i0];
        float u1 = pp.uvU[i1], v1uv = pp.uvV[i1];
        float u2 = pp.uvU[i2], v2uv = pp.uvV[i2];

        // Backface culling (CCW front-facing)
        float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (area < 0) {
            if (!prim.doubleSided) continue; // backface culled
            std::swap(x1, x2); std::swap(y1, y2); std::swap(z1, z2);
            std::swap(u1, u2); std::swap(v1uv, v2uv);
            area = -area;
        }
        if (area < 0.01f) continue;

        // Bounding box (clipped to screen)
        float minX = std::max(0.0f, std::min({x0, x1, x2}));
        float maxX = std::min((float)fb.width - 1, std::max({x0, x1, x2}));
        float minY = std::max(0.0f, std::min({y0, y1, y2}));
        float maxY = std::min((float)fb.height - 1, std::max({y0, y1, y2}));

        if (maxX < minX || maxY < minY) continue;

        int ix0 = static_cast<int>(minX);
        int ix1 = static_cast<int>(maxX);
        int iy0 = static_cast<int>(minY);
        int iy1 = static_cast<int>(maxY);

        float invArea = 1.0f / area;

        for (int py = iy0; py <= iy1; py++) {
            for (int px = ix0; px <= ix1; px++) {
                float fx = px + 0.5f;
                float fy = py + 0.5f;

                // Barycentric weights via edge functions
                // E(A,B,P) = (Bx-Ax)*(Py-Ay) - (By-Ay)*(Px-Ax)
                float e0 = (x2 - x1) * (fy - y1) - (y2 - y1) * (fx - x1);
                float e1 = (x0 - x2) * (fy - y2) - (y0 - y2) * (fx - x2);
                float e2 = (x1 - x0) * (fy - y0) - (y1 - y0) * (fx - x0);

                if (e0 < 0 || e1 < 0 || e2 < 0) continue;

                float b0 = e0 * invArea;
                float b1 = e1 * invArea;
                float b2 = e2 * invArea;

                // Interpolate Z
                float z = b0 * z0 + b1 * z1 + b2 * z2;

                int pixIdx = py * fb.width + px;
                if (z < fb.depth[pixIdx]) {
                    fb.depth[pixIdx] = z;

                    uint8_t color[4];
                    if (tex) {
                        float u = b0 * u0 + b1 * u1 + b2 * u2;
                        float v = b0 * v0uv + b1 * v1uv + b2 * v2uv;
                        // VRM textures typically use V-flipped UVs, but stb_image loads top-to-bottom
                        // For now, use V as-is
                        sampleTextureNearest(*tex, u, v, color);
                    } else {
                        color[0] = (uint8_t)(prim.baseColor.r * 255);
                        color[1] = (uint8_t)(prim.baseColor.g * 255);
                        color[2] = (uint8_t)(prim.baseColor.b * 255);
                        color[3] = 255;
                    }
                    fb.color[pixIdx * 4 + 0] = color[0];
                    fb.color[pixIdx * 4 + 1] = color[1];
                    fb.color[pixIdx * 4 + 2] = color[2];
                    fb.color[pixIdx * 4 + 3] = color[3];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 SIMD rasterizer — processes 8 pixels per iteration
// ---------------------------------------------------------------------------
void rasterizePrimSIMD(
    const ProcessedPrim& pp,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int yMin, int yMax)
{
    const MeshPrimitive& prim = *pp.prim;
    const uint32_t* indices = prim.indices.data();
    int numTris = prim.triangleCount();
    bool hasTex = prim.textureIndex >= 0 && prim.textureIndex < (int)textures.size();
    const TextureData* tex = hasTex ? &textures[prim.textureIndex] : nullptr;

    // Constants
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one  = _mm256_set1_ps(1.0f);
    const __m256i v_inc = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    const __m256 v_inc_f = _mm256_cvtepi32_ps(v_inc);

    for (int t = 0; t < numTris; t++) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        float x0 = pp.screenX[i0], y0 = pp.screenY[i0], z0 = pp.clipZ[i0];
        float x1 = pp.screenX[i1], y1 = pp.screenY[i1], z1 = pp.clipZ[i1];
        float x2 = pp.screenX[i2], y2 = pp.screenY[i2], z2 = pp.clipZ[i2];
        float u0 = pp.uvU[i0], v0uv = pp.uvV[i0];
        float u1 = pp.uvU[i1], v1uv = pp.uvV[i1];
        float u2 = pp.uvU[i2], v2uv = pp.uvV[i2];

        // Backface culling
        float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (area < 0) {
            if (!prim.doubleSided) continue;
            std::swap(x1, x2); std::swap(y1, y2); std::swap(z1, z2);
            std::swap(u1, u2); std::swap(v1uv, v2uv);
            area = -area;
        }
        if (area < 0.01f) continue;

        float minX = std::max(0.0f, std::min({x0, x1, x2}));
        float maxX = std::min((float)fb.width - 1, std::max({x0, x1, x2}));
        float minY = std::max(0.0f, std::min({y0, y1, y2}));
        float maxY = std::min((float)fb.height - 1, std::max({y0, y1, y2}));
        if (maxX < minX || maxY < minY) continue;

        int ix0 = static_cast<int>(minX);
        int ix1 = static_cast<int>(maxX);
        int iy0 = std::max(static_cast<int>(minY), yMin);
        int iy1 = std::min(static_cast<int>(maxY), yMax - 1);
        if (iy1 < iy0) continue;

        float invArea = 1.0f / area;

        // Edge function is linear in fx: e(fx) = C + D*fx
        // de0/dx = -(y2-y1), etc.
        float step0 = -(y2 - y1);
        float step1 = -(y0 - y2);
        float step2 = -(y1 - y0);

        __m256 v_step0 = _mm256_set1_ps(step0);
        __m256 v_step1 = _mm256_set1_ps(step1);
        __m256 v_step2 = _mm256_set1_ps(step2);
        __m256 v_invArea = _mm256_set1_ps(invArea);
        __m256 v_z0 = _mm256_set1_ps(z0);
        __m256 v_z1 = _mm256_set1_ps(z1);
        __m256 v_z2 = _mm256_set1_ps(z2);

        // UV broadcast
        __m256 v_u0 = _mm256_set1_ps(u0), v_u1 = _mm256_set1_ps(u1), v_u2 = _mm256_set1_ps(u2);
        __m256 v_v0 = _mm256_set1_ps(v0uv), v_v1 = _mm256_set1_ps(v1uv), v_v2 = _mm256_set1_ps(v2uv);

        // Texture constants
        __m256 v_tw = tex ? _mm256_set1_ps((float)tex->width) : v_zero;
        __m256 v_th = tex ? _mm256_set1_ps((float)tex->height) : v_zero;
        __m256i v_tw_m1 = tex ? _mm256_set1_epi32(tex->width - 1) : _mm256_setzero_si256();
        __m256i v_th_m1 = tex ? _mm256_set1_epi32(tex->height - 1) : _mm256_setzero_si256();
        __m256i v_tw_int = tex ? _mm256_set1_epi32(tex->width) : _mm256_setzero_si256();
        const int* tex_data = tex ? reinterpret_cast<const int*>(tex->pixels.data()) : nullptr;

        for (int py = iy0; py <= iy1; py++) {
            float fy = py + 0.5f;
            int rowBase = py * fb.width;

            // --- SIMD blocks (8 pixels each) ---
            int bx = ix0;
            for (; bx + 7 <= ix1; bx += 8) {
                float fxBase = bx + 0.5f;

                // Edge function at first pixel center
                float e0b = (x2 - x1) * (fy - y1) - (y2 - y1) * (fxBase - x1);
                float e1b = (x0 - x2) * (fy - y2) - (y0 - y2) * (fxBase - x2);
                float e2b = (x1 - x0) * (fy - y0) - (y1 - y0) * (fxBase - x0);

                // Compute 8 edge values: base + step * {0..7}
                __m256 v_e0 = _mm256_fmadd_ps(v_step0, v_inc_f, _mm256_set1_ps(e0b));
                __m256 v_e1 = _mm256_fmadd_ps(v_step1, v_inc_f, _mm256_set1_ps(e1b));
                __m256 v_e2 = _mm256_fmadd_ps(v_step2, v_inc_f, _mm256_set1_ps(e2b));

                // Coverage mask: all edges >= 0
                __m256 mask = _mm256_and_ps(
                    _mm256_cmp_ps(v_e0, v_zero, _CMP_GE_OQ),
                    _mm256_and_ps(
                        _mm256_cmp_ps(v_e1, v_zero, _CMP_GE_OQ),
                        _mm256_cmp_ps(v_e2, v_zero, _CMP_GE_OQ)));
                int mmask = _mm256_movemask_ps(mask);
                if (!mmask) continue;

                // Barycentric weights
                __m256 v_b0 = _mm256_mul_ps(v_e0, v_invArea);
                __m256 v_b1 = _mm256_mul_ps(v_e1, v_invArea);
                __m256 v_b2 = _mm256_mul_ps(v_e2, v_invArea);

                // Interpolate Z
                __m256 v_z = _mm256_fmadd_ps(v_b0, v_z0,
                               _mm256_fmadd_ps(v_b1, v_z1,
                                      _mm256_mul_ps(v_b2, v_z2)));

                // Z-test via gather
                __m256i v_pixIdx = _mm256_add_epi32(_mm256_set1_epi32(rowBase + bx), v_inc);
                __m256 v_depth = _mm256_i32gather_ps(&fb.depth[0], v_pixIdx, 4);
                mask = _mm256_and_ps(mask, _mm256_cmp_ps(v_z, v_depth, _CMP_LT_OQ));
                mmask = _mm256_movemask_ps(mask);
                if (!mmask) continue;

                // Write depth + color for passing lanes
                alignas(32) float z_arr[8];
                _mm256_store_ps(z_arr, v_z);

                if (tex_data) {
                    // Interpolate UV
                    __m256 v_u = _mm256_fmadd_ps(v_b0, v_u0,
                                   _mm256_fmadd_ps(v_b1, v_u1,
                                          _mm256_mul_ps(v_b2, v_u2)));
                    __m256 v_v = _mm256_fmadd_ps(v_b0, v_v0,
                                   _mm256_fmadd_ps(v_b1, v_v1,
                                          _mm256_mul_ps(v_b2, v_v2)));
                    // Clamp [0,1]
                    v_u = _mm256_min_ps(_mm256_max_ps(v_u, v_zero), v_one);
                    v_v = _mm256_min_ps(_mm256_max_ps(v_v, v_zero), v_one);
                    // Texel coords
                    __m256i v_tx = _mm256_cvttps_epi32(_mm256_mul_ps(v_u, v_tw));
                    __m256i v_ty = _mm256_cvttps_epi32(_mm256_mul_ps(v_v, v_th));
                    v_tx = _mm256_min_epi32(_mm256_max_epi32(v_tx, _mm256_setzero_si256()), v_tw_m1);
                    v_ty = _mm256_min_epi32(_mm256_max_epi32(v_ty, _mm256_setzero_si256()), v_th_m1);
                    // Texel offset = ty * width + tx
                    __m256i v_off = _mm256_add_epi32(_mm256_mullo_epi32(v_ty, v_tw_int), v_tx);
                    // Gather RGBA32 pixels
                    __m256i v_texels = _mm256_i32gather_epi32(tex_data, v_off, 4);
                    alignas(32) int tex_arr[8];
                    _mm256_store_si256((__m256i*)tex_arr, v_texels);
                    for (int lane = 0; lane < 8; lane++) {
                        if (mmask & (1 << lane)) {
                            int idx = rowBase + bx + lane;
                            fb.depth[idx] = z_arr[lane];
                            *reinterpret_cast<int*>(&fb.color[idx * 4]) = tex_arr[lane];
                        }
                    }
                } else {
                    uint8_t cr = (uint8_t)(prim.baseColor.r * 255);
                    uint8_t cg = (uint8_t)(prim.baseColor.g * 255);
                    uint8_t cb = (uint8_t)(prim.baseColor.b * 255);
                    int packed = (255 << 24) | (cb << 16) | (cg << 8) | cr;
                    for (int lane = 0; lane < 8; lane++) {
                        if (mmask & (1 << lane)) {
                            int idx = rowBase + bx + lane;
                            fb.depth[idx] = z_arr[lane];
                            *reinterpret_cast<int*>(&fb.color[idx * 4]) = packed;
                        }
                    }
                }
            }

            // --- Scalar remainder ---
            for (; bx <= ix1; bx++) {
                float fx = bx + 0.5f;
                float e0 = (x2 - x1) * (fy - y1) - (y2 - y1) * (fx - x1);
                float e1 = (x0 - x2) * (fy - y2) - (y0 - y2) * (fx - x2);
                float e2 = (x1 - x0) * (fy - y0) - (y1 - y0) * (fx - x0);
                if (e0 < 0 || e1 < 0 || e2 < 0) continue;
                float b0 = e0 * invArea, b1 = e1 * invArea, b2 = e2 * invArea;
                float z = b0 * z0 + b1 * z1 + b2 * z2;
                int idx = rowBase + bx;
                if (z >= fb.depth[idx]) continue;
                fb.depth[idx] = z;
                if (tex) {
                    float u = b0 * u0 + b1 * u1 + b2 * u2;
                    float v = b0 * v0uv + b1 * v1uv + b2 * v2uv;
                    u = std::max(0.0f, std::min(1.0f, u));
                    v = std::max(0.0f, std::min(1.0f, v));
                    int tx = (int)(u * tex->width);
                    int ty = (int)(v * tex->height);
                    if (tx >= tex->width) tx = tex->width - 1;
                    if (ty >= tex->height) ty = tex->height - 1;
                    const uint8_t* px = &tex->pixels[(ty * tex->width + tx) * 4];
                    fb.color[idx * 4 + 0] = px[0];
                    fb.color[idx * 4 + 1] = px[1];
                    fb.color[idx * 4 + 2] = px[2];
                    fb.color[idx * 4 + 3] = px[3];
                } else {
                    fb.color[idx * 4 + 0] = (uint8_t)(prim.baseColor.r * 255);
                    fb.color[idx * 4 + 1] = (uint8_t)(prim.baseColor.g * 255);
                    fb.color[idx * 4 + 2] = (uint8_t)(prim.baseColor.b * 255);
                    fb.color[idx * 4 + 3] = 255;
                }
            }
        }
    }
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
    int numMeshes = static_cast<int>(model.meshes.size());

    auto processRange = [&](int meshStart, int meshEnd) {
        for (int mi = meshStart; mi < meshEnd; mi++) {
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

                int weightBase = 0;
                for (int mj = 0; mj < mi; mj++) {
                    if (!model.meshes[mj].primitives.empty())
                        weightBase += model.meshes[mj].primitives[0].morphCount;
                }

                for (int v = 0; v < vc; v++) {
                    glm::vec3 pos(prim.positions[v * 3], prim.positions[v * 3 + 1], prim.positions[v * 3 + 2]);
                    if (prim.morphCount > 0) {
                        for (int t = 0; t < prim.morphCount; t++) {
                            float w = morphWeights[weightBase + t];
                            if (w > 0.0f) {
                                pos.x += prim.morphDeltas[(t * vc + v) * 3 + 0] * w;
                                pos.y += prim.morphDeltas[(t * vc + v) * 3 + 1] * w;
                                pos.z += prim.morphDeltas[(t * vc + v) * 3 + 2] * w;
                            }
                        }
                    }

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

                    glm::vec4 clip = viewProj * skinned;
                    float w = clip.w;
                    if (w <= 0.0f) w = 1e-10f;
                    float invW = 1.0f / w;

                    pp.screenX[v] = (clip.x * invW * 0.5f + 0.5f) * fbWidth;
                    pp.screenY[v] = (1.0f - (clip.y * invW * 0.5f + 0.5f)) * fbHeight;
                    pp.clipZ[v] = clip.z * invW;
                    pp.uvU[v] = prim.uvs.empty() ? 0 : prim.uvs[v * 2];
                    pp.uvV[v] = prim.uvs.empty() ? 0 : prim.uvs[v * 2 + 1];
                }
            }
        }
    };

    if (numThreads <= 1 || numMeshes <= 1) {
        processRange(0, numMeshes);
    } else {
        int nt = std::min(numThreads, numMeshes);
        std::vector<std::thread> threads;
        int perThread = (numMeshes + nt - 1) / nt;
        for (int t = 0; t < nt; t++) {
            int start = t * perThread;
            int end = std::min(start + perThread, numMeshes);
            if (start >= end) break;
            threads.emplace_back(processRange, start, end);
        }
        for (auto& th : threads) th.join();
    }

    return result;
}

// ---------------------------------------------------------------------------
// Parallel rasterization (horizontal bands)
// ---------------------------------------------------------------------------
void rasterizeParallel(
    const std::vector<ProcessedMesh>& processed,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int numThreads)
{
    if (numThreads <= 1) {
        for (size_t mi = 0; mi < processed.size(); mi++) {
            for (size_t pi = 0; pi < processed[mi].prims.size(); pi++) {
                rasterizePrimSIMD(processed[mi].prims[pi], fb, textures);
            }
        }
        return;
    }

    int bandHeight = (fb.height + numThreads - 1) / numThreads;

    auto rasterizeBand = [&](int yStart, int yEnd) {
        for (size_t mi = 0; mi < processed.size(); mi++) {
            for (size_t pi = 0; pi < processed[mi].prims.size(); pi++) {
                rasterizePrimSIMD(processed[mi].prims[pi], fb, textures, yStart, yEnd);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) {
        int yStart = t * bandHeight;
        int yEnd = std::min(yStart + bandHeight, fb.height);
        if (yStart >= yEnd) break;
        threads.emplace_back(rasterizeBand, yStart, yEnd);
    }
    for (auto& th : threads) th.join();
}
