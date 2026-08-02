#include "rasterizer.h"
#include "BS_thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <atomic>

// Persistent thread pool — created once, reused every frame
static BS::thread_pool<>& getThreadPool() {
    static BS::thread_pool<> pool;
    return pool;
}

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

// Rasterize a single triangle. xMin/xMax/yMin/yMax are inclusive clip bounds.
static inline void rasterizeTri(
    const ProcessedPrim& pp,
    const MeshPrimitive& prim,
    int triIdx,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int xMin, int xMax, int yMin, int yMax)
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

    // Backface culling (CCW front-facing)
    float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (area < 0) {
        if (!prim.doubleSided) return;
        std::swap(x1, x2); std::swap(y1, y2); std::swap(z1, z2);
        std::swap(u1, u2); std::swap(v1uv, v2uv);
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

    for (int py = iy0; py <= iy1; py++) {
        for (int px = ix0; px <= ix1; px++) {
            float fx = px + 0.5f;
            float fy = py + 0.5f;

            float e0 = (x2 - x1) * (fy - y1) - (y2 - y1) * (fx - x1);
            float e1 = (x0 - x2) * (fy - y2) - (y0 - y2) * (fx - x2);
            float e2 = (x1 - x0) * (fy - y0) - (y1 - y0) * (fx - x0);

            if (e0 < 0 || e1 < 0 || e2 < 0) continue;

            float b0 = e0 * invArea;
            float b1 = e1 * invArea;
            float b2 = e2 * invArea;
            float z = b0 * z0 + b1 * z1 + b2 * z2;

            int pixIdx = py * fb.width + px;
            if (z < fb.depth[pixIdx]) {
                fb.depth[pixIdx] = z;

                uint8_t color[4];
                if (tex) {
                    float u = b0 * u0 + b1 * u1 + b2 * u2;
                    float v = b0 * v0uv + b1 * v1uv + b2 * v2uv;
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

void rasterizePrim(
    const ProcessedPrim& pp,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int yMin, int yMax)
{
    const MeshPrimitive& prim = *pp.prim;
    int numTris = prim.triangleCount();
    for (int t = 0; t < numTris; t++)
        rasterizeTri(pp, prim, t, fb, textures, 0, fb.width - 1, yMin, yMax - 1);
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
// Parallel rasterization (horizontal bands)
// ---------------------------------------------------------------------------
void rasterizeParallel(
    const std::vector<ProcessedMesh>& processed,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int numThreads)
{
    // Flatten all prims
    const ProcessedPrim* allPrims[64];
    int numPrims = 0;
    for (size_t mi = 0; mi < processed.size(); mi++)
        for (size_t pi = 0; pi < processed[mi].prims.size() && numPrims < 64; pi++)
            allPrims[numPrims++] = &processed[mi].prims[pi];

    if (numThreads <= 1) {
        for (int i = 0; i < numPrims; i++)
            rasterizePrim(*allPrims[i], fb, textures);
        return;
    }

    // --- Tile-based parallel rasterization ---
    // 1. Bin all triangles into tiles (each tri assigned to tiles it overlaps)
    // 2. Threads pull tiles from atomic queue — only rasterize binned tris
    const int TILE = 64;
    int tilesX = (fb.width + TILE - 1) / TILE;
    int tilesY = (fb.height + TILE - 1) / TILE;
    int numTiles = tilesX * tilesY;

    struct TriRef { const ProcessedPrim* pp; int triIdx; };
    std::vector<std::vector<TriRef>> tileBins(numTiles);

    // Binning pass: assign each triangle to overlapping tiles
    for (int pi = 0; pi < numPrims; pi++) {
        const auto& pp = *allPrims[pi];
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
            for (int ty = ty0; ty <= ty1; ty++) {
                for (int tx = tx0; tx <= tx1; tx++) {
                    tileBins[ty * tilesX + tx].push_back({allPrims[pi], t});
                }
            }
        }
    }

    // Rasterize pass: atomic work queue over tiles
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
                             xStart, xEnd - 1, yStart, yEnd - 1);
        }
    };

    auto& pool = getThreadPool();
    for (int t = 0; t < numThreads; t++)
        pool.detach_task(worker);
    pool.wait();
}
