#include "rasterizer.h"
#include "rasterizer_internal.h"
#include "BS_thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <atomic>

// sRGB LUT definitions (declared extern in rasterizer_internal.h)
float sRGBToLinearLUT[256];
uint8_t linearToSRGBLUT[1025];

static struct LUTInit {
    LUTInit() {
        for (int i = 0; i < 256; i++) {
            float s = i / 255.0f;
            sRGBToLinearLUT[i] = (s <= 0.04045f) ? s / 12.92f
                                 : powf((s + 0.055f) / 1.055f, 2.4f);
        }
        for (int i = 0; i <= 1024; i++) {
            float l = i / 1024.0f;
            float s = (l <= 0.0031308f) ? l * 12.92f
                      : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f;
            linearToSRGBLUT[i] = (uint8_t)std::min(255.0f, std::max(0.0f, s * 255.0f + 0.5f));
        }
    }
} _lutInit;

// Persistent thread pool — created once, reused every frame
static BS::thread_pool<>& getThreadPool() {
    static BS::thread_pool<> pool;
    return pool;
}

Framebuffer::Framebuffer(int w, int h)
    : width(w), height(h), color(w * h * 4, 0), depth(w * h, 1.0f) {}

void Framebuffer::clear(float depthClear) {
    // White background to match browser (three-vrm with transparent canvas over white page)
    memset(color.data(), 255, width * height * 4);
    std::fill(depth.begin(), depth.end(), depthClear);
}

std::vector<glm::mat4> computeWorldMatrices(const VRMModel& model) {
    int n = static_cast<int>(model.nodes.size());
    std::vector<glm::mat4> world(n);
    std::vector<bool> computed(n, false);

    // Recursive helper
    std::function<void(int)> compute = [&](int i) {
        if (computed[i]) return;
        glm::mat4 local = glm::translate(glm::mat4(1), model.nodes[i].translation)
                        * glm::mat4_cast(model.nodes[i].rotation)
                        * glm::scale(glm::mat4(1), model.nodes[i].scale);
        if (model.nodes[i].parent >= 0) {
            compute(model.nodes[i].parent);
            world[i] = world[model.nodes[i].parent] * local;
        } else {
            world[i] = local;
        }
        computed[i] = true;
    };

    for (int i = 0; i < n; i++)
        compute(i);

    return world;
}

std::vector<glm::mat4> computeWorldMatricesWithOverrides(
    const VRMModel& model,
    const std::unordered_map<int, glm::quat>& overrides)
{
    int n = static_cast<int>(model.nodes.size());
    std::vector<glm::mat4> world(n);
    std::vector<bool> computed(n, false);

    std::function<void(int)> compute = [&](int i) {
        if (computed[i]) return;
        glm::quat rot = model.nodes[i].rotation;
        auto it = overrides.find(i);
        if (it != overrides.end())
            rot = rot * it->second;
        glm::mat4 local = glm::translate(glm::mat4(1), model.nodes[i].translation)
                        * glm::mat4_cast(rot)
                        * glm::scale(glm::mat4(1), model.nodes[i].scale);
        if (model.nodes[i].parent >= 0) {
            compute(model.nodes[i].parent);
            world[i] = world[model.nodes[i].parent] * local;
        } else {
            world[i] = local;
        }
        computed[i] = true;
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

            // Precompute active morph targets
            std::vector<std::pair<const float*, float>> activeMorphs;
            if (prim.morphCount > 0) {
                for (int t = 0; t < prim.morphCount; t++) {
                    float w = morphWeights[weightBase + t];
                    if (w > 0.0f)
                        activeMorphs.push_back({&prim.morphDeltas[(size_t)t * vc * 3], w});
                }
            }

            for (int v = 0; v < vc; v++) {
                // --- 1. Morph targets ---
                glm::vec3 pos(prim.positions[v * 3], prim.positions[v * 3 + 1], prim.positions[v * 3 + 2]);
                for (const auto& [deltas, w] : activeMorphs) {
                    const float* dp = &deltas[v * 3];
                    pos.x += dp[0] * w;
                    pos.y += dp[1] * w;
                    pos.z += dp[2] * w;
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
static void rasterizePrim(
    const ProcessedPrim& pp,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int yMin, int yMax,
    bool depthTest, bool depthWrite)
{
    const MeshPrimitive& prim = *pp.prim;
    int numTris = prim.triangleCount();
    for (int t = 0; t < numTris; t++)
        rasterizeTri(pp, prim, t, fb, textures, 0, fb.width - 1, yMin, yMax - 1,
                      depthTest, depthWrite);
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
    int numThreads,
    float turnStrength,
    bool leftIsFront)
{
    std::vector<ProcessedMesh> result(model.meshes.size());

    // Build arm-joint lookup early (needed for prim tagging + vertex bias)
    std::vector<uint8_t> armSide(model.jointNodes.size(), 0);
    for (size_t ji = 0; ji < model.jointNodes.size(); ji++) {
        int cur = model.jointNodes[ji];
        while (cur >= 0) {
            const std::string& nm = model.nodes[cur].name;
            bool isArm = nm.find("UpperArm") != std::string::npos ||
                         nm.find("LowerArm") != std::string::npos ||
                         nm.find("Hand") != std::string::npos;
            if (isArm) {
                // VRM 0.x uses "J_Bip_L_UpperArm" (single L/R letter),
                // VRM 1.0 uses "leftUpperArm". Handle both.
                bool isLeft = nm.find("left") != std::string::npos ||
                              nm.find("Left") != std::string::npos ||
                              nm.find("_L_") != std::string::npos ||
                              nm.find("_L.") != std::string::npos;
                armSide[ji] = isLeft ? 1 : 2;
                break;
            }
            cur = model.nodes[cur].parent;
        }
    }

    // --- Phase 1: Pre-allocate + build flat arrays for cache-friendly access ---
    struct ActiveMorph { const float* deltas; float weight; };
    struct PrimInfo {
        const MeshPrimitive* prim;
        ProcessedPrim* pp;
        int weightBase;
        std::vector<ActiveMorph> activeMorphs;
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
            pp.armSideV.resize(vc);

            PrimInfo info;
            info.prim = &prim;
            info.pp = &pp;
            info.weightBase = weightBase;
            // Precompute active morph targets (avoid iterating all 122 per vertex)
            if (prim.morphCount > 0) {
                for (int t = 0; t < prim.morphCount; t++) {
                    float w = morphWeights[weightBase + t];
                    if (w > 0.0f)
                        info.activeMorphs.push_back({&prim.morphDeltas[(size_t)t * vc * 3], w});
                }
            }
            primsFlat.push_back(std::move(info));
        }
    }

    const int numPrims = static_cast<int>(primsFlat.size());
    const glm::mat4* jm = jointMatrices.data();
    const uint8_t* aSide = armSide.data();

    // --- Phase 2: Parallel vertex processing with lock-free work queue ---
    auto processPrimVertices = [&](const PrimInfo& info, int v) {
        const MeshPrimitive& prim = *info.prim;
        ProcessedPrim& pp = *info.pp;

        glm::vec3 pos(prim.positions[v * 3], prim.positions[v * 3 + 1], prim.positions[v * 3 + 2]);
        for (const auto& m : info.activeMorphs) {
            const float* dp = &m.deltas[v * 3];
            pos.x += dp[0] * m.weight;
            pos.y += dp[1] * m.weight;
            pos.z += dp[2] * m.weight;
        }

        const uint16_t* j = &prim.joints[v * 4];
        const float* wt = &prim.weights[v * 4];
        glm::vec4 hp(pos, 1.0f);
        glm::vec4 skinned = (jm[j[0]] * hp) * wt[0];
        skinned += (jm[j[1]] * hp) * wt[1];
        skinned += (jm[j[2]] * hp) * wt[2];
        skinned += (jm[j[3]] * hp) * wt[3];

        // Push arm vertices toward camera (-Z in model space) to prevent
        // arm-through-torso clipping. Front arm gets larger separation.
        float leftW = 0, rightW = 0;
        for (int b = 0; b < 4; b++) {
            if (aSide[j[b]] == 1) leftW += wt[b];
            else if (aSide[j[b]] == 2) rightW += wt[b];
        }
        float totalArm = leftW + rightW;
        if (totalArm > 0.3f && turnStrength > 0.01f) {
            float frontW = leftIsFront ? leftW : rightW;
            skinned.z += frontW * -0.12f * turnStrength / totalArm;
            pp.armSideV[v] = (frontW >= (totalArm - frontW)) ? 1 : 2;
        } else {
            pp.armSideV[v] = 0;
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
// Parallel rasterization with VRM render ordering (opaque → transparent)
// ---------------------------------------------------------------------------

struct PrimEntry {
    const ProcessedPrim* pp;
    int renderQueue;
    int origIndex;
};

// Tile-based binning + parallel rasterization for a subset of prims.
static void rasterizePassTiles(
    const PrimEntry* entries, int startIdx, int endIdx,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int numThreads,
    bool depthTest, bool depthWrite)
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
                              depthTest, depthWrite);
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
    // 1. Flatten all prims
    std::vector<PrimEntry> entries;
    int origIdx = 0;
    for (const auto& mesh : processed)
        for (const auto& prim : mesh.prims) {
            entries.push_back({&prim, prim.prim->renderQueue, origIdx});
            origIdx++;
        }
    int numEntries = (int)entries.size();
    if (numEntries == 0) return;

    // 2. Split into 2 groups (matching three.js):
    //    - opaque: alphaMode != 2 → depthTest=true, depthWrite=true
    //    - transparent: alphaMode == 2 → depthTest=true, depthWrite=false
    std::vector<PrimEntry> opaque, transparent;
    for (const auto& e : entries) {
        if (e.pp->prim->alphaMode == 2)
            transparent.push_back(e);
        else
            opaque.push_back(e);
    }

    auto sortByQueue = [](std::vector<PrimEntry>& v) {
        std::sort(v.begin(), v.end(), [](const PrimEntry& a, const PrimEntry& b) {
            if (a.renderQueue != b.renderQueue) return a.renderQueue < b.renderQueue;
            return a.origIndex < b.origIndex;
        });
    };
    sortByQueue(opaque);
    sortByQueue(transparent);

    // 3. Opaque pass: depthTest=true, depthWrite=true
    if (!opaque.empty()) {
        if (numThreads <= 1) {
            for (auto& e : opaque)
                rasterizePrim(*e.pp, fb, textures, 0, 0x7FFFFFFF, true, true);
        } else {
            rasterizePassTiles(opaque.data(), 0, (int)opaque.size(), fb, textures, numThreads,
                               true, true);
        }
    }

    // 4. Transparent pass: depthTest=true, depthWrite=false (back-to-front blending)
    if (!transparent.empty()) {
        if (numThreads <= 1) {
            for (auto& e : transparent)
                rasterizePrim(*e.pp, fb, textures, 0, 0x7FFFFFFF, true, false);
        } else {
            rasterizePassTiles(transparent.data(), 0, (int)transparent.size(), fb, textures, numThreads,
                               true, false);
        }
    }
}

// ---------------------------------------------------------------------------
// 2× box-filter downsample (SSAA resolve)
// ---------------------------------------------------------------------------
void downsample2x2(const Framebuffer& ss, Framebuffer& out, int numThreads) {
    int ow = out.width, oh = out.height;
    int sw = ss.width;

    auto processRows = [&](int yStart, int yEnd) {
        for (int y = yStart; y < yEnd; y++) {
            int sy0 = y * 2, sy1 = sy0 + 1;
            const uint8_t* row0 = &ss.color[(sy0 * sw) * 4];
            const uint8_t* row1 = &ss.color[(sy1 * sw) * 4];
            uint8_t* outRow = &out.color[(y * ow) * 4];
            const float* drow0 = &ss.depth[sy0 * sw];
            const float* drow1 = &ss.depth[sy1 * sw];
            float* outDepth = &out.depth[y * ow];
            for (int x = 0; x < ow; x++) {
                int sx0 = x * 2, sx1 = sx0 + 1;
                const uint8_t* p0 = &row0[sx0 * 4];
                const uint8_t* p1 = &row0[sx1 * 4];
                const uint8_t* p2 = &row1[sx0 * 4];
                const uint8_t* p3 = &row1[sx1 * 4];
                int oi = x * 4;
                outRow[oi+0] = (uint8_t)((p0[0] + p1[0] + p2[0] + p3[0] + 2) >> 2);
                outRow[oi+1] = (uint8_t)((p0[1] + p1[1] + p2[1] + p3[1] + 2) >> 2);
                outRow[oi+2] = (uint8_t)((p0[2] + p1[2] + p2[2] + p3[2] + 2) >> 2);
                outRow[oi+3] = 255;
                float dmin = drow0[sx0];
                if (drow0[sx1] < dmin) dmin = drow0[sx1];
                if (drow1[sx0] < dmin) dmin = drow1[sx0];
                if (drow1[sx1] < dmin) dmin = drow1[sx1];
                outDepth[x] = dmin;
            }
        }
    };

    if (numThreads <= 1) {
        processRows(0, oh);
    } else {
        auto& pool = getThreadPool();
        int chunkSize = std::max(1, (oh + numThreads - 1) / numThreads);
        for (int t = 0; t < numThreads; t++) {
            int yStart = t * chunkSize;
            int yEnd = std::min(yStart + chunkSize, oh);
            if (yStart >= yEnd) break;
            pool.detach_task([&, yStart, yEnd]() { processRows(yStart, yEnd); });
        }
        pool.wait();
    }
}
