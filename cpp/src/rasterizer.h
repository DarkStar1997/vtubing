#pragma once

#include "vrm_loader.h"
#include <vector>
#include <cstdint>
#include <chrono>

struct Framebuffer {
    int width, height;
    std::vector<uint8_t> color;    // RGBA8, width*height*4
    std::vector<float> depth;      // float32, width*height

    Framebuffer(int w, int h);
    void clear(float depthClear = 1.0f);
};

struct Timer {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// Compute world matrices for all nodes (recursive TRS).
std::vector<glm::mat4> computeWorldMatrices(const VRMModel& model);

// Compute skinning joint matrices: worldMatrix[jointNode] * IBM[joint]
std::vector<glm::mat4> computeJointMatrices(
    const VRMModel& model,
    const std::vector<glm::mat4>& worldMatrices);

// Process all vertices: morph + skin + MVP → screen space
// Outputs: screenX[], screenY[], clipW[], uvOut[], textureIndex per prim
struct ProcessedPrim {
    // Per-vertex outputs (aligned for SIMD)
    std::vector<float> screenX;
    std::vector<float> screenY;
    std::vector<float> clipZ;
    std::vector<float> uvU;
    std::vector<float> uvV;
    std::vector<float> worldNX;        // skinned world-space normal X (unnormalized)
    std::vector<float> worldNY;
    std::vector<float> worldNZ;
    const MeshPrimitive* prim = nullptr;
};

struct ProcessedMesh {
    std::vector<ProcessedPrim> prims;
};

// Transform all vertices for all meshes
std::vector<ProcessedMesh> processVertices(
    const VRMModel& model,
    const std::vector<glm::mat4>& jointMatrices,
    const glm::mat4& viewProj,
    const std::vector<float>& morphWeights,  // [meshIdx][targetIdx] flattened
    int fbWidth, int fbHeight);

// --- Multithreaded variants ---

// Parallel vertex processing: splits meshes across numThreads threads
std::vector<ProcessedMesh> processVerticesParallel(
    const VRMModel& model,
    const std::vector<glm::mat4>& jointMatrices,
    const glm::mat4& viewProj,
    const std::vector<float>& morphWeights,
    int fbWidth, int fbHeight,
    int numThreads);

// Parallel rasterization: splits framebuffer into horizontal bands
void rasterizeParallel(
    const std::vector<ProcessedMesh>& processed,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int numThreads);
