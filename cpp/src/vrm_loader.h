#pragma once

#include <vector>
#include <string>
#include <cstdint>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct MeshPrimitive {
    std::vector<float> positions;      // xyz interleaved
    std::vector<float> normals;        // xyz interleaved
    std::vector<float> uvs;            // xy interleaved
    std::vector<uint16_t> joints;      // 4 per vertex
    std::vector<float> weights;        // 4 per vertex
    std::vector<uint32_t> indices;

    // Morph targets: [target][vertex][xyz] flattened
    int morphCount = 0;
    std::vector<float> morphDeltas;    // morphCount * vertCount * 3

    int textureIndex = -1;             // index into VRMModel.textures
    glm::vec4 baseColor = {1, 1, 1, 1};
    bool doubleSided = false;
    std::string matName;               // VRM material name "{name}_{order}_{TAG}"

    int vertexCount() const { return static_cast<int>(positions.size() / 3); }
    int triangleCount() const { return static_cast<int>(indices.size() / 3); }
};

struct Mesh {
    std::string name;
    int nodeIndex = -1;                // glTF node that owns this mesh
    std::vector<MeshPrimitive> primitives;
};

struct TextureData {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;       // RGBA8
};

struct VRMModel {
    std::string path;
    std::vector<Mesh> meshes;
    std::vector<TextureData> textures;

    // Node hierarchy
    struct Node {
        std::string name;
        glm::vec3 translation = {0, 0, 0};
        glm::quat rotation = {1, 0, 0, 0};  // wxyz
        glm::vec3 scale = {1, 1, 1};
        int parent = -1;
        std::vector<int> children;
    };
    std::vector<Node> nodes;

    // Skin data (one skin, joints reference node indices)
    std::vector<int> jointNodes;       // node index for each joint
    std::vector<glm::mat4> inverseBindMatrices;

    // Model bounding box (from all positions)
    glm::vec3 bboxMin = {0, 0, 0};
    glm::vec3 bboxMax = {0, 0, 0};

    int totalTriangles() const;
    int totalVertices() const;
};

VRMModel loadVRM(const std::string& path);
