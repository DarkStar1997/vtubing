#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

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
    int alphaMode = 0;                 // 0=opaque, 1=mask, 2=blend
    int renderQueue = 2450;            // VRM renderQueue (2450=opaque, 3000+=transparent)
    std::string matName;               // VRM material name "{name}_{order}_{TAG}"

    // VRM/MToon shader parameters (transformed per three-vrm convention)
    glm::vec3 mtoonShadeColor = {1, 1, 1};
    float mtoonRampScale = 10.0f;      // 1 / (2*(1-toony))
    float mtoonRampBias = 0.0f;        // shift + 1 - toony

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

    int headNodeIndex = -1;             // VRM humanoid "head" bone node

    // VRM humanoid bone name → node index
    // Keys: hips, spine, chest, upperChest, neck, head,
    //       leftShoulder, rightShoulder,
    //       leftUpperArm, rightUpperArm, leftLowerArm, rightLowerArm,
    //       leftHand, rightHand,
    //       leftUpperLeg, rightUpperLeg, leftLowerLeg, rightLowerLeg,
    //       leftFoot, rightFoot, leftToes, rightToes,
    //       leftEye, rightEye,
    //       + finger bones (leftThumbProximal, etc.)
    std::unordered_map<std::string, int> boneNodes;

    // VRM 0.x blendshape groups (VRM 1.0 expressions are converted into the
    // same representation: presetName = preset key or "" for custom)
    struct BlendShapeBind {
        int mesh;      // glTF mesh index
        int index;     // morph target index within mesh
        float weight;  // 0-100
    };
    struct BlendShapeGroup {
        std::string name;
        std::string presetName;  // e.g. "a", "i", "blink", "happy"
        std::vector<BlendShapeBind> binds;
    };
    std::vector<BlendShapeGroup> blendShapeGroups;

    // VRM lookAt configuration (0.x firstPerson / 1.0 VRMC_vrm.lookAt)
    struct LookAtConfig {
        std::string type = "bone";  // "bone" | "expression"
        float hOut = 10.0f;         // horizontalInner rangeMap outputScale (degrees)
        float vUpOut = 10.0f;       // verticalUp outputScale
        float vDownOut = 10.0f;     // verticalDown outputScale
    };
    LookAtConfig lookAt;

    // VRM secondary animation (springbones): hair / clothes physics
    // Unified over VRM 0.x secondaryAnimation and VRM 1.0 VRMC_springBone.
    struct SpringCollider {
        int node = -1;
        glm::vec3 offset = {0.0f, 0.0f, 0.0f};
        float radius = 0.08f;
    };
    struct SpringChain {
        std::vector<int> joints;  // node indices, root → tail
        float stiffness = 1.0f;
        float gravityPower = 0.0f;
        glm::vec3 gravityDir = {0.0f, -1.0f, 0.0f};
        float dragForce = 0.4f;
        float hitRadius = 0.02f;
        std::vector<SpringCollider> colliders;  // colliders for this chain
    };
    std::vector<SpringChain> springChains;

    int totalTriangles() const;
    int totalVertices() const;
};

VRMModel loadVRM(const std::string& path);
