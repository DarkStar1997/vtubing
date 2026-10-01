#pragma once
#include "vrm_loader.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <unordered_map>
#include <vector>

// Simplified VRM springbone simulation (hair / clothes physics).
//
// Port of the classic UniVRM verlet approach: each joint keeps its previous
// and current world-space position; per frame the next position is derived
// from inertia (drag), a stiffness pull toward the bind direction under the
// current parent rotation, gravity, and sphere-collider push-out. The parent
// bone is then rotated so its child lands on the new direction.
//
// The solver only needs the model + the current world matrices (which
// already include tracking overrides such as the head rotation), and returns
// local rotation deltas to be merged into the renderer's overrides.
class SpringBoneSolver {
public:
    SpringBoneSolver(const VRMModel& model);

    bool empty() const { return chains_.empty(); }

    // (Re)initialize verlet state from the given world matrices (bind pose
    // or a tracking-modified pose).
    void reset(const VRMModel& model, const std::vector<glm::mat4>& world);

    // Advance the simulation. `world` must contain the current (tracking-
    // overridden) world matrices. Returns per-node local rotation deltas.
    // stiffnessScale / gravityScale are user multipliers (settings GUI).
    std::unordered_map<int, glm::quat> update(
        const VRMModel& model,
        const std::vector<glm::mat4>& world,
        float dt,
        float stiffnessScale = 1.0f,
        float gravityScale = 1.0f);

private:
    struct JointState {
        int node = -1;              // the bone being rotated
        int childNode = -1;         // bone whose tail position drives it
        glm::vec3 bindOffset{0.0f}; // child position - bone position (bind, local bone space)
        float bindLen = 0.0f;
        glm::vec3 prevPos{0.0f};
        glm::vec3 curPos{0.0f};
        bool initialized = false;
    };
    struct ChainState {
        VRMModel::SpringChain cfg;
        std::vector<JointState> joints;
    };
    std::vector<ChainState> chains_;
    std::vector<VRMModel::Node> workNodes_;  // working locals (per-frame bind reset)
};
