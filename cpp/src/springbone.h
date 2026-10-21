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

    // Sync bones (KawaiiPhysics pattern, deejay port): a chain ROOT
    // joint whose rest direction partially follows a source bone's
    // animated rotation delta from bind - e.g. skirt panels rooted
    // under the hips following the nearest thigh, so the skirt tracks
    // leg motion instead of lagging into the legs. `follow` is the
    // slerp weight (0 = free, 1 = rigid). The hem joints still spring
    // normally.
    struct SyncBone {
        int target;    // chain root node the sync drives
        int source;    // driving node (e.g. upperLeg)
        float follow;  // 0..1
    };
    void setSyncBones(std::vector<SyncBone> s) { sync_ = std::move(s); }

    // Subdivision (deejay port): chains whose ROOT joint node is listed
    // get every segment split in two. A VIRTUAL mid-joint simulates the
    // proximal half, colliders shape BOTH halves, and the real bone
    // adopts the composed rotation pointing along the wrapped
    // distal-half direction. Skirt panels are commonly SINGLE-bone
    // chains - one rigid plank per panel whose chord cuts through the
    // thigh colliders its tail is pushed out of. The mesh cannot bend
    // mid-bone, so this raises the simulation's wrap resolution without
    // re-rigging avatars. Call before reset().
    void setSubdivideRoots(const std::vector<int>& roots);

private:
    struct JointState {
        int node = -1;              // the bone being rotated
        int childNode = -1;         // bone whose tail position drives it
        glm::vec3 bindOffset{0.0f}; // child position - bone position (bind, local bone space)
        float bindLen = 0.0f;
        glm::vec3 prevPos{0.0f};
        glm::vec3 curPos{0.0f};
        bool initialized = false;
        // Teleport guard + warm-up (KawaiiPhysics patterns, deejay
        // port): the parent frame's world position last step (a jump
        // past the threshold snaps the tail to expected, killing the
        // inertia that would otherwise explode into a swing), and
        // settle frames of inertia-free damped pull after a reset or
        // snap.
        glm::vec3 lastBonePos{0.0f};
        bool haveLastBone = false;
        int settle = 0;
        // Subdivision: subdivided = this real joint's segment is
        // simulated as two halves; virtualJoint = no node, the
        // distal-half verlet partner of the preceding real joint.
        bool subdivided = false;
        bool virtualJoint = false;
    };
    struct ChainState {
        VRMModel::SpringChain cfg;
        std::vector<JointState> joints;
        bool subdiv = false;   // segments split in two
    };
    std::vector<ChainState> chains_;
    std::vector<VRMModel::Node> workNodes_;  // working locals (per-frame bind reset)
    // Sync bones (see setSyncBones): config + each source's world
    // rotation captured at reset() (the delta reference).
    std::vector<SyncBone> sync_;
    std::unordered_map<int, glm::quat> syncBindRot_;
    // Subdivision (see setSubdivideRoots).
    std::vector<int> subdivideRoots_;
};
