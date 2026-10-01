#include "springbone.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <algorithm>
#include <cmath>

// Design notes
// ------------
// State per joint: verlet positions (prev/current tail position in world
// space). Node rotations are NOT persisted across frames — each frame the
// bone-local rotation delta is recomputed fresh as "rotate the child from
// its bind direction (under the current parent/tracking rotation) to the
// verlet position". This keeps the returned overrides replaceable, matching
// how the renderer applies overrides (bind * delta), and keeps the model
// const.
//
// The simulation runs on an internal working copy of the node locals so
// chains fold correctly root→tail within one frame.

SpringBoneSolver::SpringBoneSolver(const VRMModel& model) {
    workNodes_ = model.nodes;
    for (const auto& sc : model.springChains) {
        if (sc.joints.size() < 2) continue;
        ChainState cs;
        cs.cfg = sc;
        // joints[N-1] is the tail: it has no child to steer by, so only
        // joints[0..N-2] get states (child = joints[i+1]).
        for (size_t i = 0; i + 1 < sc.joints.size(); i++) {
            JointState js;
            js.node = sc.joints[i];
            js.childNode = sc.joints[i + 1];
            cs.joints.push_back(js);
        }
        chains_.push_back(std::move(cs));
    }
}

static glm::mat4 localTRS(const VRMModel::Node& n) {
    glm::mat4 t = glm::translate(glm::mat4(1.0f), n.translation);
    glm::mat4 r = glm::mat4_cast(n.rotation);
    glm::mat4 s = glm::scale(glm::mat4(1.0f), n.scale);
    return t * r * s;
}

void SpringBoneSolver::reset(const VRMModel& model, const std::vector<glm::mat4>& world) {
    // `world` should be the BIND-pose world matrices so bind offsets are
    // captured before any tracking overrides are applied.
    workNodes_ = model.nodes;
    for (auto& cs : chains_) {
        for (auto& js : cs.joints) {
            if (js.node < 0 || js.childNode < 0 ||
                js.node >= (int)world.size() || js.childNode >= (int)world.size())
                continue;
            glm::vec3 bonePos(world[js.node][3]);
            glm::vec3 childPos(world[js.childNode][3]);
            // Bind offset in the bone's local space, so it can be rotated by
            // the bone's current world rotation each frame (UniVRM-style
            // "expected child position").
            glm::quat worldRot = glm::quat_cast(world[js.node]);
            glm::vec3 worldOffset = childPos - bonePos;
            js.bindOffset = glm::inverse(worldRot) * worldOffset;
            js.bindLen = std::max(glm::length(worldOffset), 1e-5f);
            js.curPos = childPos;
            js.prevPos = childPos;
            js.initialized = true;
        }
    }
}

std::unordered_map<int, glm::quat> SpringBoneSolver::update(
    const VRMModel& model,
    const std::vector<glm::mat4>& world,
    float dt,
    float stiffnessScale,
    float gravityScale) {
    std::unordered_map<int, glm::quat> overrides;
    if (chains_.empty()) return overrides;

    // Restore working node locals to bind; deltas are recomputed fresh.
    workNodes_ = model.nodes;
    std::vector<glm::mat4> w = world;

    float dtSec = std::clamp(dt, 0.001f, 0.05f);
    float step = dtSec * 60.0f;  // parameters are tuned per-60fps-step

    // Recompute world matrices for `node` and its ancestors from the working
    // locals (spring nodes may already carry this frame's deltas).
    auto recomputeWorld = [&](int node) {
        std::vector<int> path;
        int cur = node;
        while (cur >= 0) {
            path.push_back(cur);
            int parent = (cur < (int)workNodes_.size()) ? workNodes_[cur].parent : -1;
            if (parent < 0) break;
            cur = parent;
        }
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            int ni = *it;
            glm::mat4 local = localTRS(workNodes_[ni]);
            int parent = workNodes_[ni].parent;
            if (parent >= 0 && parent < (int)w.size())
                w[ni] = w[parent] * local;
            else
                w[ni] = local;
        }
    };

    for (auto& cs : chains_) {
        for (auto& js : cs.joints) {
            if (!js.initialized) continue;
            if (js.node >= (int)w.size() || js.childNode >= (int)w.size()) continue;

            glm::vec3 bonePos(w[js.node][3]);
            glm::quat boneWorldRot = glm::quat_cast(w[js.node]);

            // Expected child position: the bind direction under the bone's
            // current world rotation (follows head/body tracking).
            glm::vec3 expected = bonePos + boneWorldRot * js.bindOffset;

            // Verlet: inertia (drag) + stiffness pull + gravity.
            glm::vec3 inertia = (js.curPos - js.prevPos) * (1.0f - cs.cfg.dragForce);
            glm::vec3 stiff =
                (expected - js.curPos) * cs.cfg.stiffness * stiffnessScale * step;
            glm::vec3 grav =
                cs.cfg.gravityDir * (cs.cfg.gravityPower * gravityScale);
            glm::vec3 next = js.curPos + inertia + stiff + grav * dtSec * dtSec;

            // Collider push-out (spheres in world space)
            for (const auto& c : cs.cfg.colliders) {
                if (c.node < 0 || c.node >= (int)w.size()) continue;
                glm::vec3 cPos = glm::vec3(w[c.node][3]) + c.offset;
                glm::vec3 d = next - cPos;
                float dist = glm::length(d);
                float minDist = c.radius + cs.cfg.hitRadius;
                if (dist < minDist && dist > 1e-6f)
                    next = cPos + d * (minDist / dist);
            }

            // Length constraint: keep the tail at bind distance from the bone.
            glm::vec3 dir = next - bonePos;
            float dl = glm::length(dir);
            if (dl > 1e-6f)
                next = bonePos + dir * (js.bindLen / dl);

            // Rotate the bone from its bind direction to the verlet direction.
            glm::vec3 a = expected - bonePos;
            glm::vec3 b = next - bonePos;
            float al = glm::length(a), bl = glm::length(b);
            if (al > 1e-6f && bl > 1e-6f) {
                a /= al;
                b /= bl;
                float dot = glm::clamp(glm::dot(a, b), -1.0f, 1.0f);
                float angle = std::acos(dot);
                if (angle > 1e-5f) {
                    glm::vec3 axis = glm::cross(a, b);
                    float axl = glm::length(axis);
                    if (axl > 1e-6f) {
                        axis /= axl;
                        glm::quat worldDelta = glm::angleAxis(angle, axis);
                        glm::quat localDelta =
                            glm::inverse(boneWorldRot) * worldDelta * boneWorldRot;
                        overrides[js.node] = localDelta;
                        workNodes_[js.node].rotation =
                            workNodes_[js.node].rotation * localDelta;
                        recomputeWorld(js.childNode);
                    }
                }
            }

            js.prevPos = js.curPos;
            js.curPos = next;
        }
    }
    return overrides;
}
