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

void SpringBoneSolver::setSubdivideRoots(const std::vector<int>& roots) {
    subdivideRoots_ = roots;
    for (auto& cs : chains_) {
        // Idempotent: never interleave a second time.
        bool already = false;
        for (const auto& j : cs.joints) already = already || j.virtualJoint;
        if (already || cs.cfg.joints.empty()) continue;
        bool hit = false;
        const int root = cs.cfg.joints[0];
        for (int r : roots) hit = hit || (r == root);
        if (!hit) continue;
        cs.subdiv = true;
        // Interleave virtual mid-joints: real[0], virtual[0], real[1],
        // virtual[1], ... Every real simulating joint gets a verlet
        // partner for the distal half of its segment.
        std::vector<JointState> nj;
        nj.reserve(cs.joints.size() * 2);
        for (auto& js : cs.joints) {
            js.subdivided = true;
            nj.push_back(js);
            JointState v;
            v.virtualJoint = true;
            nj.push_back(v);
        }
        cs.joints = std::move(nj);
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
        for (size_t ji = 0; ji < cs.joints.size(); ji++) {
            JointState& js = cs.joints[ji];
            if (js.virtualJoint) continue;  // initialized with its parent
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
            // Subdivided joints simulate the PROXIMAL HALF: the verlet
            // tail starts at the segment midpoint; the virtual partner
            // (distal half) starts at the true child position.
            if (js.subdivided) {
                js.curPos = bonePos + worldRot * (js.bindOffset * 0.5f);
                js.prevPos = js.curPos;
            }
            js.lastBonePos = bonePos;
            js.haveLastBone = true;
            // Warm-up: settle with inertia-free damped pull for the
            // first ~0.75 s (45 60fps-reference steps) so the chain
            // adopts the live pose without a startup swing.
            js.settle = 45;
            js.initialized = true;
            if (js.subdivided && ji + 1 < cs.joints.size() &&
                cs.joints[ji + 1].virtualJoint) {
                JointState& v = cs.joints[ji + 1];
                v.curPos = childPos;
                v.prevPos = childPos;
                v.settle = 45;
                v.initialized = true;
            }
        }
    }
    // Capture sync sources' world rotations as the delta reference.
    syncBindRot_.clear();
    for (const auto& s : sync_) {
        if (s.source >= 0 && s.source < (int)world.size())
            syncBindRot_[s.source] = glm::quat_cast(world[s.source]);
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

    // Frame-rate-independent, stability-capped parameters.
    //
    // The naive "gap * stiffness * (dt*60)" spring correction overshoots
    // whenever stiffness*steps > 1 (e.g. stiffness 0.5 at a 15 fps render
    // cap ⇒ 1.5× the gap per frame) and drives a sustained side-to-side
    // oscillation. Cap the per-frame correction well inside the stable
    // region and make the drag decay per-second-equivalent, with a base
    // damping floor so chains always settle instead of ringing forever.
    float dtSec = std::clamp(dt, 0.001f, 0.05f);
    float steps = dtSec * 60.0f;                 // 60fps-reference step count
    auto springK = [&](float stiffness, float scale) {
        // Per-frame fraction of the gap to close (≤ 0.25 keeps the verlet
        // spring well inside the stable region for any retention ≤ 1).
        return std::min(stiffness * scale * steps * 0.12f, 0.25f);
    };
    auto dragRetention = [&](float dragForce) {
        // (1-drag) per 60fps step → per-frame retention, plus a damping
        // floor (×0.85/frame) so low-drag models still settle quickly.
        float base = 1.0f - std::clamp(dragForce, 0.0f, 0.98f);
        return std::pow(base, steps) * 0.85f;
    };

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

    // Sync-bone deltas (deejay port): each source's animated rotation
    // delta from the reset reference, slerped by `follow`
    // (half-follow = the panel tilts with the leg but keeps its own
    // spring character).
    std::unordered_map<int, glm::quat> syncDq;
    for (const auto& s : sync_) {
        if (s.source < 0 || s.source >= (int)w.size()) continue;
        auto bit = syncBindRot_.find(s.source);
        if (bit == syncBindRot_.end()) continue;
        const glm::quat delta =
            glm::normalize(glm::quat_cast(w[s.source]) * glm::inverse(bit->second));
        syncDq[s.target] = glm::slerp(glm::quat(1, 0, 0, 0), delta,
                                      glm::clamp(s.follow, 0.0f, 1.0f));
    }

    // Subdivision helper (deejay port): minimal rotation taking unit
    // vector a onto unit vector b (identity when degenerate).
    auto rotFromTo = [](const glm::vec3& a, const glm::vec3& b) -> glm::quat {
        float d = glm::clamp(glm::dot(a, b), -1.0f, 1.0f);
        if (d > 0.99999f) return glm::quat(1, 0, 0, 0);
        if (d < -0.99999f) {
            glm::vec3 ax = glm::cross(a, glm::vec3(1, 0, 0));
            if (glm::length(ax) < 1e-4f)
                ax = glm::cross(a, glm::vec3(0, 1, 0));
            float l = glm::length(ax);
            return l > 1e-6f
                       ? glm::angleAxis(3.14159265f, ax / l)
                       : glm::quat(1, 0, 0, 0);
        }
        glm::vec3 ax = glm::cross(a, b);
        float l = glm::length(ax);
        if (l < 1e-8f) return glm::quat(1, 0, 0, 0);
        return glm::angleAxis(std::acos(d), ax / l);
    };

    for (auto& cs : chains_) {
        // Collider push-out shared by real and virtual verlet points
        // (spheres in world space; same semantics as before the port:
        // offset added in world axes, UniVRM summed radius).
        auto pushOut = [&](glm::vec3& p) {
            for (const auto& c : cs.cfg.colliders) {
                if (c.node < 0 || c.node >= (int)w.size()) continue;
                glm::vec3 cPos = glm::vec3(w[c.node][3]) + c.offset;
                glm::vec3 d = p - cPos;
                float dist = glm::length(d);
                float minDist = c.radius + cs.cfg.hitRadius;
                if (dist < minDist && dist > 1e-6f)
                    p = cPos + d * (minDist / dist);
            }
        };
        for (size_t ji = 0; ji < cs.joints.size(); ji++) {
            JointState& js = cs.joints[ji];
            if (!js.initialized || js.virtualJoint) continue;
            if (js.node >= (int)w.size() || js.childNode >= (int)w.size()) continue;

            glm::vec3 bonePos(w[js.node][3]);
            glm::quat boneWorldRot = glm::quat_cast(w[js.node]);

            // Subdivided segments simulate the proximal HALF; the
            // virtual partner (distal half) runs right after this
            // joint's verlet step below.
            const bool sub =
                js.subdivided && ji + 1 < cs.joints.size() &&
                cs.joints[ji + 1].virtualJoint &&
                cs.joints[ji + 1].initialized;

            // Bind direction under the bone's current world rotation
            // (follows head/body tracking); chain ROOTS carrying a
            // sync bone ride the source's partial rotation delta on
            // their SPRING TARGET first (skirt panels following the
            // thighs). The verlet pulls toward the synced target,
            // but the emitted rotation is still measured against
            // the UNSYNCED bind direction below - mixing the two
            // instantly counter-rotates the bone and decays.
            const glm::vec3 bindDirW = boneWorldRot * js.bindOffset;
            const glm::vec3 halfDirW = boneWorldRot * (js.bindOffset * 0.5f);
            glm::vec3 expected = bonePos + (sub ? halfDirW : bindDirW);
            if (ji == 0) {
                auto dit = syncDq.find(js.node);
                if (dit != syncDq.end())
                    expected = bonePos + dit->second *
                                             (sub ? halfDirW : bindDirW);
            }

            // Teleport guard (KawaiiPhysics pattern): a parent-frame
            // jump far beyond any physical motion (avatar re-anchor,
            // tracking glitch, stage cut) turns the verlet inertia
            // into a violent whip. Snap the tail onto the expected
            // position and settle briefly with damped pull instead.
            // Threshold: 0.35 m or 8x the bone length in one step.
            if (js.haveLastBone &&
                glm::length(bonePos - js.lastBonePos) >
                    std::max(0.35f, js.bindLen * 8.0f)) {
                js.curPos = expected;
                js.prevPos = expected;
                js.settle = std::max(js.settle, 8);
                if (sub) {
                    // The virtual partner's frame jumped with it: snap
                    // it onto the full-length bind direction.
                    JointState& v = cs.joints[ji + 1];
                    v.curPos = expected + (expected - bonePos);
                    v.prevPos = v.curPos;
                    v.settle = std::max(v.settle, 8);
                }
            }
            js.lastBonePos = bonePos;
            js.haveLastBone = true;

            // Verlet: inertia (drag) + stiffness pull + gravity.
            glm::vec3 inertia =
                (js.curPos - js.prevPos) * dragRetention(cs.cfg.dragForce);
            if (js.settle > 0) {
                // Warm-up / post-snap settle: no inertia, pure
                // damped pull toward the live pose.
                js.settle--;
                inertia = glm::vec3(0.0f);
            }
            glm::vec3 stiff =
                (expected - js.curPos) * springK(cs.cfg.stiffness, stiffnessScale);
            glm::vec3 grav =
                cs.cfg.gravityDir * (cs.cfg.gravityPower * gravityScale);
            glm::vec3 next = js.curPos + inertia + stiff + grav * dtSec * dtSec;

            // Collider push-out (spheres in world space).
            pushOut(next);

            // Length constraint: keep the tail at bind distance from
            // the bone — the proximal HALF for subdivided segments.
            const float segLen = sub ? js.bindLen * 0.5f : js.bindLen;
            glm::vec3 dir = next - bonePos;
            float dl = glm::length(dir);
            if (dl > 1e-6f)
                next = bonePos + dir * (segLen / dl);

            // Rotate the bone from its (unsynced) bind direction to
            // the verlet direction.
            glm::quat worldDelta(1, 0, 0, 0);
            bool emit = false;
            if (!sub) {
                glm::vec3 a = bindDirW;
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
                            worldDelta = glm::angleAxis(angle, axis);
                            emit = true;
                        }
                    }
                }
            } else {
                // Subdivided: run the virtual distal half, then emit
                // the COMPOSED rotation. rotA swings the bind
                // direction onto the proximal half; the virtual tail
                // (bone at P1, frame rotA) springs and collides on
                // its own; rotB swings the proximal onto the distal
                // half. rotB * rotA maps bind -> distal direction, so
                // the real bone renders along the wrapped distal
                // tangent - the chord a single coarse bone would draw
                // through the collider tilts clear of it.
                JointState& v = cs.joints[ji + 1];
                const glm::vec3 P1 = next;
                const glm::vec3 d1 = (P1 - bonePos) / segLen;
                const glm::quat rotA = rotFromTo(bindDirW / js.bindLen, d1);
                const glm::vec3 halfOff = js.bindOffset * 0.5f;
                const glm::vec3 vExp = P1 + rotA * halfOff;
                glm::vec3 vin =
                    (v.curPos - v.prevPos) * dragRetention(cs.cfg.dragForce);
                if (v.settle > 0) {
                    v.settle--;
                    vin = glm::vec3(0.0f);
                }
                const glm::vec3 vstiff =
                    (vExp - v.curPos) * springK(cs.cfg.stiffness, stiffnessScale);
                glm::vec3 vnext = v.curPos + vin + vstiff + grav * dtSec * dtSec;
                pushOut(vnext);
                glm::vec3 vdir = vnext - P1;
                float vdl = glm::length(vdir);
                if (vdl > 1e-6f)
                    vnext = P1 + vdir * (segLen / vdl);
                const glm::quat rotB = rotFromTo(d1, (vnext - P1) / segLen);
                v.prevPos = v.curPos;
                v.curPos = vnext;
                worldDelta = rotB * rotA;
                emit = true;
            }
            if (emit) {
                glm::quat localDelta =
                    glm::inverse(boneWorldRot) * worldDelta * boneWorldRot;
                overrides[js.node] = localDelta;
                workNodes_[js.node].rotation =
                    workNodes_[js.node].rotation * localDelta;
                recomputeWorld(js.childNode);
            }

            js.prevPos = js.curPos;
            js.curPos = next;
        }
    }
    return overrides;
}
