// Springbone solver regression tests (no camera / no GPU).
//
// 1. A chain whose bind direction opposes gravity bends downward.
// 2. With zero gravity and zero stiffness, a chain at rest produces no
//    rotation overrides.
// 3. A collider sphere overlapping the tail pushes the chain out of it.
#include "springbone.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <cstdio>
#include <cmath>

static int g_failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            printf("PASS: %s\n", msg);                                       \
        } else {                                                             \
            printf("FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);          \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

// Build: root at origin; n1 child of root; n2 child of n1; n3 child of n2.
// Translations configurable to shape the bind pose.
static VRMModel makeModel(const glm::vec3& n1T, const glm::vec3& n2T,
                          const glm::vec3& n3T) {
    VRMModel m;
    m.nodes.resize(4);
    m.nodes[0].name = "root";
    m.nodes[1].name = "j1";
    m.nodes[2].name = "j2";
    m.nodes[3].name = "j3";
    m.nodes[1].parent = 0;
    m.nodes[2].parent = 1;
    m.nodes[3].parent = 2;
    m.nodes[1].translation = n1T;
    m.nodes[2].translation = n2T;
    m.nodes[3].translation = n3T;
    return m;
}

// Hand-computed bind world matrices for a no-rotation chain.
static std::vector<glm::mat4> bindWorld(const VRMModel& m) {
    std::vector<glm::mat4> w(m.nodes.size(), glm::mat4(1.0f));
    for (size_t i = 1; i < m.nodes.size(); i++) {
        glm::mat4 local = glm::translate(glm::mat4(1.0f), m.nodes[i].translation);
        w[i] = w[m.nodes[i].parent] * local;
    }
    return w;
}

int main() {
    // --- 1. Gravity bend --------------------------------------------------
    {
        // Chain extends +X from the origin; gravity (0,-1,0) pulls it down.
        VRMModel m = makeModel({1, 0, 0}, {1, 0, 0}, {1, 0, 0});
        VRMModel::SpringChain sc;
        sc.joints = {1, 2, 3};
        sc.stiffness = 0.0f;
        sc.gravityPower = 3.0f;
        sc.gravityDir = {0, -1, 0};
        sc.dragForce = 0.4f;
        sc.hitRadius = 0.01f;
        m.springChains.push_back(sc);

        SpringBoneSolver solver(m);
        CHECK(!solver.empty(), "chain registered");
        std::vector<glm::mat4> w = bindWorld(m);
        solver.reset(m, w);

        std::unordered_map<int, glm::quat> over;
        for (int i = 0; i < 180; i++)  // 3 s at 60 fps
            over = solver.update(m, w, 1.0f / 60.0f);
        CHECK(!over.empty(), "gravity produces rotation overrides");

        // Node 1 must rotate its child direction downward: applying the
        // override to the bind rotation, +X gains negative Y.
        auto it = over.find(1);
        CHECK(it != over.end(), "root joint of the chain rotated");
        if (it != over.end()) {
            glm::quat bindRot = m.nodes[1].rotation;  // identity
            glm::quat finalRot = bindRot * it->second;
            glm::vec3 childDir = finalRot * glm::vec3(1, 0, 0);
            CHECK(childDir.y < -0.25f,
                  "chain bends downward under gravity");
            CHECK(std::abs(glm::length(childDir) - 1.0f) < 1e-4f,
                  "child direction stays normalized");
        }
    }

    // --- 2. Rest chain: no forces, no overrides ---------------------------
    {
        VRMModel m = makeModel({0, -1, 0}, {0, -1, 0}, {0, -1, 0});
        VRMModel::SpringChain sc;
        sc.joints = {1, 2, 3};
        sc.stiffness = 0.5f;
        sc.gravityPower = 0.0f;
        sc.dragForce = 0.4f;
        sc.hitRadius = 0.01f;
        m.springChains.push_back(sc);

        SpringBoneSolver solver(m);
        std::vector<glm::mat4> w = bindWorld(m);
        solver.reset(m, w);
        auto over = solver.update(m, w, 1.0f / 60.0f);
        bool allIdentity = true;
        for (auto& [n, q] : over)
            if (glm::angle(q) > 0.01f) allIdentity = false;
        CHECK(allIdentity, "chain at rest produces no rotation");
    }

    // --- 3. Collider push-out ---------------------------------------------
    {
        // Bind child at (+1, 0, 0). A collider sphere overlaps that point
        // from above, so the tail is pushed downward and out.
        VRMModel m = makeModel({1, 0, 0}, {1, 0, 0}, {1, 0, 0});
        VRMModel::SpringChain sc;
        sc.joints = {1, 2, 3};
        sc.stiffness = 0.0f;
        sc.gravityPower = 0.0f;
        sc.dragForce = 1.0f;  // kill inertia for a deterministic result
        sc.hitRadius = 0.01f;
        VRMModel::SpringCollider col;
        col.node = 0;
        // Sphere near the bind child position (node 2 sits at world (2,0,0))
        col.offset = glm::vec3(1.9f, 0.15f, 0.0f);
        col.radius = 0.6f;
        sc.colliders.push_back(col);
        m.springChains.push_back(sc);

        SpringBoneSolver solver(m);
        std::vector<glm::mat4> w = bindWorld(m);
        solver.reset(m, w);
        auto over = solver.update(m, w, 1.0f / 60.0f);
        auto it = over.find(1);
        CHECK(it != over.end(), "collider produces a rotation override");
        if (it != over.end()) {
            glm::quat finalRot = m.nodes[1].rotation * it->second;
            glm::vec3 childDir = finalRot * glm::vec3(1, 0, 0);
            // Pushed away from the collider: downward component appears.
            CHECK(childDir.y < -0.2f, "tail pushed out of the collider sphere");
            // And the tail stays on the unit sphere around the bone (length).
            CHECK(std::abs(glm::length(childDir) - 1.0f) < 1e-4f,
                  "bone length preserved after push-out");
        }
    }

    // --- 4. Ring-down: no sustained oscillation ---------------------------
    // Regression test for the "avatar keeps shaking left-to-right" bug: the
    // naive stiffness term (gap × stiffness × steps) overshoots at low frame
    // rates (stiffness 0.5 × 3 steps at 15 fps = 1.5× the gap per frame) and
    // drag 0.05 removes almost no energy, so chains rang indefinitely.
    // Excite the chain with a brief parent rotation, then hold the parent
    // still: the chain must settle back near its rest direction.
    {
        VRMModel m = makeModel({1, 0, 0}, {1, 0, 0}, {1, 0, 0});
        VRMModel::SpringChain sc;
        sc.joints = {1, 2, 3};
        sc.stiffness = 0.5f;      // parameters of the shipped default avatar
        sc.gravityPower = 0.0f;
        sc.gravityDir = {0, -1, 0};
        sc.dragForce = 0.05f;
        sc.hitRadius = 0.02f;
        m.springChains.push_back(sc);

        SpringBoneSolver solver(m);
        std::vector<glm::mat4> bind = bindWorld(m);
        solver.reset(m, bind);

        // World matrices with the chain's root parent rotated by `deg` (Y):
        // the hair root turns with the head, like tracking input.
        auto worldWithRootRot = [&](float deg) {
            std::vector<glm::mat4> w = bind;
            glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(deg),
                                      glm::vec3(0, 1, 0));
            for (size_t i = 1; i < w.size(); i++)
                w[i] = r * w[i];
            return w;
        };
        // Deviation of joint 1's tail from the bind (+X) direction, degrees
        auto deviationDeg = [&](const std::unordered_map<int, glm::quat>& over) {
            auto it = over.find(1);
            if (it == over.end()) return 0.0f;
            glm::quat rot = m.nodes[1].rotation * it->second;
            glm::vec3 dir = rot * glm::vec3(1, 0, 0);
            return glm::degrees(std::acos(glm::clamp(dir.x, -1.0f, 1.0f)));
        };

        const float dt = 1.0f / 15.0f;  // default render cap
        // Excite: 5 frames with the head turned 20°
        std::unordered_map<int, glm::quat> over;
        auto excited = worldWithRootRot(20.0f);
        for (int i = 0; i < 5; i++) over = solver.update(m, excited, dt);
        CHECK(deviationDeg(over) > 1.0f, "excitation bends the chain");
        // Then hold the head still for 3 s. A couple of decaying swings in
        // the first ~1.5 s are natural; what must NOT happen is sustained
        // oscillation. Measure the final second.
        float maxDevAfter = 0.0f;
        for (int i = 0; i < 45; i++) {
            over = solver.update(m, bind, dt);
            if (i >= 30)  // ignore the ring-down swings of seconds 1–2
                maxDevAfter = std::max(maxDevAfter, deviationDeg(over));
        }
        printf("  ring-down: max deviation in final second = %.2f deg\n", maxDevAfter);
        CHECK(maxDevAfter < 3.0f,
              "chain settles after excitation (no sustained oscillation)");
    }

    printf("\n%s: %d failure(s)\n", g_failures ? "FAILURES" : "ALL OK", g_failures);
    return g_failures;
}
