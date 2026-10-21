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

    // --- 5. Teleport guard -------------------------------------------------
    // A parent-frame jump far beyond physical motion must snap the
    // tail to the expected position (small immediate deltas), not
    // whip the verlet inertia through a huge swing. (deejay port)
    {
        VRMModel m = makeModel({0, -0.1f, 0}, {0, -0.1f, 0}, {0, -0.1f, 0});
        VRMModel::SpringChain sc;
        sc.joints = {1, 2, 3};
        sc.stiffness = 0.3f;
        sc.gravityPower = 0.0f;
        sc.dragForce = 0.4f;
        sc.hitRadius = 0.01f;
        m.springChains.push_back(sc);

        SpringBoneSolver solver(m);
        std::vector<glm::mat4> bind = bindWorld(m);
        solver.reset(m, bind);
        for (int i = 0; i < 60; i++)
            solver.update(m, bind, 1.0f / 60.0f);

        // Jump the whole rig 1 m sideways in one frame.
        std::vector<glm::mat4> jumped = bind;
        glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        for (size_t i = 1; i < jumped.size(); i++)
            jumped[i] = t * jumped[i];
        float maxAngle = 0.0f;
        for (int i = 0; i < 10; i++) {
            auto over = solver.update(m, jumped, 1.0f / 60.0f);
            for (auto& [n, q] : over)
                maxAngle = std::max(maxAngle, glm::degrees(glm::angle(q)));
        }
        CHECK(maxAngle < 10.0f,
              "teleport snaps the tail (no inertia whip)");
    }

    // --- 6. Warm-up: settle without overshoot ------------------------------
    // After reset the chain adopts a rotated pose with inertia-free
    // damped pull: the deviation from the live expected direction
    // must never INCREASE during the settle window. (deejay port)
    {
        VRMModel m = makeModel({1, 0, 0}, {1, 0, 0}, {1, 0, 0});
        VRMModel::SpringChain sc;
        sc.joints = {1, 2, 3};
        sc.stiffness = 0.5f;
        sc.gravityPower = 0.0f;
        sc.dragForce = 0.2f;
        sc.hitRadius = 0.01f;
        m.springChains.push_back(sc);

        SpringBoneSolver solver(m);
        std::vector<glm::mat4> bind = bindWorld(m);
        solver.reset(m, bind);

        std::vector<glm::mat4> excited = bind;
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(30.0f),
                                  glm::vec3(0, 1, 0));
        for (size_t i = 1; i < excited.size(); i++)
            excited[i] = r * excited[i];

        const glm::vec3 wantDir = glm::normalize(
            glm::vec3(excited[2][3]) - glm::vec3(excited[1][3]));
        auto deviationDeg = [&](const std::unordered_map<int, glm::quat>& over) {
            auto it = over.find(1);
            if (it == over.end()) return 0.0f;
            glm::quat rot = m.nodes[1].rotation * it->second;
            glm::vec3 dir = rot * glm::vec3(1, 0, 0);
            return glm::degrees(std::acos(
                glm::clamp(glm::dot(glm::normalize(dir), wantDir),
                           -1.0f, 1.0f)));
        };

        bool monotone = true;
        float prevDev = 1e9f;
        for (int i = 0; i < 40; i++) {
            auto over = solver.update(m, excited, 1.0f / 60.0f);
            float d = deviationDeg(over);
            if (d > prevDev + 0.01f) monotone = false;
            prevDev = d;
        }
        CHECK(monotone,
              "warm-up settles without overshoot (deviation never rises)");
    }

    // --- 7. Sync bone: skirt root follows the thigh ------------------------
    // A chain rooted under the hips (a skirt panel) with a sync bone
    // on the thigh: a 40-deg thigh swing tilts the panel's rest
    // direction by the follow weight (20 deg). Without the sync the
    // panel ignores the leg entirely. (deejay port)
    {
        // root(0) -> hips(1) -> thighL(2); hips -> sk1(4) -> sk2(5)
        // -> sk3(6) hangs below the hips beside the thigh.
        VRMModel m;
        m.nodes.resize(7);
        m.nodes[0].name = "root";
        m.nodes[1].name = "hips";   m.nodes[1].parent = 0;
        m.nodes[1].translation = {0, -0.5f, 0};
        m.nodes[2].name = "thighL"; m.nodes[2].parent = 1;
        m.nodes[2].translation = {-0.1f, -0.05f, 0};
        m.nodes[3].name = "shinL";  m.nodes[3].parent = 2;
        m.nodes[3].translation = {0, -0.3f, 0};
        m.nodes[4].name = "sk1";    m.nodes[4].parent = 1;
        m.nodes[4].translation = {-0.1f, -0.06f, 0};
        m.nodes[5].name = "sk2";    m.nodes[5].parent = 4;
        m.nodes[5].translation = {0, -0.1f, 0};
        m.nodes[6].name = "sk3";    m.nodes[6].parent = 5;
        m.nodes[6].translation = {0, -0.1f, 0};
        VRMModel::SpringChain sc;
        sc.joints = {4, 5, 6};
        sc.stiffness = 0.3f;
        sc.gravityPower = 0.0f;
        sc.dragForce = 1.0f;  // deterministic settle
        sc.hitRadius = 0.01f;
        m.springChains.push_back(sc);

        std::vector<glm::mat4> w = bindWorld(m);

        // Rotate the thigh 40 deg about x (world); everything else
        // stays at bind (the skirt hangs from the static hips).
        std::vector<glm::mat4> legUp = w;
        legUp[2] = glm::rotate(glm::mat4(1.0f), glm::radians(40.0f),
                               glm::vec3(1, 0, 0)) * legUp[2];

        // Control: no sync bone -> panel ignores the leg.
        {
            SpringBoneSolver solver(m);
            solver.reset(m, w);
            std::unordered_map<int, glm::quat> over;
            for (int i = 0; i < 90; i++)
                over = solver.update(m, legUp, 1.0f / 60.0f);
            auto it = over.find(4);
            const float a =
                it == over.end() ? 0.0f : glm::degrees(glm::angle(it->second));
            CHECK(a < 3.0f, "no sync: skirt panel ignores the thigh");
        }
        // Sync at follow 0.5 -> panel tilts ~20 deg with the leg.
        {
            SpringBoneSolver solver(m);
            solver.setSyncBones({{4, 2, 0.5f}});
            solver.reset(m, w);
            std::unordered_map<int, glm::quat> over;
            for (int i = 0; i < 90; i++)
                over = solver.update(m, legUp, 1.0f / 60.0f);
            auto it = over.find(4);
            CHECK(it != over.end(), "sync: skirt panel drives an override");
            if (it != over.end()) {
                const float a = glm::degrees(glm::angle(it->second));
                CHECK(a > 14.0f && a < 26.0f,
                      "sync: skirt root follows the thigh at half "
                      "weight (~20 deg)");
                glm::quat rot = m.nodes[4].rotation * it->second;
                glm::vec3 dir = rot * glm::vec3(0, -1, 0);
                CHECK(dir.z < -0.2f,
                      "sync: panel tilts forward with the thigh swing");
            }
        }
    }

    // --- 8. Subdivision: single-bone panel wraps the collider --------------
    // Skirt panels are commonly SINGLE-bone chains. When the thigh
    // collider sphere straddles the middle of the segment, the coarse
    // bone's tail is pushed out radially and the rendered chord still
    // plows through the sphere. Subdivision simulates a virtual
    // mid-joint: the real bone adopts the wrapped direction, so the
    // chord only TANGENTS the sphere. (deejay port; hitRadius 0 keeps
    // the summed collider radius equal to the raw radius here)
    {
        VRMModel m;
        m.nodes.resize(3);
        m.nodes[0].name = "root";
        m.nodes[1].name = "sk1";  m.nodes[1].parent = 0;
        m.nodes[1].translation = {0, 1.0f, 0};
        m.nodes[2].name = "sk2";  m.nodes[2].parent = 1;
        m.nodes[2].translation = {0, -0.2f, 0};
        VRMModel::SpringChain sc;
        sc.joints = {1, 2};
        sc.stiffness = 0.3f;
        sc.gravityPower = 0.0f;
        sc.dragForce = 1.0f;  // deterministic settle
        sc.hitRadius = 0.0f;
        VRMModel::SpringCollider col;
        col.node = 0;
        col.offset = glm::vec3(0.0f, 0.87f, 0.03f);
        col.radius = 0.08f;
        sc.colliders.push_back(col);
        m.springChains.push_back(sc);

        std::vector<glm::mat4> w = bindWorld(m);
        const glm::vec3 cPos = col.offset;
        const glm::vec3 sk1(w[1][3]);

        // Closest approach of the RENDERED segment (sk1 -> sk1 +
        // 0.2*dir) to the collider center.
        auto chordMinDist = [&](const glm::vec3& dir) {
            const glm::vec3 d = dir * 0.2f;
            float t = glm::clamp(glm::dot(cPos - sk1, d) / glm::dot(d, d),
                                 0.0f, 1.0f);
            return glm::length(sk1 + t * d - cPos);
        };
        auto run = [&](bool subdiv, float* rotDeg, float* minDist) {
            SpringBoneSolver solver(m);
            if (subdiv) solver.setSubdivideRoots({1});
            solver.reset(m, w);
            std::unordered_map<int, glm::quat> over;
            for (int i = 0; i < 90; i++)
                over = solver.update(m, w, 1.0f / 60.0f);
            auto it = over.find(1);
            *rotDeg = it == over.end() ? 0.0f : glm::degrees(glm::angle(it->second));
            glm::quat finalRot = m.nodes[1].rotation *
                                (it == over.end() ? glm::quat(1, 0, 0, 0)
                                                   : it->second);
            *minDist = chordMinDist(finalRot * glm::vec3(0, -1, 0));
        };

        float rot0, dist0, rot1, dist1;
        run(false, &rot0, &dist0);  // control: coarse bone
        run(true, &rot1, &dist1);   // subdivided
        printf("  subdiv: coarse rot=%.2f deg chord=%.4f m | subdiv "
               "rot=%.2f deg chord=%.4f m (sphere r=%.2f)\n",
               rot0, dist0, rot1, dist1, col.radius);
        CHECK(rot0 < 3.0f, "coarse panel barely tilts (radial push-out)");
        CHECK(dist0 < 0.6f * col.radius,
              "coarse panel's rendered chord cuts through the sphere");
        CHECK(rot1 > 10.0f, "subdivided panel adopts the wrapped direction");
        CHECK(dist1 > 0.9f * col.radius,
              "subdivided panel's chord clears the sphere surface");
        CHECK(dist1 > dist0 + 0.02f,
              "subdivision strictly improves the chord clearance");
    }

    // --- 9. Subdivided chain at rest produces no rotation -----------------
    // Regression for the half-offset math: with the mid-joint verlet
    // initialized to the true segment midpoint, a force-free chain
    // must sit exactly on its bind line. (deejay port)
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
        solver.setSubdivideRoots({1});
        std::vector<glm::mat4> w = bindWorld(m);
        solver.reset(m, w);
        bool allIdentity = true;
        for (int i = 0; i < 60; i++) {
            auto over = solver.update(m, w, 1.0f / 60.0f);
            for (auto& [n, q] : over)
                if (glm::angle(q) > 0.01f) allIdentity = false;
        }
        CHECK(allIdentity,
              "subdivided chain at rest produces no rotation");
    }

    printf("\n%s: %d failure(s)\n", g_failures ? "FAILURES" : "ALL OK", g_failures);
    return g_failures;
}
