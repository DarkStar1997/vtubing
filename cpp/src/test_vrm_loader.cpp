// VRM loader regression tests: VRM 1.0 expression parsing
// (VRMC_vrmExpressions), lookAt range maps, VRMC_springBone chains/colliders,
// and humanoid bone resolution — validated against a checked-in minimal
// fixture (cpp/tests/fixtures/vrm1_minimal.vrm).
//
// Usage: test_vrm_loader <fixture.vrm>
#include "vrm_loader.h"
#include <cstdio>
#include <string>

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

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/fixtures/vrm1_minimal.vrm";
    VRMModel model = loadVRM(path);
    CHECK(!model.meshes.empty(), "fixture loads (mesh present)");
    if (model.meshes.empty()) {
        printf("\nFAILURES: %d failure(s)\n", g_failures + 1);
        return 1;
    }
    CHECK(model.meshes[0].primitives[0].morphCount == 2,
          "morph targets parsed (2 targets)");

    // --- VRMC_vrmExpressions -------------------------------------------
    auto findGroup = [&](const std::string& name) -> const VRMModel::BlendShapeGroup* {
        for (const auto& g : model.blendShapeGroups)
            if (g.name == name || g.presetName == name) return &g;
        return (const VRMModel::BlendShapeGroup*)nullptr;
    };

    const auto* happy = findGroup("happy");
    CHECK(happy != nullptr, "VRM 1.0 preset expression 'happy' parsed");
    if (happy) {
        CHECK(happy->binds.size() == 1 && happy->binds[0].index == 0 &&
                  happy->binds[0].weight == 100.0f,
              "'happy' binds morph 0 at weight 1.0");
    }
    const auto* aa = findGroup("aa");
    CHECK(aa != nullptr, "VRM 1.0 preset expression 'aa' parsed");
    if (aa) {
        CHECK(aa->binds.size() == 1 && aa->binds[0].index == 1 &&
                  aa->binds[0].weight == 75.0f,
              "'aa' binds morph 1 at weight 0.75");
    }
    const auto* custom = findGroup("CustomSmile");
    CHECK(custom != nullptr, "VRM 1.0 custom expression parsed");
    if (custom) {
        CHECK(custom->binds.size() == 1 && custom->binds[0].weight == 50.0f,
              "custom expression carries its 0.5 weight");
    }

    // --- lookAt ----------------------------------------------------------
    CHECK(model.lookAt.type == "bone", "lookAt type parsed");
    CHECK(model.lookAt.hOut == 10.0f, "lookAt horizontal outputScale = 10");
    CHECK(model.lookAt.vUpOut == 12.0f, "lookAt verticalUp outputScale = 12");
    CHECK(model.lookAt.vDownOut == 8.0f, "lookAt verticalDown outputScale = 8");

    // --- humanoid bones ---------------------------------------------------
    CHECK(model.boneNodes.count("head") && model.boneNodes.count("hips"),
          "humanBones head/hips resolved");
    CHECK(model.boneNodes.count("leftEye") && model.boneNodes.count("rightEye"),
          "humanBones eyes resolved");
    CHECK(model.boneNodes["head"] == 1 && model.boneNodes["leftEye"] == 2,
          "bone node indices correct");

    // --- VRMC_springBone --------------------------------------------------
    CHECK(model.springChains.size() == 1, "one spring chain parsed");
    if (!model.springChains.empty()) {
        const auto& sc = model.springChains[0];
        CHECK(sc.joints.size() == 3 && sc.joints[0] == 4 && sc.joints[2] == 6,
              "spring joints [4,5,6]");
        CHECK(sc.stiffness == 0.5f, "spring stiffness parsed");
        CHECK(sc.gravityPower == 1.5f, "spring gravityPower parsed");
        CHECK(sc.dragForce == 0.4f, "spring dragForce parsed");
        CHECK(sc.gravityDir.y == -1.0f, "spring gravityDir parsed");
        CHECK(sc.colliders.size() == 1 && sc.colliders[0].node == 1 &&
                  sc.colliders[0].radius == 0.05f,
              "spring collider (head sphere) resolved");
    }

    printf("\n%s: %d failure(s)\n", g_failures ? "FAILURES" : "ALL OK", g_failures);
    return g_failures;
}
