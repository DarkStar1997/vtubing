// Rig solver regression tests (no camera / no MediaPipe / no GPU).
//
// Covers the ARKit→VRM expression mapping (visemes, emotions, blink
// combining, look helpers), VRM 0.x preset renaming, direct ARKit-name
// passthrough, eye-gaze angle math, head gain/clamp knobs and detection
// loss decay — the features ported from the Python pipeline.
#include "rig_solver.h"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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

// ARKit blendshape indices (mirror of RigSolver::kArkitNames)
namespace arkit {
enum {
    BROW_INNER_UP = 1,
    BROW_DOWN_L = 2, BROW_DOWN_R = 3,
    BROW_OUTER_UP_L = 4, BROW_OUTER_UP_R = 5,
    EYE_BLINK_L = 9, EYE_BLINK_R = 10,
    EYE_LOOK_DOWN_L = 11, EYE_LOOK_DOWN_R = 12,
    EYE_LOOK_IN_L = 13, EYE_LOOK_IN_R = 14,
    EYE_LOOK_OUT_L = 15, EYE_LOOK_OUT_R = 16,
    EYE_LOOK_UP_L = 17, EYE_LOOK_UP_R = 18,
    EYE_WIDE_L = 21, EYE_WIDE_R = 22,
    JAW_OPEN = 25,
    MOUTH_FROWN_L = 30, MOUTH_FROWN_R = 31,
    MOUTH_FUNNEL = 32, MOUTH_PUCKER = 38,
    MOUTH_SMILE_L = 44, MOUTH_SMILE_R = 45,
    MOUTH_STRETCH_L = 46, MOUTH_STRETCH_R = 47,
    NOSE_SNEER_L = 50, NOSE_SNEER_R = 51,
};
}

// ---------------------------------------------------------------------------
// Synthetic avatar: one mesh with N morphs and configurable blendshape
// groups (standard presets, legacy 0.x presets, and/or ARKit-named groups).
// ---------------------------------------------------------------------------
struct TestAvatar {
    VRMModel model;
    // original preset/name key → morph index in mesh 0
    std::unordered_map<std::string, int> morphOf;

    TestAvatar(bool legacy0xPresets = false, bool arkitPassthrough = false) {
        Mesh mesh;
        mesh.nodeIndex = -1;
        MeshPrimitive prim;
        prim.morphCount = 64;
        mesh.primitives.push_back(prim);
        model.meshes.push_back(mesh);

        int nextMorph = 0;
        auto addGroup = [&](const std::string& preset, const std::string& name) {
            VRMModel::BlendShapeGroup g;
            g.name = name;
            g.presetName = preset;
            VRMModel::BlendShapeBind b;
            b.mesh = 0;
            b.index = nextMorph++;
            b.weight = 100.0f;
            g.binds.push_back(b);
            model.blendShapeGroups.push_back(g);
            std::string key = (preset.empty() || preset == "unknown") ? name : preset;
            for (auto& c : key) c = (char)std::tolower(c);
            morphOf[key] = b.index;
        };

        if (legacy0xPresets) {
            // VRM 0.x legacy presets (renamed by the solver)
            addGroup("a", "Legacy_A");
            addGroup("joy", "Legacy_Joy");
            addGroup("blink_l", "Legacy_BlinkL");
        } else {
            // Standard VRM preset expressions (a "modern" avatar without
            // ARKit Perfect Sync groups)
            addGroup("aa", "A");
            addGroup("ih", "I");
            addGroup("ou", "U");
            addGroup("ee", "E");
            addGroup("oh", "O");
            addGroup("happy", "Happy");
            addGroup("angry", "Angry");
            addGroup("sad", "Sad");
            addGroup("surprised", "Surprised");
            addGroup("blink", "Blink");
            addGroup("blinkLeft", "Blink_L");
            addGroup("blinkRight", "Blink_R");
            addGroup("lookUp", "LookUp");
            addGroup("lookDown", "LookDown");
            addGroup("lookLeft", "LookLeft");
            addGroup("lookRight", "LookRight");
        }
        if (arkitPassthrough) {
            // ARKit-named group (Perfect-Sync style direct passthrough)
            addGroup("unknown", "jawopen");
        }

        model.lookAt.type = "bone";
        model.lookAt.hOut = 10.0f;
        model.lookAt.vUpOut = 12.0f;
        model.lookAt.vDownOut = 8.0f;
        model.nodes.resize(3);
        model.nodes[0].name = "head";
        model.nodes[1].name = "leftEye";
        model.nodes[2].name = "rightEye";
        model.boneNodes["head"] = 0;
        model.boneNodes["leftEye"] = 1;
        model.boneNodes["rightEye"] = 2;
        model.headNodeIndex = 0;
    }
};

static FaceResult makeFace(const std::vector<std::pair<int, float>>& bs, float yaw = 0,
                           float pitch = 0, float roll = 0) {
    FaceResult f;
    f.detected = true;
    for (auto& [idx, v] : bs) f.blendshapes[idx] = v;
    f.yaw = yaw;
    f.pitch = pitch;
    f.roll = roll;
    return f;
}

// Drive the solver to convergence with a constant input.
static void runFrames(RigSolver& solver, const FaceResult& face, int n,
                      float dt = 1.0f / 15.0f) {
    for (int i = 0; i < n; i++) solver.update(face, dt);
}

static void calibrate(RigSolver& solver) {
    solver.startCalibration();
    FaceResult neutral;
    neutral.detected = true;
    for (int i = 0; i < 35; i++) solver.calibrate(neutral);
    solver.calibratePose();
}

int main() {
    // --- Expression mapping (standard-preset avatar) ---------------------
    {
        TestAvatar av(/*legacy0xPresets=*/false, /*arkitPassthrough=*/true);
        RigSolver solver(av.model);
        calibrate(solver);

        // aa viseme from jawOpen; ih = min(jaw*0.3, 0.3)
        runFrames(solver, makeFace({{arkit::JAW_OPEN, 1.0f}}), 40);
        const auto& w1 = solver.morphWeights();
        CHECK(w1[av.morphOf["aa"]] > 0.95f, "aa viseme = jawOpen");
        CHECK(w1[av.morphOf["ih"]] > 0.25f && w1[av.morphOf["ih"]] < 0.35f,
              "ih = min(jaw*0.3, 0.3)");
        // Direct ARKit passthrough (Perfect-Sync groups)
        CHECK(w1[av.morphOf["jawopen"]] > 0.95f, "ARKit jawopen passthrough");

        // Blink combining: blink = max(L, R)
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::EYE_BLINK_L, 1.0f},
                                    {arkit::EYE_BLINK_R, 0.2f}}), 40);
        const auto& w2 = solver.morphWeights();
        CHECK(w2[av.morphOf["blink"]] > 0.9f, "blink = max(blinkL, blinkR)");
        CHECK(w2[av.morphOf["blinkleft"]] > 0.9f, "blinkLeft = eyeBlinkLeft");
        CHECK(w2[av.morphOf["blinkright"]] > 0.15f && w2[av.morphOf["blinkright"]] < 0.3f,
              "blinkRight = eyeBlinkRight");

        // happy = min(smileL, smileR)
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::MOUTH_SMILE_L, 0.8f},
                                    {arkit::MOUTH_SMILE_R, 0.6f}}), 40);
        const auto& w3 = solver.morphWeights();
        CHECK(w3[av.morphOf["happy"]] > 0.5f && w3[av.morphOf["happy"]] < 0.7f,
              "happy = min(smileL, smileR)");

        // ou = max(funnel, pucker)
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::MOUTH_FUNNEL, 0.4f},
                                    {arkit::MOUTH_PUCKER, 0.9f}}), 40);
        CHECK(solver.morphWeights()[av.morphOf["ou"]] > 0.8f, "ou = max(funnel, pucker)");

        // surprised = brow*0.3 + eyeWide*0.3 + jaw*0.4
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::BROW_OUTER_UP_L, 1.0f},
                                    {arkit::BROW_OUTER_UP_R, 1.0f},
                                    {arkit::EYE_WIDE_L, 1.0f},
                                    {arkit::EYE_WIDE_R, 1.0f},
                                    {arkit::JAW_OPEN, 0.5f}}), 40);
        float s = solver.morphWeights()[av.morphOf["surprised"]];
        CHECK(s > 0.7f && s < 0.9f, "surprised = brow*0.3 + eyeWide*0.3 + jaw*0.4");

        // sad = max(frown)*0.7 + browInnerUp*0.3
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::MOUTH_FROWN_L, 1.0f},
                                    {arkit::BROW_INNER_UP, 1.0f}}), 40);
        float sad = solver.morphWeights()[av.morphOf["sad"]];
        CHECK(sad > 0.9f && sad <= 1.0f, "sad = frown*0.7 + browInnerUp*0.3 (clamped)");

        // angry = browDown*0.7 + sneer*0.3
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::BROW_DOWN_L, 1.0f},
                                    {arkit::BROW_DOWN_R, 1.0f},
                                    {arkit::NOSE_SNEER_L, 0.5f},
                                    {arkit::NOSE_SNEER_R, 0.5f}}), 40);
        float angry = solver.morphWeights()[av.morphOf["angry"]];
        CHECK(angry > 0.75f && angry < 0.95f, "angry = browDown*0.7 + sneer*0.3");

        // look* expression weights (for expression-type lookAt models)
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::EYE_LOOK_IN_L, 1.0f},
                                    {arkit::EYE_LOOK_UP_L, 1.0f},
                                    {arkit::EYE_LOOK_UP_R, 1.0f}}), 40);
        const auto& w4 = solver.morphWeights();
        CHECK(w4[av.morphOf["lookright"]] > 0.9f,
              "lookRight = max(eyeLookInLeft, eyeLookOutRight)");
        CHECK(w4[av.morphOf["lookup"]] > 0.9f, "lookUp = max(eyeLookUp L/R)");
        CHECK(w4[av.morphOf["lookleft"]] < 0.05f, "lookLeft unaffected");
    }

    // --- VRM 0.x preset renaming -----------------------------------------
    {
        TestAvatar av(/*legacy0xPresets=*/true, /*arkitPassthrough=*/false);
        RigSolver solver(av.model);
        calibrate(solver);

        // a → aa, driven by jawOpen
        runFrames(solver, makeFace({{arkit::JAW_OPEN, 1.0f}}), 40);
        CHECK(solver.morphWeights()[av.morphOf["a"]] > 0.95f,
              "0.x rename: 'a' preset driven by aa viseme");

        // joy → happy
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::MOUTH_SMILE_L, 0.8f},
                                    {arkit::MOUTH_SMILE_R, 0.8f}}), 40);
        CHECK(solver.morphWeights()[av.morphOf["joy"]] > 0.7f,
              "0.x rename: 'joy' preset driven by happy");

        // blink_l → blinkLeft
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::EYE_BLINK_L, 1.0f}}), 40);
        CHECK(solver.morphWeights()[av.morphOf["blink_l"]] > 0.9f,
              "0.x rename: 'blink_l' preset driven by blinkLeft");
    }

    // --- Gaze math -------------------------------------------------------
    {
        TestAvatar av;
        RigSolver solver(av.model);
        calibrate(solver);

        // Look right: eyeLookInLeft=1 → yaw = +1 * hScale(10) = +10°
        runFrames(solver, makeFace({{arkit::EYE_LOOK_IN_L, 1.0f}}), 40);
        CHECK(solver.eyeYawDeg() > 8.0f && solver.eyeYawDeg() < 12.0f,
              "gaze yaw = (lookRight - lookLeft) * hOut");
        CHECK(solver.lookAtBoneType(), "bone-type lookAt detected");

        // Rotation convention (matches Python renderer): yaw → -Y, pitch → +X
        glm::quat expect = glm::angleAxis(glm::radians(-10.0f), glm::vec3(0, 1, 0));
        glm::quat got = solver.eyeRotation();
        float dot = std::abs(glm::dot(got, expect));
        CHECK(dot > 0.999f, "eyeRotation() = R_y(-yaw) * R_x(pitch)");

        // Look up: pitch = lookUp * vUp(12)
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::EYE_LOOK_UP_L, 1.0f},
                                    {arkit::EYE_LOOK_UP_R, 1.0f}}), 40);
        CHECK(solver.eyePitchDeg() > 10.0f && solver.eyePitchDeg() < 14.0f,
              "gaze pitch = lookUp * vUpOut");

        // Gaze scale knob
        solver.setGazeScale(0.5f);
        runFrames(solver, makeFace({{arkit::EYE_LOOK_UP_L, 1.0f},
                                    {arkit::EYE_LOOK_UP_R, 1.0f}}), 40);
        CHECK(solver.eyePitchDeg() > 5.0f && solver.eyePitchDeg() < 7.0f,
              "setGazeScale scales gaze angles");

        // Expression-type lookAt → no bone rotation; look* groups carry it
        TestAvatar avExpr;
        avExpr.model.lookAt.type = "expression";
        RigSolver solver2(avExpr.model);
        calibrate(solver2);
        runFrames(solver2, makeFace({{arkit::EYE_LOOK_IN_L, 1.0f}}), 40);
        CHECK(!solver2.lookAtBoneType(), "expression-type lookAt detected");
        CHECK(solver2.morphWeights()[avExpr.morphOf["lookright"]] > 0.9f,
              "expression lookAt: lookRight group driven");
    }

    // --- Head knobs -------------------------------------------------------
    {
        TestAvatar av;
        RigSolver solver(av.model);
        calibrate(solver);

        // Default: gain 0.65, clamp 35 → yaw 40 ⇒ 26°
        runFrames(solver, makeFace({}, 40.0f, 0, 0), 60);
        float angle = glm::degrees(glm::angle(solver.headRotation()));
        CHECK(angle > 24.0f && angle < 28.0f, "head yaw = input * 0.65 gain");

        // Tighter clamp
        solver.setHeadClamps(5.0f, 20.0f, 15.0f);
        runFrames(solver, makeFace({}, 40.0f, 0, 0), 60);
        angle = glm::degrees(glm::angle(solver.headRotation()));
        CHECK(angle > 3.5f && angle < 6.5f, "setHeadClamps caps head rotation");

        // Gain knob
        solver.setHeadClamps(35.0f, 20.0f, 15.0f);
        solver.setHeadGains(0.3f, 0.3f, 0.3f);
        runFrames(solver, makeFace({}, 40.0f, 0, 0), 60);
        angle = glm::degrees(glm::angle(solver.headRotation()));
        CHECK(angle > 10.0f && angle < 14.0f, "setHeadGains scales head rotation");
    }

    // --- Detection loss decay ---------------------------------------------
    {
        TestAvatar av;
        RigSolver solver(av.model);
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::JAW_OPEN, 1.0f}}), 40);
        CHECK(solver.morphWeights()[av.morphOf["aa"]] > 0.9f, "aa set before loss");

        FaceResult lost;  // detected = false
        for (int i = 0; i < 30; i++) solver.update(lost, 1.0f / 15.0f);
        CHECK(solver.morphWeights()[av.morphOf["aa"]] < 0.05f,
              "expressions decay to neutral on detection loss");

        // Gaze decays too
        calibrate(solver);
        runFrames(solver, makeFace({{arkit::EYE_LOOK_IN_L, 1.0f}}), 40);
        CHECK(solver.eyeYawDeg() > 8.0f, "gaze set before loss");
        for (int i = 0; i < 30; i++) solver.update(lost, 1.0f / 15.0f);
        CHECK(std::abs(solver.eyeYawDeg()) < 0.1f, "gaze decays on detection loss");
    }

    printf("\n%s: %d failure(s)\n", g_failures ? "FAILURES" : "ALL OK", g_failures);
    return g_failures;
}
