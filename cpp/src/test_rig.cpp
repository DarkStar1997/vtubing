// Rig solver regression tests (no camera / no MediaPipe / no GPU).
//
// Covers the ARKit→VRM expression mapping (visemes, emotions, blink
// combining, look helpers), VRM 0.x preset renaming, direct ARKit-name
// passthrough, eye-gaze angle math, head gain/clamp knobs and detection
// loss decay — the features ported from the Python pipeline.
#include "rig_solver.h"
#include "pose_tracker.h"
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

// Synthetic MediaPipe pose frame: symmetric standing figure whose arm
// directions (raw MediaPipe world-space unit vectors, pre-AXIS_FLIP) are the
// parameters. Hips sit 15 cm behind the shoulders so torso lean is nonzero.
static PoseResult makePose(const glm::vec3& uaLraw, const glm::vec3& laLraw,
                           const glm::vec3& uaRraw, const glm::vec3& laRraw) {
    constexpr int LS = PoseLandmarkIdx::L_SHOULDER, RS = PoseLandmarkIdx::R_SHOULDER;
    constexpr int LE = PoseLandmarkIdx::L_ELBOW, RE = PoseLandmarkIdx::R_ELBOW;
    constexpr int LW = PoseLandmarkIdx::L_WRIST, RW = PoseLandmarkIdx::R_WRIST;
    constexpr int LH = PoseLandmarkIdx::L_HIP, RH = PoseLandmarkIdx::R_HIP;
    constexpr int LK = PoseLandmarkIdx::L_KNEE, RK = PoseLandmarkIdx::R_KNEE;
    constexpr int LA = PoseLandmarkIdx::L_ANKLE, RA = PoseLandmarkIdx::R_ANKLE;
    PoseResult p;
    p.detected = true;
    p.presence = 1.0f;
    p.frameW = 640;
    p.frameH = 480;
    auto setWl = [&](int i, const glm::vec3& v) {
        p.worldLandmarks[i * 3 + 0] = v.x;
        p.worldLandmarks[i * 3 + 1] = v.y;
        p.worldLandmarks[i * 3 + 2] = v.z;
    };
    auto setLm = [&](int i, float xN, float yN) {
        p.landmarks[i * 5 + 0] = xN;
        p.landmarks[i * 5 + 1] = yN;
        p.landmarks[i * 5 + 2] = 0.0f;
        p.landmarks[i * 5 + 3] = 1.0f;  // visibility
        p.landmarks[i * 5 + 4] = 1.0f;  // presence
    };
    glm::vec3 ls(-0.18f, 0.45f, 0.0f), rs(0.18f, 0.45f, 0.0f);
    glm::vec3 lh(-0.10f, 0.05f, -0.15f), rh(0.10f, 0.05f, -0.15f);
    glm::vec3 le = ls + uaLraw * 0.25f, re = rs + uaRraw * 0.25f;
    glm::vec3 lw = le + laLraw * 0.25f, rw = re + laRraw * 0.25f;
    setWl(LS, ls); setWl(RS, rs);
    setWl(LE, le);   setWl(RE, re);
    setWl(LW, lw);   setWl(RW, rw);
    setWl(LH, lh);     setWl(RH, rh);
    setLm(LS, 0.42f, 0.30f); setLm(RS, 0.58f, 0.30f);
    setLm(LE, 0.44f, 0.45f);    setLm(RE, 0.56f, 0.45f);
    setLm(LW, 0.46f, 0.60f);    setLm(RW, 0.54f, 0.60f);
    setLm(LH, 0.45f, 0.55f);      setLm(RH, 0.55f, 0.55f);
    setLm(LK, 0.46f, 0.75f);     setLm(RK, 0.54f, 0.75f);
    setLm(LA, 0.47f, 0.90f);    setLm(RA, 0.53f, 0.90f);
    return p;
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

        // Explicit gain 0.65 → yaw 40 ⇒ 26° (independent of default tuning)
        solver.setHeadGains(0.65f, 0.65f, 0.65f);
        runFrames(solver, makeFace({}, 40.0f, 0, 0), 60);
        float angle = glm::degrees(glm::angle(solver.headRotation()));
        CHECK(angle > 24.0f && angle < 28.0f, "head yaw = input * gain");

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

        // Smoothing knob: higher smoothing must suppress a jittering input.
        // Feed alternating ±2° yaw around a 20° base; the smoothed output's
        // peak-to-peak variation must shrink as smoothing increases.
        auto jitterVar = [&](float smoothing) {
            RigSolver s2(av.model);
            calibrate(s2);
            s2.setHeadGains(1.0f, 1.0f, 1.0f);  // isolate filtering from gain
            s2.setHeadSmoothing(smoothing);
            float lo = 1e9f, hi = -1e9f;
            for (int i = 0; i < 120; i++) {
                float yawDeg = 20.0f + ((i % 2 == 0) ? 2.0f : -2.0f);
                s2.update(makeFace({}, yawDeg, 0, 0), 1.0f / 15.0f);
                float e = glm::degrees(glm::angle(s2.headRotation()));
                if (i >= 60) {  // skip convergence
                    lo = std::min(lo, e);
                    hi = std::max(hi, e);
                }
            }
            return hi - lo;
        };
        float varSnappy = jitterVar(0.0f);
        float varSmooth = jitterVar(1.0f);
        printf("  jitter peak-to-peak: smoothing 0 → %.2f deg, smoothing 1 → %.2f deg\n",
               varSnappy, varSmooth);
        CHECK(varSmooth < varSnappy * 0.6f,
              "setHeadSmoothing suppresses tracking jitter");
        CHECK(varSmooth < 1.0f,
              "max smoothing nearly eliminates jitter");
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

    // --- Body pose: smooth, flicker-resistant ----------------------------
    {
        TestAvatar av;
        RigSolver solver(av.model);
        calibrate(solver);

        const float dt = 1.0f / 15.0f;
        glm::vec3 down(0.0f, -1.0f, 0.0f);  // arms hanging down (raw = flipped)
        PoseResult armsDown = makePose(down, down, down, down);

        // Converge: arms 90° from rest, hips behind shoulders → lean
        for (int i = 0; i < 60; i++) solver.updatePose(armsDown, dt);
        const BodyPose bp0 = solver.bodyPose();
        CHECK(bp0.valid, "body pose becomes valid");
        float angleDown = glm::degrees(glm::angle(bp0.leftUpperArm));
        CHECK(angleDown > 70.0f && angleDown < 110.0f,
              "arms-down pose tracked (~90° from rest)");
        CHECK(bp0.lean > 0.10f, "forward lean tracked from shoulder/hip depth");
        printf("  arms %.1f°, lean %.3f\n", angleDown, bp0.lean);

        // Brief detection dropout (< grace): pose must hold, no snap
        PoseResult lost;  // detected = false
        solver.updatePose(lost, dt);
        solver.updatePose(lost, dt);  // ~0.13 s lost
        const BodyPose bp1 = solver.bodyPose();
        CHECK(std::abs(glm::degrees(glm::angle(bp1.leftUpperArm)) - angleDown)
                  < 2.0f,
              "brief dropout holds arm pose (grace period)");
        CHECK(std::abs(bp1.lean - bp0.lean) < 0.005f,
              "brief dropout holds torso lean");

        // Recovery from the flicker: continuous with the pre-dropout pose
        solver.updatePose(armsDown, dt);
        float angleRec = glm::degrees(glm::angle(solver.bodyPose().leftUpperArm));
        CHECK(std::abs(angleRec - angleDown) < 3.0f,
              "recovery after dropout is continuous (no snap)");

        // Sustained loss (> grace): everything relaxes toward rest
        for (int i = 0; i < 20; i++) solver.updatePose(lost, dt);  // 1.33 s
        const BodyPose bp2 = solver.bodyPose();
        CHECK(glm::degrees(glm::angle(bp2.leftUpperArm)) < angleDown * 0.3f,
              "sustained loss relaxes arms toward rest");
        CHECK(std::abs(bp2.lean) < 0.02f, "sustained loss relaxes lean");

        // Body smoothing knob: alternating ±5° elbow-direction wobble must
        // shrink as smoothing rises (pose-estimation jitter resistance).
        auto jitterP2P = [&](float smoothing) {
            RigSolver s2(av.model);
            calibrate(s2);
            s2.setBodySmoothing(smoothing);
            float lo = 1e9f, hi = -1e9f;
            for (int i = 0; i < 120; i++) {
                float a = glm::radians(90.0f + ((i % 2 == 0) ? 5.0f : -5.0f));
                glm::vec3 d(std::cos(a), std::sin(a), 0.0f);
                s2.updatePose(makePose(d, d, d, d), dt);
                float e = glm::degrees(glm::angle(s2.bodyPose().leftUpperArm));
                if (i >= 60) {  // skip convergence
                    lo = std::min(lo, e);
                    hi = std::max(hi, e);
                }
            }
            return hi - lo;
        };
        float p2pSnappy = jitterP2P(0.0f);
        float p2pSmooth = jitterP2P(1.0f);
        printf("  pose jitter peak-to-peak: smoothing 0 → %.2f deg, "
               "smoothing 1 → %.2f deg\n", p2pSnappy, p2pSmooth);
        CHECK(p2pSmooth < p2pSnappy * 0.5f,
              "setBodySmoothing suppresses pose jitter");
        CHECK(p2pSmooth < 3.0f, "max smoothing nearly eliminates pose jitter");
    }

    // --- Procedural idle life + head parallax ----------------------------
    {
        TestAvatar av;
        const float dt = 1.0f / 15.0f;
        FaceResult neutral = makeFace({}, 0, 0, 0);

        // Auto-blink fires on a Poisson schedule when the user doesn't blink
        {
            RigSolver s(av.model);
            calibrate(s);
            s.setIdleEnabled(true);
            s.setBlinkRatePerMin(60.0f);  // mean interval 1 s
            float maxBlink = 0.0f;
            for (int i = 0; i < 90; i++) {  // 6 s
                s.update(neutral, dt);
                maxBlink = std::max(maxBlink, s.morphWeights()[av.morphOf["blink"]]);
            }
            CHECK(maxBlink > 0.5f, "procedural blink fires without user blink");
            // ... and the envelope closes again
            float endBlink = s.morphWeights()[av.morphOf["blink"]];
            CHECK(endBlink < 0.35f, "procedural blink envelope closes");
        }

        // No procedural blink while the user is blinking (0.75 s guard)
        {
            RigSolver s(av.model);
            calibrate(s);
            s.setIdleEnabled(true);
            s.setBlinkRatePerMin(60.0f);
            float minBlink = 1.0f;
            int firstOnset = -1;
            for (int i = 0; i < 60; i++) {  // 4 s
                FaceResult f = neutral;
                if (i < 5) f.blendshapes[arkit::EYE_BLINK_L] = 1.0f;
                s.update(f, dt);
                float b = s.morphWeights()[av.morphOf["blink"]];
                if (b > 0.5f && firstOnset < 0) firstOnset = i;
                if (i >= 5) minBlink = std::min(minBlink, b);
            }
            CHECK(firstOnset <= 2, "user blink passes through");
            CHECK(minBlink < 0.5f,
                  "no procedural blink during the user-blink guard window");
        }

        // Idle disabled: nothing procedural happens
        {
            RigSolver s(av.model);
            calibrate(s);
            s.setIdleEnabled(false);
            s.setBlinkRatePerMin(60.0f);
            float maxBlink = 0.0f;
            for (int i = 0; i < 90; i++) {
                s.update(neutral, dt);
                maxBlink = std::max(maxBlink, s.morphWeights()[av.morphOf["blink"]]);
            }
            CHECK(maxBlink < 0.05f, "idle off: no procedural blink");
            CHECK(std::abs(s.breath()) < 1e-6f && std::abs(s.sway()) < 1e-6f,
                  "idle off: breath/sway are zero");
            float gaze = glm::degrees(glm::angle(s.eyeRotation()));
            CHECK(gaze < 1e-4f, "idle off: no saccade offset");
        }

        // Breathing: 0.2 Hz sine at full intensity spans ±1
        {
            RigSolver s(av.model);
            calibrate(s);
            s.setIdleEnabled(true);
            s.setIdleIntensity(1.0f);
            float lo = 1e9f, hi = -1e9f, mean = 0.0f;
            for (int i = 0; i < 75; i++) {  // 5 s = one full period
                s.update(neutral, dt);
                lo = std::min(lo, s.breath());
                hi = std::max(hi, s.breath());
                mean += s.breath();
            }
            CHECK(hi - lo > 1.5f, "breathing oscillates (0.2 Hz)");
            CHECK(std::abs(mean / 75.0f) < 0.3f, "breathing is zero-mean");
        }

        // Weight shift: slow sway covers most of its range in ~14 s
        {
            RigSolver s(av.model);
            calibrate(s);
            s.setIdleEnabled(true);
            s.setIdleIntensity(1.0f);
            float maxAbs = 0.0f;
            for (int i = 0; i < 210; i++) {  // 14 s ≈ full period
                s.update(neutral, dt);
                maxAbs = std::max(maxAbs, std::abs(s.sway()));
            }
            CHECK(maxAbs > 0.8f, "weight-shift sway reaches full amplitude");
        }

        // Micro-saccades: gaze wiggles without eyeLook input, bounded
        {
            RigSolver s(av.model);
            calibrate(s);
            s.setIdleEnabled(true);
            s.setIdleIntensity(1.0f);
            float maxGaze = 0.0f;
            for (int i = 0; i < 120; i++) {  // 8 s
                s.update(neutral, dt);
                maxGaze = std::max(
                    maxGaze, glm::degrees(glm::angle(s.eyeRotation())));
            }
            printf("  saccade-driven max gaze deviation: %.2f deg\n", maxGaze);
            CHECK(maxGaze > 0.15f, "micro-saccades move the gaze");
            CHECK(maxGaze < 4.0f, "saccade offsets stay small");
        }

        // Head translation / parallax (gain 1.0)
        auto faceAt = [](float noseX, float noseY, float eyeLX, float eyeRX) {
            FaceResult f = makeFace({}, 0, 0, 0);
            f.landmarks[1 * 3 + 0] = noseX;
            f.landmarks[1 * 3 + 1] = noseY;
            f.landmarks[33 * 3 + 0] = eyeLX;
            f.landmarks[33 * 3 + 1] = 240.0f;
            f.landmarks[263 * 3 + 0] = eyeRX;
            f.landmarks[263 * 3 + 1] = 240.0f;
            return f;
        };
        const float eyeL = 280.0f, eyeR = 360.0f;  // span 80 px
        {
            RigSolver s(av.model);
            s.startCalibration();
            for (int i = 0; i < 35; i++) s.calibrate(faceAt(320, 240, eyeL, eyeR));
            s.setHeadPosGain(1.0f);
            for (int i = 0; i < 60; i++) s.update(faceAt(400, 240, eyeL, eyeR), dt);
            float x = s.headPosition().x;
            printf("  head pos after +1 face-span right shift: x=%.3f\n", x);
            CHECK(x > 0.03f && x < 0.06f, "head shifts right (clamped)");
        }
        {
            RigSolver s(av.model);
            s.startCalibration();
            for (int i = 0; i < 35; i++) s.calibrate(faceAt(320, 240, eyeL, eyeR));
            s.setHeadPosGain(1.0f);
            for (int i = 0; i < 60; i++) s.update(faceAt(320, 300, eyeL, eyeR), dt);
            float y = s.headPosition().y;
            CHECK(y < -0.025f && y > -0.05f, "head shifts down in image");
        }
        {
            RigSolver s(av.model);
            s.startCalibration();
            for (int i = 0; i < 35; i++) s.calibrate(faceAt(320, 240, eyeL, eyeR));
            s.setHeadPosGain(1.0f);
            // Lean in: face 25% bigger (span 80 -> 100)
            for (int i = 0; i < 60; i++) s.update(faceAt(320, 240, 270, 370), dt);
            float z = s.headPosition().z;
            printf("  head pos after lean-in: z=%.3f\n", z);
            CHECK(z > 0.015f && z < 0.045f, "leaning in brings the head forward");
        }
        {
            RigSolver s(av.model);
            s.startCalibration();
            for (int i = 0; i < 35; i++) s.calibrate(faceAt(320, 240, eyeL, eyeR));
            s.setHeadPosGain(0.0f);
            for (int i = 0; i < 60; i++) s.update(faceAt(400, 300, 270, 370), dt);
            CHECK(glm::length(s.headPosition()) < 1e-5f,
                  "head position gain 0 disables parallax");
        }
    }

    printf("\n%s: %d failure(s)\n", g_failures ? "FAILURES" : "ALL OK", g_failures);
    return g_failures;
}
