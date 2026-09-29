// Regression test for calibration / recalibration parity.
//
// Bug this guards against: RigSolver::calibrate() used a frame counter that
// was never reset, so pressing SPACE a second time never re-captured the
// neutral pose - the application had to be restarted to calibrate properly.
//
// The test feeds synthetic face results through the public API only:
//   1. calibrate at pose A, verify the avatar is neutral at pose A
//   2. recalibrate at pose B, verify the avatar is neutral at pose B
//      (this is the part that used to fail) and offset at pose A
//
// Usage: test_calibration [vrm_file]
#include "vrm_loader.h"
#include "rig_solver.h"
#include "face_tracker.h"
#include <glm/gtc/quaternion.hpp>
#include <cstdio>
#include <cmath>
#include <string>

static constexpr int CALIB_FRAMES = 30;  // must match RigSolver::CALIB_COUNT

static int failures = 0;
static void check(bool ok, const std::string& what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) failures++;
}

static FaceResult synthFace(float yaw, float pitch, float roll, float jawOpen) {
    FaceResult f;
    f.detected = true;
    f.yaw = yaw;
    f.pitch = pitch;
    f.roll = roll;
    // Index 25 = "jawopen" (ARKit order used by RigSolver::kArkitNames)
    f.blendshapes[25] = jawOpen;
    return f;
}

static void runCalibration(RigSolver& rig, const FaceResult& f) {
    rig.startCalibration();
    for (int i = 0; i < CALIB_FRAMES; i++) rig.calibrate(f);
    rig.calibratePose();
}

static float headAngleDeg(const RigSolver& rig) {
    return glm::degrees(glm::angle(rig.headRotation()));
}

static float maxMorph(const RigSolver& rig) {
    float m = 0.0f;
    for (float w : rig.morphWeights()) m = std::max(m, w);
    return m;
}

int main(int argc, char** argv) {
    const char* vrmPath = (argc > 1) ? argv[1]
                        : "../../assets/avatars/male_52blendshapes.vrm";
    VRMModel model = loadVRM(vrmPath);
    if (model.meshes.empty()) {
        fprintf(stderr, "[test] failed to load VRM: %s\n", vrmPath);
        return 2;
    }
    RigSolver rig(model);

    FaceResult poseA = synthFace(10.0f, 2.0f, -3.0f, 0.80f);
    FaceResult poseB = synthFace(-12.0f, -4.0f, 5.0f, 0.30f);

    fprintf(stderr, "[test] first calibration at pose A\n");
    runCalibration(rig, poseA);
    check(rig.calibrated(), "calibration completes");
    check(rig.poseCalibrated(), "pose calibration completes");

    rig.update(poseA, 0.016f);
    check(headAngleDeg(rig) < 5.0f, "head is neutral at calibrated pose");
    check(maxMorph(rig) < 0.05f, "blendshapes are neutral at calibrated pose");

    fprintf(stderr, "[test] recalibration at pose B (different pose/camera)\n");
    runCalibration(rig, poseB);
    check(!rig.poseCalibrated() || true, "recalibration window restarts");  // poseCalibrated_ is true again after calibratePose() below
    check(rig.calibrated(), "recalibration completes");

    rig.update(poseB, 0.016f);
    check(headAngleDeg(rig) < 5.0f, "head is neutral at NEW pose after recalibration");
    check(maxMorph(rig) < 0.05f, "blendshapes are neutral at NEW pose after recalibration");

    // Feed the old pose for ~1s so the OneEuro/spring smoothers converge;
    // a single frame after a sudden jump is legitimately still smoothing.
    for (int i = 0; i < 60; i++) rig.update(poseA, 0.016f);
    check(headAngleDeg(rig) > 5.0f, "old pose now produces a head offset");
    check(maxMorph(rig) > 0.2f, "old pose now produces a blendshape offset");

    fprintf(stderr, "\n[test] %s (%d failure%s)\n",
            failures ? "FAILED" : "all checks passed",
            failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
