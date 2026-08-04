#include "face_tracker.h"
#include "vrm_loader.h"
#include "rig_solver.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image.jpg> [models_dir]\n", argv[0]);
        return 1;
    }
    std::string imagePath = argv[1];
    std::string modelDir = argc > 2 ? argv[2] : "../../assets/models";

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        fprintf(stderr, "Failed to load: %s\n", imagePath.c_str());
        return 1;
    }
    fprintf(stderr, "[test] Image: %dx%d\n", img.cols, img.rows);

    FaceTracker tracker(modelDir);
    fprintf(stderr, "[test] Models loaded.\n");

    FaceResult result;
    tracker.detect(img, result);

    if (!result.detected) {
        fprintf(stderr, "[test] No face detected!\n");

        // Save letterboxed detection input for debugging
        cv::imwrite("/tmp/debug_input.jpg", img);
        fprintf(stderr, "[test] Saved input to /tmp/debug_input.jpg\n");
        return 1;
    }

    fprintf(stderr, "[test] Face detected! score=%.3f\n", result.score);
    fprintf(stderr, "  bbox: (%.0f,%.0f)-(%.0f,%.0f)\n",
            result.bboxX1, result.bboxY1, result.bboxX2, result.bboxY2);
    fprintf(stderr, "  roll=%.3f pitch=%.3f yaw=%.3f\n",
            result.roll, result.pitch, result.yaw);

    // Print non-zero blendshapes
    static const char* bsNames[52] = {
        "_neutral","browInnerUp","browDownLeft","browDownRight",
        "browOuterUpLeft","browOuterUpRight","cheekPuff","cheekSquintLeft",
        "cheekSquintRight","eyeBlinkLeft","eyeBlinkRight",
        "eyeLookDownLeft","eyeLookDownRight","eyeLookInLeft","eyeLookInRight",
        "eyeLookOutLeft","eyeLookOutRight","eyeLookUpLeft","eyeLookUpRight",
        "eyeSquintLeft","eyeSquintRight","eyeWideLeft","eyeWideRight",
        "jawForward","jawLeft","jawOpen","jawRight",
        "mouthClose","mouthDimpleLeft","mouthDimpleRight",
        "mouthFrownLeft","mouthFrownRight","mouthFunnel",
        "mouthLeft","mouthLowerDownLeft","mouthLowerDownRight",
        "mouthPressLeft","mouthPressRight","mouthPucker",
        "mouthRight","mouthRollLower","mouthRollUpper",
        "mouthShrugLower","mouthShrugUpper",
        "mouthSmileLeft","mouthSmileRight",
        "mouthStretchLeft","mouthStretchRight",
        "mouthUpperUpLeft","mouthUpperUpRight",
        "noseSneerLeft","noseSneerRight"
    };

    fprintf(stderr, "\n  Blendshapes (non-zero):\n");
    for (int i = 0; i < 52; i++) {
        if (result.blendshapes[i] > 0.01f)
            fprintf(stderr, "    [%2d] %-20s = %.3f\n", i, bsNames[i], result.blendshapes[i]);
    }

    // Draw annotated output
    cv::Mat annotated;
    cv::cvtColor(img, annotated, cv::COLOR_BGR2RGB);
    cv::cvtColor(annotated, annotated, cv::COLOR_RGB2BGR);

    // Draw bbox
    cv::rectangle(annotated,
        cv::Point((int)result.bboxX1, (int)result.bboxY1),
        cv::Point((int)result.bboxX2, (int)result.bboxY2),
        cv::Scalar(0, 255, 0), 2);

    // Draw all 478 landmarks
    for (int i = 0; i < 478; i++) {
        cv::circle(annotated,
            cv::Point((int)result.landmarks[i*3], (int)result.landmarks[i*3+1]),
            1, cv::Scalar(0, 255, 255), -1);
    }

    // Test VRM mapping
    VRMModel vrm = loadVRM("../../assets/avatars/male_52blendshapes.vrm");
    RigSolver rig(vrm);
    rig.calibrate(result); // single-frame "calibration" (not ideal but works for testing)
    // Manually set calibrated
    rig.update(result, 0.016f);

    fprintf(stderr, "\n  VRM morph weights (non-zero):\n");
    const auto& mw = rig.morphWeights();
    for (size_t i = 0; i < mw.size(); i++) {
        if (mw[i] > 0.01f)
            fprintf(stderr, "    morph[%3zu] = %.3f\n", i, mw[i]);
    }

    // Save
    cv::imwrite("/tmp/tracker_output.jpg", annotated);
    fprintf(stderr, "\n[test] Saved annotated output to /tmp/tracker_output.jpg\n");

    return 0;
}
