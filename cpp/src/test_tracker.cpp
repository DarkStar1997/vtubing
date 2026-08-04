#include "face_tracker.h"
#include "vrm_loader.h"
#include "rig_solver.h"
#include "image.h"
#include <stb_image.h>   // declarations only (impl compiled in vrm_loader.cpp)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image.jpg> [models_dir]\n", argv[0]);
        return 1;
    }
    std::string imagePath = argv[1];
    std::string modelDir = argc > 2 ? argv[2] : "../../assets/models";

    // Load image (stb returns RGB); convert to BGR for the pipeline.
    int iw = 0, ih = 0, ic = 0;
    stbi_uc* pixels = stbi_load(imagePath.c_str(), &iw, &ih, &ic, 3);
    if (!pixels) {
        fprintf(stderr, "Failed to load: %s\n", imagePath.c_str());
        return 1;
    }
    Image img(iw, ih, 3);
    for (int i = 0; i < iw * ih; i++) {
        img.data[i*3+0] = pixels[i*3+2];  // B
        img.data[i*3+1] = pixels[i*3+1];  // G
        img.data[i*3+2] = pixels[i*3+0];  // R
    }
    stbi_image_free(pixels);
    fprintf(stderr, "[test] Image: %dx%d\n", img.width, img.height);

    FaceTracker tracker(modelDir);
    fprintf(stderr, "[test] Models loaded.\n");

    FaceResult result;
    tracker.detect(img, result);

    if (!result.detected) {
        fprintf(stderr, "[test] No face detected!\n");

        // Save input for debugging (convert BGR→RGB for stb)
        std::vector<uint8_t> rgb(img.width * img.height * 3);
        for (int i = 0; i < img.width * img.height; i++) {
            rgb[i*3+0] = img.data[i*3+2];
            rgb[i*3+1] = img.data[i*3+1];
            rgb[i*3+2] = img.data[i*3+0];
        }
        stbi_write_jpg("/tmp/debug_input.jpg", img.width, img.height, 3, rgb.data(), 90);
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

    // Draw annotated output (work on a copy of the input)
    Image annotated = img;
    uint8_t green[3] = {0, 255, 0};
    uint8_t yellow[3] = {0, 255, 255};

    // Draw bbox
    drawRect(annotated,
        (int)result.bboxX1, (int)result.bboxY1,
        (int)result.bboxX2, (int)result.bboxY2,
        green, 2);

    // Draw all 478 landmarks
    for (int i = 0; i < 478; i++) {
        drawCircleFilled(annotated,
            (int)result.landmarks[i*3],
            (int)result.landmarks[i*3+1],
            1, yellow);
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

    // Save (convert BGR→RGB for stb)
    std::vector<uint8_t> rgb(annotated.width * annotated.height * 3);
    for (int i = 0; i < annotated.width * annotated.height; i++) {
        rgb[i*3+0] = annotated.data[i*3+2];
        rgb[i*3+1] = annotated.data[i*3+1];
        rgb[i*3+2] = annotated.data[i*3+0];
    }
    stbi_write_jpg("/tmp/tracker_output.jpg", annotated.width, annotated.height, 3, rgb.data(), 90);
    fprintf(stderr, "\n[test] Saved annotated output to /tmp/tracker_output.jpg\n");

    return 0;
}
