#pragma once
#include "mediapipe_c_api.h"
#include "image.h"
#include <array>
#include <cmath>
#include <string>

struct FaceResult {
    bool detected = false;
    float score = 0.0f;
    std::array<float, 478 * 3> landmarks{};  // x,y,z in full-image pixel coords
    std::array<float, 52> blendshapes{};
    // Head rotation Euler angles in DEGREES: yaw (Y), pitch (X), roll (Z)
    float yaw = 0, pitch = 0, roll = 0;
    // Bounding box in image coords
    float bboxX1 = 0, bboxY1 = 0, bboxX2 = 0, bboxY2 = 0;
};

class FaceTracker {
public:
    FaceTracker(const std::string& modelDir);
    ~FaceTracker();

    void detect(const Image& bgr, FaceResult& result);

private:
    MpFaceLandmarkerPtr landmarker_ = nullptr;
    int64_t timestampMs_ = 0;
};
