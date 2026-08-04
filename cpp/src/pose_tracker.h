#pragma once
#include "onnx_session.h"
#include "image.h"
#include <array>
#include <cmath>

// COCO 17-keypoint indices (RTMO output)
struct PoseLandmarkIdx {
    static constexpr int NOSE = 0;
    static constexpr int L_EYE = 1, R_EYE = 2;
    static constexpr int L_EAR = 3, R_EAR = 4;
    static constexpr int L_SHOULDER = 5, R_SHOULDER = 6;
    static constexpr int L_ELBOW = 7, R_ELBOW = 8;
    static constexpr int L_WRIST = 9, R_WRIST = 10;
    static constexpr int L_HIP = 11, R_HIP = 12;
    static constexpr int L_KNEE = 13, R_KNEE = 14;
    static constexpr int L_ANKLE = 15, R_ANKLE = 16;
    static constexpr int NUM = 17;
};

// Pose detection result from RTMO (single-stage, detector+pose in one model)
struct PoseResult {
    bool detected = false;
    float score = 0.0f;
    float bboxX = 0.0f, bboxY = 0.0f, bboxW = 0.0f, bboxH = 0.0f;

    // 17 COCO keypoints: x, y, score — in ORIGINAL frame pixel coordinates
    std::array<float, PoseLandmarkIdx::NUM * 3> keypoints{};

    float kpX(int i) const { return keypoints[i * 3 + 0]; }
    float kpY(int i) const { return keypoints[i * 3 + 1]; }
    float kpScore(int i) const { return keypoints[i * 3 + 2]; }
};

class PoseTracker {
public:
    PoseTracker(const std::string& modelPath);

    void detect(const Image& bgr, PoseResult& result);

private:
    std::unique_ptr<OnnxSession> session_;
    static constexpr int INPUT_SIZE = 640;
};
