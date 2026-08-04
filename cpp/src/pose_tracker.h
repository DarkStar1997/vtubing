#pragma once
#include "onnx_session.h"
#include "image.h"
#include <array>
#include <cmath>

// BlazePose 33 body landmark indices
struct PoseLandmarkIdx {
    static constexpr int NOSE = 0;
    static constexpr int L_SHOULDER = 11;
    static constexpr int R_SHOULDER = 12;
    static constexpr int L_ELBOW = 13;
    static constexpr int R_ELBOW = 14;
    static constexpr int L_WRIST = 15;
    static constexpr int R_WRIST = 16;
    static constexpr int L_HIP = 23;
    static constexpr int R_HIP = 24;
    static constexpr int L_KNEE = 25;
    static constexpr int R_KNEE = 26;
    static constexpr int L_ANKLE = 27;
    static constexpr int R_ANKLE = 28;
};

struct PoseResult {
    bool detected = false;
    float presence = 0.0f;  // person presence [0,1]

    // 39 landmarks, each: x, y, z, visibility (image-space, normalized [0,1])
    // Only first 33 are body landmarks (BlazePose), 33-38 are auxiliary
    std::array<float, 39 * 4> landmarks{};

    // 39 world landmarks, each: x, y, z (metric, meters)
    std::array<float, 39 * 3> worldLandmarks{};

    // Convenience accessors
    float lmX(int i) const { return landmarks[i * 4 + 0]; }
    float lmY(int i) const { return landmarks[i * 4 + 1]; }
    float lmZ(int i) const { return landmarks[i * 4 + 2]; }
    float lmVis(int i) const { return landmarks[i * 4 + 3]; }
    float wlX(int i) const { return worldLandmarks[i * 3 + 0]; }
    float wlY(int i) const { return worldLandmarks[i * 3 + 1]; }
    float wlZ(int i) const { return worldLandmarks[i * 3 + 2]; }
};

class PoseTracker {
public:
    PoseTracker(const std::string& modelPath);

    void detect(const Image& bgr, PoseResult& result);

private:
    std::unique_ptr<OnnxSession> landmarker_;

    // Letterbox image to target size, returning scale and padding
    Image letterbox(const Image& src, int size, float& scale, float& padX, float& padY);
};
