#pragma once
#include "image.h"
#include "mediapipe_c_api.h"
#include <array>
#include <cmath>
#include <string>

// BlazePose 33-landmark indices (MediaPipe convention)
struct PoseLandmarkIdx {
    static constexpr int NOSE = 0;
    static constexpr int L_EYE_INNER = 1, R_EYE_INNER = 4;
    static constexpr int L_EYE = 2, R_EYE = 5;
    static constexpr int L_EYE_OUTER = 3, R_EYE_OUTER = 6;
    static constexpr int L_EAR = 7, R_EAR = 8;
    static constexpr int MOUTH_L = 9, MOUTH_R = 10;
    static constexpr int L_SHOULDER = 11, R_SHOULDER = 12;
    static constexpr int L_ELBOW = 13, R_ELBOW = 14;
    static constexpr int L_WRIST = 15, R_WRIST = 16;
    static constexpr int L_PINKY = 17, R_PINKY = 18;
    static constexpr int L_INDEX = 19, R_INDEX = 20;
    static constexpr int L_THUMB = 21, R_THUMB = 22;
    static constexpr int L_HIP = 23, R_HIP = 24;
    static constexpr int L_KNEE = 25, R_KNEE = 26;
    static constexpr int L_ANKLE = 27, R_ANKLE = 28;
    static constexpr int L_HEEL = 29, R_HEEL = 30;
    static constexpr int L_FOOT = 31, R_FOOT = 32;
    static constexpr int NUM = 33;
};

struct PoseResult {
    bool detected = false;
    float presence = 0.0f;

    int frameW = 0, frameH = 0;

    // 33 normalized image landmarks: x, y, z, visibility, presence
    std::array<float, PoseLandmarkIdx::NUM * 5> landmarks{};

    // 33 world landmarks: x, y, z (metric meters, hip-origin)
    std::array<float, PoseLandmarkIdx::NUM * 3> worldLandmarks{};

    float lmX(int i) const { return landmarks[i * 5 + 0]; }
    float lmY(int i) const { return landmarks[i * 5 + 1]; }
    float lmZ(int i) const { return landmarks[i * 5 + 2]; }
    float lmVis(int i) const { return landmarks[i * 5 + 3]; }
    float lmPres(int i) const { return landmarks[i * 5 + 4]; }

    // Frame pixel coords from normalized [0,1]
    float frameX(int i) const { return lmX(i) * frameW; }
    float frameY(int i) const { return lmY(i) * frameH; }

    float wlX(int i) const { return worldLandmarks[i * 3 + 0]; }
    float wlY(int i) const { return worldLandmarks[i * 3 + 1]; }
    float wlZ(int i) const { return worldLandmarks[i * 3 + 2]; }
};

class PoseTracker {
public:
    PoseTracker(const std::string& modelDir);
    ~PoseTracker();

    void detect(const Image& bgr, PoseResult& result);

private:
    MpPoseLandmarkerPtr landmarker_ = nullptr;
    int64_t timestampMs_ = 0;
};
