#pragma once
#include "mediapipe_c_api.h"
#include "image.h"
#include <array>
#include <cmath>
#include <string>

// MediaPipe 21 hand landmark indices
struct HandLandmarkIdx {
    static constexpr int WRIST = 0;
    static constexpr int THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
    static constexpr int INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
    static constexpr int MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
    static constexpr int RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
    static constexpr int LITTLE_MCP = 17, LITTLE_PIP = 18, LITTLE_DIP = 19, LITTLE_TIP = 20;
};

struct HandResult {
    bool detected = false;

    // 21 normalized landmarks (x,y,z in [0,1] image space)
    std::array<float, 21 * 3> landmarks{};

    // 21 world landmarks (x,y,z in metric meters)
    std::array<float, 21 * 3> worldLandmarks{};

    int frameW = 0, frameH = 0;

    float lmX(int i) const { return landmarks[i * 3 + 0]; }
    float lmY(int i) const { return landmarks[i * 3 + 1]; }
    float lmZ(int i) const { return landmarks[i * 3 + 2]; }
    float wlX(int i) const { return worldLandmarks[i * 3 + 0]; }
    float wlY(int i) const { return worldLandmarks[i * 3 + 1]; }
    float wlZ(int i) const { return worldLandmarks[i * 3 + 2]; }
    float pxX(int i) const { return lmX(i) * frameW; }
    float pxY(int i) const { return lmY(i) * frameH; }
};

class HandTracker {
public:
    HandTracker(const std::string& modelDir);
    ~HandTracker();

    // Detect hands on the full frame. Separates into left/right by
    // handedness classification.
    void detect(const Image& bgr, HandResult& left, HandResult& right);

private:
    MpHandLandmarkerPtr landmarker_ = nullptr;
    int64_t timestampMs_ = 0;
};
