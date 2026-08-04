#pragma once
#include "onnx_session.h"
#include "image.h"
#include <array>
#include <cmath>

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
    float handedness = 0.0f;  // sigmoid score (0=right, 1=left typically)
    float presence = 0.0f;    // sigmoid score (confidence a hand is in frame)

    // 21 landmarks, each x,y,z (pixel coords in the 224×224 letterbox canvas)
    std::array<float, 21 * 3> landmarks{};

    // 21 world landmarks, each x,y,z (metric, meters)
    std::array<float, 21 * 3> worldLandmarks{};

    // Letterbox params for un-projecting landmarks to original frame coords
    float lbScale = 1.0f, lbPadX = 0.0f, lbPadY = 0.0f;
    int roiX = 0, roiY = 0;

    // Anchor position (e.g. pose tracker's wrist keypoint) in original frame
    // pixel coords. Used to lock the hand wrist (lm 0) to the arm endpoint so
    // the hand never appears detached from the wrist.
    float anchorX = -1.0f, anchorY = -1.0f;

    // Elbow→wrist distance in frame pixels (from pose tracker). Used to
    // rescale landmarks: the landmarker outputs fixed-scale landmarks
    // (~115px wrist→midtip on 224-canvas) regardless of actual hand size,
    // so we rescale to the anthropometric hand length (≈0.7× forearm).
    float armLen = 0.0f;

    float lmX(int i) const { return landmarks[i * 3 + 0]; }
    float lmY(int i) const { return landmarks[i * 3 + 1]; }
    float lmZ(int i) const { return landmarks[i * 3 + 2]; }
    float wlX(int i) const { return worldLandmarks[i * 3 + 0]; }
    float wlY(int i) const { return worldLandmarks[i * 3 + 1]; }
    float wlZ(int i) const { return worldLandmarks[i * 3 + 2]; }
};

class HandTracker {
public:
    HandTracker(const std::string& modelPath);

    void detect(const Image& bgr, HandResult& result);

private:
    std::unique_ptr<OnnxSession> landmarker_;

    Image letterbox(const Image& src, int size, float& scale, float& padX, float& padY);
};
