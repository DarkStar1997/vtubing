#include "hand_tracker.h"
#include <algorithm>

HandTracker::HandTracker(const std::string& modelPath) {
    landmarker_ = std::make_unique<OnnxSession>(modelPath);
}

Image HandTracker::letterbox(const Image& src, int size, float& scale, float& padX, float& padY) {
    int h = src.height, w = src.width;
    scale = (float)size / std::max(h, w);
    padX = (size - w * scale) / 2.0f;
    padY = (size - h * scale) / 2.0f;
    float M[6] = {scale, 0, padX, 0, scale, padY};
    return warpAffine(src, M, size, size);
}

void HandTracker::detect(const Image& bgr, HandResult& result) {
    result.detected = false;
    if (bgr.empty()) return;

    const int SIZE = 224;
    float scale, padX, padY;
    Image canvas = letterbox(bgr, SIZE, scale, padX, padY);
    result.lbScale = scale;
    result.lbPadX = padX;
    result.lbPadY = padY;

    std::vector<float> blob(SIZE * SIZE * 3);
    for (int y = 0; y < SIZE; y++) {
        for (int x = 0; x < SIZE; x++) {
            const uint8_t* px = canvas.ptr(y, x);
            int idx = (y * SIZE + x) * 3;
            blob[idx + 0] = px[2] / 255.0f; // R
            blob[idx + 1] = px[1] / 255.0f; // G
            blob[idx + 2] = px[0] / 255.0f; // B
        }
    }

    auto outputs = landmarker_->run(blob.data(), blob.size());
    if (outputs.size() < 4) return;

    // Output 0: [1, 63] = 21 landmarks × 3 (pixel coords in 224×224 canvas)
    // Output 1: [1, 1]  = handedness (logit, sigmoid)
    // Output 2: [1, 1]  = presence (logit, sigmoid)
    // Output 3: [1, 63] = 21 world landmarks × 3 (metric)
    auto& lmRaw = outputs[0];
    auto& handedRaw = outputs[1];
    auto& presenceRaw = outputs[2];
    auto& worldRaw = outputs[3];

    float presence = 1.0f / (1.0f + std::exp(-presenceRaw[0]));
    // Threshold 0.5: empirically real hands (even with widely-visible fingers)
    // score in the 0.55-0.9 range on this landmarker-without-detector setup.
    // Higher thresholds reject valid detections; lower accepts garbage.
    if (presence < 0.5f) return;

    result.detected = true;
    result.presence = presence;
    result.handedness = 1.0f / (1.0f + std::exp(-handedRaw[0]));

    for (int i = 0; i < 21; i++) {
        result.landmarks[i * 3 + 0] = lmRaw[i * 3 + 0];
        result.landmarks[i * 3 + 1] = lmRaw[i * 3 + 1];
        result.landmarks[i * 3 + 2] = lmRaw[i * 3 + 2];
    }
    for (int i = 0; i < 21; i++) {
        result.worldLandmarks[i * 3 + 0] = worldRaw[i * 3 + 0];
        result.worldLandmarks[i * 3 + 1] = worldRaw[i * 3 + 1];
        result.worldLandmarks[i * 3 + 2] = worldRaw[i * 3 + 2];
    }
}
