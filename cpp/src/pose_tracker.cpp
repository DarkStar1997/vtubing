#include "pose_tracker.h"
#include <algorithm>

PoseTracker::PoseTracker(const std::string& modelPath) {
    landmarker_ = std::make_unique<OnnxSession>(modelPath);
}

Image PoseTracker::letterbox(const Image& src, int size, float& scale, float& padX, float& padY) {
    int h = src.height, w = src.width;
    scale = (float)size / std::max(h, w);
    padX = (size - w * scale) / 2.0f;
    padY = (size - h * scale) / 2.0f;
    float M[6] = {scale, 0, padX, 0, scale, padY};
    return warpAffine(src, M, size, size);
}

void PoseTracker::detect(const Image& bgr, PoseResult& result) {
    result.detected = false;

    if (bgr.empty()) return;

    const int SIZE = 256;
    float scale, padX, padY;
    Image canvas = letterbox(bgr, SIZE, scale, padX, padY);

    // Preprocess: NHWC float32 [0,1], BGR→RGB
    // Input shape: [1, 256, 256, 3]
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

    // Output[0]: [1, 195] = 39 landmarks × 5 (x, y, z, visibility, presence)
    // Output[1]: [1, 1] = person presence (logit, needs sigmoid)
    // Output[4]: [1, 117] = 39 landmarks × 3 world coords (metric, meters)
    if (outputs.size() < 5) return;

    auto& lmRaw = outputs[0];   // 195 values
    auto& presenceRaw = outputs[1]; // 1 value
    auto& worldRaw = outputs[4];    // 117 values

    // Person presence: sigmoid
    float presenceLogit = presenceRaw[0];
    result.presence = 1.0f / (1.0f + std::exp(-presenceLogit));

    if (result.presence < 0.3f) return;

    result.detected = true;

    // Parse 39 landmarks
    for (int i = 0; i < 39; i++) {
        float px = lmRaw[i * 5 + 0];
        float py = lmRaw[i * 5 + 1];
        float pz = lmRaw[i * 5 + 2];
        float visLogit = lmRaw[i * 5 + 3];
        float vis = 1.0f / (1.0f + std::exp(-visLogit));

        // Normalize pixel coords to [0,1] relative to input canvas
        result.landmarks[i * 4 + 0] = px / SIZE;
        result.landmarks[i * 4 + 1] = py / SIZE;
        result.landmarks[i * 4 + 2] = pz / SIZE;
        result.landmarks[i * 4 + 3] = vis;
    }

    // Parse world landmarks
    for (int i = 0; i < 39; i++) {
        result.worldLandmarks[i * 3 + 0] = worldRaw[i * 3 + 0];
        result.worldLandmarks[i * 3 + 1] = worldRaw[i * 3 + 1];
        result.worldLandmarks[i * 3 + 2] = worldRaw[i * 3 + 2];
    }
}
