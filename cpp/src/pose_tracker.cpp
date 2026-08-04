#include "pose_tracker.h"
#include <algorithm>

PoseTracker::PoseTracker(const std::string& modelPath) {
    session_ = std::make_unique<OnnxSession>(modelPath);
}

void PoseTracker::detect(const Image& bgr, PoseResult& result) {
    result.detected = false;
    if (bgr.empty()) return;

    const int S = INPUT_SIZE;  // 640

    // Letterbox: resize maintaining aspect ratio, pad to S×S
    float scale = (float)S / std::max(bgr.width, bgr.height);
    int newW = (int)(bgr.width * scale);
    int newH = (int)(bgr.height * scale);
    int padX = (S - newW) / 2;
    int padY = (S - newH) / 2;

    Image resized = resizeBilinear(bgr, newW, newH);
    Image canvas(S, S, bgr.channels);
    memset(canvas.data.data(), 0, canvas.data.size());
    for (int y = 0; y < newH; y++) {
        const uint8_t* src = resized.ptr(y, 0);
        uint8_t* dst = canvas.ptr(y + padY, padX);
        memcpy(dst, src, newW * bgr.channels);
    }

    // Preprocess: NCHW float32 [0,1], BGR→RGB
    std::vector<float> blob(3 * S * S);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            const uint8_t* px = canvas.ptr(y, x);
            blob[(0 * S + y) * S + x] = px[2] / 255.0f;  // R
            blob[(1 * S + y) * S + x] = px[1] / 255.0f;  // G
            blob[(2 * S + y) * S + x] = px[0] / 255.0f;  // B
        }
    }

    auto outputs = session_->run(blob.data(), blob.size());
    if (outputs.size() < 2) return;

    // Output[0]: dets [1, N, 5] — cx, cy, w, h, score (in 640×640 pixel space)
    // Output[1]: keypoints [1, N, 17, 3] — x, y, score (in 640×640 pixel space)
    auto& dets = outputs[0];
    auto& kps = outputs[1];

    int nPersons = (int)dets.size() / 5;
    if (nPersons == 0) return;

    // Find best person
    int bestIdx = 0;
    float bestScore = dets[4];
    for (int i = 1; i < nPersons; i++) {
        if (dets[i * 5 + 4] > bestScore) {
            bestScore = dets[i * 5 + 4];
            bestIdx = i;
        }
    }

    if (bestScore < 0.3f) return;

    result.detected = true;
    result.score = bestScore;

    // Un-letterbox: convert from 640×640 space to original frame coords
    auto unlbX = [&](float v) { return (v - padX) / scale; };
    auto unlbY = [&](float v) { return (v - padY) / scale; };

    // Bounding box
    result.bboxX = unlbX(dets[bestIdx * 5 + 0]);
    result.bboxY = unlbY(dets[bestIdx * 5 + 1]);
    result.bboxW = dets[bestIdx * 5 + 2] / scale;
    result.bboxH = dets[bestIdx * 5 + 3] / scale;

    // Keypoints
    const float* kpBase = kps.data() + bestIdx * PoseLandmarkIdx::NUM * 3;
    for (int i = 0; i < PoseLandmarkIdx::NUM; i++) {
        result.keypoints[i * 3 + 0] = unlbX(kpBase[i * 3 + 0]);
        result.keypoints[i * 3 + 1] = unlbY(kpBase[i * 3 + 1]);
        result.keypoints[i * 3 + 2] = kpBase[i * 3 + 2];
    }
}
