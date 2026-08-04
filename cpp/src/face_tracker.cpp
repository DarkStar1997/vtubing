#include "face_tracker.h"
#include <algorithm>
#include <cstring>

FaceTracker::FaceTracker(const std::string& modelDir) {
    detector_ = std::make_unique<OnnxSession>(modelDir + "/face_detection_short_range.onnx");
    landmarker_ = std::make_unique<OnnxSession>(modelDir + "/face_landmarker.onnx");
    blendshape_ = std::make_unique<OnnxSession>(modelDir + "/face_blendshapes.onnx");
    generateAnchors();
}

void FaceTracker::generateAnchors() {
    // MediaPipe SSD anchors for BlazeFace 128×128
    // strides = [8, 16, 16, 16], grouped by stride
    int strides[] = {8, 16, 16, 16};
    int idx = 0, anchorIdx = 0;
    while (idx < 4) {
        int last = idx;
        while (last < 4 && strides[last] == strides[idx]) last++;
        int repeats = 2 * (last - idx);
        int cells = 128 / strides[idx];
        for (int y = 0; y < cells; y++) {
            for (int x = 0; x < cells; x++) {
                float cx = (x + 0.5f) / cells;
                float cy = (y + 0.5f) / cells;
                for (int r = 0; r < repeats; r++) {
                    anchors_[anchorIdx * 2] = cx;
                    anchors_[anchorIdx * 2 + 1] = cy;
                    anchorIdx++;
                }
            }
        }
        idx = last;
    }
}

cv::Mat FaceTracker::letterbox(const cv::Mat& src, int size, float& scale, float& padX, float& padY) {
    int h = src.rows, w = src.cols;
    scale = (float)size / std::max(h, w);
    padX = (size - w * scale) / 2.0f;
    padY = (size - h * scale) / 2.0f;
    float mdata[6] = {scale, 0, padX, 0, scale, padY};
    cv::Mat m(2, 3, CV_32F, mdata);
    cv::Mat out;
    cv::warpAffine(src, out, m, cv::Size(size, size));
    return out;
}

std::vector<FaceTracker::Detection> FaceTracker::runDetector(const cv::Mat& bgr, int imgW, int imgH) {
    float scale, padX, padY;
    cv::Mat canvas = letterbox(bgr, 128, scale, padX, padY);

    // Preprocess: BGR→RGB, normalize [-1,1], transpose CHW
    std::vector<float> blob(3 * 128 * 128);
    for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
            auto* px = canvas.ptr<cv::Vec3b>(y, x);
            blob[0 * 128 * 128 + y * 128 + x] = ((*px)[2] - 127.5f) / 127.5f; // R
            blob[1 * 128 * 128 + y * 128 + x] = ((*px)[1] - 127.5f) / 127.5f; // G
            blob[2 * 128 * 128 + y * 128 + x] = ((*px)[0] - 127.5f) / 127.5f; // B
        }
    }

    auto outputs = detector_->run(blob.data(), blob.size());
    // regressors: [1, 896, 16], scores: [1, 896, 1]
    auto& regs = outputs[0];  // 896*16
    auto& scs = outputs[1];   // 896*1

    float threshold = 0.5f;
    auto unletterbox = [&](float nx, float ny) {
        return std::make_pair(
            (nx * 128.0f - padX) / scale,
            (ny * 128.0f - padY) / scale);
    };

    std::vector<Detection> dets;
    for (int i = 0; i < 896; i++) {
        float logit = scs[i];
        float score = 1.0f / (1.0f + std::exp(-logit));
        if (score < threshold) continue;

        Detection d;
        d.score = score;
        float ax = anchors_[i * 2], ay = anchors_[i * 2 + 1];
        d.cx = regs[i * 16 + 0] / 128.0f + ax;
        d.cy = regs[i * 16 + 1] / 128.0f + ay;
        d.w = regs[i * 16 + 2] / 128.0f;
        d.h = regs[i * 16 + 3] / 128.0f;
        for (int k = 0; k < 6; k++) {
            d.kp[k][0] = regs[i * 16 + 4 + 2 * k] / 128.0f + ax;
            d.kp[k][1] = regs[i * 16 + 5 + 2 * k] / 128.0f + ay;
        }
        dets.push_back(d);
    }
    return weightedNms(dets);
}

std::vector<FaceTracker::Detection> FaceTracker::weightedNms(std::vector<Detection>& dets) {
    if (dets.empty()) return {};

    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });

    std::vector<Detection> output;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        Detection blended = dets[i];
        float weightSum = dets[i].score;

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            float iou;
            {
                float ix1 = std::max(blended.cx - blended.w / 2, dets[j].cx - dets[j].w / 2);
                float iy1 = std::max(blended.cy - blended.h / 2, dets[j].cy - dets[j].h / 2);
                float ix2 = std::min(blended.cx + blended.w / 2, dets[j].cx + dets[j].w / 2);
                float iy2 = std::min(blended.cy + blended.h / 2, dets[j].cy + dets[j].h / 2);
                float iw = std::max(0.0f, ix2 - ix1);
                float ih = std::max(0.0f, iy2 - iy1);
                float inter = iw * ih;
                float uni = blended.w * blended.h + dets[j].w * dets[j].h - inter;
                iou = inter / std::max(uni, 1e-9f);
            }
            if (iou > 0.3f) {
                suppressed[j] = true;
                float w = dets[j].score;
                blended.cx = (blended.cx * weightSum + dets[j].cx * w) / (weightSum + w);
                blended.cy = (blended.cy * weightSum + dets[j].cy * w) / (weightSum + w);
                blended.w = (blended.w * weightSum + dets[j].w * w) / (weightSum + w);
                blended.h = (blended.h * weightSum + dets[j].h * w) / (weightSum + w);
                for (int k = 0; k < 6; k++) {
                    blended.kp[k][0] = (blended.kp[k][0] * weightSum + dets[j].kp[k][0] * w) / (weightSum + w);
                    blended.kp[k][1] = (blended.kp[k][1] * weightSum + dets[j].kp[k][1] * w) / (weightSum + w);
                }
                weightSum += w;
            }
        }
        output.push_back(blended);
    }
    return output;
}

void FaceTracker::detect(const cv::Mat& bgr, FaceResult& result) {
    result.detected = false;
    int imgW = bgr.cols, imgH = bgr.rows;

    // 1. Run BlazeFace detector
    auto dets = runDetector(bgr, imgW, imgH);
    if (dets.empty()) return;

    auto& det = dets[0]; // take highest-score face

    // 2. ROI extraction: square, 1.5× scale, rotated by eye angle
    // Un-letterbox detection from 128px normalized to full image coords
    float scale, padX, padY;
    {
        int h = bgr.rows, w = bgr.cols;
        scale = 128.0f / std::max(h, w);
        padX = (128 - w * scale) / 2.0f;
        padY = (128 - h * scale) / 2.0f;
    }

    auto unletterbox = [&](float nx, float ny) {
        return std::make_pair(
            (nx * 128.0f - padX) / scale,
            (ny * 128.0f - padY) / scale);
    };

    auto [fullCx, fullCy] = unletterbox(det.cx, det.cy);
    float fullW = det.w * 128.0f / scale;
    float fullH = det.h * 128.0f / scale;
    float margin = 0.25f;
    float fullSide = (1.0f + 2.0f * margin) * std::max(fullW, fullH);

    // Keypoints to full image coords (for rotation angle)
    float kpFull[6][2];
    for (int k = 0; k < 6; k++) {
        auto [kx, ky] = unletterbox(det.kp[k][0], det.kp[k][1]);
        kpFull[k][0] = kx;
        kpFull[k][1] = ky;
    }

    // ROI rotation angle from eye keypoints (row 1 - row 0 = right - left eye)
    float dx = kpFull[1][0] - kpFull[0][0];
    float dy = kpFull[1][1] - kpFull[0][1];
    float angle = std::atan2(dy, dx) * 180.0f / 3.14159265f;

    // 3. Warp ROI to 256×256
    int modelSize = 256;
    cv::Mat rotM = cv::getRotationMatrix2D(cv::Point2f(fullCx, fullCy), angle,
                                            (double)modelSize / fullSide);
    rotM.at<double>(0, 2) += (double)modelSize / 2.0 - fullCx;
    rotM.at<double>(1, 2) += (double)modelSize / 2.0 - fullCy;

    cv::Mat crop;
    cv::warpAffine(bgr, crop, rotM, cv::Size(modelSize, modelSize));

    // Inverse affine for landmark un-projection
    cv::Mat invM(2, 3, CV_64F);
    cv::invertAffineTransform(rotM, invM);

    // 4. Preprocess crop: BGR→RGB, normalize [0,1], transpose CHW
    std::vector<float> meshBlob(3 * modelSize * modelSize);
    for (int y = 0; y < modelSize; y++) {
        for (int x = 0; x < modelSize; x++) {
            auto* px = crop.ptr<cv::Vec3b>(y, x);
            meshBlob[0 * modelSize * modelSize + y * modelSize + x] = (*px)[2] / 255.0f; // R
            meshBlob[1 * modelSize * modelSize + y * modelSize + x] = (*px)[1] / 255.0f; // G
            meshBlob[2 * modelSize * modelSize + y * modelSize + x] = (*px)[0] / 255.0f; // B
        }
    }

    // 5. Run FaceLandmarker
    auto meshOut = landmarker_->run(meshBlob.data(), meshBlob.size());
    // landmarks: [1, 478, 3], score: [1, 1]
    auto& lms = meshOut[0];
    float presenceScore = 1.0f / (1.0f + std::exp(-meshOut[1][0]));

    if (presenceScore < 0.5f) return;

    result.detected = true;
    result.score = presenceScore;

    // 6. Un-project landmarks to full image coords
    for (int i = 0; i < 478; i++) {
        float lx = lms[i * 3], ly = lms[i * 3 + 1], lz = lms[i * 3 + 2];
        // Apply inverse affine to x,y
        float fx = (float)(invM.at<double>(0, 0) * lx + invM.at<double>(0, 1) * ly + invM.at<double>(0, 2));
        float fy = (float)(invM.at<double>(1, 0) * lx + invM.at<double>(1, 1) * ly + invM.at<double>(1, 2));
        // Scale z to image pixel scale
        float fz = lz * fullSide / modelSize;
        result.landmarks[i * 3] = fx;
        result.landmarks[i * 3 + 1] = fy;
        result.landmarks[i * 3 + 2] = fz;
    }

    // 7. BlendshapeV2: select 146 indices, denormalize to pixel coords
    std::vector<float> bsInput(146 * 2);
    for (int i = 0; i < 146; i++) {
        int idx = kBlendshapeIdxs[i];
        bsInput[i * 2] = result.landmarks[idx * 3];
        bsInput[i * 2 + 1] = result.landmarks[idx * 3 + 1];
    }

    auto bsOut = blendshape_->run(bsInput.data(), bsInput.size());
    std::memcpy(result.blendshapes.data(), bsOut[0].data(), sizeof(float) * 52);

    // 8. Head pose estimation (simplified — not full solvePnP)
    // Roll from eye line angle (landmarks 33=left eye, 263=right eye)
    float lex = result.landmarks[33 * 3], ley = result.landmarks[33 * 3 + 1];
    float rex = result.landmarks[263 * 3], rey = result.landmarks[263 * 3 + 1];
    result.roll = std::atan2(rey - ley, rex - lex);

    // Pitch from nose tip vertical position relative to face height
    float noseY = result.landmarks[1 * 3 + 1];
    float foreheadY = result.landmarks[10 * 3 + 1];
    float chinY = result.landmarks[152 * 3 + 1];
    float faceH = chinY - foreheadY;
    if (faceH > 1.0f) {
        float noseRel = (noseY - foreheadY) / faceH;  // ~0.5 neutral
        result.pitch = (noseRel - 0.5f) * 1.5f;  // simplified
    }
    result.yaw = 0;  // TODO: proper solvePnP

    // Bounding box
    result.bboxX1 = fullCx - fullW / 2;
    result.bboxY1 = fullCy - fullH / 2;
    result.bboxX2 = fullCx + fullW / 2;
    result.bboxY2 = fullCy + fullH / 2;
}
