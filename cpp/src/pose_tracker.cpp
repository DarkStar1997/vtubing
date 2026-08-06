#include "pose_tracker.h"
#include <cstdio>
#include <cstring>

PoseTracker::PoseTracker(const std::string& modelDir) {
    std::string modelPath = modelDir + "/pose_landmarker_full.task";

    struct MpPoseLandmarkerOptions opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.base_options.model_asset_path = modelPath.c_str();
    opts.base_options.delegate = MP_DELEGATE_CPU;
    opts.running_mode = MP_RUNNING_MODE_VIDEO;
    opts.num_poses = 1;
    opts.min_pose_detection_confidence = 0.5f;
    opts.min_pose_presence_confidence = 0.5f;
    opts.min_tracking_confidence = 0.5f;
    opts.output_segmentation_masks = false;
    opts.result_callback = nullptr;

    char* errorMsg = nullptr;
    MpStatus st = MpPoseLandmarkerCreate(&opts, &landmarker_, &errorMsg);
    if (st != kMpOk) {
        fprintf(stderr, "[pose] MpPoseLandmarkerCreate failed: %s\n",
                errorMsg ? errorMsg : "(unknown)");
        if (errorMsg) MpErrorFree(errorMsg);
        landmarker_ = nullptr;
    }
}

PoseTracker::~PoseTracker() {
    if (landmarker_) {
        char* errorMsg = nullptr;
        MpPoseLandmarkerClose(landmarker_, &errorMsg);
        if (errorMsg) MpErrorFree(errorMsg);
    }
}

void PoseTracker::detect(const Image& bgr, PoseResult& result) {
    result.detected = false;
    if (!landmarker_ || bgr.empty()) return;

    // MediaPipe expects RGB; our Image is BGR → swap channels
    int w = bgr.width, h = bgr.height;
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        rgb[i * 3 + 0] = bgr.data[i * 3 + 2];  // R
        rgb[i * 3 + 1] = bgr.data[i * 3 + 1];  // G
        rgb[i * 3 + 2] = bgr.data[i * 3 + 0];  // B
    }

    MpImagePtr image = nullptr;
    char* errorMsg = nullptr;
    MpStatus st = MpImageCreateFromUint8Data(
        kMpImageFormatSrgb, w, h, rgb.data(), (int)rgb.size(),
        &image, &errorMsg);
    if (st != kMpOk || !image) {
        if (errorMsg) MpErrorFree(errorMsg);
        return;
    }

    MpPoseLandmarkerResult mpResult;
    std::memset(&mpResult, 0, sizeof(mpResult));

    timestampMs_ += 33;  // ~30fps
    st = MpPoseLandmarkerDetectForVideo(
        landmarker_, image, nullptr, timestampMs_, &mpResult, &errorMsg);

    MpImageFree(image);

    if (st != kMpOk) {
        if (errorMsg) MpErrorFree(errorMsg);
        return;
    }

    // Extract landmarks (single pose expected)
    if (mpResult.pose_landmarks_count > 0 && mpResult.pose_world_landmarks_count > 0) {
        const auto& nlm = mpResult.pose_landmarks[0];
        const auto& wlm = mpResult.pose_world_landmarks[0];

        int n = (int)nlm.landmarks_count;
        if (n > PoseLandmarkIdx::NUM) n = PoseLandmarkIdx::NUM;
        for (int i = 0; i < n; i++) {
            const auto& lm = nlm.landmarks[i];
            result.landmarks[i * 5 + 0] = lm.x;
            result.landmarks[i * 5 + 1] = lm.y;
            result.landmarks[i * 5 + 2] = lm.z;
            result.landmarks[i * 5 + 3] = lm.visibility;
            result.landmarks[i * 5 + 4] = lm.presence;
        }

        int wn = (int)wlm.landmarks_count;
        if (wn > PoseLandmarkIdx::NUM) wn = PoseLandmarkIdx::NUM;
        for (int i = 0; i < wn; i++) {
            const auto& lm = wlm.landmarks[i];
            result.worldLandmarks[i * 3 + 0] = lm.x;
            result.worldLandmarks[i * 3 + 1] = lm.y;
            result.worldLandmarks[i * 3 + 2] = lm.z;
        }

        result.detected = true;
        result.frameW = w;
        result.frameH = h;
        result.presence = nlm.landmarks[0].presence;
    }

    MpPoseLandmarkerCloseResult(&mpResult);
}
