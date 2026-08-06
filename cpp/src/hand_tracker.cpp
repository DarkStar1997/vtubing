#include "hand_tracker.h"
#include <cstdio>
#include <cstring>
#include <cstring>

HandTracker::HandTracker(const std::string& modelDir) {
    std::string modelPath = modelDir + "/hand_landmarker.task";

    struct MpHandLandmarkerOptions opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.base_options.model_asset_path = modelPath.c_str();
    opts.base_options.delegate = MP_DELEGATE_CPU;
    opts.running_mode = MP_RUNNING_MODE_VIDEO;
    opts.num_hands = 2;
    opts.min_hand_detection_confidence = 0.5f;
    opts.min_hand_presence_confidence = 0.5f;
    opts.min_tracking_confidence = 0.5f;
    opts.result_callback = nullptr;

    char* errorMsg = nullptr;
    MpStatus st = MpHandLandmarkerCreate(&opts, &landmarker_, &errorMsg);
    if (st != kMpOk) {
        fprintf(stderr, "[hand] MpHandLandmarkerCreate failed: %s\n",
                errorMsg ? errorMsg : "(unknown)");
        if (errorMsg) MpErrorFree(errorMsg);
        landmarker_ = nullptr;
    }
}

HandTracker::~HandTracker() {
    if (landmarker_) {
        char* errorMsg = nullptr;
        MpHandLandmarkerClose(landmarker_, &errorMsg);
        if (errorMsg) MpErrorFree(errorMsg);
    }
}

void HandTracker::detect(const Image& bgr, HandResult& left, HandResult& right) {
    left.detected = false;
    right.detected = false;
    if (!landmarker_ || bgr.empty()) return;

    int w = bgr.width, h = bgr.height;

    // MediaPipe expects RGB; our Image is BGR → swap channels
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

    MpHandLandmarkerResult mpResult;
    std::memset(&mpResult, 0, sizeof(mpResult));

    timestampMs_ += 33;
    st = MpHandLandmarkerDetectForVideo(
        landmarker_, image, nullptr, timestampMs_, &mpResult, &errorMsg);

    MpImageFree(image);

    if (st != kMpOk) {
        if (errorMsg) MpErrorFree(errorMsg);
        return;
    }

    int numHands = (int)mpResult.hand_landmarks_count;
    for (int hi = 0; hi < numHands; hi++) {
        const auto& nlm = mpResult.hand_landmarks[hi];

        // Determine handedness from category name.
        // MediaPipe labels match the user's hands for un-mirrored feeds.
        bool isLeft = false;
        if (hi < (int)mpResult.handedness_count) {
            const auto& cats = mpResult.handedness[hi];
            if (cats.categories_count > 0) {
                const char* name = cats.categories[0].category_name;
                if (name && name[0] == 'L') isLeft = true;
            }
        }

        HandResult* dst = isLeft ? &left : &right;
        dst->detected = true;
        dst->frameW = w;
        dst->frameH = h;

        int n = (int)nlm.landmarks_count;
        if (n > 21) n = 21;
        for (int i = 0; i < n; i++) {
            const auto& lm = nlm.landmarks[i];
            dst->landmarks[i * 3 + 0] = lm.x;
            dst->landmarks[i * 3 + 1] = lm.y;
            dst->landmarks[i * 3 + 2] = lm.z;
        }

        // World landmarks (metric)
        if (hi < (int)mpResult.hand_world_landmarks_count) {
            const auto& wlm = mpResult.hand_world_landmarks[hi];
            int wn = (int)wlm.landmarks_count;
            if (wn > 21) wn = 21;
            for (int i = 0; i < wn; i++) {
                const auto& lm = wlm.landmarks[i];
                dst->worldLandmarks[i * 3 + 0] = lm.x;
                dst->worldLandmarks[i * 3 + 1] = lm.y;
                dst->worldLandmarks[i * 3 + 2] = lm.z;
            }
        }
    }

    MpHandLandmarkerCloseResult(&mpResult);
}
