#include "face_tracker.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

FaceTracker::FaceTracker(const std::string& modelDir) {
    std::string modelPath = modelDir + "/face_landmarker.task";

    struct MpFaceLandmarkerOptions opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.base_options.model_asset_path = modelPath.c_str();
    opts.base_options.delegate = MP_DELEGATE_CPU;
    opts.running_mode = MP_RUNNING_MODE_VIDEO;
    opts.num_faces = 1;
    opts.min_face_detection_confidence = 0.5f;
    opts.min_face_presence_confidence = 0.5f;
    opts.min_tracking_confidence = 0.5f;
    opts.output_face_blendshapes = true;
    opts.output_facial_transformation_matrixes = true;
    opts.result_callback = nullptr;

    char* errorMsg = nullptr;
    MpStatus st = MpFaceLandmarkerCreate(&opts, &landmarker_, &errorMsg);
    if (st != kMpOk) {
        fprintf(stderr, "[face] MpFaceLandmarkerCreate failed: %s\n",
                errorMsg ? errorMsg : "(unknown)");
        if (errorMsg) MpErrorFree(errorMsg);
        landmarker_ = nullptr;
    }
}

FaceTracker::~FaceTracker() {
    if (landmarker_) {
        char* errorMsg = nullptr;
        MpFaceLandmarkerClose(landmarker_, &errorMsg);
        if (errorMsg) MpErrorFree(errorMsg);
    }
}

void FaceTracker::detect(const Image& bgr, FaceResult& result) {
    result.detected = false;
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

    MpFaceLandmarkerResult mpResult;
    std::memset(&mpResult, 0, sizeof(mpResult));

    timestampMs_ += 33;
    st = MpFaceLandmarkerDetectForVideo(
        landmarker_, image, nullptr, timestampMs_, &mpResult, &errorMsg);

    MpImageFree(image);

    if (st != kMpOk) {
        if (errorMsg) MpErrorFree(errorMsg);
        return;
    }

    // Extract normalized landmarks → pixel coords
    if (mpResult.face_landmarks_count > 0) {
        const auto& nlm = mpResult.face_landmarks[0];
        int n = (int)nlm.landmarks_count;
        if (n > 478) n = 478;
        result.detected = true;
        result.score = 1.0f;

        float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
        for (int i = 0; i < n; i++) {
            const auto& lm = nlm.landmarks[i];
            float px = lm.x * w;
            float py = lm.y * h;
            result.landmarks[i * 3] = px;
            result.landmarks[i * 3 + 1] = py;
            result.landmarks[i * 3 + 2] = lm.z * w;
            minX = std::min(minX, px); maxX = std::max(maxX, px);
            minY = std::min(minY, py); maxY = std::max(maxY, py);
        }
        // Pad bbox slightly
        float padX = (maxX - minX) * 0.05f;
        float padY = (maxY - minY) * 0.05f;
        result.bboxX1 = minX - padX;
        result.bboxY1 = minY - padY;
        result.bboxX2 = maxX + padX;
        result.bboxY2 = maxY + padY;
    }

    // Extract 52 blendshapes
    if (mpResult.face_blendshapes_count > 0) {
        const auto& cats = mpResult.face_blendshapes[0];
        int n = (int)cats.categories_count;
        if (n > 52) n = 52;
        for (int i = 0; i < n; i++)
            result.blendshapes[i] = cats.categories[i].score;
    }

    // Extract head rotation from 4×4 facial transformation matrix.
    // Matrix is column-major (OpenGL/GLM convention).
    // Decompose as YXZ intrinsic Euler: [yaw, pitch, roll] in degrees.
    if (mpResult.facial_transformation_matrixes_count > 0) {
        const auto& mat = mpResult.facial_transformation_matrixes[0];
        const float* d = mat.data;
        // R[row][col] = d[col*4 + row]  (column-major)
        float r02 = d[8], r10 = d[1], r11 = d[5], r12 = d[9], r22 = d[10];
        float pitch = std::asin(std::clamp(-r12, -1.0f, 1.0f));
        float roll  = std::atan2(r10, r11);
        float yaw   = std::atan2(r02, r22);
        // Negate yaw: non-mirrored camera (matches Python pipeline)
        yaw = -yaw;
        // Store in degrees
        result.yaw   = yaw   * 180.0f / 3.14159265f;
        result.pitch = pitch * 180.0f / 3.14159265f;
        result.roll  = roll  * 180.0f / 3.14159265f;
    }

    MpFaceLandmarkerCloseResult(&mpResult);
}
