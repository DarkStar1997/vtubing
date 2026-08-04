#pragma once
#include "onnx_session.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>
#include <array>
#include <cmath>

struct FaceResult {
    bool detected = false;
    float score = 0.0f;
    std::array<float, 478 * 3> landmarks{};  // x,y,z in full-image pixel coords
    std::array<float, 52> blendshapes{};
    // Head rotation as Euler angles (radians): yaw, pitch, roll
    float yaw = 0, pitch = 0, roll = 0;
    // Bounding box in image coords
    float bboxX1, bboxY1, bboxX2, bboxY2;
};

class FaceTracker {
public:
    FaceTracker(const std::string& modelDir);

    // Process a BGR frame and fill result. imgW/imgH = frame dimensions.
    void detect(const cv::Mat& bgr, FaceResult& result);

private:
    std::unique_ptr<OnnxSession> detector_;
    std::unique_ptr<OnnxSession> landmarker_;
    std::unique_ptr<OnnxSession> blendshape_;

    // BlazeFace anchors (896 × 2: x, y centers normalized [0,1])
    std::array<float, 896 * 2> anchors_;

    // Blendshape model 146-index subset
    static constexpr std::array<int, 146> kBlendshapeIdxs = {
        0,1,4,5,6,7,8,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,63,65,66,67,70,78,80,
        81,82,84,87,88,91,93,95,103,105,107,109,127,132,133,136,144,145,146,148,149,150,152,
        153,154,155,157,158,159,160,161,162,163,168,172,173,176,178,181,185,191,195,197,234,
        246,249,251,263,267,269,270,276,282,283,284,285,288,291,293,295,296,297,300,308,310,
        311,312,314,317,318,321,323,324,332,334,336,338,356,361,362,365,373,374,375,377,378,
        379,380,381,382,384,385,386,387,388,389,390,397,398,400,402,405,409,415,454,466,468,
        469,470,471,472,473,474,475,476,477};

    void generateAnchors();
    cv::Mat letterbox(const cv::Mat& src, int size, float& scale, float& padX, float& padY);

    // BlazeFace detection: returns (x1,y1,x2,y2) bbox + 6 keypoints (each x,y) in image coords
    struct Detection {
        float score;
        float cx, cy, w, h;
        float kp[6][2];  // 6 keypoints
    };
    std::vector<Detection> runDetector(const cv::Mat& bgr, int imgW, int imgH);
    std::vector<Detection> weightedNms(std::vector<Detection>& dets);
};
