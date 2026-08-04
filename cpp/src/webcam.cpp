#include "webcam.h"
#include <chrono>

WebcamCapture::WebcamCapture(int index, int width, int height, int fps)
    : index_(index), width_(width), height_(height), fps_(fps) {}

WebcamCapture::~WebcamCapture() { stop(); }

bool WebcamCapture::start() {
    cap_.open(index_, cv::CAP_V4L2);
    if (!cap_.isOpened()) return false;
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap_.set(cv::CAP_PROP_FPS, fps_);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

    running_ = true;
    thread_ = std::thread(&WebcamCapture::loop, this);
    return true;
}

void WebcamCapture::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (cap_.isOpened()) cap_.release();
}

void WebcamCapture::loop() {
    while (running_) {
        cv::Mat frame;
        if (cap_.read(frame) && !frame.empty()) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                frame.copyTo(latestFrame_);
                isNew_ = true;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

cv::Mat WebcamCapture::getLatest(bool& isNew) {
    std::lock_guard<std::mutex> lock(mutex_);
    isNew = isNew_;
    isNew_ = false;
    cv::Mat result;
    if (!latestFrame_.empty()) latestFrame_.copyTo(result);
    return result;
}
