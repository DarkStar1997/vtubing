#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>

class WebcamCapture {
public:
    WebcamCapture(int index = 0, int width = 640, int height = 480, int fps = 30);
    ~WebcamCapture();

    bool start();
    void stop();

    // Returns latest frame (BGR). isNew is true only on first call after a fresh capture.
    cv::Mat getLatest(bool& isNew);

private:
    int index_, width_, height_, fps_;
    cv::VideoCapture cap_;
    std::thread thread_;
    std::mutex mutex_;
    cv::Mat latestFrame_;
    bool isNew_ = false;
    std::atomic<bool> running_{false};

    void loop();
};
