#pragma once
#include "image.h"
#include <SDL3/SDL.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>

struct CameraDeviceInfo {
    SDL_CameraID id = 0;
    std::string name;
    int index = 0;
};

class WebcamCapture {
public:
    WebcamCapture(int index = 0, int width = 640, int height = 480, int fps = 30);
    ~WebcamCapture();

    bool start();
    void stop();

    // Query all video capture devices detected by SDL
    static std::vector<CameraDeviceInfo> getAvailableCameras();

    // Switch capture to another camera index (0-based)
    bool switchCamera(int index);

    int getIndex() const { return index_; }
    std::string getCurrentCameraName() const;

    // Returns latest frame (BGR). isNew is true only on first call after a fresh capture.
    Image getLatest(bool& isNew);

private:
    int index_, width_, height_, fps_;
    SDL_Camera* camera_ = nullptr;
    std::thread thread_;
    std::mutex mutex_;
    Image latestFrame_;
    bool isNew_ = false;
    std::atomic<bool> running_{false};
    std::string currentCameraName_;

    void loop();
};

