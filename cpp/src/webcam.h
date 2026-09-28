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

    bool start(SDL_CameraID devId = 0);
    void stop();

    // Query all video capture devices detected by SDL
    static std::vector<CameraDeviceInfo> getAvailableCameras();

    // Switch capture to another camera by device ID or positional index
    bool switchCamera(SDL_CameraID devId, int index = -1);
    bool switchCamera(int index);

    SDL_CameraID getActiveCameraId() const { return activeDevId_; }
    int getIndex() const { return index_; }
    std::string getCurrentCameraName() const;

    // Returns latest frame (BGR). isNew is true only on first call after a fresh capture.
    Image getLatest(bool& isNew);

private:
    int index_, width_, height_, fps_;
    SDL_CameraID activeDevId_ = 0;
    SDL_Camera* camera_ = nullptr;
    std::thread thread_;
    std::mutex mutex_;
    Image latestFrame_;
    bool isNew_ = false;
    std::atomic<bool> running_{false};
    std::string currentCameraName_;

    void loop();
};

