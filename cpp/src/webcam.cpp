#include "webcam.h"
#include <chrono>
#include <cstdio>

WebcamCapture::WebcamCapture(int index, int width, int height, int fps)
    : index_(index), width_(width), height_(height), fps_(fps) {}

WebcamCapture::~WebcamCapture() { stop(); }

bool WebcamCapture::start() {
    if (!SDL_Init(SDL_INIT_CAMERA)) {
        fprintf(stderr, "[webcam] SDL_INIT_CAMERA failed: %s\n", SDL_GetError());
        return false;
    }

    int numCameras = 0;
    SDL_CameraID* cameras = SDL_GetCameras(&numCameras);
    if (!cameras || numCameras == 0) {
        fprintf(stderr, "[webcam] No cameras found\n");
        if (cameras) SDL_free(cameras);
        return false;
    }
    if (index_ >= numCameras) index_ = 0;
    SDL_CameraID devId = cameras[index_];
    SDL_free(cameras);

    SDL_CameraSpec spec = {};
    spec.format = SDL_PIXELFORMAT_BGR24;
    spec.width = width_;
    spec.height = height_;
    spec.framerate_numerator = fps_;
    spec.framerate_denominator = 1;

    camera_ = SDL_OpenCamera(devId, &spec);
    if (!camera_) {
        fprintf(stderr, "[webcam] SDL_OpenCamera failed: %s\n", SDL_GetError());
        return false;
    }

    running_ = true;
    thread_ = std::thread(&WebcamCapture::loop, this);
    return true;
}

void WebcamCapture::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (camera_) {
        SDL_CloseCamera(camera_);
        camera_ = nullptr;
    }
}

void WebcamCapture::loop() {
    // Wait for permission/approval and first frame.
    while (running_) {
        if (SDL_GetCameraPermissionState(camera_) <= 0) {
            // Still pending or denied; keep waiting briefly.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        break;
    }

    while (running_) {
        Uint64 ts = 0;
        SDL_Surface* surf = SDL_AcquireCameraFrame(camera_, &ts);
        if (!surf) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        int w = surf->w, h = surf->h;
        Image frame(w, h, 3);
        bool ok = SDL_ConvertPixels(w, h,
            (SDL_PixelFormat)surf->format, surf->pixels, surf->pitch,
            SDL_PIXELFORMAT_BGR24, frame.data.data(), w * 3);
        SDL_ReleaseCameraFrame(camera_, surf);

        if (ok) {
            std::lock_guard<std::mutex> lock(mutex_);
            latestFrame_ = std::move(frame);
            isNew_ = true;
        }
    }
}

Image WebcamCapture::getLatest(bool& isNew) {
    std::lock_guard<std::mutex> lock(mutex_);
    isNew = isNew_;
    isNew_ = false;
    Image result;
    if (!latestFrame_.empty()) result = latestFrame_;
    return result;
}
