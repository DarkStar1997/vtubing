#include "vrm_loader.h"
#include "rasterizer.h"
#include "webcam.h"
#include "face_tracker.h"
#include "rig_solver.h"

#include <SDL3/SDL.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

int main(int argc, char** argv) {
    std::string vrmPath = "../../assets/avatars/male_52blendshapes.vrm";
    std::string modelDir = "../../assets/models";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) { modelDir = argv[++i]; }
        else vrmPath = a;
    }

    int fbWidth = 1280, fbHeight = 720;
    int ss = 1;
    int renderW = fbWidth * ss, renderH = fbHeight * ss;

    fprintf(stderr, "[live] loading VRM: %s\n", vrmPath.c_str());
    VRMModel model = loadVRM(vrmPath);
    if (model.meshes.empty()) { fprintf(stderr, "Failed to load model\n"); return 1; }
    fprintf(stderr, "[live] %d blendshape groups, headNode=%d\n",
            (int)model.blendShapeGroups.size(), model.headNodeIndex);

    // Compute bind-pose matrices (kept as reference)
    std::vector<glm::mat4> bindWorldMatrices = computeWorldMatrices(model);
    std::vector<glm::mat4> bindJointMatrices = computeJointMatrices(model, bindWorldMatrices);
    // Working copy that gets modified each frame
    std::vector<glm::mat4> jointMatrices = bindJointMatrices;

    // Camera (same as main.cpp bbox-based framing)
    glm::vec3 bboxMin(1e9f), bboxMax(-1e9f);
    for (size_t mi = 0; mi < model.meshes.size(); mi++) {
        const auto& mesh = model.meshes[mi];
        glm::mat4 nodeMat = (mesh.nodeIndex >= 0 && mesh.nodeIndex < (int)bindWorldMatrices.size())
            ? bindWorldMatrices[mesh.nodeIndex] : glm::mat4(1.0f);
        for (const auto& prim : mesh.primitives) {
            int vc = prim.vertexCount();
            for (int vi = 0; vi < vc; vi++) {
                glm::vec4 p(prim.positions[vi*3], prim.positions[vi*3+1], prim.positions[vi*3+2], 1.0f);
                glm::vec3 wp = glm::vec3(nodeMat * p);
                bboxMin = glm::min(bboxMin, wp);
                bboxMax = glm::max(bboxMax, wp);
            }
        }
    }
    glm::vec3 centre = (bboxMin + bboxMax) * 0.5f;
    glm::vec3 size = bboxMax - bboxMin;
    float targetY = centre.y + size.y * 0.28f;
    float camDist = std::max(size.y * 0.85f, 1.1f);
    glm::vec3 eye(0, targetY + 0.10f, -camDist);
    glm::vec3 target(0, targetY - 0.02f, 0);
    float fovY = glm::radians(28.0f);
    float aspect = (float)fbWidth / fbHeight;
    glm::mat4 proj = glm::perspective(fovY, aspect, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0, 1, 0));
    glm::mat4 viewProj = proj * view;

    // Init systems
    int numThreads = std::min((int)std::thread::hardware_concurrency(), 16);
    Framebuffer ssfb(renderW, renderH), fb(fbWidth, fbHeight);

    WebcamCapture webcam(0, 640, 480, 30);
    if (!webcam.start()) {
        fprintf(stderr, "[live] WARNING: webcam not available, running without tracking\n");
    }

    FaceTracker faceTracker(modelDir);
    RigSolver rigSolver(model);

    fprintf(stderr, "[live] Face tracker initialized. Press SPACE to calibrate.\n");

    // SDL window
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("VTuber Live", fbWidth, fbHeight, 0);
    SDL_Surface* winSurface = SDL_GetWindowSurface(window);

    bool showPiP = true, calibrating = false;
    SDL_Event event;
    Image lastAnnotated;  // cached PiP frame, reused between webcam updates
    Image pipImg;         // pre-resized PiP (updated at webcam rate, not render rate)
    const int pipW = 320, pipH = 240;
    std::vector<uint8_t> bgraBuf(fbWidth * fbHeight * 4);  // pre-allocated BGRA buffer

    // --- Async detection thread ---
    struct TrackUpdate {
        FaceResult result;
        Image annotated;
    };
    TrackUpdate latestTrack;
    std::mutex trackMutex;
    std::atomic<bool> hasNewTrack{false};
    std::atomic<bool> running{true};
    std::atomic<bool> showPiPAtomic{true};

    std::thread detectThread([&]() {
        while (running.load()) {
            bool isNew = false;
            Image frame = webcam.getLatest(isNew);
            if (frame.empty() || !isNew) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            FaceResult result;
            faceTracker.detect(frame, result);

            Image annotated;
            if (showPiPAtomic.load()) {
                annotated = frame;
                if (result.detected) {
                    uint8_t green[3] = {0, 255, 0};
                    uint8_t yellow[3] = {0, 255, 255};
                    drawRect(annotated,
                        (int)result.bboxX1, (int)result.bboxY1,
                        (int)result.bboxX2, (int)result.bboxY2,
                        green, 2);
                    for (int i = 0; i < 478; i += 10) {
                        drawCircleFilled(annotated,
                            (int)result.landmarks[i*3],
                            (int)result.landmarks[i*3+1],
                            1, yellow);
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(trackMutex);
                latestTrack.result = std::move(result);
                latestTrack.annotated = std::move(annotated);
            }
            hasNewTrack.store(true, std::memory_order_release);
        }
    });

    auto lastTime = std::chrono::steady_clock::now();
    auto lastTrackTime = lastTime;  // for correct dt between face-tracking updates
    int frameCount = 0;
    int detectCount = 0;
    auto statTime = lastTime;

    // Pre-render a frame with default pose for initial display
    auto proc = processVerticesParallel(model, jointMatrices, viewProj,
                                        rigSolver.morphWeights(), renderW, renderH, numThreads);
    ssfb.clear();
    rasterizeParallel(proc, ssfb, model.textures, numThreads);

    while (running) {
        // Events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_ESCAPE) running = false;
                if (event.key.key == SDLK_W) { showPiP = !showPiP; showPiPAtomic.store(showPiP); }
                if (event.key.key == SDLK_SPACE) { calibrating = true; fprintf(stderr, "[live] Calibrating...\n"); }
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        if (dt > 0.1f) dt = 0.1f;

        // Consume async detection result (non-blocking)
        FaceResult faceResult;
        bool hasNew = false;
        if (hasNewTrack.exchange(false, std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(trackMutex);
            faceResult = latestTrack.result;
            if (showPiP && !latestTrack.annotated.empty()) {
                lastAnnotated = latestTrack.annotated;
                pipImg = resizeBilinear(lastAnnotated, pipW, pipH);
            }
            hasNew = true;

            float trackDt = std::chrono::duration<float>(now - lastTrackTime).count();
            lastTrackTime = now;
            if (trackDt > 0.1f) trackDt = 0.1f;

            if (faceResult.detected) {
                detectCount++;
                if (calibrating) {
                    rigSolver.calibrate(faceResult);
                    if (rigSolver.calibrated()) {
                        calibrating = false;
                        fprintf(stderr, "[live] Calibration complete.\n");
                    }
                }
                rigSolver.update(faceResult, trackDt);
            } else {
                rigSolver.update(faceResult, trackDt);
            }
        }

        // Apply head rotation to joint matrices.
        // Only recompute on new detection; between updates, persist the
        // last computed pose to avoid snapping back to bind pose (stutter).
        if (hasNew && faceResult.detected) {
            jointMatrices = bindJointMatrices;
            glm::quat headRot = rigSolver.headRotation();
            if (headRot != glm::quat(1, 0, 0, 0)) {
                // Modify head node world matrix and propagate to children
                std::vector<glm::mat4> modWorld = bindWorldMatrices;
                int headNode = model.headNodeIndex;
                if (headNode >= 0 && headNode < (int)modWorld.size()) {
                    // Apply local rotation at head node
                    modWorld[headNode] = modWorld[headNode] * glm::mat4_cast(headRot);
                    // Recompute children's world matrices
                    std::function<void(int)> updateChildren = [&](int parent) {
                        for (int c = parent + 1; c < (int)model.nodes.size(); c++) {
                            if (model.nodes[c].parent == parent) {
                                glm::mat4 local = glm::translate(glm::mat4(1), model.nodes[c].translation)
                                    * glm::mat4_cast(model.nodes[c].rotation)
                                    * glm::scale(glm::mat4(1), model.nodes[c].scale);
                                modWorld[c] = modWorld[parent] * local;
                                updateChildren(c);
                            }
                        }
                    };
                    updateChildren(headNode);
                    // Recompute joint matrices for affected nodes
                    for (int ji = 0; ji < (int)model.jointNodes.size(); ji++) {
                        int ni = model.jointNodes[ji];
                        if (ni >= 0 && ni < (int)modWorld.size()) {
                            glm::mat4 ibm = (ji < (int)model.inverseBindMatrices.size())
                                ? model.inverseBindMatrices[ji] : glm::mat4(1);
                            jointMatrices[ji] = modWorld[ni] * ibm;
                        }
                    }
                }
            }
        }

        // Render avatar
        auto proc = processVerticesParallel(model, jointMatrices, viewProj,
                                            rigSolver.morphWeights(), renderW, renderH, numThreads);
        ssfb.clear();
        rasterizeParallel(proc, ssfb, model.textures, numThreads);

        // Convert framebuffer RGBA to BGRA8888 for SDL
        // On little-endian, SDL_PIXELFORMAT_BGRA8888 reads bytes as [A,R,G,B]
        Framebuffer& out = ssfb;
        for (int i = 0; i < fbWidth * fbHeight; i++) {
            bgraBuf[i*4+0] = 255;               // A
            bgraBuf[i*4+1] = out.color[i*4+0];  // R
            bgraBuf[i*4+2] = out.color[i*4+1];  // G
            bgraBuf[i*4+3] = out.color[i*4+2];  // B
        }

        // PiP overlay
        if (showPiP && !pipImg.empty()) {
            int pipX = fbWidth - pipW - 10;
            int pipY = fbHeight - pipH - 10;
            for (int py = 0; py < pipH; py++) {
                for (int px = 0; px < pipW; px++) {
                    const uint8_t* src = pipImg.ptr(py, px);
                    int di = ((pipY + py) * fbWidth + (pipX + px)) * 4;
                    bgraBuf[di+0] = 255;        // A
                    bgraBuf[di+1] = src[2];     // R (from BGR)
                    bgraBuf[di+2] = src[1];     // G
                    bgraBuf[di+3] = src[0];     // B
                }
            }
        }

        SDL_Surface* fbSurface = SDL_CreateSurfaceFrom(
            fbWidth, fbHeight, SDL_PIXELFORMAT_BGRA8888, bgraBuf.data(), fbWidth * 4);
        SDL_BlitSurface(fbSurface, nullptr, winSurface, nullptr);
        SDL_UpdateWindowSurface(window);
        SDL_DestroySurface(fbSurface);

        frameCount++;
        auto elapsed = std::chrono::duration<float>(now - statTime).count();
        if (elapsed >= 2.0f) {
            float fps = frameCount / elapsed;
            float detRate = frameCount > 0 ? (float)detectCount / frameCount * 100.0f : 0.0f;
            fprintf(stderr, "[live] %.1f fps, detect: %.0f%% (%d/%d)\n",
                    fps, detRate, detectCount, frameCount);
            // Print a few blendshape values when detected
            if (faceResult.detected) {
                fprintf(stderr, "  jawOpen=%.2f mouthSmileL=%.2f eyeBlinkL=%.2f eyeBlinkR=%.2f browInnerUp=%.2f\n",
                    faceResult.blendshapes[25], faceResult.blendshapes[44],
                    faceResult.blendshapes[9], faceResult.blendshapes[10],
                    faceResult.blendshapes[1]);
            }
            frameCount = 0;
            detectCount = 0;
            statTime = now;
        }
    }

    running = false;
    detectThread.join();
    webcam.stop();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
