#include "vrm_loader.h"
#include "rasterizer.h"
#include "webcam.h"
#include "face_tracker.h"
#include "pose_tracker.h"
#include "hand_tracker.h"
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

#ifdef _WIN32
static inline void setEnvVar(const char* name, const char* value) { _putenv_s(name, value); }
#else
static inline void setEnvVar(const char* name, const char* value) { setenv(name, value, 1); }
#endif

int main(int argc, char** argv) {
    // Suppress MediaPipe verbose logs. On Linux, EGL stubs are compiled in
    // (with -rdynamic) to force pure-CPU inference.
    setEnvVar("GLOG_minloglevel", "2");

    auto printUsage = []() {
        fprintf(stderr,
            "Usage: vtuber_live [options] [vrm_file]\n"
            "\n"
            "Options:\n"
            "  --models <dir>   Path to MediaPipe model directory\n"
            "                   (default: ../../assets/models)\n"
            "  --threads <N>    Limit CPU threads, minimum 2 (default: auto)\n"
            "  --fps <N>        Cap frame rate to reduce CPU usage (default: unlimited)\n"
            "  -h, --help       Show this help message\n"
            "\n"
            "Controls:\n"
            "  SPACE            Calibrate neutral pose\n"
            "  W                Toggle picture-in-picture overlay\n"
            "  ESC              Quit\n");
    };

    std::string vrmPath = "../../assets/avatars/male_52blendshapes.vrm";
    std::string modelDir = "../../assets/models";
    int maxThreads = 0;  // 0 = auto (hardware_concurrency capped at 16)
    int targetFps = 0;   // 0 = unlimited

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) { modelDir = argv[++i]; }
        else if (a == "--threads" && i + 1 < argc) { maxThreads = std::max(2, std::stoi(argv[++i])); }
        else if (a == "--fps" && i + 1 < argc) { targetFps = std::max(1, std::stoi(argv[++i])); }
        else if (a == "-h" || a == "--help") { printUsage(); return 0; }
        else if (a.substr(0, 2) == "--") {
            fprintf(stderr, "Unknown option: %s\n\n", a.c_str());
            printUsage();
            return 1;
        }
        else vrmPath = a;
    }

    if (maxThreads > 0) {
        setEnvVar("OMP_NUM_THREADS", std::to_string(maxThreads).c_str());
        setEnvVar("TF_NUM_INTRAOP_THREADS", std::to_string(maxThreads).c_str());
        setEnvVar("TF_NUM_INTEROP_THREADS", "1");
    }

    int fbWidth = 1280, fbHeight = 720;
    int ss = 2;  // 2×2 supersampling anti-aliasing (SSAA)
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

    // Bone node indices for front-arm detection
    int leftUpperArmNode = -1, rightUpperArmNode = -1;
    {
        auto it = model.boneNodes.find("leftUpperArm");
        if (it != model.boneNodes.end()) leftUpperArmNode = it->second;
        it = model.boneNodes.find("rightUpperArm");
        if (it != model.boneNodes.end()) rightUpperArmNode = it->second;
    }
    bool leftIsFront = true;
    float turnStrength = 0.0f;

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

    // Camera params: sit-cam (bust shot) + stand-cam (full body)
    struct CamParams { float targetY; float dist; float fov; };
    CamParams sitCam{centre.y + size.y * 0.28f, std::max(size.y * 0.85f, 1.1f), 28.0f};
    CamParams standCam{centre.y, (size.y * 1.1f) / (2.0f * std::tan(glm::radians(15.0f))), 30.0f};

    // Pre-compute body geometry for dynamic camera (matching Python avatar.js)
    float headBoneY = centre.y + size.y * 0.45f;
    float hipsBoneY = centre.y;
    {
        int headIdx = model.headNodeIndex;
        if (headIdx >= 0) headBoneY = bindWorldMatrices[headIdx][3].y;
        auto hipsIt = model.boneNodes.find("hips");
        if (hipsIt != model.boneNodes.end())
            hipsBoneY = bindWorldMatrices[hipsIt->second][3].y;
    }
    float modelH = size.y;
    float shoulderBoneY = headBoneY - modelH * 0.05f;
    float torsoUnit = std::abs(hipsBoneY - shoulderBoneY);
    float feetY = bboxMin.y;
    float aspect = (float)fbWidth / fbHeight;
    float curFov = sitCam.fov;
    float curTargetY = sitCam.targetY;
    float curDist = sitCam.dist;
    glm::mat4 viewProj;

    // Init systems
    int numThreads = (maxThreads > 0) ? maxThreads : std::min((int)std::thread::hardware_concurrency(), 16);
    fprintf(stderr, "[live] threads: %d\n", numThreads);
    Framebuffer ssfb(renderW, renderH), fb(fbWidth, fbHeight);

    WebcamCapture webcam(0, 640, 480, 30);
    if (!webcam.start()) {
        fprintf(stderr, "[live] WARNING: webcam not available, running without tracking\n");
    }

    FaceTracker faceTracker(modelDir);
    PoseTracker poseTracker(modelDir);
    HandTracker handTracker(modelDir);
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
    bool framingApplied = false;
    float calibFaceMinY = 1.0f, calibFaceMaxY = 0.0f;
    int calibFaceFrames = 0;
    SDL_Event event;
    Image lastAnnotated;  // cached PiP frame, reused between webcam updates
    Image pipImg;         // pre-resized PiP (updated at webcam rate, not render rate)
    const int pipW = 320, pipH = 240;
    std::vector<uint8_t> bgraBuf(fbWidth * fbHeight * 4);  // pre-allocated BGRA buffer

    // --- Async detection thread ---
    struct TrackUpdate {
        FaceResult result;
        PoseResult pose;
        HandResult handLeft;
        HandResult handRight;
        Image annotated;
    };
    TrackUpdate latestTrack;
    std::mutex trackMutex;
    std::atomic<bool> hasNewTrack{false};
    std::atomic<bool> running{true};
    std::atomic<bool> showPiPAtomic{true};

    std::thread detectThread([&]() {
        auto lastDetectTime = std::chrono::steady_clock::now();
        while (running.load()) {
            bool isNew = false;
            Image frame = webcam.getLatest(isNew);
            if (frame.empty() || !isNew) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            // Throttle detection to match render FPS cap — no point running
            // 3 MediaPipe inferences faster than the display can consume.
            if (targetFps > 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastDetectTime).count();
                auto detectPeriod = 1000000 / targetFps;
                if (elapsed < detectPeriod) {
                    std::this_thread::sleep_for(std::chrono::microseconds(detectPeriod - elapsed));
                }
            }
            lastDetectTime = std::chrono::steady_clock::now();
            FaceResult result;
            faceTracker.detect(frame, result);

            // RTMO: single-stage full-frame pose detection (no detector needed)
            PoseResult pose;
            poseTracker.detect(frame, pose);

            // Hand tracking: MediaPipe HandLandmarker on full frame
            HandResult handLeft, handRight;
            handTracker.detect(frame, handLeft, handRight);

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
                    for (int i = 0; i < 478; i++) {
                        drawCircleFilled(annotated,
                            (int)result.landmarks[i*3],
                            (int)result.landmarks[i*3+1],
                            1, yellow);
                    }
                }
                if (pose.detected) {
                    uint8_t cyan[3] = {255, 255, 0};
                    static const int POSE_CONN[][2] = {
                        {11,12},{11,13},{13,15},{12,14},{14,16},
                        {11,23},{12,24},{23,24},
                        {23,25},{25,27},{24,26},{26,28},
                        {27,29},{29,31},{28,30},{30,32},
                        {15,17},{15,19},{15,21},{16,18},{16,20},{16,22},
                    };
                    for (auto& c : POSE_CONN) {
                        if (pose.lmVis(c[0]) > 0.15f && pose.lmVis(c[1]) > 0.15f) {
                            drawLine(annotated,
                                (int)pose.frameX(c[0]), (int)pose.frameY(c[0]),
                                (int)pose.frameX(c[1]), (int)pose.frameY(c[1]),
                                cyan, 2);
                        }
                    }
                    uint8_t dotColor[3] = {0, 0, 255};
                    for (int i = 0; i < 33; i++) {
                        if (pose.lmVis(i) > 0.15f)
                            drawCircleFilled(annotated, (int)pose.frameX(i), (int)pose.frameY(i), 3, dotColor);
                    }
                }
                // Hand skeleton — normalized landmarks → frame pixel coords
                auto drawHandSkeleton = [&](const HandResult& hr) {
                    if (!hr.detected) return;
                    uint8_t orange[3] = {0, 200, 255};
                    uint8_t dotBlue[3] = {255, 100, 0};
                    static const int HAND_CONN[][2] = {
                        {0,1},{1,2},{2,3},{3,4},
                        {0,5},{5,6},{6,7},{7,8},
                        {5,9},{9,10},{10,11},{11,12},
                        {9,13},{13,14},{14,15},{15,16},
                        {13,17},{17,18},{18,19},{19,20},
                        {0,17},
                    };
                    for (auto& c : HAND_CONN) {
                        drawLine(annotated,
                            (int)hr.pxX(c[0]), (int)hr.pxY(c[0]),
                            (int)hr.pxX(c[1]), (int)hr.pxY(c[1]),
                            orange, 2);
                    }
                    for (int i = 0; i < 21; i++)
                        drawCircleFilled(annotated, (int)hr.pxX(i), (int)hr.pxY(i), 3, dotBlue);
                };
                drawHandSkeleton(handLeft);
                drawHandSkeleton(handRight);
            }

            {
                std::lock_guard<std::mutex> lock(trackMutex);
                latestTrack.result = std::move(result);
                latestTrack.pose = std::move(pose);
                latestTrack.handLeft = std::move(handLeft);
                latestTrack.handRight = std::move(handRight);
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

    // Compute viewProj from current camera params
    auto computeViewProj = [&]() {
        glm::vec3 eye(0, curTargetY + 0.10f, -curDist);
        glm::vec3 tgt(0, curTargetY - 0.02f, 0);
        glm::mat4 p = glm::perspective(glm::radians(curFov), aspect, 0.1f, 100.0f);
        glm::mat4 v = glm::lookAt(eye, tgt, glm::vec3(0, 1, 0));
        viewProj = p * v;
    };
    computeViewProj();

    // Pre-render a frame with default pose for initial display
    auto proc = processVerticesParallel(model, jointMatrices, viewProj,
                                        rigSolver.morphWeights(), renderW, renderH, numThreads);
    ssfb.clear();
    rasterizeParallel(proc, ssfb, model.textures, numThreads);
    if (ss > 1) downsample2x2(ssfb, fb, numThreads);

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
        PoseResult poseResult;
        HandResult handLeft, handRight;
        bool hasNew = false;
        if (hasNewTrack.exchange(false, std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(trackMutex);
            faceResult = latestTrack.result;
            poseResult = latestTrack.pose;
            handLeft = latestTrack.handLeft;
            handRight = latestTrack.handRight;
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
                    rigSolver.calibratePose();
                    // Collect face framing data (normalized Y coords)
                    float y1 = faceResult.bboxY1 / 480.0f;
                    float y2 = faceResult.bboxY2 / 480.0f;
                    calibFaceMinY = std::min(calibFaceMinY, y1);
                    calibFaceMaxY = std::max(calibFaceMaxY, y2);
                    calibFaceFrames++;
                    if (rigSolver.calibrated() && rigSolver.poseCalibrated()) {
                        calibrating = false;
                        fprintf(stderr, "[live] Calibration complete.\n");
                        // Compute framing-calibrated sitCam (matching Python avatar.js)
                        if (!framingApplied && calibFaceFrames > 0) {
                            float userFaceH = calibFaceMaxY - calibFaceMinY;
                            float userFaceCenterY = (calibFaceMinY + calibFaceMaxY) * 0.5f;
                            float modelFaceH = modelH * 0.13f;
                            float modelFaceCenter = headBoneY + modelH * 0.035f;
                            float fov = 28.0f;
                            float tanHalf = std::tan(glm::radians(fov * 0.5f));
                            sitCam.dist = modelFaceH / (std::max(userFaceH, 0.01f) * 2.0f * tanHalf);
                            sitCam.targetY = modelFaceCenter + (2.0f * userFaceCenterY - 1.0f) * sitCam.dist * tanHalf;
                            sitCam.fov = fov;
                            curFov = sitCam.fov;
                            curTargetY = sitCam.targetY;
                            curDist = sitCam.dist;
                            framingApplied = true;
                            fprintf(stderr, "[cam] Framing applied: userFaceH=%.2f center=%.2f "
                                    "→ sitCam(ty=%.2f,d=%.2f,fov=%.0f)\n",
                                    userFaceH, userFaceCenterY,
                                    sitCam.targetY, sitCam.dist, sitCam.fov);
                        }
                    }
                }
                rigSolver.update(faceResult, trackDt);
            } else {
                rigSolver.update(faceResult, trackDt);
            }

            if (poseResult.detected) {
                rigSolver.updatePose(poseResult, trackDt);
            } else {
                rigSolver.updatePose(poseResult, trackDt);
            }

            rigSolver.updateHands(handLeft, handRight, trackDt);
        }

        // Build bone rotation overrides from face + pose + hands
        if (hasNew) {
            std::unordered_map<int, glm::quat> overrides;
            glm::quat headRot = rigSolver.headRotation();
            if (headRot != glm::quat(1, 0, 0, 0)) {
                int headNode = model.headNodeIndex;
                if (headNode >= 0) overrides[headNode] = headRot;
            }
            const BodyPose& bp = rigSolver.bodyPose();
            if (bp.valid) {
                auto setIf = [&](const std::string& bone, const glm::quat& q) {
                    auto it = model.boneNodes.find(bone);
                    if (it != model.boneNodes.end() && q != glm::quat(1, 0, 0, 0))
                        overrides[it->second] = q;
                };
                setIf("leftUpperArm", bp.leftUpperArm);
                setIf("rightUpperArm", bp.rightUpperArm);
                setIf("leftLowerArm", bp.leftLowerArm);
                setIf("rightLowerArm", bp.rightLowerArm);
                // Spine: distribute lean/twist/lateral across upperChest, chest, spine
                auto setEuler = [&](const std::string& bone, float s) {
                    auto it = model.boneNodes.find(bone);
                    if (it != model.boneNodes.end())
                        overrides[it->second] = glm::quat(
                            glm::vec3(bp.lean * s, bp.twist * s, bp.lateral * s));
                };
                setEuler("upperChest", 0.5f);
                setEuler("chest", 0.3f);
                setEuler("spine", 0.2f);
            } else if (rigSolver.calibrated()) {
                // Rest arm pose: A-pose (relaxed) when sitting
                auto setRot = [&](const std::string& bone, float x, float y, float z) {
                    auto it = model.boneNodes.find(bone);
                    if (it != model.boneNodes.end())
                        overrides[it->second] = glm::quat(glm::vec3(x, y, z));
                };
                setRot("leftUpperArm", 0, 0.20f, 1.30f);
                setRot("rightUpperArm", 0, -0.20f, -1.30f);
                setRot("leftLowerArm", 0, 0, -1.15f);
                setRot("rightLowerArm", 0, 0, 1.15f);
            }
            // Merge hand finger overrides
            for (const auto& [nodeIdx, rot] : rigSolver.handOverrides())
                overrides[nodeIdx] = rot;

            if (!overrides.empty()) {
                std::vector<glm::mat4> modWorld =
                    computeWorldMatricesWithOverrides(model, overrides);
                jointMatrices = computeJointMatrices(model, modWorld);

                // Determine front arm from world Z of upper arm bones
                if (leftUpperArmNode >= 0 && rightUpperArmNode >= 0) {
                    float lz = modWorld[leftUpperArmNode][3].z;
                    float rz = modWorld[rightUpperArmNode][3].z;
                    float zDiff = lz - rz;
                    if (zDiff < -0.02f) leftIsFront = true;
                    else if (zDiff > 0.02f) leftIsFront = false;
                    turnStrength = std::clamp((std::abs(zDiff) - 0.04f) / 0.06f, 0.0f, 1.0f);
                }
            } else {
                jointMatrices = bindJointMatrices;
            }
        }

        // Dynamic camera: blend sit-cam (bust shot) ↔ body-fit cam
        // matching Python avatar.js applyDynamicCamera()
        {
            const BodyPose& bp = rigSolver.bodyPose();
            float standing = bp.valid ? bp.standing : 0.0f;
            float be = bp.valid ? bp.bodyExtent : 0.0f;
            // Standing but no body extent → assume full body visible
            if (be < 0.1f && standing > 0.5f) be = 3.5f;

            // Body-fit camera: frames head-top to visible-body-bottom
            float modelBottomY = shoulderBoneY - be * torsoUnit;
            modelBottomY = std::max(modelBottomY, feetY);
            float headTopY = headBoneY + modelH * 0.06f;
            float bodyH = headTopY - modelBottomY;
            float bfCenterY = (headTopY + modelBottomY) * 0.5f;
            float bfFov = 30.0f;
            float bfDist = std::max(
                (bodyH * 1.1f) / (2.0f * std::tan(glm::radians(bfFov * 0.5f))), 0.8f);

            // Blend sit-cam ↔ body-fit cam
            float frac = std::min(standing + std::max(0.0f, be - 0.5f) * 0.3f, 1.0f);
            float sit = 1.0f - frac;
            float newFov = sitCam.fov * sit + bfFov * frac;
            float newTargetY = sitCam.targetY * sit + bfCenterY * frac;
            float newDist = sitCam.dist * sit + bfDist * frac;

            // Smooth transitions
            float camAlpha = std::min(1.0f, dt * 3.0f);
            curFov += (newFov - curFov) * camAlpha;
            curTargetY += (newTargetY - curTargetY) * camAlpha;
            curDist += (newDist - curDist) * camAlpha;
            computeViewProj();
        }

        // Render avatar
        auto proc = processVerticesParallel(model, jointMatrices, viewProj,
                                            rigSolver.morphWeights(), renderW, renderH, numThreads,
                                             turnStrength, leftIsFront);
        ssfb.clear();
        rasterizeParallel(proc, ssfb, model.textures, numThreads);
        if (ss > 1) downsample2x2(ssfb, fb, numThreads);

        // Convert framebuffer RGBA to BGRA8888 for SDL
        // On little-endian, SDL_PIXELFORMAT_BGRA8888 reads bytes as [A,R,G,B]
        Framebuffer& out = (ss > 1) ? fb : ssfb;
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

        // Frame rate cap
        if (targetFps > 0) {
            auto frameEnd = std::chrono::steady_clock::now();
            auto frameMicros = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - now).count();
            auto targetMicros = 1000000 / targetFps;
            if (frameMicros < targetMicros)
                std::this_thread::sleep_for(std::chrono::microseconds(targetMicros - frameMicros));
        }
    }

    running = false;
    detectThread.join();
    webcam.stop();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
