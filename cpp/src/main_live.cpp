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

int main(int argc, char** argv) {
    std::string vrmPath = "../../assets/avatars/male_52blendshapes.vrm";
    std::string modelDir = "../../assets/models";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) { modelDir = argv[++i]; }
        else vrmPath = a;
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
    float aspect = (float)fbWidth / fbHeight;
    float curFov = sitCam.fov;
    float curTargetY = sitCam.targetY;
    float curDist = sitCam.dist;
    glm::mat4 viewProj;

    // Init systems
    int numThreads = std::min((int)std::thread::hardware_concurrency(), 16);
    Framebuffer ssfb(renderW, renderH), fb(fbWidth, fbHeight);

    WebcamCapture webcam(0, 640, 480, 30);
    if (!webcam.start()) {
        fprintf(stderr, "[live] WARNING: webcam not available, running without tracking\n");
    }

    FaceTracker faceTracker(modelDir);
    PoseTracker poseTracker(modelDir + "/rtmo-s.onnx");
    HandTracker handTracker(modelDir + "/hand_landmarker.onnx");
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
        while (running.load()) {
            bool isNew = false;
            Image frame = webcam.getLatest(isNew);
            if (frame.empty() || !isNew) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            FaceResult result;
            faceTracker.detect(frame, result);

            // RTMO: single-stage full-frame pose detection (no detector needed)
            PoseResult pose;
            poseTracker.detect(frame, pose);

            // Hand tracking: crop ROIs from RTMO's accurate wrist positions.
            // ROI size targets ~1× the elbow→wrist distance so the hand fills
            // most of the 224×224 input the landmarker is trained on (hand
            // length ≈ 0.7× forearm; ROI = 1× armLen gives ~30% margin).
            // Larger ROIs leave the hand too small and the model falls back
            // to its extended-finger prior, especially on curled fists.
            HandResult handLeft, handRight;
            if (pose.detected) {
                float shoulderDist = std::abs(pose.kpX(PoseLandmarkIdx::L_SHOULDER) -
                                              pose.kpX(PoseLandmarkIdx::R_SHOULDER));
                auto detectHand = [&](int wristIdx, int elbowIdx, HandResult& outResult) {
                    if (pose.kpScore(wristIdx) < 0.3f) return;
                    int wx = (int)pose.kpX(wristIdx);
                    int wy = (int)pose.kpY(wristIdx);
                    float armLen = 0.0f;
                    if (pose.kpScore(elbowIdx) > 0.3f) {
                        float ex = pose.kpX(elbowIdx), ey = pose.kpY(elbowIdx);
                        armLen = std::sqrt((wx - ex) * (wx - ex) + (wy - ey) * (wy - ey));
                    }
                    int hs = (int)std::max({armLen * 1.0f, shoulderDist * 0.65f, 120.0f});
                    int hrx = wx - hs / 2, hry = wy - hs / 2;
                    HandResult h;
                    handTracker.detect(cropImage(frame, hrx, hry, hs, hs), h);
                    if (h.detected) {
                        h.roiX = hrx; h.roiY = hry;
                        h.anchorX = (float)wx; h.anchorY = (float)wy;
                        h.armLen = armLen;
                        outResult = h;
                    }
                };
                detectHand(PoseLandmarkIdx::L_WRIST, PoseLandmarkIdx::L_ELBOW, handLeft);
                detectHand(PoseLandmarkIdx::R_WRIST, PoseLandmarkIdx::R_ELBOW, handRight);
            }

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
                    // RTMO keypoints are already in original frame pixel coords
                    uint8_t cyan[3] = {255, 255, 0};
                    static const int POSE_CONN[][2] = {
                        {5,6},{5,7},{7,9},{6,8},{8,10},
                        {5,11},{6,12},{11,12},
                        {11,13},{13,15},{12,14},{14,16},
                    };
                    for (auto& c : POSE_CONN) {
                        if (pose.kpScore(c[0]) > 0.3f && pose.kpScore(c[1]) > 0.3f) {
                            drawLine(annotated,
                                (int)pose.kpX(c[0]), (int)pose.kpY(c[0]),
                                (int)pose.kpX(c[1]), (int)pose.kpY(c[1]),
                                cyan, 2);
                        }
                    }
                    uint8_t dotColor[3] = {0, 0, 255};
                    for (int i = 0; i < 17; i++) {
                        if (pose.kpScore(i) > 0.3f)
                            drawCircleFilled(annotated, (int)pose.kpX(i), (int)pose.kpY(i), 3, dotColor);
                    }
                }
                // Hand skeleton
                auto drawHandSkeleton = [&](const HandResult& hr) {
                    if (!hr.detected) return;
                    // Hand landmarker outputs landmarks in 224-canvas pixel
                    // coords (NOT normalized); un-letterbox back to ROI frame.
                    auto toFrame = [&](float px224, float py224) {
                        int px = (int)((px224 - hr.lbPadX) / hr.lbScale) + hr.roiX;
                        int py = (int)((py224 - hr.lbPadY) / hr.lbScale) + hr.roiY;
                        return std::make_pair(px, py);
                    };

                    // The landmarker returns FIXED-SCALE landmarks (~115px
                    // wrist→midtip on the 224-canvas) regardless of the
                    // actual hand size in the input. Without MediaPipe's
                    // detector (which provides a tight rotation-normalized
                    // crop), the landmarks are always ~50% of canvas scale.
                    // Rescale to the anthropometric hand length (0.7× forearm).
                    std::pair<int,int> frameLm[21];
                    for (int i = 0; i < 21; i++)
                        frameLm[i] = toFrame(hr.lmX(i), hr.lmY(i));

                    // Anchor: lock the hand wrist (lm 0) to the pose-tracker
                    // wrist so the hand never appears detached/elongated.
                    int anchorX = frameLm[0].first;
                    int anchorY = frameLm[0].second;
                    if (hr.anchorX >= 0.0f) { anchorX = (int)hr.anchorX; anchorY = (int)hr.anchorY; }

                    // Scale factor: expected hand length vs model's output
                    float modelLen = 0;
                    static const int tips[] = {4, 8, 12, 16, 20};
                    for (int t : tips) {
                        float dx = frameLm[t].first - frameLm[0].first;
                        float dy = frameLm[t].second - frameLm[0].second;
                        float d = std::sqrt(dx*dx + dy*dy);
                        if (d > modelLen) modelLen = d;
                    }
                    // Hand length ≈ 0.7 × forearm; fall back to 0.6×ROI if no armLen
                    float expectedLen = (hr.armLen > 1.0f)
                        ? hr.armLen * 0.7f
                        : (hr.lbScale > 0 ? (224.0f / hr.lbScale) * 0.6f : modelLen);
                    float scale = (modelLen > 1.0f) ? expectedLen / modelLen : 1.0f;

                    auto toFrameScaled = [&](int i) {
                        float dx = (frameLm[i].first - frameLm[0].first) * scale;
                        float dy = (frameLm[i].second - frameLm[0].second) * scale;
                        return std::make_pair((int)(anchorX + dx), (int)(anchorY + dy));
                    };

                    uint8_t orange[3] = {0, 200, 255};   // BGR orange lines
                    uint8_t dotBlue[3] = {255, 100, 0};   // BGR blue dots
                    static const int HAND_CONN[][2] = {
                        {0,1},{1,2},{2,3},{3,4},
                        {0,5},{5,6},{6,7},{7,8},
                        {5,9},{9,10},{10,11},{11,12},
                        {9,13},{13,14},{14,15},{15,16},
                        {13,17},{17,18},{18,19},{19,20},
                        {0,17},
                    };
                    for (auto& c : HAND_CONN) {
                        auto [x0, y0] = toFrameScaled(c[0]);
                        auto [x1, y1] = toFrameScaled(c[1]);
                        drawLine(annotated, x0, y0, x1, y1, orange, 2);
                    }
                    for (int i = 0; i < 21; i++) {
                        auto [px, py] = toFrameScaled(i);
                        drawCircleFilled(annotated, px, py, 3, dotBlue);
                    }
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
                    if (rigSolver.calibrated() && rigSolver.poseCalibrated()) {
                        calibrating = false;
                        fprintf(stderr, "[live] Calibration complete.\n");
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
                setIf("spine", bp.spine);
                setIf("leftUpperArm", bp.leftUpperArm);
                setIf("rightUpperArm", bp.rightUpperArm);
                setIf("leftLowerArm", bp.leftLowerArm);
                setIf("rightLowerArm", bp.rightLowerArm);
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
            } else {
                jointMatrices = bindJointMatrices;
            }
        }

        // Dynamic camera: blend sit-cam (bust shot) ↔ stand-cam (full body)
        {
            const BodyPose& bp = rigSolver.bodyPose();
            float frac = bp.valid ? bp.standing : 0.0f;
            if (bp.valid && bp.bodyExtent > 0.5f)
                frac = std::min(1.0f, frac + (bp.bodyExtent - 0.5f) * 0.3f);
            float sit = 1.0f - frac;
            float newFov = sitCam.fov * sit + standCam.fov * frac;
            float newTargetY = sitCam.targetY * sit + standCam.targetY * frac;
            float newDist = sitCam.dist * sit + standCam.dist * frac;
            // Smooth transitions
            float camAlpha = std::min(1.0f, dt * 3.0f);
            curFov += (newFov - curFov) * camAlpha;
            curTargetY += (newTargetY - curTargetY) * camAlpha;
            curDist += (newDist - curDist) * camAlpha;
            computeViewProj();
        }

        // Render avatar
        auto proc = processVerticesParallel(model, jointMatrices, viewProj,
                                            rigSolver.morphWeights(), renderW, renderH, numThreads);
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
    }

    running = false;
    detectThread.join();
    webcam.stop();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
