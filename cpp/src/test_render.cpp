// Expression-driven render regression tests — C++ port of Python
// test_rig.py. Renders the avatar offline (no camera / no MediaPipe) driven
// by the rig solver with synthetic faces, and pixel-diffs neutral vs
// aa/blink/happy/gaze/head-rotation frames, plus transparent-background
// alpha output.
//
// Usage: test_render [path/to/avatar.vrm]
#include "vrm_loader.h"
#include "rasterizer.h"
#include "rig_solver.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdio>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>

namespace fs = std::filesystem;
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

static int g_failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        const std::string m_(msg);                                           \
        if (cond) {                                                          \
            printf("PASS: %s\n", m_.c_str());                                \
        } else {                                                             \
            printf("FAIL: %s  (%s:%d)\n", m_.c_str(), __FILE__, __LINE__);   \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

// ARKit indices used by the synthetic faces
enum {
    EYE_BLINK_L = 9, EYE_BLINK_R = 10,
    EYE_LOOK_IN_L = 13,
    JAW_OPEN = 25,
    MOUTH_SMILE_L = 44, MOUTH_SMILE_R = 45,
};

static FaceResult makeFace(const std::vector<std::pair<int, float>>& bs,
                           float yaw = 0) {
    FaceResult f;
    f.detected = true;
    for (auto& [idx, v] : bs) f.blendshapes[idx] = v;
    f.yaw = yaw;
    return f;
}

struct RenderCtx {
    VRMModel model;
    std::vector<glm::mat4> bindWorld;
    std::vector<glm::mat4> bindJoint;
    glm::mat4 viewProj;
    int width, height;
    int threads = 4;
    Framebuffer fb{1, 1};

    RenderCtx(const std::string& vrmPath, int w, int h)
        : model(loadVRM(vrmPath)), width(w), height(h), fb(w, h) {
        bindWorld = computeWorldMatrices(model);
        bindJoint = computeJointMatrices(model, bindWorld);

        // Bust-shot camera (same framing as the live app's sit-cam)
        glm::vec3 bmin(1e9f), bmax(-1e9f);
        for (const auto& mesh : model.meshes) {
            glm::mat4 nodeMat = (mesh.nodeIndex >= 0 &&
                                 mesh.nodeIndex < (int)bindWorld.size())
                ? bindWorld[mesh.nodeIndex] : glm::mat4(1.0f);
            for (const auto& prim : mesh.primitives) {
                for (size_t vi = 0; vi < prim.positions.size(); vi += 3) {
                    glm::vec4 p(prim.positions[vi], prim.positions[vi + 1],
                                prim.positions[vi + 2], 1.0f);
                    glm::vec3 wp = glm::vec3(nodeMat * p);
                    bmin = glm::min(bmin, wp);
                    bmax = glm::max(bmax, wp);
                }
            }
        }
        glm::vec3 centre = (bmin + bmax) * 0.5f;
        glm::vec3 size = bmax - bmin;
        float targetY = centre.y + size.y * 0.28f;
        float dist = std::max(size.y * 0.85f, 1.1f);
        float fovY = 28.0f;
        float aspect = (float)width / height;
        glm::mat4 proj = glm::perspective(glm::radians(fovY), aspect, 0.1f, 100.0f);
        glm::vec3 eye(0, targetY + 0.10f, -dist);
        glm::vec3 tgt(0, targetY - 0.02f, 0);
        glm::mat4 view = glm::lookAt(eye, tgt, glm::vec3(0, 1, 0));
        viewProj = proj * view;
    }

    // Render one solver pose into fb
    void render(const RigSolver& solver, bool transparentBg = false) {
        if (transparentBg)
            fb.setClearColor(0, 0, 0, 0);
        else
            fb.setClearColor(255, 255, 255, 255);

        std::unordered_map<int, glm::quat> overrides;
        glm::quat headRot = solver.headRotation();
        if (model.headNodeIndex >= 0 && headRot != glm::quat(1, 0, 0, 0))
            overrides[model.headNodeIndex] = headRot;
        if (solver.lookAtBoneType()) {
            glm::quat eyeRot = solver.eyeRotation();
            if (eyeRot != glm::quat(1, 0, 0, 0)) {
                auto it = model.boneNodes.find("leftEye");
                if (it != model.boneNodes.end()) overrides[it->second] = eyeRot;
                it = model.boneNodes.find("rightEye");
                if (it != model.boneNodes.end()) overrides[it->second] = eyeRot;
            }
        }

        std::vector<glm::mat4> joint =
            overrides.empty() ? bindJoint
                              : [&]() {
                                    auto w = computeWorldMatricesWithOverrides(model, overrides);
                                    return computeJointMatrices(model, w);
                                }();

        auto proc = processVerticesParallel(model, joint, viewProj,
                                            solver.morphWeights(), width, height,
                                            threads);
        fb.clear();
        rasterizeParallel(proc, fb, model.textures, threads);
    }
};

static int diffPixels(const Framebuffer& a, const Framebuffer& b, int tol = 30) {
    int diff = 0;
    for (size_t i = 0; i < a.color.size(); i += 4) {
        int d = std::abs(a.color[i] - b.color[i]) +
                std::abs(a.color[i + 1] - b.color[i + 1]) +
                std::abs(a.color[i + 2] - b.color[i + 2]);
        if (d > tol) diff++;
    }
    return diff;
}

// Count foreground pixels: for a transparent render that's alpha > 127;
// for an opaque render, pixels differing from the background clear color.
static int countFg(const Framebuffer& fb, bool transparent) {
    int fg = 0;
    for (size_t i = 0; i < fb.color.size(); i += 4) {
        if (transparent) {
            if (fb.color[i + 3] > 127) fg++;
        } else {
            int dr = std::abs((int)fb.color[i] - 255);
            int dg = std::abs((int)fb.color[i + 1] - 255);
            int db = std::abs((int)fb.color[i + 2] - 255);
            if (dr + dg + db > 30) fg++;
        }
    }
    return fg;
}

static void savePNG(const std::string& name, const Framebuffer& fb) {
    stbi_write_png(name.c_str(), fb.width, fb.height, 4, fb.color.data(),
                   fb.width * 4);
}

static std::string findAvatar(int argc, char** argv) {
    if (argc > 1) return argv[1];
    const char* candidates[] = {
        "assets/avatars/male_52blendshapes.vrm",
        "../../assets/avatars/male_52blendshapes.vrm",
    };
    for (const char* c : candidates)
        if (fs::exists(c)) return c;
    return "";
}

int main(int argc, char** argv) {
    std::string vrmPath = findAvatar(argc, argv);
    if (vrmPath.empty() || !fs::exists(vrmPath)) {
        fprintf(stderr, "test_render: no VRM file found (pass one as argv[1])\n");
        return 2;
    }
    printf("test_render: model %s\n", vrmPath.c_str());

    const int W = 640, H = 360;
    RenderCtx ctx(vrmPath, W, H);
    if (ctx.model.meshes.empty()) {
        fprintf(stderr, "test_render: failed to load model\n");
        return 2;
    }
    printf("  %d meshes, %d triangles, %d blendshape groups, %d spring chains\n",
           (int)ctx.model.meshes.size(), ctx.model.totalTriangles(),
           (int)ctx.model.blendShapeGroups.size(),
           (int)ctx.model.springChains.size());

    // Build solver states (drive filters to convergence)
    auto makeSolver = [&](const std::vector<std::pair<int, float>>& bs, float yaw = 0) {
        RigSolver solver(ctx.model);
        solver.startCalibration();
        FaceResult neutral;
        neutral.detected = true;
        for (int i = 0; i < 35; i++) solver.calibrate(neutral);
        solver.calibratePose();
        FaceResult face = makeFace(bs, yaw);
        for (int i = 0; i < 40; i++) solver.update(face, 1.0f / 15.0f);
        return solver;
    };

    RigSolver solverNeutral = makeSolver({});
    RigSolver solverAa = makeSolver({{JAW_OPEN, 0.7f}});
    RigSolver solverBlink = makeSolver({{EYE_BLINK_L, 0.9f}, {EYE_BLINK_R, 0.9f}});
    RigSolver solverHappy = makeSolver({{MOUTH_SMILE_L, 0.7f}, {MOUTH_SMILE_R, 0.7f}});
    RigSolver solverGaze = makeSolver({{EYE_LOOK_IN_L, 1.0f}});
    RigSolver solverYaw = makeSolver({}, 30.0f);

    // Render all cases
    Framebuffer fbNeutral(W, H), fbAa(W, H), fbBlink(W, H), fbHappy(W, H),
        fbGaze(W, H), fbYaw(W, H), fbAlpha(W, H);
    struct Case { RigSolver& solver; Framebuffer& fb; const char* name; };
    // (solvers are copied into place; render reads their state)
    Case cases[] = {
        {solverNeutral, fbNeutral, "neutral"},
        {solverAa, fbAa, "aa"},
        {solverBlink, fbBlink, "blink"},
        {solverHappy, fbHappy, "happy"},
        {solverGaze, fbGaze, "gaze"},
        {solverYaw, fbYaw, "headyaw"},
    };
    for (auto& c : cases) {
        ctx.fb = Framebuffer(W, H);
        ctx.render(c.solver);
        c.fb = ctx.fb;  // move rendered pixels into the case buffer
        std::string png = std::string("output_test_") + c.name + ".png";
        savePNG(png, c.fb);
    }

    // 1. Avatar visible in the neutral frame
    int fg = countFg(fbNeutral, /*transparent=*/false);
    printf("  neutral fg pixels: %d\n", fg);
    CHECK(fg > 15000, "avatar renders (neutral frame has foreground)");

    // 2-6. Expression / pose pixel diffs vs neutral. Thresholds are
    // calibrated for a 640x360 software render with per-pixel tolerance 30.
    struct DiffCase { const char* name; Framebuffer& fb; int expect; };
    DiffCase diffs[] = {
        {"aa (jawOpen)", fbAa, 300},
        {"blink", fbBlink, 200},
        {"happy (smile)", fbHappy, 200},
        {"head yaw 30deg", fbYaw, 1500},
    };
    for (auto& d : diffs) {
        int n = diffPixels(fbNeutral, d.fb);
        printf("  diff neutral vs %-16s: %d pixels\n", d.name, n);
        CHECK(n > d.expect, std::string(d.name) + " changes the render");
    }

    // Gaze: eye bone rotation must move eye pixels
    bool hasEyes = ctx.model.boneNodes.count("leftEye") &&
                   ctx.model.boneNodes.count("rightEye");
    if (hasEyes && solverGaze.lookAtBoneType()) {
        int n = diffPixels(fbNeutral, fbGaze);
        printf("  diff neutral vs %-16s: %d pixels\n", "gaze right", n);
        CHECK(n > 100, "eye-gaze rotation changes the render");
    } else {
        printf("  SKIP: gaze render test (no eye bones or expression lookAt)\n");
    }

    // 7. Transparent background: alpha channel output
    ctx.fb = Framebuffer(W, H);
    ctx.render(solverNeutral, /*transparentBg=*/true);
    fbAlpha = ctx.fb;
    savePNG("output_test_alpha.png", fbAlpha);
    int alpha0 = 0, alpha255 = 0;
    for (size_t i = 0; i < fbAlpha.color.size(); i += 4) {
        if (fbAlpha.color[i + 3] == 0) alpha0++;
        else if (fbAlpha.color[i + 3] == 255) alpha255++;
    }
    printf("  transparent bg: %d fully-clear + %d opaque pixels\n", alpha0, alpha255);
    CHECK(alpha0 > 100000, "transparent background → mostly alpha-0 pixels");
    CHECK(alpha255 > 15000, "avatar stays opaque over transparent background");

    printf("\n%s: %d failure(s)\n", g_failures ? "FAILURES" : "ALL OK", g_failures);
    return g_failures;
}
