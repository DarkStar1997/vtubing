#include "vrm_loader.h"
#include "rasterizer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <SDL3/SDL.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>

int main(int argc, char** argv) {
    std::string vrmPath = "../../assets/avatars/male_52blendshapes.vrm";

    // Parse args: [--bench N] [--ss N] [vrm_path]
    int benchFrames = 0;
    int ss = 2;  // supersampling factor (2 = 4× SSAA)
    int argIdx = 1;
    while (argIdx < argc) {
        std::string arg = argv[argIdx];
        if (arg == "--bench" && argIdx + 1 < argc) {
            benchFrames = std::atoi(argv[argIdx + 1]);
            argIdx += 2;
        } else if (arg == "--ss" && argIdx + 1 < argc) {
            ss = std::atoi(argv[argIdx + 1]);
            argIdx += 2;
        } else {
            vrmPath = arg;
            argIdx++;
        }
    }

    int fbWidth = 1280;
    int fbHeight = 720;
    int renderW = fbWidth * ss;
    int renderH = fbHeight * ss;

    fprintf(stderr, "[main] loading VRM: %s\n", vrmPath.c_str());
    VRMModel model = loadVRM(vrmPath);
    if (model.meshes.empty()) {
        fprintf(stderr, "[main] failed to load model\n");
        return 1;
    }

    // --- Compute joint matrices (bind pose, no bone overrides) ---
    std::vector<glm::mat4> worldMatrices = computeWorldMatrices(model);
    std::vector<glm::mat4> jointMatrices = computeJointMatrices(model, worldMatrices);

    // --- Camera: match browser loadVRM() bbox-based framing (avatar.js:114-126) ---
    // Compute world-space bounding box of all mesh vertices
    glm::vec3 bboxMin(1e9f), bboxMax(-1e9f);
    for (size_t mi = 0; mi < model.meshes.size(); mi++) {
        const auto& mesh = model.meshes[mi];
        glm::mat4 nodeMat = (mesh.nodeIndex >= 0 && mesh.nodeIndex < (int)worldMatrices.size())
            ? worldMatrices[mesh.nodeIndex] : glm::mat4(1.0f);
        for (const auto& prim : mesh.primitives) {
            int vc = prim.vertexCount();
            for (int vi = 0; vi < vc; vi++) {
                glm::vec4 p(prim.positions[vi * 3], prim.positions[vi * 3 + 1],
                            prim.positions[vi * 3 + 2], 1.0f);
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
    glm::vec3 eye(0.0f, targetY + 0.10f, -camDist);
    glm::vec3 target(0.0f, targetY - 0.02f, 0.0f);
    glm::vec3 up(0, 1, 0);
    float fovY = glm::radians(28.0f);
    float aspect = (float)fbWidth / fbHeight;
    float nearZ = 0.1f;
    float farZ = 100.0f;

    glm::mat4 proj = glm::perspective(fovY, aspect, nearZ, farZ);
    glm::mat4 view = glm::lookAt(eye, target, up);
    glm::mat4 viewProj = proj * view;

    // --- Morph weights (flat array, one weight per morph target per mesh) ---
    std::vector<float> morphWeights;
    int totalMorphs = 0;
    for (size_t mi = 0; mi < model.meshes.size(); mi++) {
        if (!model.meshes[mi].primitives.empty())
            totalMorphs += model.meshes[mi].primitives[0].morphCount;
    }
    morphWeights.resize(totalMorphs, 0.0f);

    // --- Framebuffer ---
    Framebuffer ssfb(renderW, renderH);  // supersampled render target
    Framebuffer fb(fbWidth, fbHeight);   // final output (downsampled)

    // --- Render one frame and time it ---
    int numThreads = std::min((int)std::thread::hardware_concurrency(), 16);
    fprintf(stderr, "[main] threads: %d, SS: %dx (%dx%d → %dx%d)\n",
            numThreads, ss, renderW, renderH, fbWidth, fbHeight);

    if (benchFrames > 0) {
        fprintf(stderr, "\n=== BENCHMARK: %d frames ===\n", benchFrames);
        Timer btimer, timer;
        double totalVert = 0, totalRast = 0, totalDown = 0;
        for (int f = 0; f < benchFrames; f++) {
            timer.reset();
            auto proc = processVerticesParallel(model, jointMatrices, viewProj, morphWeights, renderW, renderH, numThreads);
            double vms = timer.elapsedMs();
            timer.reset();
            ssfb.clear();
            rasterizeParallel(proc, ssfb, model.textures, numThreads);
            double rms = timer.elapsedMs();
            timer.reset();
            if (ss > 1) downsample2x2(ssfb, fb, numThreads);
            double dms = timer.elapsedMs();
            totalVert += vms;
            totalRast += rms;
            totalDown += dms;
        }
        double wallMs = btimer.elapsedMs();
        fprintf(stderr, "Avg vertex: %.3f ms | Avg raster: %.3f ms | Avg downsample: %.3f ms\n",
                totalVert / benchFrames, totalRast / benchFrames, totalDown / benchFrames);
        fprintf(stderr, "Avg total:  %.3f ms (%.1f fps)\n",
                wallMs / benchFrames, benchFrames * 1000.0 / wallMs);

        // Verify last frame: foreground pixel count
        Framebuffer& out = (ss > 1) ? fb : ssfb;
        int fgCount = 0;
        for (int i = 0; i < fbWidth * fbHeight; i++)
            if (out.depth[i] < 1.0f) fgCount++;
        fprintf(stderr, "Foreground pixels: %d / %d (%.1f%%)\n",
                fgCount, fbWidth * fbHeight, 100.0f * fgCount / (fbWidth * fbHeight));

        // Save last frame for visual comparison
        std::vector<uint8_t> rgb(fbWidth * fbHeight * 3);
        for (int i = 0; i < fbWidth * fbHeight; i++) {
            rgb[i * 3 + 0] = out.color[i * 4 + 0];
            rgb[i * 3 + 1] = out.color[i * 4 + 1];
            rgb[i * 3 + 2] = out.color[i * 4 + 2];
        }
        stbi_write_png("output_bench.png", fbWidth, fbHeight, 3, rgb.data(), fbWidth * 3);
        fprintf(stderr, "Saved: output_bench.png\n");
        fprintf(stderr, "Wall time:  %.1f ms\n", wallMs);
        return 0;
    }

    Timer timer;

    // Step 1: Vertex processing (parallel)
    timer.reset();
    auto processed = processVerticesParallel(model, jointMatrices, viewProj, morphWeights, renderW, renderH, numThreads);
    double vertexMs = timer.elapsedMs();

    // Step 2: Rasterization (parallel, supersampled)
    timer.reset();
    ssfb.clear();
    rasterizeParallel(processed, ssfb, model.textures, numThreads);
    double rasterMs = timer.elapsedMs();

    // Step 3: Downsample SSAA → output
    timer.reset();
    if (ss > 1) downsample2x2(ssfb, fb, numThreads);
    double downMs = timer.elapsedMs();

    double totalMs = vertexMs + rasterMs + downMs;

    fprintf(stderr, "\n=== SSAA %dx (%d threads) ===\n", ss, numThreads);
    fprintf(stderr, "Vertex processing: %.2f ms\n", vertexMs);
    fprintf(stderr, "Rasterization:     %.2f ms\n", rasterMs);
    fprintf(stderr, "Downsample:        %.2f ms\n", downMs);
    fprintf(stderr, "Total:             %.2f ms (%.1f fps)\n", totalMs, 1000.0 / totalMs);
    fprintf(stderr, "Resolution:        %dx%d (render: %dx%d)\n", fbWidth, fbHeight, renderW, renderH);

    // Count foreground pixels
    Framebuffer& out = (ss > 1) ? fb : ssfb;
    int fgCount = 0;
    for (int i = 0; i < fbWidth * fbHeight; i++) {
        if (out.depth[i] < 1.0f) fgCount++;
    }
    fprintf(stderr, "Foreground pixels: %d / %d (%.1f%%)\n",
            fgCount, fbWidth * fbHeight, 100.0f * fgCount / (fbWidth * fbHeight));

    // --- Save PNG ---
    // Convert RGBA to RGB for stb_image_write
    std::vector<uint8_t> rgb(fbWidth * fbHeight * 3);
    for (int i = 0; i < fbWidth * fbHeight; i++) {
        rgb[i * 3 + 0] = out.color[i * 4 + 0];
        rgb[i * 3 + 1] = out.color[i * 4 + 1];
        rgb[i * 3 + 2] = out.color[i * 4 + 2];
    }
    stbi_write_png("output_cpp.png", fbWidth, fbHeight, 3, rgb.data(), fbWidth * 3);
    fprintf(stderr, "[main] saved output_cpp.png\n");

    // --- SDL3 window display ---
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "[main] SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("VTuber CPU Rasterizer", fbWidth, fbHeight, 0);
    SDL_Surface* winSurface = SDL_GetWindowSurface(window);
    if (!winSurface) {
        fprintf(stderr, "[main] SDL_GetWindowSurface failed: %s\n", SDL_GetError());
        return 1;
    }

    std::vector<uint8_t> bgra(fbWidth * fbHeight * 4);
    for (int i = 0; i < fbWidth * fbHeight; i++) {
        bgra[i * 4 + 0] = out.color[i * 4 + 2];
        bgra[i * 4 + 1] = out.color[i * 4 + 1];
        bgra[i * 4 + 2] = out.color[i * 4 + 0];
        bgra[i * 4 + 3] = 255;
    }

    SDL_Surface* fbSurface = SDL_CreateSurfaceFrom(
        fbWidth, fbHeight, SDL_PIXELFORMAT_BGRA8888, bgra.data(), fbWidth * 4);

    bool running = true;
    SDL_Event event;
    fprintf(stderr, "\n[main] SDL window open. ESC or close to quit.\n");

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) running = false;
        }

        SDL_BlitSurface(fbSurface, nullptr, winSurface, nullptr);
        SDL_UpdateWindowSurface(window);
        SDL_Delay(16);
    }

    SDL_DestroySurface(fbSurface);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}