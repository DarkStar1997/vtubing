# Bare-Metal CPU VTubing Utility: Technical Implementation Plan

## 1. Project Overview
**Goal:** Build an ultra-low latency, zero-GPU 3D VTubing utility that captures webcam input, tracks full-body/hand/face movements, maps them to an arbitrary VRM model, and outputs the rendered frames to OBS via shared memory.
**Core Constraint:** 100% CPU-based rendering and processing. The GPU must remain completely free for gaming.
**Target Performance:** 60 FPS internal processing, <30ms end-to-end latency, <15% total CPU usage on a modern 6-core processor.

---

## 2. Technology Stack

| Component | Technology | Justification |
| :--- | :--- | :--- |
| **Language** | C++20 | Maximum performance, direct hardware control. |
| **Build System** | CMake | Industry standard, handles complex dependencies easily. |
| **Windowing/Pixels** | **SDL3** | `SDL_CreateSurface` provides a raw, fast RAM pixel buffer. |
| **AI Inference** | **ONNX Runtime (C++)** | Ultra-fast CPU inference. Must use **INT8 quantized** models. |
| **3D Model Parsing** | **cgltf** | Single-header, blazing fast glTF/VRM parser. |
| **Scene Math** | **DirectXMath** (Win) / **GLM** (Cross) | Optimized 4x4 matrix and Quaternion math for bone transforms. |
| **SIMD Rasterizer** | **Google Highway** OR **VCL 2** | Zero-overhead SIMD abstraction for the pixel-pushing inner loop. |
| **Virtual Camera** | **Shared Memory** + OBS Plugin | Memory-mapped files bypass OS compositor, zero latency. |

---

## 3. High-Level Architecture (The 4-Stage Pipeline)

To maximize CPU cache utilization and prevent thread-blocking, the app is divided into 4 distinct stages/threads.

1. **Ingestion & AI Thread:** Captures webcam frame, downscales, extracts ROIs (Regions of Interest), runs ONNX INT8 models.
2. **Retargeting & Physics Thread:** Filters jitter (One Euro), calculates Quaternions for bones, calculates Blendshape weights, runs lightweight Verlet physics for VRM Spring Bones (hair/clothes).
3. **Rasterization Thread(s) (Multithreaded):** Transforms vertices, splits the screen into tiles, uses SIMD (AVX2) to rasterize triangles, sample textures, and write to the Z-Buffer/Color Buffer.
4. **Output Thread:** Copies the completed RAM Framebuffer to the Shared Memory block for OBS to read.

---

## 4. Phase-by-Phase Implementation Plan

### Phase 1: Core Pipeline & Shared Memory (Weeks 1-2)
*Objective: Get a solid color rendering to OBS with zero latency.*
- [ ] Set up CMake project with SDL3 and fetch dependencies.
- [ ] Create a raw RGBA pixel buffer in RAM (e.g., 1280x720).
- [ ] Implement Windows Memory-Mapped Files (`CreateFileMapping`).
- [ ] Define the Shared Memory Header (Width, Height, Frame ID, Double Buffer Index).
- [ ] Write a basic loop that fills the RAM buffer with a test color and pushes it to Shared Memory.
- [ ] Write/Install a minimal OBS Source Plugin to read this Shared Memory and display it.

### Phase 2: VRM Parsing & The Software Rasterizer (Weeks 3-6)
*Objective: Load a 3D model and draw it to the RAM buffer without a GPU.*
- [ ] Integrate `cgltf` to parse `.vrm` files. Extract vertices, UVs, normals, and skeleton hierarchy.
- [ ] Parse VRM-specific extensions (`VRMC_vrm` for humanoid bones, `VRMC_springBone` for physics).
- [ ] **Build the Software Rasterizer:**
  - Implement Vertex Shading (Model -> World -> View -> Clip -> Screen space).
  - Implement Triangle Setup (Bounding box calculation, Backface culling).
  - Implement the Inner Loop: Barycentric coordinates / Edge functions to test pixel inclusion.
  - Implement Z-Buffer (Depth testing) using 16-bit or 32-bit floats.
  - Implement Texture Sampling (Bilinear filtering) and basic MToon Cel-Shading (N-dot-L + Ramp Texture).
- [ ] **Optimize Rasterizer:** Convert the inner pixel loop to **Structure of Arrays (SoA)** and optimize using **Google Highway** (processing 8 pixels per clock cycle).

### Phase 3: AI Tracking Integration (Weeks 7-9)
*Objective: Extract 3D keypoints from the webcam with minimal CPU overhead.*
- [ ] Integrate OpenCV or Media Foundation for raw webcam capture.
- [ ] Integrate ONNX Runtime C++ API. Load INT8 quantized models:
  - *Body:* MoveNet Lightning (192x192)
  - *Face:* Lightweight FaceMesh (256x256)
  - *Hands:* MediaPipe Hands (256x256)
- [ ] Implement the **Cascaded ROI Pipeline**:
  - Run Body model on full downscaled frame.
  - Extract wrist/neck coordinates -> Crop 256x256 boxes -> Run Hand/Face models *only* on those boxes.
- [ ] Implement Confidence Thresholding (discard keypoints with < 0.3 confidence to prevent rubber-banding).

### Phase 4: Retargeting Math & Animation (Weeks 10-12)
*Objective: Map AI data to the VRM skeleton smoothly and accurately.*
- [ ] Implement the **One Euro Filter** in C++ to smooth all raw AI keypoints.
- [ ] Implement the **Auto-Retargeting Engine**:
  - Map VRM Humanoid bone indices to AI keypoint indices.
  - Calculate Rest Pose vectors (`tail - head`).
  - Calculate Target vectors from filtered AI points.
  - Use `Quaternion::FromToRotation` to calculate delta rotations.
- [ ] Implement Facial Blendshape mapping (Calculate distances between face points -> map to VRM `aa`, `blink`, etc.).
- [ ] Implement **State-Based Overrides**:
  - Detect "Sitting" state via hip height/torso angle.
  - Apply procedural sitting pose to leg bones when AI leg confidence is low.
- [ ] Implement lightweight CPU Verlet integration for VRM `SpringBone` (hair/clothing physics).

### Phase 5: Polish, Framing & Edge Cases (Weeks 13-14)
*Objective: Make it look professional and broadcast-ready.*
- [ ] Implement Dynamic Virtual Camera Framing (Auto-zoom/pan based on user's hip height in the webcam).
- [ ] Implement "Deadzones" for micro-jitter (e.g., ignore head tilts < 3 degrees).
- [ ] Implement Non-Linear Exaggeration Curves (Ease-out functions for snappy, expressive movements).
- [ ] Profile the entire app using **Tracy Profiler**. Identify cache misses and optimize memory layouts.
- [ ] Final UI implementation using SDL3 or Dear ImGui (rendered directly to the RAM buffer).

---

## 5. Project Directory Structure

```text
vtuber-cpu-engine/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                # Entry point, thread management, main loop
│   ├── capture/
│   │   └── webcam.cpp          # Media Foundation / V4L2 wrapper
│   ├── tracking/
│   │   ├── onnx_manager.cpp    # ONNX Runtime session setup
│   │   ├── roi_cascader.cpp    # Crop extraction logic
│   │   └── one_euro_filter.cpp # Jitter smoothing
│   ├── retargeting/
│   │   ├── vrm_parser.cpp      # cgltf wrapper + VRM extension parsing
│   │   ├── skeleton_mapper.cpp # Heuristic bone matching & Quaternion math
│   │   └── blendshape_calc.cpp # Face point distance -> weight math
│   ├── rendering/
│   │   ├── rasterizer.cpp      # The core SIMD pixel-pusher (Highway/VCL)
│   │   ├── vertex_shader.cpp   # Matrix transformations
│   │   ├── mtoon_shader.cpp    # Cel-shading math
│   │   └── spring_bones.cpp    # Verlet physics for hair/clothes
│   └── output/
│       └── shared_memory.cpp   # Memory-mapped file writer for OBS
├── include/                    # Headers for third-party libs (SDL3, cgltf, etc.)
├── models/                     # Default lightweight VRM model & ONNX files
└── obs_plugin/                 # Source code for the custom OBS C++ plugin