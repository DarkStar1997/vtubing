#pragma once
#include "rasterizer.h"
#include <cstdint>

// MToon lighting constants (shared between rasterizer.cpp and rasterizer_tri.cpp)
static constexpr float KX = 0.371391f, KY = 0.742781f, KZ = -0.557086f;
static constexpr float KEY_INT = 0.9f;
static constexpr float FX = -0.577350f, FY = 0.577350f, FZ = -0.577350f;
static constexpr float FILL_INT = 0.3f;
static constexpr float HEMI_GROUND = 0.267f;
static constexpr float HEMI_DIFF = 1.0f - 0.267f;
static constexpr float INV_PI = 0.318309886f;
static constexpr float KEY_PI = KEY_INT * INV_PI;
static constexpr float FILL_PI = FILL_INT * INV_PI;

// sRGB LUTs
extern float sRGBToLinearLUT[256];
extern uint8_t linearToSRGBLUT[1025];

static inline uint8_t linToSRGB(float v) {
    int idx = (int)(v * 1024.0f);
    if (idx <= 0) return 0;
    if (idx >= 1025) return 255;
    return linearToSRGBLUT[idx];
}

// rasterizeTri — defined in rasterizer_tri.cpp (compiled with -ffast-math)
void rasterizeTri(
    const ProcessedPrim& pp,
    const MeshPrimitive& prim,
    int triIdx,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int xMin, int xMax, int yMin, int yMax,
    bool depthTest, bool depthWrite);
