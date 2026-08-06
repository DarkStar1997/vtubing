#include "rasterizer_internal.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

void rasterizeTri(
    const ProcessedPrim& pp,
    const MeshPrimitive& prim,
    int triIdx,
    Framebuffer& fb,
    const std::vector<TextureData>& textures,
    int xMin, int xMax, int yMin, int yMax,
    bool depthTest, bool depthWrite)
{
    const uint32_t* indices = prim.indices.data();
    uint32_t i0 = indices[triIdx * 3 + 0];
    uint32_t i1 = indices[triIdx * 3 + 1];
    uint32_t i2 = indices[triIdx * 3 + 2];

    float x0 = pp.screenX[i0], y0 = pp.screenY[i0], z0 = pp.clipZ[i0];
    float x1 = pp.screenX[i1], y1 = pp.screenY[i1], z1 = pp.clipZ[i1];
    float x2 = pp.screenX[i2], y2 = pp.screenY[i2], z2 = pp.clipZ[i2];

    float u0 = pp.uvU[i0], v0uv = pp.uvV[i0];
    float u1 = pp.uvU[i1], v1uv = pp.uvV[i1];
    float u2 = pp.uvU[i2], v2uv = pp.uvV[i2];

    float n0x = pp.worldNX[i0], n0y = pp.worldNY[i0], n0z = pp.worldNZ[i0];
    float n1x = pp.worldNX[i1], n1y = pp.worldNY[i1], n1z = pp.worldNZ[i1];
    float n2x = pp.worldNX[i2], n2y = pp.worldNY[i2], n2z = pp.worldNZ[i2];

    float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (area < 0) {
        std::swap(x1, x2); std::swap(y1, y2); std::swap(z1, z2);
        std::swap(u1, u2); std::swap(v1uv, v2uv);
        std::swap(n1x, n2x); std::swap(n1y, n2y); std::swap(n1z, n2z);
        area = -area;
    } else if (!prim.doubleSided) {
        return;  // Back-face culling for single-sided materials
    }
    if (area < 0.01f) return;

    int ix0 = std::max(xMin, static_cast<int>(std::min({x0, x1, x2})));
    int ix1 = std::min(xMax, static_cast<int>(std::max({x0, x1, x2})));
    int iy0 = std::max(yMin, static_cast<int>(std::min({y0, y1, y2})));
    int iy1 = std::min(yMax, static_cast<int>(std::max({y0, y1, y2})));
    if (iy1 < iy0 || ix1 < ix0) return;

    float invArea = 1.0f / area;
    bool hasTex = prim.textureIndex >= 0 && prim.textureIndex < (int)textures.size();
    const TextureData* tex = hasTex ? &textures[prim.textureIndex] : nullptr;

    const uint8_t* texPixels = nullptr;
    int texW = 0, texH = 0;
    if (tex) {
        texPixels = tex->pixels.data();
        texW = tex->width;
        texH = tex->height;
    }

    float bcr = prim.baseColor.r, bcg = prim.baseColor.g, bcb = prim.baseColor.b;
    bool alphaBlend = (prim.alphaMode == 2);
    bool alphaTest = (prim.alphaMode == 1);
    float rampScale = prim.mtoonRampScale;
    float rampBias = prim.mtoonRampBias;
    float shadeR = prim.mtoonShadeColor.r;
    float shadeG = prim.mtoonShadeColor.g;
    float shadeB = prim.mtoonShadeColor.b;

    const float dE0dx = y1 - y2, dE0dy = x2 - x1;
    const float dE1dx = y2 - y0, dE1dy = x0 - x2;
    const float dE2dx = y0 - y1, dE2dy = x1 - x0;

    const float dz_dx  = (dE0dx*z0  + dE1dx*z1  + dE2dx*z2)  * invArea;
    const float du_dx  = (dE0dx*u0  + dE1dx*u1  + dE2dx*u2)  * invArea;
    const float dv_dx  = (dE0dx*v0uv + dE1dx*v1uv + dE2dx*v2uv) * invArea;
    const float dnx_dx = (dE0dx*n0x + dE1dx*n1x + dE2dx*n2x) * invArea;
    const float dny_dx = (dE0dx*n0y + dE1dx*n1y + dE2dx*n2y) * invArea;
    const float dnz_dx = (dE0dx*n0z + dE1dx*n1z + dE2dx*n2z) * invArea;
    const float dz_dy  = (dE0dy*z0  + dE1dy*z1  + dE2dy*z2)  * invArea;
    const float du_dy  = (dE0dy*u0  + dE1dy*u1  + dE2dy*u2)  * invArea;
    const float dv_dy  = (dE0dy*v0uv + dE1dy*v1uv + dE2dy*v2uv) * invArea;
    const float dnx_dy = (dE0dy*n0x + dE1dy*n1x + dE2dy*n2x) * invArea;
    const float dny_dy = (dE0dy*n0y + dE1dy*n1y + dE2dy*n2y) * invArea;
    const float dnz_dy = (dE0dy*n0z + dE1dy*n1z + dE2dy*n2z) * invArea;

    float fx0 = ix0 + 0.5f, fy0 = iy0 + 0.5f;
    float e0_row = (x2 - x1) * (fy0 - y1) - (y2 - y1) * (fx0 - x1);
    float e1_row = (x0 - x2) * (fy0 - y2) - (y0 - y2) * (fx0 - x2);
    float e2_row = (x1 - x0) * (fy0 - y0) - (y1 - y0) * (fx0 - x0);
    float z_row  = (e0_row*z0  + e1_row*z1  + e2_row*z2)  * invArea;
    float u_row  = (e0_row*u0  + e1_row*u1  + e2_row*u2)  * invArea;
    float v_row  = (e0_row*v0uv + e1_row*v1uv + e2_row*v2uv) * invArea;
    float nx_row = (e0_row*n0x + e1_row*n1x + e2_row*n2x) * invArea;
    float ny_row = (e0_row*n0y + e1_row*n1y + e2_row*n2y) * invArea;
    float nz_row = (e0_row*n0z + e1_row*n1z + e2_row*n2z) * invArea;

    const float KEY_FILL = KEY_PI + FILL_PI;

    // Scanline rasterization: compute exact covered range per row
    // Edge function e(px) = e_row + (px - ix0) * dEdx, evaluated at pixel center px+0.5
    // Covered when e >= 0. Crossing at offset = -e_row / dEdx from ix0.
    int rowOffset = iy0 * fb.width;
    for (int py = iy0; py <= iy1; py++) {
        // Compute covered pixel range [pxStart, pxEnd] using edge crossings
        float pxEnter = (float)ix0;
        float pxExit = (float)ix1;

        // Edge 0
        if (dE0dx > 0) {
            float cross = (float)ix0 - e0_row / dE0dx;
            if (cross > pxEnter) pxEnter = cross;
        } else if (dE0dx < 0) {
            float cross = (float)ix0 - e0_row / dE0dx;
            if (cross < pxExit) pxExit = cross;
        } else if (e0_row < 0) goto next_row;

        // Edge 1
        if (dE1dx > 0) {
            float cross = (float)ix0 - e1_row / dE1dx;
            if (cross > pxEnter) pxEnter = cross;
        } else if (dE1dx < 0) {
            float cross = (float)ix0 - e1_row / dE1dx;
            if (cross < pxExit) pxExit = cross;
        } else if (e1_row < 0) goto next_row;

        // Edge 2
        if (dE2dx > 0) {
            float cross = (float)ix0 - e2_row / dE2dx;
            if (cross > pxEnter) pxEnter = cross;
        } else if (dE2dx < 0) {
            float cross = (float)ix0 - e2_row / dE2dx;
            if (cross < pxExit) pxExit = cross;
        } else if (e2_row < 0) goto next_row;

        {
            int pxStart = std::max(ix0, (int)ceilf(pxEnter));
            int pxEnd = std::min(ix1, (int)floorf(pxExit));
            if (pxEnd >= pxStart) {
                // Advance interpolated values to pxStart
                float dx = (float)(pxStart - ix0);
                float z = z_row + dx * dz_dx;
                float u = u_row + dx * du_dx;
                float v = v_row + dx * dv_dx;
                float nx = nx_row + dx * dnx_dx;
                float ny = ny_row + dx * dny_dx;
                float nz = nz_row + dx * dnz_dx;
                int pixIdx = rowOffset + pxStart;

                for (int px = pxStart; px <= pxEnd; px++) {
                    if (!depthTest || z < fb.depth[pixIdx]) {

                        float nlen2 = nx*nx + ny*ny + nz*nz;
                        __m128 tmp = _mm_rsqrt_ss(_mm_set_ss(nlen2));
                        float invNlen = _mm_cvtss_f32(tmp);
                        float nnx = nx * invNlen, nny = ny * invNlen, nnz = nz * invNlen;

                        float diffR = bcr, diffG = bcg, diffB = bcb;
                        float texAlpha = 1.0f;
                        if (texPixels) {
                            float tfx = u * texW - 0.5f;
                            float tfy = v * texH - 0.5f;
                            int tx0 = (int)floorf(tfx);
                            int ty0 = (int)floorf(tfy);
                            float wx = tfx - tx0;
                            float wy = tfy - ty0;
                            int tx1i = tx0 + 1, ty1i = ty0 + 1;
                            tx0 = std::max(0, std::min(tx0, texW - 1));
                            tx1i = std::max(0, std::min(tx1i, texW - 1));
                            ty0 = std::max(0, std::min(ty0, texH - 1));
                            ty1i = std::max(0, std::min(ty1i, texH - 1));
                            const uint8_t *t00 = &texPixels[(ty0 * texW + tx0) * 4];
                            const uint8_t *t01 = &texPixels[(ty0 * texW + tx1i) * 4];
                            const uint8_t *t10 = &texPixels[(ty1i * texW + tx0) * 4];
                            const uint8_t *t11 = &texPixels[(ty1i * texW + tx1i) * 4];
                            float w00 = (1 - wx) * (1 - wy);
                            float w01 = wx * (1 - wy);
                            float w10 = (1 - wx) * wy;
                            float w11 = wx * wy;
                            diffR = (sRGBToLinearLUT[t00[0]]*w00 + sRGBToLinearLUT[t01[0]]*w01 + sRGBToLinearLUT[t10[0]]*w10 + sRGBToLinearLUT[t11[0]]*w11) * bcr;
                            diffG = (sRGBToLinearLUT[t00[1]]*w00 + sRGBToLinearLUT[t01[1]]*w01 + sRGBToLinearLUT[t10[1]]*w10 + sRGBToLinearLUT[t11[1]]*w11) * bcg;
                            diffB = (sRGBToLinearLUT[t00[2]]*w00 + sRGBToLinearLUT[t01[2]]*w01 + sRGBToLinearLUT[t10[2]]*w10 + sRGBToLinearLUT[t11[2]]*w11) * bcb;
                            texAlpha = (t00[3]*w00 + t01[3]*w01 + t10[3]*w10 + t11[3]*w11) * (1.0f/255.0f);
                        }

                        bool writePixel = true;
                        if (alphaTest && texAlpha < 0.5f) writePixel = false;
                        else if (alphaBlend && texAlpha <= 0.0f) writePixel = false;

                        if (writePixel) {
                            if (depthWrite) fb.depth[pixIdx] = z;
                            float dotNK = nnx*KX + nny*KY + nnz*KZ;
                            float shK = (dotNK + rampBias) * rampScale;
                            if (shK < 0) shK = 0; else if (shK > 1) shK = 1;
                            float dotNF = nnx*FX + nny*FY + nnz*FZ;
                            float shF = (dotNF + rampBias) * rampScale;
                            if (shF < 0) shF = 0; else if (shF > 1) shF = 1;

                            float litW = KEY_PI * shK + FILL_PI * shF;
                            float hemiW = 0.5f + 0.5f * nny;
                            float hemiC = (HEMI_GROUND + HEMI_DIFF * hemiW) * INV_PI;
                            float sFac = KEY_FILL - litW;
                            float dFac = litW + hemiC;
                            float r  = shadeR * sFac + diffR * dFac;
                            float g  = shadeG * sFac + diffG * dFac;
                            float b_ = shadeB * sFac + diffB * dFac;

                            int ci = pixIdx * 4;
                            if (alphaBlend && texAlpha < 1.0f) {
                                float invA = 1.0f - texAlpha;
                                uint8_t sr = linToSRGB(r);
                                uint8_t sg = linToSRGB(g);
                                uint8_t sb = linToSRGB(b_);
                                fb.color[ci+0] = (uint8_t)(sr * texAlpha + fb.color[ci+0] * invA);
                                fb.color[ci+1] = (uint8_t)(sg * texAlpha + fb.color[ci+1] * invA);
                                fb.color[ci+2] = (uint8_t)(sb * texAlpha + fb.color[ci+2] * invA);
                            } else {
                                fb.color[ci+0] = linToSRGB(r);
                                fb.color[ci+1] = linToSRGB(g);
                                fb.color[ci+2] = linToSRGB(b_);
                            }
                            fb.color[ci+3] = 255;
                        }
                    }
                    z += dz_dx; u += du_dx; v += dv_dx;
                    nx += dnx_dx; ny += dny_dx; nz += dnz_dx;
                    pixIdx++;
                }
            }
        }
    next_row:
        e0_row += dE0dy; e1_row += dE1dy; e2_row += dE2dy;
        z_row += dz_dy; u_row += du_dy; v_row += dv_dy;
        nx_row += dnx_dy; ny_row += dny_dy; nz_row += dnz_dy;
        rowOffset += fb.width;
    }
}
