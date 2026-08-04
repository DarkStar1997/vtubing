#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

// Lightweight replacement for cv::Mat. 3-channel BGR by default (matches
// the existing pipeline's assumptions). Owns its pixel storage (deep copy).
struct Image {
    int width = 0, height = 0, channels = 3;
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h, int ch = 3)
        : width(w), height(h), channels(ch), data((size_t)w * h * ch, 0) {}

    uint8_t* ptr(int y, int x) { return &data[((size_t)y * width + x) * channels]; }
    const uint8_t* ptr(int y, int x) const { return &data[((size_t)y * width + x) * channels]; }
    bool empty() const { return data.empty(); }
    int rows() const { return height; }
    int cols() const { return width; }
    void create(int w, int h, int ch = 3) {
        width = w; height = h; channels = ch;
        data.assign((size_t)w * h * ch, 0);
    }
};

// --- Affine transform helpers (drop-in replacements for the OpenCV calls) ---

// Matches cv::getRotationMatrix2D: builds a 2x3 matrix (row-major [a,b,tx,c,d,ty])
// that maps src coords -> dst coords. angleDeg in degrees, positive = CCW.
inline void getRotationMatrix2D(float cx, float cy, float angleDeg, float scale, float M[6]) {
    float angle = angleDeg * 3.14159265358979f / 180.0f;
    float alpha = scale * std::cos(angle);
    float beta  = scale * std::sin(angle);
    M[0] = alpha;
    M[1] = beta;
    M[2] = (1.0f - alpha) * cx - beta * cy;
    M[3] = -beta;
    M[4] = alpha;
    M[5] = beta * cx + (1.0f - alpha) * cy;
}

// Matches cv::invertAffineTransform: inverts a 2x3 affine matrix.
inline void invertAffine(const float M[6], float invM[6]) {
    float a = M[0], b = M[1], tx = M[2];
    float c = M[3], d = M[4], ty = M[5];
    float det = a * d - b * c;
    float invDet = 1.0f / det;
    invM[0] =  d * invDet;
    invM[1] = -b * invDet;
    invM[3] = -c * invDet;
    invM[4] =  a * invDet;
    invM[2] = -(invM[0] * tx + invM[1] * ty);
    invM[5] = -(invM[3] * tx + invM[4] * ty);
}

// Matches cv::warpAffine with INTER_LINEAR + BORDER_CONSTANT(0).
// M maps src->dst. Internally inverts M to map dst->src, bilinear samples.
inline Image warpAffine(const Image& src, const float M[6], int dstW, int dstH) {
    float invM[6];
    invertAffine(M, invM);
    int ch = src.channels;
    Image dst(dstW, dstH, ch);
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            float sx = invM[0] * x + invM[1] * y + invM[2];
            float sy = invM[3] * x + invM[4] * y + invM[5];
            uint8_t* out = dst.ptr(y, x);
            if (sx < 0.0f || sy < 0.0f || sx > src.width - 1.0f || sy > src.height - 1.0f) {
                for (int c = 0; c < ch; c++) out[c] = 0;
            } else {
                int x0 = (int)sx, y0 = (int)sy;
                int x1 = std::min(x0 + 1, src.width - 1);
                int y1 = std::min(y0 + 1, src.height - 1);
                float fx = sx - x0, fy = sy - y0;
                const uint8_t *p00 = src.ptr(y0, x0), *p01 = src.ptr(y0, x1);
                const uint8_t *p10 = src.ptr(y1, x0), *p11 = src.ptr(y1, x1);
                for (int c = 0; c < ch; c++) {
                    float v = (1 - fx) * (1 - fy) * p00[c] + fx * (1 - fy) * p01[c]
                            + (1 - fx) * fy * p10[c] + fx * fy * p11[c];
                    out[c] = (uint8_t)(v + 0.5f);
                }
            }
        }
    }
    return dst;
}

// --- Drawing helpers ---

inline void drawRect(Image& img, int x1, int y1, int x2, int y2, const uint8_t color[3], int thickness = 1) {
    int ch = img.channels;
    int w = img.width, h = img.height;
    auto setPx = [&](int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        uint8_t* p = img.ptr(y, x);
        for (int c = 0; c < ch && c < 3; c++) p[c] = color[c];
    };
    int xa = std::min(x1, x2), xb = std::max(x1, x2);
    int ya = std::min(y1, y2), yb = std::max(y1, y2);
    for (int t = 0; t < thickness; t++) {
        int tt = ya + t, bb = yb - t, ll = xa + t, rr = xb - t;
        for (int x = xa; x <= xb; x++) { setPx(x, tt); setPx(x, bb); }
        for (int y = ya; y <= yb; y++) { setPx(ll, y); setPx(rr, y); }
    }
}

inline void drawCircleFilled(Image& img, int cx, int cy, int radius, const uint8_t color[3]) {
    int ch = img.channels;
    int r2 = radius * radius;
    for (int y = cy - radius; y <= cy + radius; y++) {
        for (int x = cx - radius; x <= cx + radius; x++) {
            int dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy <= r2 && x >= 0 && x < img.width && y >= 0 && y < img.height) {
                uint8_t* p = img.ptr(y, x);
                for (int c = 0; c < ch && c < 3; c++) p[c] = color[c];
            }
        }
    }
}

inline Image cropImage(const Image& src, int x, int y, int w, int h) {
    Image out(w, h, src.channels);
    for (int dy = 0; dy < h; dy++) {
        int sy = y + dy;
        if (sy < 0 || sy >= src.height) continue;
        for (int dx = 0; dx < w; dx++) {
            int sx = x + dx;
            if (sx < 0 || sx >= src.width) continue;
            const uint8_t* sp = src.ptr(sy, sx);
            uint8_t* dp = out.ptr(dy, dx);
            for (int c = 0; c < src.channels; c++) dp[c] = sp[c];
        }
    }
    return out;
}

inline void drawLine(Image& img, int x0, int y0, int x1, int y1, const uint8_t color[3], int thickness = 1) {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int w = img.width, h = img.height;
    auto setPx = [&](int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        uint8_t* p = img.ptr(y, x);
        for (int c = 0; c < img.channels && c < 3; c++) p[c] = color[c];
    };
    for (;;) {
        for (int ty = 0; ty < thickness; ty++) {
            for (int tx = 0; tx < thickness; tx++) {
                setPx(x0 + tx - thickness / 2, y0 + ty - thickness / 2);
            }
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}

// Bilinear resize. Matches cv::resize(INTER_LINEAR) closely enough for the PiP overlay.
inline Image resizeBilinear(const Image& src, int dstW, int dstH) {
    int ch = src.channels;
    Image dst(dstW, dstH, ch);
    float sxRatio = (float)src.width / dstW;
    float syRatio = (float)src.height / dstH;
    for (int y = 0; y < dstH; y++) {
        float sy = (y + 0.5f) * syRatio - 0.5f;
        int y0 = std::max(0, std::min((int)sy, src.height - 1));
        int y1 = std::min(y0 + 1, src.height - 1);
        float fy = sy - y0; if (fy < 0) fy = 0;
        for (int x = 0; x < dstW; x++) {
            float sx = (x + 0.5f) * sxRatio - 0.5f;
            int x0 = std::max(0, std::min((int)sx, src.width - 1));
            int x1 = std::min(x0 + 1, src.width - 1);
            float fx = sx - x0; if (fx < 0) fx = 0;
            const uint8_t *p00 = src.ptr(y0, x0), *p01 = src.ptr(y0, x1);
            const uint8_t *p10 = src.ptr(y1, x0), *p11 = src.ptr(y1, x1);
            uint8_t* out = dst.ptr(y, x);
            for (int c = 0; c < ch; c++) {
                float v = (1 - fx) * (1 - fy) * p00[c] + fx * (1 - fy) * p01[c]
                        + (1 - fx) * fy * p10[c] + fx * fy * p11[c];
                out[c] = (uint8_t)(v + 0.5f);
            }
        }
    }
    return dst;
}
