#pragma once
#include <cmath>

// One-Euro adaptive low-pass filter (ported from Python one_euro.py)
// Usage: filter each signal independently with its own OneEuroFilter instance.
class OneEuroFilter {
public:
    OneEuroFilter(float minCutoff = 1.0f, float beta = 0.0f, float dCutoff = 1.0f)
        : minCutoff_(minCutoff), beta_(beta), dCutoff_(dCutoff) {}

    float filter(float x, float dt) {
        if (dt <= 0.0f) dt = 1e-5f;
        float prevX = x_;
        bool first = !initialized_;
        x_ = x;

        float dx = first ? 0.0f : (x - prevX) / dt;
        float edx = alpha(dt, dCutoff_) * dx + (1.0f - alpha(dt, dCutoff_)) * dx_;
        dx_ = edx;

        float cutoff = minCutoff_ + beta_ * std::fabs(edx);
        float alphaVal = alpha(dt, cutoff);
        float fx = first ? x : (alphaVal * x + (1.0f - alphaVal) * prevFiltX_);
        prevFiltX_ = fx;
        initialized_ = true;
        return fx;
    }

    void reset() { initialized_ = false; }

private:
    static float alpha(float dt, float cutoff) {
        float tau = 1.0f / (2.0f * 3.14159265f * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }

    float minCutoff_, beta_, dCutoff_;
    float x_ = 0.0f, prevFiltX_ = 0.0f, dx_ = 0.0f;
    bool initialized_ = false;
};
