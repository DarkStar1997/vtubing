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
    float lastFiltered() const { return prevFiltX_; }

private:
    static float alpha(float dt, float cutoff) {
        float tau = 1.0f / (2.0f * 3.14159265f * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }

    float minCutoff_, beta_, dCutoff_;
    float x_ = 0.0f, prevFiltX_ = 0.0f, dx_ = 0.0f;
    bool initialized_ = false;
};

// Critically-damped spring smoother (Unity SmoothDamp algorithm).
// Produces natural ease-in/ease-out motion with acceleration & deceleration.
// smoothTime ≈ time to reach ~63% of the way to the target.
class SmoothFloat {
public:
    SmoothFloat(float smoothTime = 0.1f)
        : smoothTime_(smoothTime), value_(0), velocity_(0), initialized_(false) {}

    float update(float target, float dt) {
        if (!initialized_) {
            value_ = target;
            velocity_ = 0;
            initialized_ = true;
            return value_;
        }
        if (dt <= 0.0f) return value_;
        float omega = 2.0f / smoothTime_;
        float x = omega * dt;
        float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
        float change = value_ - target;
        float temp = (velocity_ + omega * change) * dt;
        velocity_ = (velocity_ - omega * temp) * exp;
        value_ = target + (change + temp) * exp;
        return value_;
    }

    float get() const { return value_; }
    void reset() { initialized_ = false; velocity_ = 0; }

private:
    float smoothTime_;
    float value_, velocity_;
    bool initialized_;
};
