#pragma once
#include "one_euro.h"
#include "vrm_loader.h"
#include "face_tracker.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <map>

class RigSolver {
public:
    RigSolver(const VRMModel& model);

    // Update rig from face tracking result. dt = time since last call.
    void update(const FaceResult& face, float dt);

    // Calibration: collect neutral baseline
    void calibrate(const FaceResult& face);
    bool calibrated() const { return calibrated_; }

    // Output: morph weights (for processVerticesParallel) and head rotation
    const std::vector<float>& morphWeights() const { return morphWeights_; }
    glm::quat headRotation() const { return headRot_; }

private:
    // VRM blendshape group mapping: preset name → list of (meshIdx, morphTargetIdx, weight)
    struct MorphBind { int meshIdx; int targetIdx; float weight; };
    std::map<std::string, std::vector<MorphBind>> presetToMorphs_;
    std::vector<int> meshMorphBase_;  // base offset per mesh in morphWeights_

    // One-Euro filters for blendshapes
    std::array<OneEuroFilter, 52> bsFilters_;
    OneEuroFilter yawFilter_{1.0f, 0.0f}, pitchFilter_{1.0f, 0.0f}, rollFilter_{1.0f, 0.0f};

    // Calibration baselines
    std::array<float, 52> neutralBs_{};
    float neutralYaw_ = 0, neutralPitch_ = 0, neutralRoll_ = 0;
    int calibFrames_ = 0;
    static constexpr int CALIB_COUNT = 30;
    bool calibrated_ = false;

    // Output
    std::vector<float> morphWeights_;
    glm::quat headRot_ = glm::quat(1, 0, 0, 0);

    // VRM preset name list (18 presets from Python map_arkit_to_vrm)
    // Maps ARKit blendshape indices to VRM preset expressions
    void applyArkitToVrm(const float* arkit, std::map<std::string, float>& vrmPresets);
};
