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

    void update(const FaceResult& face, float dt);
    void calibrate(const FaceResult& face);
    bool calibrated() const { return calibrated_; }

    const std::vector<float>& morphWeights() const { return morphWeights_; }
    glm::quat headRotation() const { return headRot_; }

private:
    struct MorphBind { int meshIdx; int targetIdx; float weight; };
    // Key: lowercased group name (preset name, or group name if preset is empty/unknown)
    std::map<std::string, std::vector<MorphBind>> groupToMorphs_;
    std::vector<int> meshMorphBase_;

    std::array<OneEuroFilter, 52> bsFilters_;
    OneEuroFilter yawFilter_{1.5f, 0.05f}, pitchFilter_{1.5f, 0.05f}, rollFilter_{1.5f, 0.05f};

    std::array<float, 52> neutralBs_{};
    float neutralYaw_ = 0, neutralPitch_ = 0, neutralRoll_ = 0;
    int calibFrames_ = 0;
    static constexpr int CALIB_COUNT = 30;
    bool calibrated_ = false;

    std::vector<float> morphWeights_;
    glm::quat headRot_ = glm::quat(1, 0, 0, 0);

    // 52 ARKit blendshape names (lowercased) matching VRM group names
    static constexpr const char* kArkitNames[52] = {
        "neutral",
        "browinnerup", "browdownleft", "browdownright",
        "browouterupleft", "browouterupright",
        "cheekpuff", "cheeksquintleft", "cheeksquintright",
        "eyeblinkleft", "eyeblinkright",
        "eyelookdownleft", "eyelookdownright",
        "eyelookinleft", "eyelookinright",
        "eyelookoutleft", "eyelookoutright",
        "eyelookupleft", "eyelookupright",
        "eyesquintleft", "eyesquintright",
        "eyewideleft", "eyewideright",
        "jawforward", "jawleft", "jawopen", "jawright",
        "mouthclose",
        "mouthdimpleleft", "mouthdimpleright",
        "mouthfrownleft", "mouthfrownright",
        "mouthfunnel", "mouthleft",
        "mouthlowerdownleft", "mouthlowerdownright",
        "mouthpressleft", "mouthpressright",
        "mouthpucker", "mouthright",
        "mouthrolllower", "mouthrollupper",
        "mouthshruglower", "mouthshrugupper",
        "mouthsmileleft", "mouthsmileright",
        "mouthstretchleft", "mouthstretchright",
        "mouthupperupleft", "mouthupperupright",
        "nosesneerleft", "nosesneerright",
    };

    // Head rotation gain & clamp (keep thin-shell avatar front-facing)
    static constexpr float HEAD_GAIN = 0.65f;
    static constexpr float MAX_YAW = 35.0f;
    static constexpr float MAX_PITCH = 20.0f;
    static constexpr float MAX_ROLL = 15.0f;
};
