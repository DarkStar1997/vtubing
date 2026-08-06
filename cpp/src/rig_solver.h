#pragma once
#include "one_euro.h"
#include "vrm_loader.h"
#include "face_tracker.h"
#include "pose_tracker.h"
#include "hand_tracker.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <map>
#include <unordered_map>

struct BodyPose {
    bool valid = false;
    glm::quat leftUpperArm = glm::quat(1, 0, 0, 0);
    glm::quat rightUpperArm = glm::quat(1, 0, 0, 0);
    glm::quat leftLowerArm = glm::quat(1, 0, 0, 0);
    glm::quat rightLowerArm = glm::quat(1, 0, 0, 0);
    float lean = 0.0f;     // X-axis: forward lean
    float twist = 0.0f;    // Y-axis: torso twist
    float lateral = 0.0f;  // Z-axis: lateral bend
    float standing = 0.0f;    // 0=sitting, 1=standing
    float bodyExtent = 0.0f;  // torso-lengths of body visible
};

class RigSolver {
public:
    RigSolver(const VRMModel& model);

    void update(const FaceResult& face, float dt);
    void updatePose(const PoseResult& pose, float dt);
    void updateHands(const HandResult& left, const HandResult& right, float dt);
    void calibrate(const FaceResult& face);
    void calibratePose();
    bool calibrated() const { return calibrated_; }
    bool poseCalibrated() const { return poseCalibrated_; }

    const std::vector<float>& morphWeights() const { return morphWeights_; }
    glm::quat headRotation() const { return headRot_; }
    const BodyPose& bodyPose() const { return bodyPose_; }
    const std::unordered_map<int, glm::quat>& handOverrides() const { return handOverrides_; }

private:
    struct MorphBind { int meshIdx; int targetIdx; float weight; };
    // Key: lowercased group name (preset name, or group name if preset is empty/unknown)
    std::map<std::string, std::vector<MorphBind>> groupToMorphs_;
    std::vector<int> meshMorphBase_;

    std::array<OneEuroFilter, 52> bsFilters_;
    OneEuroFilter yawFilter_{4.0f, 0.15f}, pitchFilter_{4.0f, 0.15f}, rollFilter_{4.0f, 0.15f};

    // Pose filters: rotation vectors (3 per bone) for arms + spine
    OneEuroFilter poseRotFilters_[8][3] = {{
        {1.0f, 0.05f}, {1.0f, 0.05f}, {1.0f, 0.05f}
    }};
    OneEuroFilter torsoFilter_{1.0f, 0.0f};
    OneEuroFilter spineYFilter_{1.0f, 0.0f};
    OneEuroFilter spineZFilter_{1.0f, 0.0f};
    float torsoNeutral_ = 0, spineYNeutral_ = 0, spineZNeutral_ = 0;
    bool poseCalibrated_ = false;

    // Standing detection: hysteresis + smoothing (matching Python pipeline)
    bool standState_ = false;
    OneEuroFilter standingFilter_{0.4f, 0.0f};
    OneEuroFilter bodyExtentFilter_{0.8f, 0.0f};

    std::array<float, 52> neutralBs_{};
    float neutralYaw_ = 0, neutralPitch_ = 0, neutralRoll_ = 0;
    int calibFrames_ = 0;
    static constexpr int CALIB_COUNT = 30;
    bool calibrated_ = false;

    std::vector<float> morphWeights_;
    glm::quat headRot_ = glm::quat(1, 0, 0, 0);
    BodyPose bodyPose_;

    // Hand finger bone node indices (resolved from VRM boneNodes in constructor)
    // [side][finger][joint]: side 0=left 1=right, finger 0-4, joint 0-2
    int fingerBoneNodes_[2][5][3]{};   // -1 if bone not found
    int handBoneNodes_[2]{};           // leftHand/rightHand node
    std::unordered_map<int, glm::quat> handOverrides_;
    std::unordered_map<int, glm::quat> lastGoodHandOverrides_;
    float handTwistNeutral_[2] = {0, 0};
    bool handsCalibrated_ = false;

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

    // Pose: rest directions for arms (matching Python pipeline)
    // Python: REST_L=[1,0,0], REST_R=[-1,0,0], AXIS_FLIP=[1,1,-1]
    static constexpr glm::vec3 REST_L = {1.0f, 0.0f, 0.0f};
    static constexpr glm::vec3 REST_R = {-1.0f, 0.0f, 0.0f};
    static constexpr glm::vec3 AXIS_FLIP = {1.0f, 1.0f, -1.0f};

    glm::quat dirToRotation(const glm::vec3& rest, const glm::vec3& target);
    glm::quat filterRot(int idx, const glm::quat& q, float dt);
    static float jointAngle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
};
