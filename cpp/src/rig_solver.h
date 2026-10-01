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
    // Begin a fresh (re)calibration: resets accumulators and filter history so
    // the result matches a freshly started application. Must be called before
    // feeding calibrate()/calibratePose() frames for a new neutral pose.
    void startCalibration();
    bool calibrated() const { return calibrated_; }
    bool poseCalibrated() const { return poseCalibrated_; }

    const std::vector<float>& morphWeights() const { return morphWeights_; }
    glm::quat headRotation() const { return headRot_; }
    const BodyPose& bodyPose() const { return bodyPose_; }
    const std::unordered_map<int, glm::quat>& handOverrides() const { return handOverrides_; }

    // Eye gaze: local rotation delta for the leftEye/rightEye bones
    // (bone-type lookAt models). Degrees are exposed for the HUD.
    glm::quat eyeRotation() const;
    float eyeYawDeg() const { return eyeYawDeg_; }
    float eyePitchDeg() const { return eyePitchDeg_; }
    bool lookAtBoneType() const { return lookAtBoneType_; }

    // Runtime-tunable knobs (adjusted from the settings GUI)
    void setHeadGains(float yaw, float pitch, float roll);
    void setHeadClamps(float maxYaw, float maxPitch, float maxRoll);
    void setGazeScale(float s) { gazeScale_ = s; }
    // s in [0,1]: 0 responsive, 1 very smooth. Retunes the head filters
    // (OneEuro cutoff/beta + critically-damped glide time).
    void setHeadSmoothing(float s);

private:
    struct MorphBind { int meshIdx; int targetIdx; float weight; };
    // Key: lowercased group name (preset name, or group name if preset is empty/unknown)
    std::map<std::string, std::vector<MorphBind>> groupToMorphs_;
    std::vector<int> meshMorphBase_;

    std::array<OneEuroFilter, 52> bsFilters_;
    OneEuroFilter yawFilter_{1.6f, 0.05f}, pitchFilter_{1.6f, 0.05f}, rollFilter_{1.6f, 0.05f};

    // Spring smoothers for gentle ease-in/ease-out on top of OneEuro filtering
    SmoothFloat smoothYaw_{0.09f}, smoothPitch_{0.09f}, smoothRoll_{0.09f};

    // Pose filters: rotation vectors (3 per bone) for arms + spine
    OneEuroFilter poseRotFilters_[8][3];
    SmoothFloat smoothRot_[8][3];
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
    bool calibratingNow_ = false;   // active accumulation window

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

    // Head rotation gain & clamp (keep thin-shell avatar front-facing).
    // Runtime-tunable via setHeadGains()/setHeadClamps().
    float headGainYaw_ = 0.55f;
    float headGainPitch_ = 0.55f;
    float headGainRoll_ = 0.55f;
    float maxYaw_ = 35.0f;
    float maxPitch_ = 20.0f;
    float maxRoll_ = 15.0f;
    // Head smoothing 0..1: 0 = responsive (jerky tracking noise passes),
    // 1 = very smooth (more lag). Retunes the filters below.
    float headSmoothing_ = 0.5f;

    // Eye gaze (bone-type lookAt): computed from the eyeLook* blendshapes
    // scaled by the model's lookAt range maps (matching Python solver).
    OneEuroFilter gazeYawFilter_{3.0f, 0.0f};
    OneEuroFilter gazePitchFilter_{3.0f, 0.0f};
    float gazeHScale_ = 10.0f;   // degrees at eyeLook weight 1.0
    float gazeVUpScale_ = 10.0f;
    float gazeVDownScale_ = 10.0f;
    float gazeScale_ = 1.0f;     // user multiplier (settings GUI)
    bool lookAtBoneType_ = true;
    float eyeYawDeg_ = 0.0f;
    float eyePitchDeg_ = 0.0f;
    int eyeNodeL_ = -1, eyeNodeR_ = -1;

    // Pose: rest directions for arms (matching Python pipeline)
    // Python: REST_L=[1,0,0], REST_R=[-1,0,0], AXIS_FLIP=[1,1,-1]
    static constexpr glm::vec3 REST_L = {1.0f, 0.0f, 0.0f};
    static constexpr glm::vec3 REST_R = {-1.0f, 0.0f, 0.0f};
    static constexpr glm::vec3 AXIS_FLIP = {1.0f, 1.0f, -1.0f};

    glm::quat dirToRotation(const glm::vec3& rest, const glm::vec3& target);
    glm::quat filterRot(int idx, const glm::quat& q, float dt);
    static float jointAngle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);

    // --- Expression mapping (ARKit 52 → VRM standard expressions) ---
    // Port of Python solver.py map_arkit_to_vrm(): drives standard VRM preset
    // groups (aa/ih/ou/ee/oh, happy/angry/sad/surprised, blink, look*) so
    // avatars without ARKit "Perfect Sync" groups still animate.
    static std::string normalizeGroupName(std::string key);
    void applyGroupWeight(const std::string& name, float w);
    static void mapArkitToVrm(const float bs[52], float out[16]);
    static constexpr const char* kVrmExprNames[16] = {
        "aa", "ih", "ou", "ee", "oh",
        "happy", "angry", "sad", "surprised",
        "blink", "blinkleft", "blinkright",
        "lookup", "lookdown", "lookleft", "lookright",
    };
};
