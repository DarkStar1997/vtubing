#include "rig_solver.h"
#include <algorithm>
#include <cmath>

RigSolver::RigSolver(const VRMModel& model) {
    // Initialize morph weights array
    int totalMorphs = 0;
    std::vector<int> meshMorphBase;  // base offset for each mesh
    for (size_t mi = 0; mi < model.meshes.size(); mi++) {
        meshMorphBase.push_back(totalMorphs);
        if (!model.meshes[mi].primitives.empty())
            totalMorphs += model.meshes[mi].primitives[0].morphCount;
    }
    morphWeights_.resize(totalMorphs, 0.0f);
    meshMorphBase_ = meshMorphBase;

    // Parse VRM 0.x blendShapeGroups
    for (auto& group : model.blendShapeGroups) {
        std::string preset = group.presetName;
        for (auto& c : preset) c = std::tolower(c);
        if (preset.empty()) {
            preset = group.name;
            for (auto& c : preset) c = std::tolower(c);
        }
        std::vector<MorphBind> binds;
        for (auto& b : group.binds) {
            if (b.mesh < (int)meshMorphBase.size()) {
                binds.push_back({b.mesh, b.index, b.weight / 100.0f});
            }
        }
        presetToMorphs_[preset] = binds;
    }
}

void RigSolver::applyArkitToVrm(const float* bs, std::map<std::string, float>& out) {
    // Port of solver.py:54-109 map_arkit_to_vrm()
    // bs indices match the 52 ARKit blendshape order
    auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };

    // ARKit indices (standard MediaPipe 52 blendshape order):
    // 0=_neutral, 1=browInnerUp, 2=browDownLeft, 3=browDownRight,
    // 4=browOuterUpLeft, 5=browOuterUpRight, 6=cheekPuff, 7=cheekSquintLeft,
    // 8=cheekSquintRight, 9=eyeBlinkLeft, 10=eyeBlinkRight,
    // 11=eyeLookDownLeft, 12=eyeLookDownRight, 13=eyeLookInLeft, 14=eyeLookInRight,
    // 15=eyeLookOutLeft, 16=eyeLookOutRight, 17=eyeLookUpLeft, 18=eyeLookUpRight,
    // 19=eyeSquintLeft, 20=eyeSquintRight, 21=eyeWideLeft, 22=eyeWideRight,
    // 23=jawForward, 24=jawLeft, 25=jawOpen, 26=jawRight,
    // 27=mouthClose, 28=mouthDimpleLeft, 29=mouthDimpleRight,
    // 30=mouthFrownLeft, 31=mouthFrownRight, 32=mouthFunnel,
    // 33=mouthLeft, 34=mouthLowerDownLeft, 35=mouthLowerDownRight,
    // 36=mouthPressLeft, 37=mouthPressRight, 38=mouthPucker,
    // 39=mouthRight, 40=mouthRollLower, 41=mouthRollUpper,
    // 42=mouthShrugLower, 43=mouthShrugUpper,
    // 44=mouthSmileLeft, 45=mouthSmileRight,
    // 46=mouthStretchLeft, 47=mouthStretchRight,
    // 48=mouthUpperUpLeft, 49=mouthUpperUpRight,
    // 50=noseSneerLeft, 51=noseSneerRight

    float blink = std::max(bs[9], bs[10]);
    out["blink"] = clamp01(blink);
    out["blinkleft"] = clamp01(bs[9]);
    out["blinkright"] = clamp01(bs[10]);

    // Gaze expressions
    out["lookup"] = clamp01(std::max(bs[17], bs[18]));
    out["lookdown"] = clamp01(std::max(bs[11], bs[12]));
    out["lookleft"] = clamp01(std::max(bs[15], bs[14]));  // eyeLookOutLeft, eyeLookInRight
    out["lookright"] = clamp01(std::max(bs[13], bs[16])); // eyeLookInLeft, eyeLookOutRight

    // Visemes
    float jaw = bs[25];
    float stretch = std::max(bs[46], bs[47]);
    float funnel = bs[32];
    float pucker = bs[38];
    float smile = std::min(bs[44], bs[45]);
    out["a"] = clamp01(jaw);
    out["i"] = clamp01(std::min(jaw * 0.3f, 0.3f) + stretch * 0.7f);
    out["u"] = clamp01(std::max(funnel, pucker));
    out["e"] = clamp01(stretch * 0.5f + smile * 0.3f);
    out["o"] = clamp01(std::max(funnel, pucker) * 0.5f + jaw * 0.3f);

    // Emotions
    out["happy"] = clamp01(std::min(bs[44], bs[45]));
    out["angry"] = clamp01(0.7f * std::max(bs[2], bs[3]) + 0.3f * std::max(bs[50], bs[51]));
    out["sad"] = clamp01(0.7f * std::max(bs[30], bs[31]) + 0.3f * bs[1]);
    out["surprised"] = clamp01(0.3f * std::max(bs[4], bs[5]) + 0.3f * std::max(bs[21], bs[22]) + 0.4f * jaw);
    out["relaxed"] = 0;
    out["neutral"] = 0;
}

void RigSolver::calibrate(const FaceResult& face) {
    if (calibFrames_ < CALIB_COUNT) {
        for (int i = 0; i < 52; i++)
            neutralBs_[i] += face.blendshapes[i];
        neutralYaw_ += face.yaw;
        neutralPitch_ += face.pitch;
        neutralRoll_ += face.roll;
        calibFrames_++;
        if (calibFrames_ == CALIB_COUNT) {
            float inv = 1.0f / CALIB_COUNT;
            for (int i = 0; i < 52; i++)
                neutralBs_[i] *= inv;
            neutralYaw_ *= inv;
            neutralPitch_ *= inv;
            neutralRoll_ *= inv;
            calibrated_ = true;
        }
    }
}

void RigSolver::update(const FaceResult& face, float dt) {
    if (!calibrated_ || !face.detected) {
        // Gentle dt-based decay toward neutral (0.2s time constant)
        float decay = std::exp(-dt / 0.2f);
        for (auto& w : morphWeights_) w *= decay;
        return;
    }

    // Filter blendshapes and subtract neutral baseline
    float filteredBs[52];
    for (int i = 0; i < 52; i++) {
        float val = face.blendshapes[i] - neutralBs_[i];
        val = std::max(0.0f, std::min(1.0f, val));
        filteredBs[i] = bsFilters_[i].filter(val, dt);
    }

    // Map ARKit → VRM presets
    std::map<std::string, float> presets;
    applyArkitToVrm(filteredBs, presets);

    // Write morph weights from VRM preset mapping
    std::fill(morphWeights_.begin(), morphWeights_.end(), 0.0f);

    // We need mesh morph base offsets — store them as a member
    // For now compute from VRMModel. Actually we stored them in constructor.
    // Let me add meshMorphBase_ as a member.

    for (auto& [preset, weight] : presets) {
        auto it = presetToMorphs_.find(preset);
        if (it == presetToMorphs_.end()) continue;
        for (auto& bind : it->second) {
            if (bind.meshIdx < (int)meshMorphBase_.size()) {
                int idx = meshMorphBase_[bind.meshIdx] + bind.targetIdx;
                if (idx >= 0 && idx < (int)morphWeights_.size())
                    morphWeights_[idx] = std::max(morphWeights_[idx], weight * bind.weight);
            }
        }
    }

    // Simplified: directly map blendshape groups
    // We need the mesh morph offset table. Let me store it in constructor.
    // For now, set head rotation
    float yaw = yawFilter_.filter(face.yaw - neutralYaw_, dt);
    float pitch = pitchFilter_.filter(face.pitch - neutralPitch_, dt);
    float roll = rollFilter_.filter(face.roll - neutralRoll_, dt);

    // Convert to quaternion (YXZ order like Python)
    glm::quat qYaw = glm::angleAxis(yaw, glm::vec3(0, 1, 0));
    glm::quat qPitch = glm::angleAxis(pitch, glm::vec3(1, 0, 0));
    glm::quat qRoll = glm::angleAxis(roll, glm::vec3(0, 0, 1));
    headRot_ = qYaw * qPitch * qRoll;
}
