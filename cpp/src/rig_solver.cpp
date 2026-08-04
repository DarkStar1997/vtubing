#include "rig_solver.h"
#include <algorithm>
#include <cmath>

RigSolver::RigSolver(const VRMModel& model) {
    int totalMorphs = 0;
    std::vector<int> meshMorphBase;
    for (size_t mi = 0; mi < model.meshes.size(); mi++) {
        meshMorphBase.push_back(totalMorphs);
        if (!model.meshes[mi].primitives.empty())
            totalMorphs += model.meshes[mi].primitives[0].morphCount;
    }
    morphWeights_.resize(totalMorphs, 0.0f);
    meshMorphBase_ = meshMorphBase;

    for (auto& group : model.blendShapeGroups) {
        std::string key = group.presetName;
        for (auto& c : key) c = std::tolower(c);
        if (key.empty() || key == "unknown") {
            key = group.name;
            for (auto& c : key) c = std::tolower(c);
        }
        std::vector<MorphBind> binds;
        for (auto& b : group.binds) {
            if (b.mesh < (int)meshMorphBase.size()) {
                binds.push_back({b.mesh, b.index, b.weight / 100.0f});
            }
        }
        groupToMorphs_[key] = binds;
    }
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
        float decay = std::exp(-dt / 0.2f);
        for (auto& w : morphWeights_) w *= decay;
        return;
    }

    float filteredBs[52];
    for (int i = 0; i < 52; i++) {
        float val = face.blendshapes[i] - neutralBs_[i];
        val = std::max(0.0f, std::min(1.0f, val));
        filteredBs[i] = bsFilters_[i].filter(val, dt);
    }

    std::fill(morphWeights_.begin(), morphWeights_.end(), 0.0f);

    for (int i = 1; i < 52; i++) {
        float w = filteredBs[i];
        if (w <= 0.001f) continue;
        auto it = groupToMorphs_.find(kArkitNames[i]);
        if (it == groupToMorphs_.end()) continue;
        for (auto& bind : it->second) {
            if (bind.meshIdx < (int)meshMorphBase_.size()) {
                int idx = meshMorphBase_[bind.meshIdx] + bind.targetIdx;
                if (idx >= 0 && idx < (int)morphWeights_.size())
                    morphWeights_[idx] = std::max(morphWeights_[idx], w * bind.weight);
            }
        }
    }

    float yaw = face.yaw - neutralYaw_;
    float pitch = face.pitch - neutralPitch_;
    float roll = face.roll - neutralRoll_;
    yaw = std::clamp(yaw * HEAD_GAIN, -MAX_YAW, MAX_YAW);
    pitch = std::clamp(pitch * HEAD_GAIN, -MAX_PITCH, MAX_PITCH);
    roll = std::clamp(roll * HEAD_GAIN, -MAX_ROLL, MAX_ROLL);
    yaw = yawFilter_.filter(yaw, dt);
    pitch = pitchFilter_.filter(pitch, dt);
    roll = rollFilter_.filter(roll, dt);

    glm::quat qYaw = glm::angleAxis(glm::radians(yaw), glm::vec3(0, 1, 0));
    glm::quat qPitch = glm::angleAxis(glm::radians(pitch), glm::vec3(1, 0, 0));
    glm::quat qRoll = glm::angleAxis(glm::radians(roll), glm::vec3(0, 0, 1));
    headRot_ = qYaw * qPitch * qRoll;
}
