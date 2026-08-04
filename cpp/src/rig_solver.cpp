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

    // Resolve finger bone node indices from VRM humanoid bones
    // Finger names: {side}{Finger}{Joint}
    // side: left/right, Finger: Thumb/Index/Middle/Ring/Little
    // Joint: Proximal/Intermediate/Distal (thumb: Metacarpal/Proximal/Distal)
    static const char* kSidePrefix[2] = {"left", "right"};
    static const char* kFingerName[5] = {"Thumb", "Index", "Middle", "Ring", "Little"};
    // Joint suffixes per finger index
    auto jointNames = [&](int finger) -> const char* (&)[3] {
        static const char* thumbJoints[3] = {"Metacarpal", "Proximal", "Distal"};
        static const char* otherJoints[3] = {"Proximal", "Intermediate", "Distal"};
        return (finger == 0) ? thumbJoints : otherJoints;
    };

    for (int side = 0; side < 2; side++) {
        handBoneNodes_[side] = -1;
        auto it = model.boneNodes.find(std::string(kSidePrefix[side]) + "Hand");
        if (it != model.boneNodes.end()) handBoneNodes_[side] = it->second;

        for (int finger = 0; finger < 5; finger++) {
            const auto* jn = jointNames(finger);
            for (int joint = 0; joint < 3; joint++) {
                std::string name = std::string(kSidePrefix[side]) + kFingerName[finger] + jn[joint];
                auto bit = model.boneNodes.find(name);
                fingerBoneNodes_[side][finger][joint] =
                    (bit != model.boneNodes.end()) ? bit->second : -1;
            }
        }
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

glm::quat RigSolver::dirToRotation(const glm::vec3& rest, const glm::vec3& target) {
    glm::vec3 r = glm::normalize(rest);
    glm::vec3 t = glm::normalize(target);
    float dot = glm::clamp(glm::dot(r, t), -1.0f, 1.0f);
    float angle = std::acos(dot);
    // 3-degree deadzone
    if (angle < 0.05f) return glm::quat(1, 0, 0, 0);
    glm::vec3 axis = glm::cross(r, t);
    float n = glm::length(axis);
    if (n < 1e-6f) {
        if (dot < 0) return glm::angleAxis(glm::pi<float>(), glm::vec3(0, 0, 1));
        return glm::quat(1, 0, 0, 0);
    }
    axis /= n;
    return glm::angleAxis(angle, axis);
}

glm::quat RigSolver::filterRot(int idx, const glm::quat& q, float dt) {
    glm::vec3 rv = glm::axis(q) * glm::angle(q);
    for (int i = 0; i < 3; i++)
        rv[i] = poseRotFilters_[idx][i].filter(rv[i], dt);
    float a = glm::length(rv);
    if (a < 1e-6f) return glm::quat(1, 0, 0, 0);
    return glm::angleAxis(a, rv / a);
}

void RigSolver::calibratePose() {
    spineZNeutral_ = spineZFilter_.lastFiltered();
    handsCalibrated_ = true;
    poseCalibrated_ = true;
}

void RigSolver::updatePose(const PoseResult& pose, float dt) {
    if (!calibrated_ || !pose.detected) {
        return;  // keep last valid pose
    }

    auto has = [&](int idx) { return pose.kpScore(idx) > 0.3f; };
    if (!has(PoseLandmarkIdx::L_SHOULDER) || !has(PoseLandmarkIdx::R_SHOULDER))
        return;

    // 2D pixel coords → VRM direction vectors.
    // Image Y is down, VRM Y is up → negate Y. Z=0 (no depth from 2D).
    auto toVec = [&](int idx) -> glm::vec3 {
        return {pose.kpX(idx), -pose.kpY(idx), 0.0f};
    };

    glm::vec3 ls = toVec(PoseLandmarkIdx::L_SHOULDER);
    glm::vec3 rs = toVec(PoseLandmarkIdx::R_SHOULDER);
    glm::vec3 le = has(PoseLandmarkIdx::L_ELBOW) ? toVec(PoseLandmarkIdx::L_ELBOW) : ls;
    glm::vec3 re = has(PoseLandmarkIdx::R_ELBOW) ? toVec(PoseLandmarkIdx::R_ELBOW) : rs;
    glm::vec3 lw = has(PoseLandmarkIdx::L_WRIST) ? toVec(PoseLandmarkIdx::L_WRIST) : le;
    glm::vec3 rw = has(PoseLandmarkIdx::R_WRIST) ? toVec(PoseLandmarkIdx::R_WRIST) : re;

    // Upper arm rotations
    glm::quat upperL = dirToRotation(REST_L, le - ls);
    glm::quat upperR = dirToRotation(REST_R, re - rs);

    // Lower arm: world → local relative to upper arm
    glm::quat lowerL_world = dirToRotation(REST_L, lw - le);
    glm::quat lowerR_world = dirToRotation(REST_R, rw - re);
    glm::quat lowerL_local = glm::inverse(upperL) * lowerL_world;
    glm::quat lowerR_local = glm::inverse(upperR) * lowerR_world;

    bodyPose_.leftUpperArm  = filterRot(0, upperL, dt);
    bodyPose_.rightUpperArm = filterRot(2, upperR, dt);
    bodyPose_.leftLowerArm  = filterRot(1, lowerL_local, dt);
    bodyPose_.rightLowerArm = filterRot(3, lowerR_local, dt);

    // Spine lateral bend from shoulder line tilt
    glm::vec3 shoulderVec = ls - rs;
    float horiz = std::abs(shoulderVec.x);
    float spineZ = 0.0f;
    if (horiz > 1e-5f)
        spineZ = -std::atan2(shoulderVec.y, horiz);
    spineZ = spineZFilter_.filter(spineZ, dt);
    if (poseCalibrated_) spineZ -= spineZNeutral_;
    bodyPose_.spine = glm::angleAxis(spineZ, glm::vec3(0, 0, 1));

    // Standing detection from lower body keypoint scores
    if (has(PoseLandmarkIdx::L_HIP) && has(PoseLandmarkIdx::R_HIP)) {
        float lowerVis = (pose.kpScore(PoseLandmarkIdx::L_KNEE) + pose.kpScore(PoseLandmarkIdx::R_KNEE) +
                          pose.kpScore(PoseLandmarkIdx::L_ANKLE) + pose.kpScore(PoseLandmarkIdx::R_ANKLE)) / 4.0f;
        bodyPose_.standing = lowerVis > 0.3f ? 1.0f : 0.0f;

        float shoulderMidY = (ls.y + rs.y) * 0.5f;
        float hipMidY = (toVec(PoseLandmarkIdx::L_HIP).y + toVec(PoseLandmarkIdx::R_HIP).y) * 0.5f;
        float torsoUnit = std::abs(shoulderMidY - hipMidY);
        if (torsoUnit > 1e-5f) {
            float lowestY = shoulderMidY;
            for (int idx : {PoseLandmarkIdx::L_KNEE, PoseLandmarkIdx::R_KNEE,
                            PoseLandmarkIdx::L_ANKLE, PoseLandmarkIdx::R_ANKLE}) {
                if (has(idx)) lowestY = std::min(lowestY, -pose.kpY(idx));
            }
            bodyPose_.bodyExtent = (shoulderMidY - lowestY) / torsoUnit;
        }
    }

    bodyPose_.valid = true;
}

float RigSolver::jointAngle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    glm::vec3 v1 = b - a;
    glm::vec3 v2 = c - b;
    float n1 = glm::length(v1);
    float n2 = glm::length(v2);
    if (n1 < 1e-8f || n2 < 1e-8f) return 0.0f;
    float dot = glm::clamp(glm::dot(v1, v2) / (n1 * n2), -1.0f, 1.0f);
    return std::acos(dot);
}

void RigSolver::updateHands(const HandResult& left, const HandResult& right, float dt) {
    handOverrides_.clear();

    const HandResult* hands[2] = {left.detected ? &left : nullptr,
                                   right.detected ? &right : nullptr};

    for (int side = 0; side < 2; side++) {
        const HandResult* hr = hands[side];
        if (!hr) continue;

        // Build 3D point array from world landmarks (metric, correct scale)
        glm::vec3 pts[21];
        for (int i = 0; i < 21; i++)
            pts[i] = {hr->wlX(i), hr->wlY(i), hr->wlZ(i)};

        // Finger joint chains: [root, a, b, c, tip] → 3 joint angles at [a, b, c]
        static const int kChains[5][5] = {
            {HandLandmarkIdx::WRIST,    HandLandmarkIdx::THUMB_CMC,  HandLandmarkIdx::THUMB_MCP,  HandLandmarkIdx::THUMB_IP,   HandLandmarkIdx::THUMB_TIP},
            {HandLandmarkIdx::WRIST,    HandLandmarkIdx::INDEX_MCP,  HandLandmarkIdx::INDEX_PIP,  HandLandmarkIdx::INDEX_DIP,  HandLandmarkIdx::INDEX_TIP},
            {HandLandmarkIdx::WRIST,    HandLandmarkIdx::MIDDLE_MCP, HandLandmarkIdx::MIDDLE_PIP, HandLandmarkIdx::MIDDLE_DIP, HandLandmarkIdx::MIDDLE_TIP},
            {HandLandmarkIdx::WRIST,    HandLandmarkIdx::RING_MCP,   HandLandmarkIdx::RING_PIP,   HandLandmarkIdx::RING_DIP,   HandLandmarkIdx::RING_TIP},
            {HandLandmarkIdx::WRIST,    HandLandmarkIdx::LITTLE_MCP, HandLandmarkIdx::LITTLE_PIP, HandLandmarkIdx::LITTLE_DIP, HandLandmarkIdx::LITTLE_TIP},
        };

        // VRM rotation sign: left hand curls positive Z, right hand negative Z
        float zSign = (side == 0) ? 1.0f : -1.0f;

        for (int finger = 0; finger < 5; finger++) {
            const int* chain = kChains[finger];
            for (int joint = 0; joint < 3; joint++) {
                int nodeIdx = fingerBoneNodes_[side][finger][joint];
                if (nodeIdx < 0) continue;

                float flex = jointAngle(pts[chain[joint]],
                                        pts[chain[joint + 1]],
                                        pts[chain[joint + 2]]);

                if (finger == 0) {
                    // Thumb: Y-axis rotation, subtract rest baseline, scale up
                    static const float kThumbBaseline[3] = {0.35f, 0.25f, 0.15f};
                    float curl = std::max(0.0f, flex - kThumbBaseline[joint]) * 2.5f;
                    float ySign = (side == 0) ? 1.0f : -1.0f;
                    handOverrides_[nodeIdx] = glm::angleAxis(curl * ySign, glm::vec3(0, 1, 0));
                } else {
                    // Other fingers: Z-axis rotation
                    handOverrides_[nodeIdx] = glm::angleAxis(flex * zSign, glm::vec3(0, 0, 1));
                }
            }
        }

        // Palm twist on hand bone (X-axis)
        if (handBoneNodes_[side] >= 0) {
            glm::vec3 v1 = pts[HandLandmarkIdx::INDEX_MCP] - pts[HandLandmarkIdx::WRIST];
            glm::vec3 v2 = pts[HandLandmarkIdx::LITTLE_MCP] - pts[HandLandmarkIdx::WRIST];
            glm::vec3 normal = glm::cross(v1, v2);
            float n = glm::length(normal);
            if (n > 1e-8f) {
                normal /= n;
                float twist = std::asin(glm::clamp(-normal.y, -1.0f, 1.0f));
                if (handsCalibrated_) twist -= handTwistNeutral_[side];
                // Sit offset: palms face camera when sitting
                float sit = 1.0f - bodyPose_.standing;
                float final = 1.8f * sit - twist;
                final = std::clamp(final, -2.5f, 2.5f);
                handOverrides_[handBoneNodes_[side]] = glm::angleAxis(final, glm::vec3(1, 0, 0));
            }
        }
    }
}
