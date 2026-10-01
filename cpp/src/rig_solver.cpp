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
        // Normalize VRM 0.x presets to VRM 1.0 expression names (same table
        // as Python vrm_loader._VRM0X_PRESET_MAP) so standard-preset avatars
        // are driven by the ARKit→VRM expression mapping below.
        key = normalizeGroupName(key);
        std::vector<MorphBind> binds;
        for (auto& b : group.binds) {
            if (b.mesh < (int)meshMorphBase.size()) {
                binds.push_back({b.mesh, b.index, b.weight / 100.0f});
            }
        }
        // Merge (rather than overwrite) so models that carry both a standard
        // preset and a custom group with the same normalized key accumulate.
        auto& slot = groupToMorphs_[key];
        slot.insert(slot.end(), binds.begin(), binds.end());
    }

    // LookAt configuration from the model (range maps → eye degrees)
    lookAtBoneType_ = (model.lookAt.type != "expression");
    gazeHScale_ = model.lookAt.hOut;
    gazeVUpScale_ = model.lookAt.vUpOut;
    gazeVDownScale_ = model.lookAt.vDownOut;
    {
        auto it = model.boneNodes.find("leftEye");
        if (it != model.boneNodes.end()) eyeNodeL_ = it->second;
        it = model.boneNodes.find("rightEye");
        if (it != model.boneNodes.end()) eyeNodeR_ = it->second;
    }

    // Pose rotation filters: lower min_cutoff for gentle small movements,
    // beta for responsive large movements.
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++) {
            poseRotFilters_[i][j] = OneEuroFilter(0.8f, 0.05f);
            smoothRot_[i][j] = SmoothFloat(0.1f);
        }

    // Apply the default head-smoothing tuning to the head filters
    setHeadSmoothing(headSmoothing_);

    // Blinks (ARKit indices 9=eyeBlinkLeft, 10=eyeBlinkRight) are very fast
    // events (100-300ms). The default 1 Hz filter only reaches ~17% per frame
    // → eyelids barely close. Use a high cutoff for near-instant response.
    bsFilters_[9] = OneEuroFilter(20.0f, 0.0f, 10.0f);
    bsFilters_[10] = OneEuroFilter(20.0f, 0.0f, 10.0f);

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

void RigSolver::startCalibration() {
    // Reset everything the accumulation window builds on, so a recalibration
    // behaves exactly like the first calibration after application start.
    calibratingNow_ = true;
    calibFrames_ = 0;
    neutralBs_.fill(0.0f);
    neutralYaw_ = neutralPitch_ = neutralRoll_ = 0.0f;
    poseCalibrated_ = false;
    handsCalibrated_ = false;
    handTwistNeutral_[0] = handTwistNeutral_[1] = 0.0f;
    // Clear filter/smoother history so stale state from the previous session
    // does not bleed into the new neutral.
    for (auto& f : bsFilters_) f.reset();
    yawFilter_.reset();
    pitchFilter_.reset();
    rollFilter_.reset();
    gazeYawFilter_.reset();
    gazePitchFilter_.reset();
    smoothYaw_.reset();
    smoothPitch_.reset();
    smoothRoll_.reset();
    // Note: pose filters (torsoFilter_, spine*, poseRotFilters_) are left
    // running on purpose - during the window updatePose() keeps tracking the
    // live pose so calibratePose() can snapshot a meaningful neutral at the
    // end, instead of decayed/reset state.
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
        headRot_ = glm::slerp(headRot_, glm::quat(1, 0, 0, 0), 1.0f - decay);
        // Decay gaze angles to neutral along with expressions (matching
        // Python _handle_loss).
        eyeYawDeg_ *= decay;
        eyePitchDeg_ *= decay;
        yawFilter_.reset();
        pitchFilter_.reset();
        rollFilter_.reset();
        smoothYaw_.reset();
        smoothPitch_.reset();
        smoothRoll_.reset();
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

    // ARKit → VRM standard expressions: drives preset groups (aa/ih/ou/ee/oh,
    // happy/angry/sad/surprised, blink, look*) on avatars that don't ship
    // ARKit "Perfect Sync" groups. Perfect-Sync models are unaffected (their
    // groups match ARKit names directly above).
    {
        float expr[16];
        mapArkitToVrm(filteredBs, expr);
        for (int e = 0; e < 16; e++)
            if (expr[e] > 0.001f) applyGroupWeight(kVrmExprNames[e], expr[e]);
    }

    float yaw = face.yaw - neutralYaw_;
    float pitch = face.pitch - neutralPitch_;
    float roll = face.roll - neutralRoll_;
    yaw = std::clamp(yaw * headGainYaw_, -maxYaw_, maxYaw_);
    pitch = std::clamp(pitch * headGainPitch_, -maxPitch_, maxPitch_);
    roll = std::clamp(roll * headGainRoll_, -maxRoll_, maxRoll_);
    yaw = yawFilter_.filter(yaw, dt);
    pitch = pitchFilter_.filter(pitch, dt);
    roll = rollFilter_.filter(roll, dt);

    yaw = smoothYaw_.update(yaw, dt);
    pitch = smoothPitch_.update(pitch, dt);
    roll = smoothRoll_.update(roll, dt);

    glm::quat qYaw = glm::angleAxis(glm::radians(yaw), glm::vec3(0, 1, 0));
    glm::quat qPitch = glm::angleAxis(glm::radians(-pitch), glm::vec3(1, 0, 0));
    glm::quat qRoll = glm::angleAxis(glm::radians(roll), glm::vec3(0, 0, 1));
    headRot_ = qYaw * qPitch * qRoll;

    // Eye gaze from the eyeLook* blendshapes, scaled by the model's lookAt
    // range maps (Python solver._compute_gaze_angles).
    {
        float lookUp = std::max(filteredBs[17], filteredBs[18]);    // eyeLookUp L/R
        float lookDown = std::max(filteredBs[11], filteredBs[12]);  // eyeLookDown L/R
        float lookLeft = std::max(filteredBs[15], filteredBs[14]);  // eyeLookOutL / eyeLookInR
        float lookRight = std::max(filteredBs[13], filteredBs[16]); // eyeLookInL / eyeLookOutR
        float yawDeg = (lookRight - lookLeft) * gazeHScale_;
        float pitchDeg = lookUp * gazeVUpScale_ - lookDown * gazeVDownScale_;
        // User gaze-strength multiplier applies to both axes
        yawDeg *= gazeScale_;
        pitchDeg *= gazeScale_;
        eyeYawDeg_ = gazeYawFilter_.filter(yawDeg, dt);
        eyePitchDeg_ = gazePitchFilter_.filter(pitchDeg, dt);
    }
}

glm::quat RigSolver::eyeRotation() const {
    // positive yaw = look right → negative Y rotation
    // positive pitch = look up → positive X rotation
    // (matches Python renderer._compute_eye_bone_quat)
    glm::quat qYaw = glm::angleAxis(glm::radians(-eyeYawDeg_), glm::vec3(0, 1, 0));
    glm::quat qPitch = glm::angleAxis(glm::radians(eyePitchDeg_), glm::vec3(1, 0, 0));
    return qYaw * qPitch;
}

void RigSolver::setHeadGains(float yaw, float pitch, float roll) {
    headGainYaw_ = std::clamp(yaw, 0.0f, 1.0f);
    headGainPitch_ = std::clamp(pitch, 0.0f, 1.0f);
    headGainRoll_ = std::clamp(roll, 0.0f, 1.0f);
}

void RigSolver::setHeadClamps(float maxYaw, float maxPitch, float maxRoll) {
    maxYaw_ = std::max(1.0f, maxYaw);
    maxPitch_ = std::max(1.0f, maxPitch);
    maxRoll_ = std::max(1.0f, maxRoll);
}

void RigSolver::setHeadSmoothing(float s) {
    headSmoothing_ = std::clamp(s, 0.0f, 1.0f);
    float t = headSmoothing_;
    // 0 → snappy (cutoff 2.5 Hz, beta 0.10, glide 40 ms)
    // 1 → very smooth (cutoff 0.7 Hz, beta 0.00, glide 140 ms)
    float cutoff = 2.5f + (0.7f - 2.5f) * t;
    float beta = 0.10f * (1.0f - t);
    float glide = 0.04f + (0.14f - 0.04f) * t;
    yawFilter_.setParams(cutoff, beta);
    pitchFilter_.setParams(cutoff, beta);
    rollFilter_.setParams(cutoff, beta);
    smoothYaw_.setSmoothTime(glide);
    smoothPitch_.setSmoothTime(glide);
    smoothRoll_.setSmoothTime(glide);
}

std::string RigSolver::normalizeGroupName(std::string key) {
    // VRM 0.x blendshape preset → VRM 1.0 expression name
    // (Python vrm_loader._VRM0X_PRESET_MAP)
    static const std::unordered_map<std::string, std::string> kPresetMap = {
        {"a", "aa"}, {"i", "ih"}, {"u", "ou"}, {"e", "ee"}, {"o", "oh"},
        {"blink_l", "blinkleft"}, {"blink_r", "blinkright"},
        {"fun", "relaxed"}, {"joy", "happy"}, {"sorrow", "sad"},
    };
    auto it = kPresetMap.find(key);
    return (it != kPresetMap.end()) ? it->second : key;
}

void RigSolver::applyGroupWeight(const std::string& name, float w) {
    auto it = groupToMorphs_.find(name);
    if (it == groupToMorphs_.end()) return;
    for (auto& bind : it->second) {
        if (bind.meshIdx < (int)meshMorphBase_.size()) {
            int idx = meshMorphBase_[bind.meshIdx] + bind.targetIdx;
            if (idx >= 0 && idx < (int)morphWeights_.size())
                morphWeights_[idx] = std::max(morphWeights_[idx], w * bind.weight);
        }
    }
}

void RigSolver::mapArkitToVrm(const float bs[52], float out[16]) {
    auto clamp01 = [](float v) { return std::clamp(v, 0.0f, 1.0f); };
    auto mx = [&](int a, int b) { return std::max(bs[a], bs[b]); };
    auto mn = [&](int a, int b) { return std::min(bs[a], bs[b]); };

    float blink = mx(9, 10);                       // eyeBlink L/R
    float lookUp = mx(17, 18);                     // eyeLookUp L/R
    float lookDown = mx(11, 12);                   // eyeLookDown L/R
    float lookLeft = std::max(bs[15], bs[14]);     // eyeLookOutLeft / eyeLookInRight
    float lookRight = std::max(bs[13], bs[16]);    // eyeLookInLeft / eyeLookOutRight

    float jaw = bs[25];                            // jawOpen
    float stretch = mx(46, 47);                    // mouthStretch L/R
    float funnel = bs[32];                         // mouthFunnel
    float pucker = bs[38];                         // mouthPucker
    float smile = mn(44, 45);                      // mouthSmile L/R

    // Visemes
    out[0] = clamp01(jaw);                                            // aa
    out[1] = clamp01(std::min(jaw * 0.3f, 0.3f) + stretch * 0.7f);   // ih
    out[2] = clamp01(std::max(funnel, pucker));                       // ou
    out[3] = clamp01(stretch * 0.5f + smile * 0.3f);                  // ee
    out[4] = clamp01(std::max(funnel, pucker) * 0.5f + jaw * 0.3f);   // oh
    // Emotions
    out[5] = clamp01(smile);                                          // happy
    out[6] = clamp01(mx(2, 3) * 0.7f + mx(50, 51) * 0.3f);            // angry
    out[7] = clamp01(mx(30, 31) * 0.7f + bs[1] * 0.3f);               // sad
    out[8] = clamp01(mx(4, 5) * 0.3f + mx(21, 22) * 0.3f + jaw * 0.4f); // surprised
    // Blink
    out[9] = clamp01(blink);                                          // blink
    out[10] = clamp01(bs[9]);                                         // blinkleft
    out[11] = clamp01(bs[10]);                                        // blinkright
    // Look (drives look* expression groups; also used for bone gaze)
    out[12] = clamp01(lookUp);                                        // lookup
    out[13] = clamp01(lookDown);                                      // lookdown
    out[14] = clamp01(lookLeft);                                      // lookleft
    out[15] = clamp01(lookRight);                                     // lookright
}

glm::quat RigSolver::dirToRotation(const glm::vec3& rest, const glm::vec3& target) {
    float targetLen = glm::length(target);
    if (targetLen < 1e-5f) return glm::quat(1, 0, 0, 0);
    glm::vec3 r = glm::normalize(rest);
    glm::vec3 t = target / targetLen;
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
    for (int i = 0; i < 3; i++) {
        rv[i] = poseRotFilters_[idx][i].filter(rv[i], dt);
        rv[i] = smoothRot_[idx][i].update(rv[i], dt);
    }
    float a = glm::length(rv);
    if (a < 1e-6f) return glm::quat(1, 0, 0, 0);
    return glm::angleAxis(a, rv / a);
}

void RigSolver::calibratePose() {
    // Snapshot only once the face-side accumulation window is complete, so the
    // pose filters hold a stable reading of the calibration pose (startup and
    // recalibration take the same path this way).
    if (calibFrames_ < CALIB_COUNT) return;
    torsoNeutral_ = torsoFilter_.lastFiltered();
    spineYNeutral_ = spineYFilter_.lastFiltered();
    spineZNeutral_ = spineZFilter_.lastFiltered();
    handsCalibrated_ = true;
    poseCalibrated_ = true;
    calibratingNow_ = false;
}

void RigSolver::updatePose(const PoseResult& pose, float dt) {
    if ((!calibrated_ && !calibratingNow_) || !pose.detected ||
        pose.lmVis(PoseLandmarkIdx::L_SHOULDER) < 0.3f ||
        pose.lmVis(PoseLandmarkIdx::R_SHOULDER) < 0.3f) {
        // Decay body pose toward neutral and reset filters
        float decay = std::exp(-dt / 0.3f);
        bodyPose_.lean *= decay;
        bodyPose_.twist *= decay;
        bodyPose_.lateral *= decay;
        bodyPose_.standing *= decay;
        bodyPose_.bodyExtent *= decay;
        torsoFilter_.reset();
        spineYFilter_.reset();
        spineZFilter_.reset();
        standingFilter_.reset();
        bodyExtentFilter_.reset();
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 3; j++) {
                poseRotFilters_[i][j].reset();
                smoothRot_[i][j].reset();
            }
        return;
    }

    // Helper: world landmark → VRM direction (apply AXIS_FLIP)
    auto wl = [&](int idx) -> glm::vec3 {
        return {pose.wlX(idx) * AXIS_FLIP.x,
                pose.wlY(idx) * AXIS_FLIP.y,
                pose.wlZ(idx) * AXIS_FLIP.z};
    };

    glm::vec3 ls = wl(PoseLandmarkIdx::L_SHOULDER);
    glm::vec3 rs = wl(PoseLandmarkIdx::R_SHOULDER);
    glm::vec3 le = wl(PoseLandmarkIdx::L_ELBOW);
    glm::vec3 re = wl(PoseLandmarkIdx::R_ELBOW);
    glm::vec3 lw = wl(PoseLandmarkIdx::L_WRIST);
    glm::vec3 rw = wl(PoseLandmarkIdx::R_WRIST);

    // Arm direction vectors (already AXIS_FLIP'd via wl()).
    glm::vec3 uaL = le - ls;  // upper arm L direction
    glm::vec3 uaR = re - rs;  // upper arm R direction
    glm::vec3 laL = lw - le;  // lower arm L direction
    glm::vec3 laR = rw - re;  // lower arm R direction

    // BlazePose world-Z for arms is unreliable when turned sideways.
    // Without correction, Z sign can flip → arm rotates backward through torso.
    // Due to REST_L={1,0,0} vs bone-at-(-X): positive Z in direction → arm
    // toward camera. Only apply Z clamp when avatar is turned sideways
    // (shoulder Z depth difference exceeds threshold).
    float shoulderZDiff = ls.z - rs.z;
    float poseTurnStr = std::clamp((std::abs(shoulderZDiff) - 0.04f) / 0.06f, 0.0f, 1.0f);
    if (poseTurnStr > 0.01f) {
        float minZ = 0.08f * poseTurnStr;
        auto clampForwardZ = [minZ](glm::vec3& d) {
            float dl = glm::length(d);
            if (dl > 1e-5f && d.z / dl < minZ)
                d.z = minZ * dl;
        };
        clampForwardZ(uaL);
        clampForwardZ(uaR);
        clampForwardZ(laL);
        clampForwardZ(laR);
    }

    // Upper arm rotations: shoulder→elbow direction
    glm::quat upperL = dirToRotation(REST_L, uaL);
    glm::quat upperR = dirToRotation(REST_R, uaR);

    // Lower arm: world→local relative to upper arm
    glm::quat lowerL_world = dirToRotation(REST_L, laL);
    glm::quat lowerR_world = dirToRotation(REST_R, laR);
    glm::quat lowerL_local = glm::inverse(upperL) * lowerL_world;
    glm::quat lowerR_local = glm::inverse(upperR) * lowerR_world;

    bodyPose_.leftUpperArm  = filterRot(0, upperL, dt);
    bodyPose_.rightUpperArm = filterRot(2, upperR, dt);
    bodyPose_.leftLowerArm  = filterRot(1, lowerL_local, dt);
    bodyPose_.rightLowerArm = filterRot(3, lowerR_local, dt);

    // Torso lean from shoulder-hip depth difference
    if (pose.lmVis(PoseLandmarkIdx::L_HIP) > 0.3f &&
        pose.lmVis(PoseLandmarkIdx::R_HIP) > 0.3f) {
        glm::vec3 lh = wl(PoseLandmarkIdx::L_HIP);
        glm::vec3 rh = wl(PoseLandmarkIdx::R_HIP);
        glm::vec3 d = (ls + rs) * 0.5f - (lh + rh) * 0.5f;
        float vert = std::sqrt(d.x * d.x + d.y * d.y);
        float rawLean = 0.0f;
        if (vert > 1e-5f)
            rawLean = std::atan2(-d.z, vert);
        rawLean = torsoFilter_.filter(rawLean, dt);
        if (poseCalibrated_) rawLean -= torsoNeutral_;
        bodyPose_.lean = std::clamp(rawLean, -0.15f, 0.15f);
    }

    // Spine lateral bend + twist from shoulder line
    glm::vec3 shoulderVec = ls - rs;
    float horiz = std::sqrt(shoulderVec.x * shoulderVec.x +
                            shoulderVec.z * shoulderVec.z);
    float rawZ = 0.0f, rawY = 0.0f;
    if (horiz > 1e-5f) {
        rawZ = -std::atan2(shoulderVec.y, horiz);
        rawY = std::atan2(-shoulderVec.z, horiz);
    }
    rawY = spineYFilter_.filter(rawY, dt);
    rawZ = spineZFilter_.filter(rawZ, dt);
    if (poseCalibrated_) {
        rawY -= spineYNeutral_;
        rawZ -= spineZNeutral_;
    }
    // Deadzone: suppress small noise from arm-raising artifacts
    if (std::abs(rawY) < 0.04f) rawY = 0.0f;
    bodyPose_.twist = std::clamp(rawY, -0.3f, 0.3f);
    if (std::abs(rawZ) < 0.03f) rawZ = 0.0f;
    bodyPose_.lateral = std::clamp(rawZ, -0.2f, 0.2f);

    // Standing detection: hip visibility with hysteresis (matching Python)
    // Enter stand at hip_vis > 0.50, exit at hip_vis < 0.30
    // Also require hips to be in-frame (normalized Y < 0.95)
    float hipVisL = (pose.lmVis(PoseLandmarkIdx::L_HIP) > 0.3f &&
                     pose.lmY(PoseLandmarkIdx::L_HIP) < 0.95f)
                    ? pose.lmVis(PoseLandmarkIdx::L_HIP) : 0.0f;
    float hipVisR = (pose.lmVis(PoseLandmarkIdx::R_HIP) > 0.3f &&
                     pose.lmY(PoseLandmarkIdx::R_HIP) < 0.95f)
                    ? pose.lmVis(PoseLandmarkIdx::R_HIP) : 0.0f;
    float hipVis = (hipVisL + hipVisR) * 0.5f;
    if (standState_)
        standState_ = hipVis < 0.30f ? false : true;
    else
        standState_ = hipVis > 0.50f ? true : false;
    bodyPose_.standing = standingFilter_.filter(standState_ ? 1.0f : 0.0f, dt);

    // Body extent: how many torso-lengths of body are visible below shoulders
    // Uses IMAGE-space normalized Y (0=top, 1=bottom) matching Python pipeline.
    //   0 = just shoulders, 1 = to hips, 2 = to knees, 3+ = to ankles
    float rawExtent = 0.0f;
    float shImgY = (pose.lmY(PoseLandmarkIdx::L_SHOULDER) +
                    pose.lmY(PoseLandmarkIdx::R_SHOULDER)) * 0.5f;
    if (pose.lmVis(PoseLandmarkIdx::L_HIP) > 0.3f &&
        pose.lmVis(PoseLandmarkIdx::R_HIP) > 0.3f) {
        float hipImgY = (pose.lmY(PoseLandmarkIdx::L_HIP) +
                         pose.lmY(PoseLandmarkIdx::R_HIP)) * 0.5f;
        float torsoUnit = std::max(hipImgY - shImgY, 0.01f);
        float lowestImgY = shImgY;
        for (int idx : {PoseLandmarkIdx::L_ANKLE, PoseLandmarkIdx::R_ANKLE,
                        PoseLandmarkIdx::L_KNEE, PoseLandmarkIdx::R_KNEE,
                        PoseLandmarkIdx::L_HIP, PoseLandmarkIdx::R_HIP}) {
            if (pose.lmVis(idx) > 0.3f && pose.lmY(idx) < 0.95f &&
                pose.lmY(idx) > lowestImgY)
                lowestImgY = pose.lmY(idx);
        }
        rawExtent = (lowestImgY - shImgY) / torsoUnit;
    }
    bodyPose_.bodyExtent = bodyExtentFilter_.filter(rawExtent, dt);

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
    // When standing, skip finger tracking — hand landmarks are too noisy
    // at full-body distance. Set a clean straight-finger rest pose with
    // thumbs brought inline (not sticking out horizontally).
    if (bodyPose_.standing > 0.5f) {
        handOverrides_.clear();
        for (int side = 0; side < 2; side++) {
            int thumbProx = fingerBoneNodes_[side][0][1];
            if (thumbProx >= 0)
                handOverrides_[thumbProx] = glm::angleAxis(
                    glm::radians(side == 0 ? 50.0f : -50.0f), glm::vec3(0, 1, 0));
        }
        lastGoodHandOverrides_.clear();
        return;
    }

    // Detect hand overlap: when hands are close together (folded,
    // interleaved, etc.), MediaPipe landmarks become unreliable and
    // finger curls jitter wildly. Smoothly freeze at last good values.
    float overlapFactor = 0.0f;
    if (left.detected && right.detected) {
        float dx = left.lmX(0) - right.lmX(0);
        float dy = left.lmY(0) - right.lmY(0);
        float wristDist = std::sqrt(dx*dx + dy*dy);
        overlapFactor = 1.0f - std::clamp((wristDist - 0.10f) / 0.08f, 0.0f, 1.0f);
    }

    // Fully overlapping: keep last good overrides unchanged
    if (overlapFactor > 0.99f) return;

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

    // Partial overlap: blend live finger rotations toward last good values
    if (overlapFactor > 0.01f) {
        for (auto& [node, rot] : handOverrides_) {
            auto it = lastGoodHandOverrides_.find(node);
            if (it != lastGoodHandOverrides_.end())
                rot = glm::slerp(rot, it->second, overlapFactor);
        }
    }
    lastGoodHandOverrides_ = handOverrides_;
}
