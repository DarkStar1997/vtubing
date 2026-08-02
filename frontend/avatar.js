import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

// ---------------------------------------------------------------------------
// Renderer / Scene / Camera
// ---------------------------------------------------------------------------
const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.45, -1.8);
camera.lookAt(0, 1.32, 0);

// Lights — simple 3-point setup
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.0));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(1, 2, -1.5);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.3);
fill.position.set(-1, 1, -1);
scene.add(fill);

// ---------------------------------------------------------------------------
// VRM model
// ---------------------------------------------------------------------------
let vrm = null;
let currentVrmUrl = null;
let _framingApplied = false;

// Default rest pose — used when no body tracking data is available
const _REST_POSE = {
  leftUpperArm:  { x: 0,    y: 0.20,  z: 1.30 },
  rightUpperArm: { x: 0,    y: -0.20, z: -1.30 },
  leftLowerArm:  { x: 0,    y: 0,     z: -1.15 },
  rightLowerArm: { x: 0,    y: 0,     z: 1.15 },
};

function _applyPose(vrmModel, pose) {
  const h = vrmModel.humanoid;
  if (!h) return;

  // Seated base posture — natural sitting at a desk
  const hips = _getBone("hips");
  if (hips) hips.rotation.set(0.06, 0, 0);
  const spine = _getBone("spine");
  if (spine) spine.rotation.set(0.04, 0, 0);

  if (pose && pose.leftUpperArm) {
    // Tracked pose — apply arm quaternions
    for (const name of ["leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm"]) {
      const bone = _getBone(name);
      if (bone && pose[name]) {
        const [x, y, z, w] = pose[name];
        bone.quaternion.set(x, y, z, w);
      }
    }
    // Spine: forward lean + twist + lateral bend
    const sp = pose.spine;  // [lean, twist, lateral]
    const lean = sp ? sp[0] : (pose.torso ?? 0);
    const twist = sp ? sp[1] : 0;
    const lateral = sp ? sp[2] : 0;
    const upperChest = _getBone("upperChest");
    if (upperChest) upperChest.rotation.set(lean * 0.5, twist * 0.5, lateral * 0.5);
    const chest = _getBone("chest");
    if (chest) chest.rotation.set(lean * 0.3, twist * 0.3, lateral * 0.3);
    // Distribute spine bend
    if (spine) spine.rotation.set(0.04 + lean * 0.2, twist * 0.2, lateral * 0.2);
  } else {
    // Fallback rest pose — relaxed A-pose for sitting
    for (const [name, rot] of Object.entries(_REST_POSE)) {
      const bone = _getBone(name);
      if (bone) bone.rotation.set(rot.x, rot.y, rot.z);
    }
    const upperChest = _getBone("upperChest");
    if (upperChest) upperChest.rotation.set(0, 0, 0);
    const chest = _getBone("chest");
    if (chest) chest.rotation.set(0, 0, 0);
  }
}

async function loadVRM(url) {
  if (vrm) {
    VRMUtils.deepDispose(vrm.scene);
    scene.remove(vrm.scene);
  }
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(vrm.scene);
  scene.add(vrm.scene);
  for (const k of Object.keys(_BONE_CACHE)) delete _BONE_CACHE[k];
  _framingApplied = false;

  // Centre + frame the model — bust shot (head to chest), webcam from above
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  vrm.scene.position.x -= centre.x;
  // Target upper chest / neck area
  const targetY = centre.y + size.y * 0.28;
  const dist = Math.max(size.y * 0.85, 1.1);
  // Camera slightly above eye-level looking down — mimics a webcam on a monitor
  camera.position.set(0, targetY + 0.10, -dist);
  camera.lookAt(0, targetY - 0.02, 0);
  camera.fov = 28;
  camera.updateProjectionMatrix();

  currentVrmUrl = url;
  console.log("VRM loaded:", vrm.meta?.name ?? "(unknown)");
}

// ---------------------------------------------------------------------------
// Calibration-based camera framing — match the user's webcam composition
// ---------------------------------------------------------------------------
const _projV = new THREE.Vector3();

function _projectToScreenY(worldY) {
  _projV.set(0, worldY, 0).project(camera);
  return (1 - _projV.y) / 2; // 0=top, 1=bottom
}

function applyFraming(framing) {
  if (!vrm || !framing || _framingApplied) return;

  const headBone = _getBone("head");
  if (!headBone) return;

  // Model geometry
  const headPos = new THREE.Vector3();
  headBone.getWorldPosition(headPos);
  const sceneBox = new THREE.Box3().setFromObject(vrm.scene);
  const sceneSize = sceneBox.getSize(new THREE.Vector3());

  // Model face height ≈ 13% of body height (standard human proportion)
  const modelFaceH = sceneSize.y * 0.13;
  // Model face center ≈ slightly above head bone (head bone is at neck/base of skull)
  const modelFaceCenter = headPos.y + sceneSize.y * 0.035;

  // User's webcam framing (normalized 0-1, 0=top)
  const userFaceH = framing.face_h;
  const userFaceCenter = framing.face_center_y;

  const fov = 28;
  const halfFov = (fov * Math.PI) / 360;
  const tanHalfFov = Math.tan(halfFov);

  // Distance: match avatar's face size to user's face size in frame
  // userFaceH fraction of frame = modelFaceH / (2 * dist * tanHalfFov)
  let dist = modelFaceH / (Math.max(userFaceH, 0.01) * 2 * tanHalfFov);
  // Target Y: match avatar's face center to user's face center in frame
  let targetY = modelFaceCenter + (2 * userFaceCenter - 1) * dist * tanHalfFov;

  // Iterative correction for camera tilt
  const tiltOffset = 0.10;
  for (let i = 0; i < 5; i++) {
    camera.position.set(0, targetY + tiltOffset, -dist);
    camera.lookAt(0, targetY, 0);
    camera.fov = fov;
    camera.updateProjectionMatrix();

    const screenFaceCenter = _projectToScreenY(modelFaceCenter);
    const err = userFaceCenter - screenFaceCenter;
    targetY += err * dist * tanHalfFov * 0.8;
  }

  camera.position.set(0, targetY + tiltOffset, -dist);
  camera.lookAt(0, targetY, 0);
  camera.fov = fov;
  camera.updateProjectionMatrix();

  _framingApplied = true;
  const screenFaceCenter = _projectToScreenY(modelFaceCenter);
  console.log("Framing applied:", {
    modelFaceH: modelFaceH.toFixed(4),
    userFaceH: userFaceH.toFixed(4),
    dist: dist.toFixed(3),
    targetY: targetY.toFixed(3),
    screenFaceCenter: screenFaceCenter.toFixed(3),
    target: userFaceCenter.toFixed(3),
  });
}

// ---------------------------------------------------------------------------
// WebSocket rig client
// ---------------------------------------------------------------------------
const statusEl = document.getElementById("status");
let ws = null;
let latestRig = null;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => (statusEl.textContent = "connected");
  ws.onmessage = (ev) => {
    try {
      latestRig = JSON.parse(ev.data);
    } catch { /* ignore */ }
  };
  ws.onclose = () => {
    statusEl.textContent = "reconnecting...";
    setTimeout(connect, 1000);
  };
}

// ---------------------------------------------------------------------------
// Apply rig state to VRM each frame
// ---------------------------------------------------------------------------
const _v = new THREE.Vector3();

function applyRig(rig) {
  if (!vrm || !rig) return;
  const expr = rig.expressions || {};
  const em = vrm.expressionManager;
  if (em) {
    for (const name of Object.keys(expr)) {
      em.setValue(name, expr[name]);
    }
    // Reset expressions not mentioned this frame (detection loss)
    em.update();
  }

  // Head bone rotation (radians: yaw, pitch, roll, YXZ order)
  if (rig.head) {
    const head = vrm.humanoid?.getNormalizedBoneNode("head");
    if (head) {
      const [yaw, pitch, roll] = rig.head;
      head.rotation.order = "YXZ";
      head.rotation.set(pitch + 0.32, yaw, roll);
    }
  }

  // Eye gaze
  if (rig.gaze && vrm.lookAt) {
    const [yaw, pitch] = rig.gaze;
    vrm.lookAt.applier.applyYawPitch(yaw, pitch);
  }

  // Hand finger curls
  _applyHands(rig.hands);
}

// ---------------------------------------------------------------------------
// Hand tracking: wrist orientation + per-joint finger angles
// ---------------------------------------------------------------------------
const _FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"];
const _HAND_BONES = {
  left: {
    thumb:  ["leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal"],
    index:  ["leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal"],
    middle: ["leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal"],
    ring:   ["leftRingProximal", "leftRingIntermediate", "leftRingDistal"],
    little: ["leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal"],
  },
  right: {
    thumb:  ["rightThumbMetacarpal", "rightThumbProximal", "rightThumbDistal"],
    index:  ["rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal"],
    middle: ["rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal"],
    ring:   ["rightRingProximal", "rightRingIntermediate", "rightRingDistal"],
    little: ["rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal"],
  },
};
// Max flexion angle per joint position [proximal, intermediate, distal] (radians)
const _JOINT_MAX_RAD = [3.0, 3.2, 2.2];
const _THUMB_JOINT_MAX_RAD = [1.5, 2.0, 2.0];
const _BONE_CACHE = {};

function _getBone(name) {
  if (!vrm?.humanoid) return null;
  if (!(name in _BONE_CACHE)) {
    _BONE_CACHE[name] = vrm.humanoid.getNormalizedBoneNode(name);
  }
  return _BONE_CACHE[name];
}

function _applyHands(hands) {
  for (const side of ["left", "right"]) {
    const hand = hands?.[side];
    const sign = side === "left" ? -1 : 1;

    // Palm twist (pronation/supination) on the hand bone — X axis.
    // VRM hand bones are NOT mirrored: same X rotation → same palm direction.
    // Cross product gives opposite-sign twist for L/R hands, so `OFFSET - twist`
    // produces correct mirrored palm motion on both hands.
    const handBone = _getBone(side === "left" ? "leftHand" : "rightHand");
    if (handBone) {
      const twistKey = side + "_twist";
      let twist = hand?.twist;
      if (twist == null) {
        twist = (window._lastTwist ??= {})[twistKey] ?? 0;
      } else {
        (window._lastTwist ??= {})[twistKey] = twist;
      }
      const OFFSET = 1.8;
      const final = Math.max(-2.5, Math.min(2.5, OFFSET - twist));
      handBone.rotation.set(final, 0, 0);
    }

    // Per-joint finger angles
    for (const fingerName of _FINGER_NAMES) {
      const bones = _HAND_BONES[side][fingerName];
      const angles = hand?.fingers?.[fingerName] ?? [];
      const maxAngles = fingerName === "thumb" ? _THUMB_JOINT_MAX_RAD : _JOINT_MAX_RAD;
      for (let j = 0; j < bones.length; j++) {
        const bone = _getBone(bones[j]);
        if (!bone) continue;
        const flex = angles[j] ?? 0;
        const angle = sign * flex * (maxAngles[j] ?? 1.1);
        if (fingerName === "thumb") {
          bone.rotation.set(0, angle, 0);
        } else {
          bone.rotation.set(0, 0, -angle);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (vrm) {
    if (latestRig?.framing) applyFraming(latestRig.framing);
    _applyPose(vrm, latestRig?.pose);
    applyRig(latestRig);
    vrm.update(dt); // spring bones, etc.
  }
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Resize handling
// ---------------------------------------------------------------------------
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
statusEl.textContent = "loading avatar...";
loadVRM("/avatar?t=" + Date.now())
  .then(() => {
    statusEl.textContent = "avatar loaded, connecting...";
    connect();
    animate();
  })
  .catch((err) => {
    console.error("boot error:", err);
    statusEl.textContent = "error: " + (err?.message || err);
  });
