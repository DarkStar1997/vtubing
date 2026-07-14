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
const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.35, -1.8);
camera.lookAt(0, 1.35, 0);

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
  if (pose && pose.leftUpperArm) {
    // Tracked pose — apply quaternions
    for (const name of ["leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm"]) {
      const bone = _getBone(name);
      if (bone && pose[name]) {
        const [x, y, z, w] = pose[name];
        bone.quaternion.set(x, y, z, w);
      }
    }
    const lean = pose.torso ?? 0;
    for (const name of ["upperChest", "chest"]) {
      const bone = _getBone(name);
      if (bone) bone.rotation.set(lean * 0.5, 0, 0);
    }
  } else {
    // Fallback rest pose
    for (const [name, rot] of Object.entries(_REST_POSE)) {
      const bone = _getBone(name);
      if (bone) bone.rotation.set(rot.x, rot.y, rot.z);
    }
    for (const name of ["upperChest", "chest"]) {
      const bone = _getBone(name);
      if (bone) bone.rotation.set(0, 0, 0);
    }
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
  VRMUtils.combineSkeletons(vrm.scene);
  scene.add(vrm.scene);
  for (const k of Object.keys(_BONE_CACHE)) delete _BONE_CACHE[k];

  // Centre + frame the model (upper-body framing for VTubing)
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  vrm.scene.position.x -= centre.x;
  const targetY = centre.y + size.y * 0.05;
  const dist = Math.max(size.y * 1.7, 2.2);
  camera.position.set(0, targetY, -dist);
  camera.lookAt(0, targetY, 0);
  camera.fov = 30;
  camera.updateProjectionMatrix();

  currentVrmUrl = url;
  console.log("VRM loaded:", vrm.meta?.name ?? "(unknown)");
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
      head.rotation.set(pitch, yaw, roll);
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
// Hand finger curl application
// ---------------------------------------------------------------------------
const _FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"];
const _HAND_BONES = {
  left: {
    thumb:  ["leftThumbProximal", "leftThumbDistal"],
    index:  ["leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal"],
    middle: ["leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal"],
    ring:   ["leftRingProximal", "leftRingIntermediate", "leftRingDistal"],
    little: ["leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal"],
  },
  right: {
    thumb:  ["rightThumbProximal", "rightThumbDistal"],
    index:  ["rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal"],
    middle: ["rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal"],
    ring:   ["rightRingProximal", "rightRingIntermediate", "rightRingDistal"],
    little: ["rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal"],
  },
};
const _MAX_CURL_RAD = 1.1;
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
    for (let i = 0; i < _FINGER_NAMES.length; i++) {
      const curl = hand?.curls?.[i] ?? 0;
      const angle = sign * curl * _MAX_CURL_RAD;
      for (const boneName of _HAND_BONES[side][_FINGER_NAMES[i]]) {
        const bone = _getBone(boneName);
        if (bone) bone.rotation.z = angle;
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
