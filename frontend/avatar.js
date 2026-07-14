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

  // Centre + frame the model (bust-up framing for VTubing)
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  vrm.scene.position.x -= centre.x;
  // Target the upper third of the model (chest-to-head area)
  const targetY = centre.y + size.y * 0.22;
  const dist = Math.max(size.y * 0.8, 1.2);
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
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (vrm) {
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
