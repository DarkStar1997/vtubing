"""moderngl offscreen VRM renderer: skinning + morph targets + lit/textured."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from src.avatar.gltf_utils import compute_world_matrices, decode_accessor, decode_texture
from src.avatar.shaders import FRAGMENT_SHADER, VERTEX_SHADER
from src.avatar.vrm_loader import VRMModel, load_vrm

MAX_JOINTS = 256
MAX_MORPHS = 64

# VRM material naming convention: "{name}_{order}_{TAG}"
# Render order: body parts first, hair second (no depth write), face/eyes last
_HAIR_TAGS = {"HAIR"}

# Sort order for face parts (back-to-front: base skin first, highlights last)
_FACE_SORT_PRIORITY = [
    ("skin", 0), ("eyewhite", 1), ("eyeline", 2), ("brow", 3),
    ("mouth", 4), ("iris", 5), ("highlight", 6),
]


def _face_sort_key(mat_name: str | None) -> int:
    if not mat_name:
        return 99
    low = mat_name.lower()
    for keyword, priority in _FACE_SORT_PRIORITY:
        if keyword in low:
            return priority
    return 7


@dataclass
class _Prim:
    vao: moderngl.VertexArray
    index_count: int
    model: np.ndarray
    mesh_idx: int
    skin_idx: int | None
    morph_count: int
    vert_count: int
    morph_deltas: np.ndarray | None
    morph_ssbo: moderngl.Buffer | None
    has_tex: bool
    base_color: np.ndarray
    texture: moderngl.Texture | None
    render_order: int
    depth_write: bool
    node_idx: int
    mat_name: str | None = None


def _perspective(fovy_rad: float, aspect: float, near: float, far: float, flip_y: bool = True) -> np.ndarray:
    f = 1.0 / np.tan(fovy_rad / 2.0)
    m = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ], np.float32)
    if flip_y:
        m[1, 1] = -m[1, 1]
    return m


def _lookat(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    up = np.asarray(up, np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([
        [s[0], s[1], s[2], -s.dot(eye)],
        [u[0], u[1], u[2], -u.dot(eye)],
        [-f[0], -f[1], -f[2], f.dot(eye)],
        [0, 0, 0, 1],
    ], np.float32)


def _mat4_col_major(m: np.ndarray) -> bytes:
    """Convert row-major numpy mat4 to column-major bytes for OpenGL upload."""
    return np.ascontiguousarray(m.T, np.float32).tobytes()


class VRMRenderer:
    def __init__(self, model: VRMModel, width: int = 1280, height: int = 720) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._color_tex = self.ctx.texture((width, height), 4)
        self._depth_tex = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(self._color_tex, self._depth_tex)
        self.fbo.viewport = (0, 0, width, height)
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        self.world = compute_world_matrices(model.nodes)
        self._neutral_world = [w.copy() for w in self.world]
        self._bone_overrides: dict[int, np.ndarray] = {}
        self._cached_skin_data: dict[int, tuple[list[int], np.ndarray]] = {}
        self._identity_ubo = self.ctx.buffer(
            np.repeat(np.eye(4, dtype=np.float32)[None], MAX_JOINTS, 0).tobytes()
        )
        self._skin_ubos = self._build_skins()
        self._mesh_morph_count: dict[int, int] = {}
        self.primitives = self._build_primitives()
        self._mesh_weights: dict[int, np.ndarray] = {
            mi: np.zeros(mc, np.float32) for mi, mc in self._mesh_morph_count.items()
        }

        head = model.humanoid_bone_by_name.get("head")
        self.head_node = head
        self.left_eye_node = model.humanoid_bone_by_name.get("leftEye")
        self.right_eye_node = model.humanoid_bone_by_name.get("rightEye")
        self.head_pos = self._neutral_world[head][:3, 3].copy() if head is not None else np.array([0, 1.5, 0], np.float32)

    def _build_skins(self) -> dict[int, moderngl.Buffer]:
        gltf = self.model.gltf
        ubos: dict[int, moderngl.Buffer] = {}
        for si, sk in enumerate(gltf.skins):
            joints = list(sk.joints)
            ibm = decode_accessor(gltf, sk.inverseBindMatrices, as_float=False).astype(np.float32).reshape(-1, 4, 4)
            ibm = ibm.transpose(0, 2, 1)  # glTF stores matrices column-major; transpose to row-major
            self._cached_skin_data[si] = (joints, ibm)
            jm = np.stack([self.world[j] @ ibm[k] for k, j in enumerate(joints)]).astype(np.float32)
            jm = jm.transpose(0, 2, 1)  # row-major -> column-major for std140
            padded = np.repeat(np.eye(4, dtype=np.float32)[None], MAX_JOINTS, 0)
            padded[: len(jm)] = jm
            ubos[si] = self.ctx.buffer(padded.tobytes())
        return ubos

    def _build_primitives(self) -> list[_Prim]:
        gltf = self.model.gltf
        prims: list[_Prim] = []
        for ni, node in enumerate(self.model.nodes):
            if node.mesh is None:
                continue
            mesh = gltf.meshes[node.mesh]
            skin_idx = getattr(gltf.nodes[ni], "skin", None)
            model_mat = self.world[ni]
            for p in mesh.primitives:
                pos = decode_accessor(gltf, getattr(p.attributes, "POSITION"))
                norm = decode_accessor(gltf, getattr(p.attributes, "NORMAL"))
                uv = decode_accessor(gltf, getattr(p.attributes, "TEXCOORD_0"))
                joints = decode_accessor(gltf, getattr(p.attributes, "JOINTS_0"))
                weights = decode_accessor(gltf, getattr(p.attributes, "WEIGHTS_0"))
                vcount = pos.shape[0]

                targets = p.targets or []
                morph_count = min(len(targets), MAX_MORPHS)
                morph_deltas = None
                morph_ssbo = None
                if morph_count:
                    deltas = np.zeros((morph_count, vcount, 4), np.float32)
                    for t in range(morph_count):
                        pos_key = targets[t].get("POSITION")
                        if pos_key is not None:
                            deltas[t, :, :3] = decode_accessor(gltf, pos_key)
                    morph_deltas = deltas
                    morph_ssbo = self.ctx.buffer(np.ascontiguousarray(deltas).tobytes())

                idx = decode_accessor(gltf, p.indices, as_float=False).astype(np.uint32)
                ibo = self.ctx.buffer(idx.tobytes())
                vbo_pos = self.ctx.buffer(pos.astype(np.float32).tobytes())
                vbo_norm = self.ctx.buffer(norm.astype(np.float32).tobytes())
                vbo_uv = self.ctx.buffer(uv.astype(np.float32).tobytes())
                vbo_j = self.ctx.buffer(joints.astype(np.float32).tobytes())
                vbo_w = self.ctx.buffer(weights.astype(np.float32).tobytes())
                vao = self.ctx.vertex_array(self.prog, [
                    (vbo_pos, "3f", "a_pos"),
                    (vbo_norm, "3f", "a_normal"),
                    (vbo_uv, "2f", "a_uv"),
                    (vbo_j, "4f", "a_joints"),
                    (vbo_w, "4f", "a_weights"),
                ], ibo)

                mat = gltf.materials[p.material] if p.material is not None and p.material < len(gltf.materials) else None
                base_color = np.array([1.0, 1.0, 1.0], np.float32)
                texture = None
                has_tex = False
                if mat is not None and mat.pbrMetallicRoughness is not None:
                    pbr = mat.pbrMetallicRoughness
                    if pbr.baseColorFactor:
                        base_color = np.array(pbr.baseColorFactor[:3], np.float32)
                    if pbr.baseColorTexture is not None and pbr.baseColorTexture.index is not None:
                        img = decode_texture(gltf, pbr.baseColorTexture.index)
                        if img is not None:
                            texture = self.ctx.texture(img.size, 3, img.tobytes())
                            has_tex = True

                mc_prev = self._mesh_morph_count.get(node.mesh, 0)
                self._mesh_morph_count[node.mesh] = max(mc_prev, morph_count)
                mat_name = getattr(mat, "name", None) if mat is not None else None
                tag = (mat_name.split("_")[-1].upper() if mat_name else "")
                if morph_count > 0:
                    rorder, dwrite = 2, True
                elif tag in _HAIR_TAGS:
                    rorder, dwrite = 1, False
                else:
                    rorder, dwrite = 0, True
                prims.append(_Prim(
                    vao=vao, index_count=idx.size, model=model_mat, mesh_idx=node.mesh,
                    skin_idx=skin_idx, morph_count=morph_count, vert_count=vcount,
                    morph_deltas=morph_deltas, morph_ssbo=morph_ssbo,
                    has_tex=has_tex, base_color=base_color, texture=texture,
                    render_order=rorder, depth_write=dwrite, node_idx=ni, mat_name=mat_name,
                ))
        return prims

    def set_camera(self, eye, target, up=(0, 1, 0), fovy_deg: float = 28.0) -> None:
        self._view = _lookat(eye, target, up)
        self._proj = _perspective(np.radians(fovy_deg), self.width / self.height, 0.05, 100.0)

    def _default_camera(self) -> None:
        if not hasattr(self, "_view"):
            eye = self.head_pos + np.array([0.0, 0.05, -1.3], np.float32)
            target = self.head_pos + np.array([0.0, -0.18, 0.0], np.float32)
            self.set_camera(eye, target)

    def clear_expressions(self) -> None:
        for w in self._mesh_weights.values():
            w.fill(0.0)

    def apply_expression(self, name: str, weight: float = 1.0) -> None:
        expr = self.model.expression_by_name.get(name)
        if expr is None:
            return
        for bind in expr.morph_binds:
            node = self.model.nodes[bind.node]
            if node.mesh is None:
                continue
            arr = self._mesh_weights.get(node.mesh)
            if arr is not None and bind.index < len(arr):
                arr[bind.index] = bind.weight * weight

    def set_morph_weight(self, mesh_idx: int, target_idx: int, weight: float) -> None:
        arr = self._mesh_weights.get(mesh_idx)
        if arr is not None and target_idx < len(arr):
            arr[target_idx] = weight

    def set_expression_weights(self, weights: dict[str, float]) -> None:
        self.clear_expressions()
        for name, weight in weights.items():
            if weight <= 0.0:
                continue
            expr = self.model.expression_by_name.get(name)
            if expr is None:
                continue
            for bind in expr.morph_binds:
                node = self.model.nodes[bind.node]
                if node.mesh is None:
                    continue
                arr = self._mesh_weights.get(node.mesh)
                if arr is not None and bind.index < len(arr):
                    arr[bind.index] = max(arr[bind.index], bind.weight * weight)

    def set_bone_rotation(self, node_idx: int, quaternion: np.ndarray) -> None:
        self._bone_overrides[node_idx] = np.asarray(quaternion, dtype=np.float32)

    def clear_bone_overrides(self) -> None:
        self._bone_overrides.clear()

    def apply_rig_state(self, state: dict) -> None:
        self.set_expression_weights(state.get("expressions", {}))
        self.clear_bone_overrides()
        head_delta = state.get("head_delta")
        if head_delta is not None and self.head_node is not None:
            q = self._compute_head_bone_quat(head_delta)
            self.set_bone_rotation(self.head_node, q)
        yaw = state.get("eye_yaw", 0.0)
        pitch = state.get("eye_pitch", 0.0)
        if yaw != 0.0 or pitch != 0.0:
            for eye_node in (self.left_eye_node, self.right_eye_node):
                if eye_node is not None:
                    q = self._compute_eye_bone_quat(yaw, pitch, eye_node)
                    self.set_bone_rotation(eye_node, q)

    def _compute_head_bone_quat(self, delta_rot: np.ndarray) -> np.ndarray:
        head_node = self.model.nodes[self.head_node]
        orig_rot = Rotation.from_quat(head_node.rotation).as_matrix()
        parent_idx = head_node.parent
        if parent_idx is not None:
            parent_rot = self._neutral_world[parent_idx][:3, :3]
        else:
            parent_rot = np.eye(3, dtype=np.float32)
        local_delta = parent_rot.T @ delta_rot @ parent_rot
        new_rot = local_delta @ orig_rot
        return Rotation.from_matrix(new_rot).as_quat().astype(np.float32)

    def _compute_eye_bone_quat(self, yaw_deg: float, pitch_deg: float, node_idx: int) -> np.ndarray:
        # positive yaw = right -> negative Y rotation
        # positive pitch = up -> positive X rotation
        rot = Rotation.from_euler("YX", [-yaw_deg, pitch_deg], degrees=True)
        rot_mat = rot.as_matrix()
        eye_node = self.model.nodes[node_idx]
        orig_rot = Rotation.from_quat(eye_node.rotation).as_matrix()
        new_rot = rot_mat @ orig_rot
        return Rotation.from_matrix(new_rot).as_quat().astype(np.float32)

    def _update_transforms(self) -> None:
        self.world = compute_world_matrices(self.model.nodes, self._bone_overrides)
        for si, (joints, ibm) in self._cached_skin_data.items():
            jm = np.stack([self.world[j] @ ibm[k] for k, j in enumerate(joints)]).astype(np.float32)
            jm = jm.transpose(0, 2, 1)
            padded = np.repeat(np.eye(4, dtype=np.float32)[None], MAX_JOINTS, 0)
            padded[: len(jm)] = jm
            self._skin_ubos[si].write(padded.tobytes())
        for prim in self.primitives:
            prim.model = self.world[prim.node_idx]

    def render(self) -> np.ndarray:
        self._default_camera()
        if self._bone_overrides:
            self._update_transforms()
        self.fbo.clear(0.05, 0.05, 0.08, 1.0)
        self.fbo.use()
        self.prog["u_proj"].write(_mat4_col_major(self._proj))
        self.prog["u_view"].write(_mat4_col_major(self._view))
        sorted_prims = sorted(
            enumerate(self.primitives),
            key=lambda ip: (ip[1].render_order, _face_sort_key(ip[1].mat_name) if ip[1].render_order == 2 else 0, ip[0])
        )
        for _, prim in sorted_prims:
            self.prog["u_model"].write(_mat4_col_major(prim.model))
            if "u_vert_count" in self.prog:
                self.prog["u_vert_count"].value = prim.vert_count
            if "u_morph_count" in self.prog:
                self.prog["u_morph_count"].value = prim.morph_count
            if prim.morph_deltas is not None and prim.morph_ssbo is not None:
                wpad = np.zeros(MAX_MORPHS, np.float32)
                mw = self._mesh_weights.get(prim.mesh_idx)
                if mw is not None:
                    wpad[:len(mw)] = mw
                prim.morph_deltas[:, :, 3] = wpad[:prim.morph_count, None]
                prim.morph_ssbo.write(prim.morph_deltas.tobytes())
            if prim.morph_ssbo is not None:
                prim.morph_ssbo.bind_to_storage_buffer(1)
            if prim.skin_idx is not None and prim.skin_idx in self._skin_ubos:
                self._skin_ubos[prim.skin_idx].bind_to_uniform_block(0)
            else:
                self._identity_ubo.bind_to_uniform_block(0)
            self.prog["u_has_tex"].value = 1 if prim.has_tex else 0
            self.prog["u_base_color"].value = tuple(prim.base_color.tolist())
            if prim.texture is not None:
                prim.texture.use(0)
            self.ctx.depth_mask = prim.depth_write
            if prim.render_order == 2:
                self.ctx.disable(moderngl.DEPTH_TEST)
            else:
                self.ctx.enable(moderngl.DEPTH_TEST)
            prim.vao.render(moderngl.TRIANGLES, vertices=prim.index_count)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)
        data = self.fbo.read(components=3, alignment=1)
        return np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)

    def render_to_file(self, path: str | Path) -> None:
        arr = self.render()
        Image.fromarray(arr, "RGB").save(str(path))


def main() -> None:
    import sys
    from scipy.spatial.transform import Rotation

    from src.rig.solver import RigSolver

    vrm_path = sys.argv[1] if len(sys.argv) > 1 else "assets/avatars/sample.vrm"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/m4_neutral.png"
    model = load_vrm(vrm_path)
    r = VRMRenderer(model, 1280, 720)
    solver = RigSolver(model)
    dt = 1.0 / 30.0
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    def _rig(yaw=0.0, pitch=0.0, roll=0.0, bs=None):
        rot = Rotation.from_euler("YXZ", [yaw, pitch, roll], degrees=True)
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = rot.as_matrix().astype(np.float32)
        return {"detected": True, "blendshapes": bs or {}, "matrix": m,
                "euler": np.array([yaw, pitch, roll], dtype=np.float32)}

    # Calibrate
    for _ in range(35):
        solver.update(_rig(), dt)

    # Neutral
    state = solver.update(_rig(), dt)
    r.apply_rig_state(state)
    r.render_to_file(out)
    print(f"wrote {out}  (neutral)")

    # 'aa' expression
    state = solver.update(_rig(bs={"jawOpen": 0.7}), dt)
    r.apply_rig_state(state)
    r.render_to_file(str(Path(out).with_name(Path(out).stem + "_aa.png")))
    print(f"wrote {Path(out).stem}_aa.png  (jawOpen=0.7)")

    # Head rotation
    state = solver.update(_rig(yaw=20, pitch=10), dt)
    r.apply_rig_state(state)
    r.render_to_file(str(Path(out).with_name(Path(out).stem + "_head.png")))
    print(f"wrote {Path(out).stem}_head.png  (yaw=20 pitch=10)")


if __name__ == "__main__":
    main()
