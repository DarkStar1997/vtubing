"""VRM 1.0 loader.

Parses a VRM 1.0 file (a glTF 2.0 binary / .glb carrying the ``VRMC_vrm``
extension) into a structured :class:`VRMModel`. Extracts:

  * humanoid bone -> glTF node mapping
  * expressions (preset + custom) with morph-target bindings
  * lookAt configuration (eye gaze)
  * mesh / morph-target metadata (incl. ARKit morph-name detection)

Binary accessor data (vertices, indices, skinning) is left in the underlying
glTF object for the M3 renderer to decode; this module only parses structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pygltflib import GLTF2

from src.constants import ARKIT_BLENDSHAPE_NAMES


@dataclass
class MorphTargetBind:
    node: int       # glTF node index
    index: int      # morph target index within that node's mesh primitive targets
    weight: float


@dataclass
class Expression:
    name: str
    preset: str
    is_custom: bool
    morph_binds: list[MorphTargetBind]
    is_binary: bool = False
    override_blink: str | None = None
    override_lookat: str | None = None
    override_mouth: str | None = None

    @property
    def morph_target_count(self) -> int:
        return len(self.morph_binds)


@dataclass
class HumanoidBone:
    name: str
    node: int


@dataclass
class LookAtConfig:
    type: str | None  # 'bone' | 'expression'
    offset_from_head: np.ndarray
    range_maps: dict[str, dict[str, float]]


@dataclass
class NodeInfo:
    index: int
    name: str | None
    parent: int | None
    children: list[int]
    mesh: int | None
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray


@dataclass
class MeshInfo:
    index: int
    primitives: int
    morph_targets: int
    target_names: list[str]
    weights: list[float] | None
    vertex_count: int


@dataclass
class VRMModel:
    path: Path
    spec_version: str
    meta: dict
    nodes: list[NodeInfo]
    meshes: list[MeshInfo]
    humanoid_bones: list[HumanoidBone]
    humanoid_bone_by_name: dict[str, int]
    expressions: list[Expression]
    expression_by_name: dict[str, Expression]
    look_at: LookAtConfig
    arkit_morph_names: list[str]  # morph target names matching ARKit (M4 fast path)
    gltf: Any  # pygltflib.GLTF2, kept for M3 binary decode


def _extract_target_names(mesh: Any) -> list[str]:
    extras = getattr(mesh, "extras", None)
    if extras is None:
        return []
    tn = extras.get("targetNames") if isinstance(extras, dict) else getattr(extras, "targetNames", None)
    if isinstance(tn, list):
        return [str(x) for x in tn]
    return []


def load_vrm(path: str | Path) -> VRMModel:
    path = Path(path)
    gltf = GLTF2.load_binary(str(path))

    vrmc = (gltf.extensions or {}).get("VRMC_vrm")
    if vrmc is None:
        raise ValueError(f"{path}: not a VRM 1.0 model (no VRMC_vrm extension)")

    spec_version = vrmc.get("specVersion", "?")
    meta = vrmc.get("meta", {})

    # nodes (build parent map from children lists)
    parent_of: dict[int, int] = {}
    for i, n in enumerate(gltf.nodes):
        for c in (n.children or []):
            parent_of[c] = i
    nodes: list[NodeInfo] = []
    for i, n in enumerate(gltf.nodes):
        nodes.append(NodeInfo(
            index=i,
            name=n.name,
            parent=parent_of.get(i),
            children=list(n.children or []),
            mesh=n.mesh,
            translation=(np.array(n.translation, dtype=np.float32) if n.translation else np.zeros(3, np.float32)),
            rotation=(np.array(n.rotation, dtype=np.float32) if n.rotation else np.array([0, 0, 0, 1], np.float32)),
            scale=(np.array(n.scale, dtype=np.float32) if n.scale else np.ones(3, np.float32)),
        ))

    # meshes / morph targets
    meshes: list[MeshInfo] = []
    for mi, m in enumerate(gltf.meshes):
        max_targets = 0
        max_verts = 0
        for p in m.primitives:
            t = len(p.targets) if p.targets else 0
            if t > max_targets:
                max_targets = t
            pos_acc = getattr(p.attributes, "POSITION", None)
            if pos_acc is not None:
                vc = gltf.accessors[pos_acc].count
                if vc > max_verts:
                    max_verts = vc
        meshes.append(MeshInfo(
            index=mi,
            primitives=len(m.primitives),
            morph_targets=max_targets,
            target_names=_extract_target_names(m),
            weights=(list(m.weights) if m.weights else None),
            vertex_count=max_verts,
        ))

    # humanoid bones
    hb_raw = vrmc.get("humanoid", {}).get("humanBones", {}) or {}
    humanoid_bones: list[HumanoidBone] = []
    humanoid_bone_by_name: dict[str, int] = {}
    for bone_name, entry in hb_raw.items():
        node = int(entry["node"])
        humanoid_bones.append(HumanoidBone(name=bone_name, node=node))
        humanoid_bone_by_name[bone_name] = node

    # expressions (preset + custom)
    expressions: list[Expression] = []
    expression_by_name: dict[str, Expression] = {}

    def _parse(name: str, spec: dict, is_custom: bool) -> None:
        binds_raw = spec.get("morphTargetBinds", []) or []
        binds = [
            MorphTargetBind(node=int(b["node"]), index=int(b["index"]), weight=float(b.get("weight", 1.0)))
            for b in binds_raw
        ]
        expr = Expression(
            name=name,
            preset=spec.get("preset", name),
            is_custom=is_custom,
            morph_binds=binds,
            is_binary=bool(spec.get("isBinary", False)),
            override_blink=spec.get("overrideBlink"),
            override_lookat=spec.get("overrideLookAt"),
            override_mouth=spec.get("overrideMouth"),
        )
        expressions.append(expr)
        expression_by_name[name] = expr

    ext_block = vrmc.get("expressions", {}) or {}
    for name, spec in (ext_block.get("preset", {}) or {}).items():
        _parse(name, spec, is_custom=False)
    for name, spec in (ext_block.get("custom", {}) or {}).items():
        _parse(name, spec, is_custom=True)

    # lookAt
    la = vrmc.get("lookAt", {}) or {}
    range_maps = {}
    for key, raw in {
        "horizontalInner": la.get("rangeMapHorizontalInner", {}),
        "horizontalOuter": la.get("rangeMapHorizontalOuter", {}),
        "verticalDown": la.get("rangeMapVerticalDown", {}),
        "verticalUp": la.get("rangeMapVerticalUp", {}),
    }.items():
        range_maps[key] = {kk: float(vv) for kk, vv in (raw or {}).items()}
    look_at = LookAtConfig(
        type=la.get("type"),
        offset_from_head=np.array(la.get("offsetFromHeadBone", [0.0, 0.0, 0.0]), dtype=np.float32),
        range_maps=range_maps,
    )

    # ARKit morph-name detection (fast path for M4 direct mapping)
    arkit_set = set(ARKIT_BLENDSHAPE_NAMES)
    arkit_morph_names: list[str] = []
    for msh in meshes:
        for tn in msh.target_names:
            if tn in arkit_set and tn not in arkit_morph_names:
                arkit_morph_names.append(tn)

    return VRMModel(
        path=path,
        spec_version=spec_version,
        meta=meta,
        nodes=nodes,
        meshes=meshes,
        humanoid_bones=humanoid_bones,
        humanoid_bone_by_name=humanoid_bone_by_name,
        expressions=expressions,
        expression_by_name=expression_by_name,
        look_at=look_at,
        arkit_morph_names=arkit_morph_names,
        gltf=gltf,
    )


def summarize(model: VRMModel) -> str:
    lines: list[str] = []
    m = model.meta
    lines.append(f"VRM {model.spec_version}  {model.path.name}")
    lines.append(
        f"  meta: name={m.get('name')!r} authors={m.get('authors')} "
        f"licenseUrl={m.get('licenseUrl')!r}"
    )
    lines.append(
        f"        allowRedistribution={m.get('allowRedistribution')} "
        f"commercialUsage={m.get('commercialUsage')}"
    )
    lines.append(f"  nodes: {len(model.nodes)}  meshes: {len(model.meshes)}")
    for msh in model.meshes:
        if msh.morph_targets or msh.target_names:
            lines.append(
                f"    mesh[{msh.index}]: prims={msh.primitives} verts={msh.vertex_count} "
                f"morphTargets={msh.morph_targets} targetNames={len(msh.target_names)}"
            )
    key_bones = [
        "hips", "spine", "chest", "upperChest", "neck", "head",
        "leftEye", "rightEye", "jaw", "leftUpperArm", "rightUpperArm",
    ]
    lines.append(f"  humanoid: {len(model.humanoid_bones)} bones")
    for bn in key_bones:
        if bn in model.humanoid_bone_by_name:
            ni = model.humanoid_bone_by_name[bn]
            lines.append(f"    {bn:14s} -> node {ni:3d} ({model.nodes[ni].name})")
    lines.append(f"  expressions: {len(model.expressions)}")
    for e in model.expressions:
        tag = "custom" if e.is_custom else "preset"
        extra = []
        if e.is_binary:
            extra.append("binary")
        if e.override_blink:
            extra.append(f"blink={e.override_blink}")
        if e.override_mouth:
            extra.append(f"mouth={e.override_mouth}")
        lines.append(
            f"    {e.name:12s} [{tag:6s}] morphBinds={e.morph_target_count} {' '.join(extra)}"
        )
    la = model.look_at
    lines.append(f"  lookAt: type={la.type} offset={la.offset_from_head.tolist()}")
    for k, v in la.range_maps.items():
        lines.append(
            f"    {k:16s} inputMax={v.get('inputMaxValue')} outputScale={v.get('outputScale')}"
        )
    lines.append(f"  ARKit morph names found: {len(model.arkit_morph_names)}")
    if model.arkit_morph_names:
        preview = model.arkit_morph_names[:12]
        more = " ..." if len(model.arkit_morph_names) > 12 else ""
        lines.append(f"    {preview}{more}")
    return "\n".join(lines)


def main() -> None:
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/avatars/sample.vrm"
    model = load_vrm(path)
    print(summarize(model))


if __name__ == "__main__":
    main()
