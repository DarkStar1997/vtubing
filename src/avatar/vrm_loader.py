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


# VRM 0.x blendshape preset → VRM 1.0 expression name
_VRM0X_PRESET_MAP: dict[str, str] = {
    "neutral": "neutral",
    "a": "aa",
    "i": "ih",
    "u": "ou",
    "e": "ee",
    "o": "oh",
    "blink": "blink",
    "blink_l": "blinkLeft",
    "blink_r": "blinkRight",
    "angry": "angry",
    "fun": "relaxed",
    "joy": "happy",
    "sorrow": "sad",
}


def _extract_target_names(mesh: Any) -> list[str]:
    extras = getattr(mesh, "extras", None)
    if extras is None:
        return []
    tn = extras.get("targetNames") if isinstance(extras, dict) else getattr(extras, "targetNames", None)
    if isinstance(tn, list):
        return [str(x) for x in tn]
    return []


def _build_nodes(gltf: Any) -> list[NodeInfo]:
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
    return nodes


def _build_meshes(gltf: Any) -> list[MeshInfo]:
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
    return meshes


def _detect_arkit_morph_names(meshes: list[MeshInfo]) -> list[str]:
    arkit_set = set(ARKIT_BLENDSHAPE_NAMES)
    names: list[str] = []
    for msh in meshes:
        for tn in msh.target_names:
            if tn in arkit_set and tn not in names:
                names.append(tn)
    return names


def _parse_vrm0x(ext: dict, gltf: Any, nodes: list[NodeInfo], meshes: list[MeshInfo]) -> tuple[
    str, dict, list[HumanoidBone], dict[str, int], list[Expression], dict[str, Expression], LookAtConfig,
]:
    """Parse VRM 0.x extension → same dataclass structure as VRM 1.0."""

    meta_raw = ext.get("meta", {})
    meta = {
        "name": meta_raw.get("title", ""),
        "authors": [meta_raw.get("author", "")] if meta_raw.get("author") else [],
        "licenseUrl": meta_raw.get("otherLicenseUrl", ""),
        "allowRedistribution": meta_raw.get("redistribution", False),
        "commercialUsage": "corporation" if meta_raw.get("commercialUssageName") == "Allow" else "personalNonProfit",
    }

    # humanoid: list of {bone, node} → HumanoidBone list
    hb_raw = ext.get("humanoid", {}).get("humanBones", []) or []
    humanoid_bones: list[HumanoidBone] = []
    humanoid_bone_by_name: dict[str, int] = {}
    for entry in hb_raw:
        bn = entry.get("bone", "")
        ni = int(entry.get("node", -1))
        humanoid_bones.append(HumanoidBone(name=bn, node=ni))
        humanoid_bone_by_name[bn] = ni

    # mesh index → node index mapping (VRM 0.x binds reference meshes, not nodes)
    mesh_to_node: dict[int, int] = {}
    for ni, node in enumerate(nodes):
        if node.mesh is not None:
            mesh_to_node[node.mesh] = ni

    # blendShapeGroups → Expression list (VRM 1.0 semantics)
    groups = ext.get("blendShapeMaster", {}).get("blendShapeGroups", []) or []
    expressions: list[Expression] = []
    expression_by_name: dict[str, Expression] = {}

    for grp in groups:
        raw_preset = grp.get("presetName", "unknown")
        grp_name = grp.get("name", raw_preset)
        expr_name = _VRM0X_PRESET_MAP.get(raw_preset, "")
        is_custom = False
        if not expr_name:
            lname = grp_name.lower().replace(" ", "")
            expr_name = lname
            is_custom = True

        binds_raw = grp.get("binds", []) or []
        morph_binds: list[MorphTargetBind] = []
        for b in binds_raw:
            mesh_idx = int(b.get("mesh", -1))
            node_idx = mesh_to_node.get(mesh_idx, -1)
            weight = float(b.get("weight", 100.0)) / 100.0  # 0-100 → 0-1
            if node_idx >= 0:
                morph_binds.append(MorphTargetBind(node=node_idx, index=int(b["index"]), weight=weight))

        expr = Expression(
            name=expr_name,
            preset=expr_name,
            is_custom=is_custom,
            morph_binds=morph_binds,
            is_binary=bool(grp.get("isBinary", False)),
            override_blink=None,
            override_lookat=None,
            override_mouth=None,
        )
        expressions.append(expr)
        expression_by_name[expr_name] = expr

    # lookAt
    fp = ext.get("firstPerson", {}) or {}
    la_type_raw = fp.get("lookAtTypeName", "Bone")
    la_type = "bone" if la_type_raw == "Bone" else "expression"

    range_maps: dict[str, dict[str, float]] = {}
    for key, raw_key in [
        ("horizontalInner", "lookAtHorizontalInner"),
        ("horizontalOuter", "lookAtHorizontalOuter"),
        ("verticalDown", "lookAtVerticalDown"),
        ("verticalUp", "lookAtVerticalUp"),
    ]:
        raw = fp.get(raw_key, {}) or {}
        range_maps[key] = {
            "inputMaxValue": float(raw.get("xRange", 90.0)),
            "outputScale": float(raw.get("yRange", 10.0)),
        }

    offset_raw = fp.get("firstPersonBoneOffset", {})
    if isinstance(offset_raw, dict):
        offset = [offset_raw.get("x", 0.0), offset_raw.get("y", 0.0), offset_raw.get("z", 0.0)]
    else:
        offset = offset_raw if offset_raw else [0.0, 0.0, 0.0]
    look_at = LookAtConfig(
        type=la_type,
        offset_from_head=np.array(offset, dtype=np.float32),
        range_maps=range_maps,
    )

    spec_version = f"0.x ({ext.get('specVersion', '?')})"
    return spec_version, meta, humanoid_bones, humanoid_bone_by_name, expressions, expression_by_name, look_at


def _parse_vrm1x(vrmc: dict) -> tuple[
    str, dict, list[HumanoidBone], dict[str, int], list[Expression], dict[str, Expression], LookAtConfig,
]:
    """Parse VRM 1.0 (VRMC_vrm) extension."""

    spec_version = vrmc.get("specVersion", "?")
    meta = vrmc.get("meta", {})

    # humanoid
    hb_raw = vrmc.get("humanoid", {}).get("humanBones", {}) or {}
    humanoid_bones: list[HumanoidBone] = []
    humanoid_bone_by_name: dict[str, int] = {}
    for bone_name, entry in hb_raw.items():
        node = int(entry["node"])
        humanoid_bones.append(HumanoidBone(name=bone_name, node=node))
        humanoid_bone_by_name[bone_name] = node

    # expressions
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

    return spec_version, meta, humanoid_bones, humanoid_bone_by_name, expressions, expression_by_name, look_at


def load_vrm(path: str | Path) -> VRMModel:
    path = Path(path)
    gltf = GLTF2.load_binary(str(path))

    exts = gltf.extensions or {}
    vrmc = exts.get("VRMC_vrm")
    vrm0x = exts.get("VRM")

    if vrmc is not None:
        spec_version, meta, humanoid_bones, humanoid_bone_by_name, \
            expressions, expression_by_name, look_at = _parse_vrm1x(vrmc)
    elif vrm0x is not None:
        nodes = _build_nodes(gltf)
        meshes = _build_meshes(gltf)
        spec_version, meta, humanoid_bones, humanoid_bone_by_name, \
            expressions, expression_by_name, look_at = _parse_vrm0x(vrm0x, gltf, nodes, meshes)
    else:
        raise ValueError(f"{path}: not a VRM model (no VRMC_vrm or VRM extension)")

    # shared: nodes, meshes (may already be built for 0.x path)
    if vrmc is not None:
        nodes = _build_nodes(gltf)
        meshes = _build_meshes(gltf)

    # ARKit morph-name detection (fast path for M4 direct mapping)
    arkit_morph_names = _detect_arkit_morph_names(meshes)

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
