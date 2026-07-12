"""glTF binary / structure helpers for VRM rendering."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

_COMP_DTYPE = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

_BLOB_CACHE: dict[int, bytes] = {}


def _blob(gltf: Any) -> bytes:
    key = id(gltf)
    cached = _BLOB_CACHE.get(key)
    if cached is not None:
        return cached
    bb = gltf.binary_blob()
    if bb is None:
        import base64
        b0 = gltf.buffers[0]
        bb = base64.b64decode(b0.uri.split(",", 1)[1])
    b = bytes(bb)
    _BLOB_CACHE[key] = b
    return b


def decode_accessor(gltf: Any, idx: int, *, as_float: bool = True) -> np.ndarray:
    """Decode a glTF accessor into a numpy array (shape (count, ncomp) or (count,)).

    Handles sparse accessors (used for morph-target deltas). With
    ``as_float=True`` integer data is converted to float32 and, if the accessor
    is ``normalized``, mapped to [0,1] (unsigned) or [-1,1] (signed). With
    ``as_float=False`` the raw dtype is returned (for indices).
    """
    acc = gltf.accessors[idx]
    dtype = _COMP_DTYPE[acc.componentType]
    ncomp = _NCOMP[acc.type]
    count = acc.count

    def _read_dense(bv_index: int, byte_offset: int, n: int) -> np.ndarray:
        bv = gltf.bufferViews[bv_index]
        blob = _blob(gltf)
        base = (bv.byteOffset or 0) + (byte_offset or 0)
        elem = ncomp * np.dtype(dtype).itemsize
        stride = getattr(bv, "byteStride", None) or 0
        if stride == 0 or stride == elem:
            return np.frombuffer(blob, dtype=dtype, count=n * ncomp, offset=base)
        raw = np.frombuffer(blob, dtype=np.uint8, count=n * stride, offset=base).reshape(n, stride)
        return raw[:, :elem].copy().view(dtype).reshape(-1)

    # base data (None bufferView -> zero-filled, typical for pure-sparse morph deltas)
    if acc.bufferView is None:
        arr = np.zeros((count, ncomp) if ncomp > 1 else (count,), dtype=dtype)
    else:
        arr = _read_dense(acc.bufferView, acc.byteOffset or 0, count)
        arr = arr.reshape(count, ncomp) if ncomp > 1 else arr.reshape(count)

    # sparse overlay
    sparse = getattr(acc, "sparse", None)
    if sparse is not None:
        arr = np.array(arr)  # ensure writable copy
        s_count = sparse.count
        idx_dt = _COMP_DTYPE[sparse.indices.componentType]
        idx_bv = gltf.bufferViews[sparse.indices.bufferView]
        blob = _blob(gltf)
        idx_base = (idx_bv.byteOffset or 0) + (sparse.indices.byteOffset or 0)
        sparse_idx = np.frombuffer(blob, dtype=idx_dt, count=s_count, offset=idx_base)
        sparse_vals = _read_dense(sparse.values.bufferView, sparse.values.byteOffset or 0, s_count)
        sparse_vals = sparse_vals.reshape(s_count, ncomp) if ncomp > 1 else sparse_vals.reshape(s_count)
        arr[sparse_idx] = sparse_vals

    if not as_float:
        return arr
    out = arr.astype(np.float32)
    if getattr(acc, "normalized", False):
        if np.issubdtype(dtype, np.unsignedinteger):
            out /= float(np.iinfo(dtype).max)
        else:
            info = np.iinfo(dtype)
            out = np.maximum(out / float(info.max), out / float(info.min))
    return out


def decode_texture(gltf: Any, tex_index: int | None) -> Image.Image | None:
    if tex_index is None or tex_index >= len(gltf.textures):
        return None
    tex = gltf.textures[tex_index]
    if tex.source is None:
        return None
    img = gltf.images[tex.source]
    if img.bufferView is not None:
        bv = gltf.bufferViews[img.bufferView]
        blob = _blob(gltf)
        data = blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]
        return Image.open(io.BytesIO(data)).convert("RGB")
    if img.uri and img.uri.startswith("data:"):
        import base64
        data = base64.b64decode(img.uri.split(",", 1)[1])
        return Image.open(io.BytesIO(data)).convert("RGB")
    return None


def trs_to_mat4(translation, rotation, scale) -> np.ndarray:
    t = np.asarray(translation, np.float32) if translation is not None else np.zeros(3, np.float32)
    r = np.asarray(rotation, np.float32) if rotation is not None else np.array([0, 0, 0, 1], np.float32)
    s = np.asarray(scale, np.float32) if scale is not None else np.ones(3, np.float32)
    x, y, z, w = r
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = t
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        R = np.eye(4, dtype=np.float32)
    else:
        k = 2.0 / n
        xs, ys, zs = k * x, k * y, k * z
        wx, wy, wz = k * w * x, k * w * y, k * w * z
        xx, xy, xz = k * x * x, k * x * y, k * x * z
        yy, yz, zz = k * y * y, k * y * z, k * z * z
        R = np.array([
            [1 - (yy + zz), xy - wz, xz + wy, 0],
            [xy + wz, 1 - (xx + zz), yz - wx, 0],
            [xz - wy, yz + wx, 1 - (xx + yy), 0],
            [0, 0, 0, 1],
        ], np.float32)
    S = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
    return (T @ R @ S).astype(np.float32)


def compute_world_matrices(nodes) -> list[np.ndarray]:
    """Compute bind-pose world matrices for every node (parent TRS accumulation)."""
    world: list[np.ndarray | None] = [None] * len(nodes)

    def get(i: int) -> np.ndarray:
        if world[i] is not None:
            return world[i]
        n = nodes[i]
        local = trs_to_mat4(n.translation, n.rotation, n.scale)
        if n.parent is not None:
            world[i] = get(n.parent) @ local
        else:
            world[i] = local
        return world[i]

    for i in range(len(nodes)):
        get(i)
    return world
