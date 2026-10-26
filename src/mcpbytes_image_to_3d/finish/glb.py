"""The GLB container as (JSON document, binary chunk), and typed reads of its accessors (NumPy only).

For engines that edit the customer's GLB in place rather than re-export it (renaming a rig's joints, adding an
animation) and for reading bundled clip libraries: everything they do not touch keeps its bytes. The stage validates
every input and output with `validation.glb_info`, so indices read here are in range.
"""

import json
import struct

import numpy as np

JSON_CHUNK, BIN_CHUNK = 0x4E4F534A, 0x004E4942
DTYPES = {5121: np.dtype("<u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(data):
    json_length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + json_length])
    cursor = 20 + json_length
    binary = b""
    if cursor + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, cursor)
        if kind == BIN_CHUNK:
            binary = bytes(data[cursor + 8:cursor + 8 + size])
    return doc, binary


def write_glb(doc, binary):
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<II", len(encoded), JSON_CHUNK) + encoded
    if binary:
        body += struct.pack("<II", len(binary), BIN_CHUNK) + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def read_accessor(doc, binary, index):
    """An accessor as an (count, components) array: normalized integers as floats, other integers as int64."""
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    dtype, n = DTYPES[a["componentType"]], COMPONENTS[a["type"]]
    stride = view.get("byteStride", dtype.itemsize * n)
    array = np.ndarray((a["count"], n), dtype, binary, view.get("byteOffset", 0) + a.get("byteOffset", 0), (stride, dtype.itemsize))
    return array / np.iinfo(dtype).max if a.get("normalized") else array.astype(float if dtype.kind == "f" else np.int64)
