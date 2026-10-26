"""Bounded file checks, independent of model imports and provider success flags."""

import hashlib
import io
import json
import math
import struct
import re
import zipfile

from PIL import Image
import warnings
from pathlib import Path

from mcpbytes_core.errors import ToolError

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_MODEL_BYTES = 100 * 1024 * 1024  # input models: the largest plan upload
MAX_MODEL_FACES = 2_000_000
MAX_PIXELS = 25_000_000
MAX_OUTPUT_BYTES = 100 * 1024 * 1024


def image_info(data, require_alpha=False):
    from PIL import Image, ImageOps

    if len(data) > MAX_INPUT_BYTES:
        raise ToolError("invalid_image", "image exceeds the encoded byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ("PNG", "JPEG", "WEBP") or getattr(source, "n_frames", 1) != 1:
                    raise ToolError("invalid_image", "expected a still PNG, JPEG, or WebP image")
                if source.width * source.height > MAX_PIXELS or min(source.size) < 1:
                    raise ToolError("invalid_image", "image exceeds the decoded pixel limit")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGBA")
                minimum, maximum = image.getchannel("A").getextrema()
                if maximum == 0:
                    raise ToolError("invalid_image", "image has no visible foreground")
                if require_alpha and minimum == 255:
                    raise ToolError("invalid_alpha", "requested transparency is absent from the alpha channel")
                return image, {"width": image.width, "height": image.height, "alpha": minimum < 255}
    except ToolError:
        raise
    except Exception:
        raise ToolError("invalid_image", "image could not be decoded safely") from None


def model_info(data, max_faces=MAX_MODEL_FACES, rigged=False):
    """An input model: a self-contained, uncompressed GLB (glb_info's structural checks) that one material shades,
    without skins, animations or morph targets. Every refusal is one the customer can fix, so none reads as ours.
    `rigged`: the model must instead carry exactly one skin (animation); its earlier animations, morph targets and
    materials are then its own business, since the stage adds motion and leaves the rest of the file as it is."""
    if len(data) > MAX_MODEL_BYTES:
        raise ToolError("limit_exceeded", "the model exceeds the 100 MiB input limit")
    invalid = ToolError("invalid_mesh", "expected a self-contained, uncompressed GLB (glTF 2.0 binary with embedded textures)")
    try:
        doc = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
    except (ValueError, struct.error):
        raise invalid from None
    # Before the structural check: glb_info also validates skins, and a customer's rig refused as a malformed
    # file would hide the actionable answer.
    if not isinstance(doc, dict):
        raise invalid
    skins = {node["skin"] for node in doc.get("nodes", []) if isinstance(node, dict) and "skin" in node}
    if rigged:
        if not skins:
            raise ToolError("unsupported_rig", "the model has no skeleton; rig it first (rig_3d_model)")
        if len(skins) > 1:
            raise ToolError("unsupported_rig", "models with several skeletons are not supported")
    elif doc.get("skins") or doc.get("animations"):
        raise ToolError("unsupported_feature", "skinned or animated models are not supported; export a static mesh")
    try:
        info = glb_info(data)
    except ToolError:
        raise invalid from None
    if rigged:
        if info["faces"] > max_faces:
            raise ToolError("limit_exceeded", f"the model has {info['faces']} triangles; the limit is {max_faces}")
        return info
    primitives = [p for mesh in doc.get("meshes", []) for p in mesh.get("primitives", [])]
    if any(p.get("targets") for p in primitives):
        raise ToolError("unsupported_feature", "morph targets are not supported; apply or remove them before exporting")
    used = {p.get("material") for p in primitives}
    if len(used) > 1:
        raise ToolError("unsupported_feature", "models with several materials are not supported yet")
    material = doc.get("materials", [])[next(iter(used))] if used and next(iter(used)) is not None else {}
    if any("COLOR_0" in p["attributes"] for p in primitives) and "baseColorTexture" not in material.get("pbrMetallicRoughness", {}):
        raise ToolError("unsupported_feature", "models coloured by vertex colours are not supported yet")
    if info["faces"] > max_faces:
        raise ToolError("limit_exceeded", f"the model has {info['faces']} triangles; the limit is {max_faces}")
    return info


def is_model(name):
    """Stage inputs are typed by their declared name: .glb is a model, everything else an image."""
    return str(name or "").lower().endswith(".glb")


def validate_input(name, data, rigged=False):
    """The controller's pre-charge check and the stage's re-check of the same bytes."""
    return model_info(data, rigged=rigged) if is_model(name) else image_info(data)[1]


def glb_info(data):
    """Validate self-contained triangle geometry, embedded images and skins, including actual indices and weights."""
    if len(data) > MAX_OUTPUT_BYTES or len(data) < 20:
        raise ToolError("invalid_output", "GLB size is outside the allowed bounds")
    try:
        magic, version, length = struct.unpack_from("<4sII", data)
        if magic != b"glTF" or version != 2 or length != len(data):
            raise ValueError()
        chunks, cursor = [], 12
        while cursor < length:
            size, kind = struct.unpack_from("<II", data, cursor)
            cursor += 8
            if size % 4 or cursor + size > length:
                raise ValueError()
            chunks.append((kind, data[cursor:cursor + size]))
            cursor += size
        if len(chunks) != 2 or [c[0] for c in chunks] != [0x4E4F534A, 0x004E4942]:
            raise ValueError()
        doc, binary = json.loads(chunks[0][1]), chunks[1][1]
        if doc.get("asset", {}).get("version") != "2.0" or len(doc.get("buffers", [])) != 1:
            raise ValueError()
        buffer = doc["buffers"][0]
        if "uri" in buffer or not 0 <= buffer["byteLength"] <= len(binary):
            raise ValueError()
        binary = binary[:buffer["byteLength"]]
        views = doc.get("bufferViews", [])
        for view in views:
            start, size = view.get("byteOffset", 0), view["byteLength"]
            if view.get("buffer", 0) != 0 or min(start, size) < 0 or start + size > len(binary):
                raise ValueError()
        def values(index, expected):
            if type(index) is not int or not 0 <= index < len(doc["accessors"]):
                raise ValueError()
            accessor = doc["accessors"][index]
            if accessor["type"] != expected or "sparse" in accessor:
                raise ValueError()
            fmt, width = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}[accessor["componentType"]]
            components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[expected]
            view_index = accessor["bufferView"]
            if type(view_index) is not int or not 0 <= view_index < len(views):
                raise ValueError()
            view = views[view_index]
            count, offset = accessor["count"], accessor.get("byteOffset", 0)
            stride = view.get("byteStride", width * components)
            if count < 1 or count > 10_000_000 or offset < 0 or stride < width * components or offset + (count - 1) * stride + width * components > view["byteLength"]:
                raise ValueError()
            base = view.get("byteOffset", 0) + offset
            return accessor, (struct.unpack_from("<" + fmt * components, binary, base + i * stride) for i in range(count))
        def skin_attributes(attributes, joint_count):
            """glTF 2.0 "Skinned Mesh Attributes": JOINTS_n/WEIGHTS_n in pairs, every joint value within the skin, no
            negative weights, no joint twice with non-zero weight, and weights summing to 1 (exactly, when quantized)."""
            sets = sorted({name.split("_", 1)[1] for name in attributes if name.startswith(("JOINTS_", "WEIGHTS_"))})
            if not joint_count or any(f"JOINTS_{s}" not in attributes or f"WEIGHTS_{s}" not in attributes for s in sets):
                raise ValueError()
            totals, used = None, None
            for s in sets:
                joint_accessor, joints = values(attributes[f"JOINTS_{s}"], "VEC4")
                weight_accessor, weights = values(attributes[f"WEIGHTS_{s}"], "VEC4")
                unit = {5126: None, 5121: 255, 5123: 65535}.get(weight_accessor["componentType"], 0)
                if joint_accessor["componentType"] not in (5121, 5123) or unit == 0 or bool(weight_accessor.get("normalized")) != (unit is not None):
                    raise ValueError()
                sums, pairs = [], []
                for joint, weight in zip(joints, weights):
                    if any(j >= joint_count for j in joint) or min(weight) < 0:
                        raise ValueError()
                    sums.append(sum(weight))
                    pairs.append([j for j, w in zip(joint, weight) if w])
                totals = sums if totals is None else [a + b for a, b in zip(totals, sums)]
                used = pairs if used is None else [a + b for a, b in zip(used, pairs)]
            if any(len(set(p)) != len(p) for p in used) or any(t != unit if unit else abs(t - 1) > 1e-3 for t in totals):
                raise ValueError()

        nodes = doc.get("nodes", [])
        skins = doc.get("skins", [])
        for skin in skins:
            joints = skin["joints"]
            if not joints or len(set(joints)) != len(joints) or any(type(j) is not int or not 0 <= j < len(nodes) for j in joints):
                raise ValueError()
            if "skeleton" in skin and (type(skin["skeleton"]) is not int or not 0 <= skin["skeleton"] < len(nodes)):
                raise ValueError()
            if "inverseBindMatrices" in skin:
                accessor, matrices = values(skin["inverseBindMatrices"], "MAT4")
                if accessor["componentType"] != 5126 or accessor["count"] < len(joints) or not all(math.isfinite(n) for m in matrices for n in m):
                    raise ValueError()
        # A skin is instantiated by a node with both a mesh and a skin; JOINTS_n index that skin's joints.
        joint_counts = {}
        for node in nodes:
            if "skin" in node:
                if type(node["skin"]) is not int or not 0 <= node["skin"] < len(skins) or "mesh" not in node:
                    raise ValueError()
                count = len(skins[node["skin"]]["joints"])
                joint_counts[node["mesh"]] = min(count, joint_counts.get(node["mesh"], count))
        vertices, faces, primitives = 0, 0, 0
        for mesh_index, mesh in enumerate(doc.get("meshes", [])):
            for primitive in mesh["primitives"]:
                if primitive.get("mode", 4) != 4 or primitive.get("extensions"):
                    raise ValueError()
                pos, points = values(primitive["attributes"]["POSITION"], "VEC3")
                if pos["componentType"] != 5126 or not all(math.isfinite(n) for point in points for n in point):
                    raise ValueError()
                for name, index in primitive["attributes"].items():
                    if name == "POSITION":
                        continue
                    if type(index) is not int or not 0 <= index < len(doc["accessors"]):
                        raise ValueError()
                    attribute, entries = values(index, doc["accessors"][index]["type"])
                    if attribute["count"] != pos["count"] or not all(math.isfinite(n) for entry in entries for n in entry):
                        raise ValueError()
                if any(name.startswith(("JOINTS_", "WEIGHTS_")) for name in primitive["attributes"]):
                    skin_attributes(primitive["attributes"], joint_counts.get(mesh_index, 0))
                count = pos["count"]
                if "indices" in primitive:
                    indices, numbers = values(primitive["indices"], "SCALAR")
                    if indices["componentType"] not in (5121, 5123, 5125) or not all(0 <= n[0] < count for n in numbers):
                        raise ValueError()
                    count = indices["count"]
                if count % 3:
                    raise ValueError()
                if "material" in primitive and not 0 <= primitive["material"] < len(doc.get("materials", [])):
                    raise ValueError()
                vertices += pos["count"]
                faces += count // 3
                primitives += 1
        if not faces:
            raise ValueError()
        textures = []
        for embedded in doc.get("images", []):
            image_view = embedded["bufferView"]
            if "uri" in embedded or type(image_view) is not int or not 0 <= image_view < len(views):
                raise ValueError()
            view = views[image_view]
            start = view.get("byteOffset", 0)
            _, info = image_info(binary[start:start + view["byteLength"]])
            textures.append(info)
        for texture in doc.get("textures", []):
            if type(texture.get("source")) is not int or not 0 <= texture["source"] < len(textures):
                raise ValueError()
        textured = False
        for material in doc.get("materials", []):
            pbr = material.get("pbrMetallicRoughness", {})
            for source, name in ((material, "normalTexture"), (material, "occlusionTexture"), (material, "emissiveTexture"), (pbr, "baseColorTexture"), (pbr, "metallicRoughnessTexture")):
                if name in source:
                    index = source[name]["index"]
                    if type(index) is not int or not 0 <= index < len(doc.get("textures", [])):
                        raise ValueError()
                    textured = True
        for node in nodes:
            if "mesh" in node and (type(node["mesh"]) is not int or not 0 <= node["mesh"] < len(doc.get("meshes", []))):
                raise ValueError()
            if any(type(index) is not int or not 0 <= index < len(nodes) for index in node.get("children", [])):
                raise ValueError()
            for name, size in (("matrix", 16), ("translation", 3), ("rotation", 4), ("scale", 3)):
                if name in node and (len(node[name]) != size or not all(math.isfinite(n) for n in node[name])):
                    raise ValueError()
        return {"vertices": vertices, "faces": faces, "primitives": primitives, "textures": textures, "textured": textured, "self_contained": True, "joints": sum(len(s["joints"]) for s in skins)}
    except (ValueError, TypeError, KeyError, IndexError, struct.error, ToolError):
        raise ToolError("invalid_output", "generated GLB failed structural validation") from None


def web_glb_info(data):
    """Validate the web copy: one Draco primitive that decodes, WebP images, nothing external."""
    if len(data) > MAX_OUTPUT_BYTES or len(data) < 20:
        raise ToolError("invalid_output", "GLB size is outside the allowed bounds")
    try:
        import DracoPy
        import numpy as np  # here, not at module level: the media controller image has no NumPy (2026-09-22 outage)

        magic, version, length = struct.unpack_from("<4sII", data)
        if magic != b"glTF" or version != 2 or length != len(data):
            raise ValueError()
        chunks, cursor = [], 12
        while cursor < length:
            size, kind = struct.unpack_from("<II", data, cursor)
            cursor += 8
            if size % 4 or cursor + size > length:
                raise ValueError()
            chunks.append((kind, data[cursor:cursor + size]))
            cursor += size
        if len(chunks) != 2 or [c[0] for c in chunks] != [0x4E4F534A, 0x004E4942]:
            raise ValueError()
        doc, binary = json.loads(chunks[0][1]), chunks[1][1]
        if doc.get("asset", {}).get("version") != "2.0" or len(doc.get("buffers", [])) != 1 or "uri" in doc["buffers"][0]:
            raise ValueError()
        if set(doc.get("extensionsRequired", [])) - {"KHR_draco_mesh_compression", "EXT_texture_webp"}:
            raise ValueError()
        views = doc.get("bufferViews", [])
        for view in views:
            start, size = view.get("byteOffset", 0), view["byteLength"]
            if view.get("buffer", 0) != 0 or min(start, size) < 0 or start + size > len(binary):
                raise ValueError()
        primitives = [p for m in doc.get("meshes", []) for p in m["primitives"]]
        if len(primitives) != 1 or primitives[0].get("mode", 4) != 4:
            raise ValueError()
        draco = primitives[0]["extensions"]["KHR_draco_mesh_compression"]
        view = views[draco["bufferView"]]
        decoded = DracoPy.decode(bytes(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]))
        vertices, faces = int(len(decoded.points)), int(len(decoded.faces))
        if not 0 < faces <= 10_000_000 or not np.isfinite(decoded.points).all():
            raise ValueError()
        textures = []
        for image in doc.get("images", []):
            if image.get("mimeType") != "image/webp" or "uri" in image:
                raise ValueError()
            view = views[image["bufferView"]]
            with Image.open(io.BytesIO(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]])) as picture:
                picture.load()
                textures.append({"width": picture.width, "height": picture.height, "alpha": "A" in picture.getbands()})
        material = doc.get("materials", [{}])[0]
        textured = bool(textures) and "baseColorTexture" in material.get("pbrMetallicRoughness", {})
    except ToolError:
        raise
    except Exception:
        raise ToolError("invalid_output", "web GLB failed validation")
    return {"vertices": vertices, "faces": faces, "primitives": 1, "textures": textures, "textured": textured, "self_contained": True, "compression": "draco"}


def artifact(path, kind, require_alpha=False, compressed=False):
    path = Path(path)
    data = path.read_bytes()
    if len(data) > MAX_OUTPUT_BYTES:
        raise ToolError("invalid_output", "generated file exceeds the output limit")
    if kind == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ToolError("invalid_output", "generated image must be encoded as PNG")
    if kind == "application/json":
        try:
            if len(data) > 1024 * 1024 or not isinstance(json.loads(data), dict):
                raise ValueError()
        except (ValueError, UnicodeError):
            raise ToolError("invalid_output", "invalid JSON artifact") from None
        info = {"json": True}
    elif kind == "application/zip":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = archive.infolist()
                if not 1 <= len(members) <= 100 or len({m.filename for m in members}) != len(members):
                    raise ValueError()
                if sum(m.file_size for m in members) > MAX_OUTPUT_BYTES:
                    raise ValueError()
                for member in members:
                    if (member.orig_filename != member.filename or "\\" in member.orig_filename
                            or not re.fullmatch(r"[a-zA-Z0-9_-][a-zA-Z0-9_.-]*(?:/[a-zA-Z0-9_-][a-zA-Z0-9_.-]*)?", member.filename)
                            or member.flag_bits & 1 or (member.external_attr >> 16) & 0o170000 == 0o120000
                            or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                        raise ValueError()
                if archive.testzip() is not None:
                    raise ValueError()
                info = {"files": len(members), "uncompressed_bytes": sum(m.file_size for m in members)}
        except (ValueError, zipfile.BadZipFile, RuntimeError, OSError):
            raise ToolError("invalid_output", "invalid or unbounded ZIP artifact") from None
    elif kind in ("image/png", "model/gltf-binary"):
        info = image_info(data, require_alpha)[1] if kind == "image/png" else web_glb_info(data) if compressed else glb_info(data)
    else:
        raise ToolError("invalid_output", "unsupported artifact type")
    return {"name": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "content_type": kind, "validation": info}
