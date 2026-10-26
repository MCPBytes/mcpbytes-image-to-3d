"""Finishing pass for generated models: clean, de-litter, normalize pose, smooth normals across UV seams, tame
glossy materials, and write a web-delivery copy (Draco geometry + WebP textures: the base colour lossy, the data
maps lossless). Geometry is never smoothed or simplified.

Runs in the model stage after the engine writes its GLB and before validation. Both engines emit one
textured primitive; anything else is left untouched and reported.
"""

import io
import json
import struct

import numpy as np

# Below this share of the total surface, a disconnected piece is debris (a stray drop, a speck), not the object.
DEBRIS_FRACTION = 0.01
# Non-metal surfaces darker than this roughness look wet or glossy; both engines under-estimate roughness.
ROUGHNESS_FLOOR = 0.35
# Base colour only: lossy WebP's block artefacts pass for texture there, but in a metallic-roughness, normal,
# occlusion or emissive map they are shading errors, so every other image is stored lossless.
WEBP_QUALITY = 90
# Lossless WebP's effort knob (0-100): above 50 a 2048² map shrinks a fraction of a percent for twice the time.
WEBP_LOSSLESS_EFFORT = 50


def _single_mesh(path):
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    if len(scene.geometry) != 1:
        return None
    name, mesh = next(iter(scene.geometry.items()))
    node = next(n for n in scene.graph.nodes_geometry if scene.graph[n][1] == name)
    transform = scene.graph[node][0]
    if not np.allclose(transform, np.eye(4)):
        mesh.apply_transform(transform)
    return mesh


def _rebuilt(mesh, faces_mask=None):
    """A fresh Trimesh with the same UVs and material, stored normals dropped (`finish_model` sets its own)."""
    import trimesh

    if faces_mask is not None:
        mesh.update_faces(faces_mask)
        mesh.remove_unreferenced_vertices()
    visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=mesh.visual.material)
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=visual, process=False)


def _roughness_floor(material, floor):
    """Raise roughness of non-metal texels/factors to `floor`; returns how many texels changed."""
    from PIL import Image

    texture = getattr(material, "metallicRoughnessTexture", None)
    if texture is not None:
        arr = np.array(texture.convert("RGB"))
        low = (arr[:, :, 2] < 128) & (arr[:, :, 1] < floor * 255)  # B = metallic, G = roughness
        arr[low, 1] = int(round(floor * 255))
        material.metallicRoughnessTexture = Image.fromarray(arr, "RGB")
        return int(low.sum())
    metallic = material.metallicFactor if material.metallicFactor is not None else 1.0
    roughness = material.roughnessFactor if material.roughnessFactor is not None else 1.0
    if metallic < 0.5 and roughness < floor:
        material.roughnessFactor = floor
        return 1
    return 0


def finish_outputs(folder, names):
    """Finish the primary model among the declared outputs; `<name>.web.glb` is written when declared."""
    models = [n for n in names if n.endswith(".glb") and not n.endswith(".web.glb")]
    if len(models) != 1 or not (folder / models[0]).is_file():
        return None
    stem = models[0][:-4]
    web = f"{stem}.web.glb"
    return finish_model(folder / models[0], web_copy=(folder / web) if web in names else None)


def finish_model(path, web_copy=None, debris_fraction=DEBRIS_FRACTION, roughness_floor=ROUGHNESS_FLOOR):
    """Finish the GLB at `path` in place and optionally write the web copy. Returns a summary dict."""
    import trimesh

    mesh = _single_mesh(path)
    if mesh is None or not isinstance(getattr(mesh.visual, "uv", None), np.ndarray):
        return {"applied": False, "reason": "not a single textured primitive"}
    before = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "extents": [round(float(x), 6) for x in mesh.extents]}
    # 1. Weld coincident vertices that also share UVs (seams stay), drop degenerate and duplicate faces.
    mesh.merge_vertices(merge_tex=False, merge_norm=False)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    # 2. Debris: keep every connected piece worth at least `debris_fraction` of the surface, always the largest.
    # Connectivity is judged on geometry alone (UV seams split vertices, so the textured mesh's own adjacency
    # would call every UV island a separate piece); face indices are shared with `mesh`.
    welded = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    components = trimesh.graph.connected_components(welded.face_adjacency, nodes=np.arange(len(mesh.faces)))
    areas = mesh.area_faces
    total = float(areas.sum()) or 1.0
    sizes = [float(areas[c].sum()) for c in components]
    keep = np.zeros(len(mesh.faces), dtype=bool)
    largest = int(np.argmax(sizes)) if sizes else -1
    for i, component in enumerate(components):
        if i == largest or sizes[i] >= debris_fraction * total:
            keep[component] = True
    removed_components = sum(1 for i in range(len(components)) if i != largest and sizes[i] < debris_fraction * total)
    mesh = _rebuilt(mesh, keep if removed_components else None)
    # 3. Pose: centred on x/z, standing on y = 0 (glTF is +Y up), largest extent 1 (outputs are unitless).
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    mesh.apply_translation([-center[0], -bounds[0][1], -center[2]])
    scale = float(mesh.extents.max())
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    # 4. Normals: one per position, so UV-seam copies shade as one smooth surface (trimesh's per-index normals give
    # each copy only its own side's triangles: a hard crease along every seam). Never the engine's: Hunyuan writes
    # none and TRELLIS's are per index. Set after the pose so no transform touches them; trimesh's GLB export
    # (4.4 to 5.1) and `write_web_copy` both write this cached array as it is.
    from .normals import welded_vertex_normals  # here, not at the top: the simplify image ships this file alone

    mesh.vertex_normals = welded_vertex_normals(mesh.vertices, mesh.faces)
    # 5. Materials: non-metal roughness floor.
    texels = _roughness_floor(mesh.visual.material, roughness_floor)
    data = mesh.export(file_type="glb", include_normals=True)
    path.write_bytes(data)
    summary = {"applied": True, "before": before, "after": {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))},
               "debris_components_removed": removed_components, "roughness_texels_raised": texels, "roughness_floor": roughness_floor,
               "pose": "centred, grounded at y=0, largest extent 1", "normals": "area-weighted, shared across UV seams"}
    if web_copy is not None:
        summary["web_copy"] = write_web_copy(mesh, data, web_copy)
    return summary


def _draco_ids(buffer):
    """Unique attribute ids the encoder assigned (glTF's extension must name them), plus the counts."""
    import DracoPy

    decoded = DracoPy.decode(buffer)
    ids = {}
    for name, kind in (("POSITION", "POSITION"), ("NORMAL", "NORMAL"), ("TEXCOORD_0", "TEX_COORD")):
        attribute = decoded.get_attribute_by_type(getattr(DracoPy.AttributeType, kind))
        if attribute is not None:
            ids[name] = int(attribute["unique_id"])
    return ids, int(len(decoded.points)), int(len(decoded.faces))


def write_web_copy(mesh, glb, path, quality=WEBP_QUALITY):
    """`model.web.glb`: the same mesh with Draco geometry and WebP textures (base colour at `quality`, data maps
    lossless), for browsers and engines. Normals are `mesh.vertex_normals`, as the main export wrote them."""
    import DracoPy
    from PIL import Image

    doc, binary = _chunks(glb)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float64)  # DracoPy insists on float64 here
    uv = np.ascontiguousarray(mesh.visual.uv, dtype=np.float64)
    uv[:, 1] = 1.0 - uv[:, 1]  # trimesh keeps UVs bottom-left; glTF's origin is top-left
    draco = DracoPy.encode(vertices, faces=faces, quantization_bits=14, compression_level=7, normals=normals, tex_coord=uv)
    ids, points, triangles = _draco_ids(draco)
    if triangles != len(faces) or set(ids) < {"POSITION", "NORMAL", "TEXCOORD_0"}:
        raise ValueError("draco round-trip mismatch")
    body, views = bytearray(), []
    def add(blob):
        offset = len(body)
        body.extend(blob)
        body.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
    # Draco payload first, then each image re-encoded as WebP. Base colour is found through the materials (glTF does
    # not order images by role); an image nothing uses as base colour is a data map and stays exact.
    add(bytes(draco))
    sources = doc.get("textures", [])
    colour = {sources[m["pbrMetallicRoughness"]["baseColorTexture"]["index"]].get("source") for m in doc.get("materials", [])
              if "baseColorTexture" in m.get("pbrMetallicRoughness", {})}
    images = []
    for index, image in enumerate(doc.get("images", [])):
        view = doc["bufferViews"][image["bufferView"]]
        raw = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
        out = io.BytesIO()
        if index in colour:
            Image.open(io.BytesIO(raw)).save(out, "WEBP", quality=quality, method=4)
        else:  # exact: keep the RGB under zero alpha too (an occlusion or roughness value is not "invisible")
            Image.open(io.BytesIO(raw)).save(out, "WEBP", lossless=True, quality=WEBP_LOSSLESS_EFFORT, method=4, exact=True)
        add(out.getvalue())
        images.append({"mimeType": "image/webp", "bufferView": len(views) - 1})
    textures = [{k: v for k, v in t.items() if k != "source"} | {"extensions": {"EXT_texture_webp": {"source": t["source"]}}} for t in sources]
    materials, samplers = doc.get("materials", [])[:1], doc.get("samplers", [])
    # What the copied material, textures and samplers use (KHR_texture_transform, KHR_materials_*) must be declared;
    # only Draco and WebP are required (validation.py `web_glb_info` refuses any other required extension).
    required = ["KHR_draco_mesh_compression", "EXT_texture_webp"]
    used = sorted(set(required) | _extension_names([materials, textures, samplers]))
    lo, hi = vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()
    accessors = [
        {"componentType": 5126, "count": int(len(vertices)), "type": "VEC3", "min": lo, "max": hi},
        {"componentType": 5126, "count": int(len(vertices)), "type": "VEC3"},
        {"componentType": 5126, "count": int(len(vertices)), "type": "VEC2"},
        {"componentType": 5125, "count": int(len(faces)) * 3, "type": "SCALAR"},
    ]
    primitive = {"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "mode": 4,
                 "extensions": {"KHR_draco_mesh_compression": {"bufferView": 0, "attributes": {"POSITION": ids["POSITION"], "NORMAL": ids["NORMAL"], "TEXCOORD_0": ids["TEXCOORD_0"]}}}}
    if doc.get("materials"):
        primitive["material"] = 0
    body = bytes(body)
    out = {"asset": {"version": "2.0", "generator": "mcpbytes finish (draco + webp)"},
           "extensionsUsed": used, "extensionsRequired": required,
           "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}], "meshes": [{"primitives": [primitive]}],
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(body)}],
           "materials": materials, "textures": textures, "images": images, "samplers": samplers}
    payload = _pack(out, body)
    path.write_bytes(payload)
    return {"bytes": len(payload), "vertices": points, "faces": triangles, "textures": len(images)}


def _extension_names(node):
    """Every extension name used anywhere inside parsed glTF JSON `node`."""
    if isinstance(node, list):
        return set().union(*map(_extension_names, node))
    if isinstance(node, dict):
        return set(node.get("extensions", {})) | set().union(*map(_extension_names, node.values()))
    return set()


def _chunks(glb):
    magic, version, length = struct.unpack_from("<4sII", glb)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glb")
    cursor, parts = 12, {}
    while cursor < length:
        size, kind = struct.unpack_from("<II", glb, cursor)
        parts[kind] = glb[cursor + 8:cursor + 8 + size]
        cursor += 8 + size
    doc = json.loads(parts[0x4E4F534A])
    return doc, parts[0x004E4942]


def _pack(doc, body):
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body += b"\0" * (-len(body) % 4)
    total = 12 + 8 + len(text) + 8 + len(body)
    return struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(body), 0x004E4942) + body
