"""Finishing (ported from the hosted runner's test_gpu_postprocess.py and test_gpu_model_preview.py, CPU tests only)."""

import hashlib
import io
import json
import struct

import numpy as np
import pytest
import trimesh
from mcpbytes_core.errors import ToolError
from PIL import Image

from mcpbytes_image_to_3d.finish.glb import write_glb
from mcpbytes_image_to_3d.finish.model_preview import render_previews, write_model_previews
from mcpbytes_image_to_3d.finish.normals import welded_vertex_normals
from mcpbytes_image_to_3d.finish.postprocess import _chunks, _pack, finish_outputs, write_web_copy
from mcpbytes_image_to_3d.finish.validation import artifact

DracoPy = pytest.importorskip("DracoPy")


def textured_box(path):
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    box.apply_translation([3.0, 4.0, 5.0])
    uv = np.random.default_rng(0).random((len(box.vertices), 2))
    base = Image.fromarray(np.full((64, 64, 4), (200, 120, 60, 255), np.uint8), "RGBA")
    rough = Image.fromarray(np.dstack([np.zeros((64, 64), np.uint8), np.full((64, 64), 20, np.uint8), np.zeros((64, 64), np.uint8)]), "RGB")  # roughness 20/255, non-metal
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=base, metallicRoughnessTexture=rough)
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    speck = trimesh.creation.box(extents=(0.01, 0.01, 0.01))  # detached debris, far smaller than 1% of the surface
    speck.apply_translation([10.0, 10.0, 10.0])
    speck.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(speck.vertices), 2)), material=material)
    trimesh.util.concatenate([box, speck]).export(path, file_type="glb")


def textured_sphere(path, base, rough, rings=16, segments=32):
    """A unit UV sphere as exporters write one: the u = 0 and u = 1 meridians and every pole corner are bit-identical
    copies with their own UVs (a seam through a smooth surface), every triangle facing out."""
    lat, lon = np.linspace(0, np.pi, rings + 1)[:, None], np.linspace(0, 2 * np.pi, segments + 1)[None, :]
    grid = np.stack(np.broadcast_arrays(np.sin(lat) * np.cos(lon), np.cos(lat), np.sin(lat) * np.sin(lon)), axis=-1)
    grid[:, -1], grid[0], grid[-1] = grid[:, 0], (0, 1, 0), (0, -1, 0)
    uv = np.stack(np.broadcast_arrays(lon / (2 * np.pi), 1 - lat / np.pi), axis=-1).reshape(-1, 2)
    index = np.arange(grid.shape[0] * grid.shape[1]).reshape(grid.shape[:2])
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1)[:-1], np.stack([a, d, c], -1)[1:]]).reshape(-1, 3)  # no pole slivers
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=base, metallicRoughnessTexture=rough)
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    trimesh.Trimesh(grid.reshape(-1, 3), faces, visual=visual, process=False).export(path, file_type="glb")


def copy_angles(points, normals):
    """Per vertex that shares its position: degrees between its normal and the first copy's (2·asin(chord/2), exact
    near zero where arccos of a float32 dot product is not). Also returns how many positions are shared."""
    normals = np.asarray(normals, dtype=np.float64)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    _, weld, counts = np.unique(np.asarray(points), axis=0, return_inverse=True, return_counts=True)
    weld = weld.reshape(-1)
    first = np.empty(len(counts), dtype=np.int64)
    first[weld[::-1]] = np.arange(len(weld))[::-1]
    chord = np.linalg.norm(normals - normals[first[weld]], axis=1)[counts[weld] > 1]
    return np.degrees(2 * np.arcsin(np.minimum(chord / 2, 1.0))), int((counts > 1).sum())


def webp_image(doc, binary, slot):
    """The web copy's WebP bytes behind material 0's `slot` texture."""
    info = doc["materials"][0]["pbrMetallicRoughness"][slot]
    image = doc["images"][doc["textures"][info["index"]]["extensions"]["EXT_texture_webp"]["source"]]
    view = doc["bufferViews"][image["bufferView"]]
    return bytes(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]])


def webp_chunks(data):
    """RIFF chunk names of a WebP file: `VP8 ` is lossy, `VP8L` lossless."""
    names, cursor = [], 12
    while cursor + 8 <= len(data):
        size = struct.unpack_from("<I", data, cursor + 4)[0]
        names.append(data[cursor:cursor + 4])
        cursor += 8 + size + (size & 1)
    return names


def test_finish_cleans_normalizes_and_writes_web_copy(tmp_path):
    textured_box(tmp_path / "model.glb")
    summary = finish_outputs(tmp_path, ["model.glb", "model.web.glb"])
    assert summary["applied"] and summary["debris_components_removed"] == 1 and summary["roughness_texels_raised"] == 64 * 64
    mesh = next(iter(trimesh.load(tmp_path / "model.glb", force="scene").geometry.values()))
    assert len(mesh.faces) == 12  # the speck is gone, the box intact
    assert np.allclose(mesh.bounds[0][1], 0, atol=1e-6) and np.isclose(mesh.extents.max(), 1.0)
    assert np.allclose((mesh.bounds[0] + mesh.bounds[1])[[0, 2]] / 2, 0, atol=1e-6)
    rough = np.array(mesh.visual.material.metallicRoughnessTexture.convert("RGB"))
    assert rough[:, :, 1].min() >= int(round(0.35 * 255)) - 1
    main = artifact(tmp_path / "model.glb", "model/gltf-binary")
    web = artifact(tmp_path / "model.web.glb", "model/gltf-binary", compressed=True)
    assert main["validation"]["textured"] and web["validation"]["textured"] and web["validation"]["compression"] == "draco"
    assert web["validation"]["faces"] == main["validation"]["faces"] == 12
    assert web["validation"]["textures"][0]["width"] == 64


def test_finish_leaves_unknown_shapes_alone(tmp_path):
    box = trimesh.creation.box()
    trimesh.Scene([box, box.copy()]).export(tmp_path / "model.glb", file_type="glb")  # two primitives, no texture
    assert finish_outputs(tmp_path, ["model.glb"])["applied"] is False
    assert finish_outputs(tmp_path, ["image.png"]) is None


def test_finish_shades_uv_seams_smoothly(tmp_path):
    """Per-index normals gave each seam copy only its own side's triangles: 11.3° creases along this sphere's seam
    and at its poles. Both exports must carry one normal per position, and it must be the surface's."""
    plain = Image.fromarray(np.full((8, 8, 3), 128, np.uint8), "RGB")
    textured_sphere(tmp_path / "model.glb", plain, plain.copy())
    finish_outputs(tmp_path, ["model.glb", "model.web.glb"])
    data = (tmp_path / "model.glb").read_bytes()
    assert "NORMAL" in _chunks(data)[0]["meshes"][0]["primitives"][0]["attributes"]  # else trimesh would compute its own
    mesh = next(iter(trimesh.load(tmp_path / "model.glb", force="scene", process=False).geometry.values()))
    doc, binary = _chunks((tmp_path / "model.web.glb").read_bytes())
    view = doc["bufferViews"][doc["meshes"][0]["primitives"][0]["extensions"]["KHR_draco_mesh_compression"]["bufferView"]]
    web = DracoPy.decode(bytes(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]))
    for points, normals in ((mesh.vertices, mesh.vertex_normals), (web.points, web.normals)):
        angles, shared = copy_angles(points, normals)
        assert shared == 15 + 2  # the seam's inner ring points and both poles survived as copies
        assert angles.max() < 0.005  # so any two copies are within 0.01°
    radial = mesh.vertices - mesh.bounds.mean(axis=0)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    normals = mesh.vertex_normals / np.linalg.norm(mesh.vertex_normals, axis=1, keepdims=True)
    assert np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", normals, radial), -1, 1))).max() < 3.0


def test_welded_normals_skip_zero_area_faces_and_point_lone_vertices_up():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [5, 5, 5]]
    faces = [[0, 1, 2], [1, 3, 4]]  # a triangle facing +Z; a zero-area one along the x axis; vertex 5 in no face
    normals = welded_vertex_normals(vertices, faces)
    assert np.allclose(normals, [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3)


def test_web_copy_keeps_data_maps_exact_and_base_colour_lossy(tmp_path):
    rng = np.random.default_rng(1)
    base = Image.fromarray(rng.integers(0, 256, (32, 32, 4), dtype=np.uint8), "RGBA")
    rough = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), "RGB")  # noise: any lossy step shows
    textured_sphere(tmp_path / "source.glb", base, rough)
    doc, binary = _chunks((tmp_path / "source.glb").read_bytes())
    # Roles come from the material, not the image order: reverse the images so metallic-roughness comes first.
    count = len(doc["images"])
    doc["images"] = doc["images"][::-1]
    for texture in doc["textures"]:
        texture["source"] = count - 1 - texture["source"]
    pbr = doc["materials"][0]["pbrMetallicRoughness"]
    assert doc["textures"][pbr["metallicRoughnessTexture"]["index"]]["source"] == 0
    pbr["baseColorTexture"]["extensions"] = {"KHR_texture_transform": {"scale": [1, 1]}}
    source = _pack(doc, bytes(binary))
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    original = np.array(Image.open(io.BytesIO(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]])).convert("RGB"))
    mesh = next(iter(trimesh.load(io.BytesIO(source), file_type="glb", force="scene", process=False).geometry.values()))
    write_web_copy(mesh, source, tmp_path / "model.web.glb")
    web, body = _chunks((tmp_path / "model.web.glb").read_bytes())
    colour, data = webp_image(web, body, "baseColorTexture"), webp_image(web, body, "metallicRoughnessTexture")
    assert b"VP8 " in webp_chunks(colour) and b"VP8L" in webp_chunks(data)
    assert np.array_equal(np.array(Image.open(io.BytesIO(data)).convert("RGB")), original)
    assert "KHR_texture_transform" in web["extensionsUsed"] and "KHR_texture_transform" not in web["extensionsRequired"]
    assert artifact(tmp_path / "model.web.glb", "model/gltf-binary", compressed=True)["validation"]["textured"]


VIEWS = ["front", "back", "left", "right", "top", "bottom"]
PREVIEWS = ["model_preview.png"] + [f"model_{v}.png" for v in VIEWS] + ["model_views.json"]
BACKGROUND = np.array([0xE9, 0xEA, 0xEC])
COLORS = {"red": (1, 0, 0), "cyan": (0, 1, 1), "green": (0, 1, 0), "magenta": (1, 0, 1), "blue": (0, 0, 1), "yellow": (1, 1, 0)}
# The axis box's colours: +X, -X, +Y, -Y, +Z, -Z (box_quads' order), then its black marker.
CELLS = list(COLORS.values()) + [(0, 0, 0)]


def box_quads(center, half):
    """A box as 6 separate quads (+X, -X, +Y, -Y, +Z, -Z): (24, 3) corners counter-clockwise from outside, (24, 3)
    normals, (24, 2) corner offsets in -1..1 across each quad."""
    points, normals, offsets = [], [], []
    for axis in range(3):
        for sign in (1, -1):
            n = np.zeros(3)
            n[axis] = sign
            u = np.roll([1.0, 0, 0], axis + 1)
            v = np.cross(n, u)  # u × v = n: counter-clockwise seen from outside
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                points.append(np.asarray(center) + np.asarray(half) * (n + a * u + b * v))
                normals.append(n)
                offsets.append((a, b))
    return np.array(points), np.array(normals), np.array(offsets, float)


def quad_faces(quads, first=0):
    return np.concatenate([[[q, q + 1, q + 2], [q, q + 2, q + 3]] for q in range(first, first + 4 * quads, 4)])


def build_glb(positions, faces, normals=None, uv=None, texture=None, material=None, nodes=None, sampler=None):
    """One indexed primitive with an optional embedded PNG base-colour texture; `nodes` replace the single identity
    node (the node holding `"mesh": 0` draws it)."""
    body, views, accessors = bytearray(), [], []

    def add(blob, **accessor):
        views.append({"buffer": 0, "byteOffset": len(body), "byteLength": len(blob)})
        body.extend(blob)
        body.extend(b"\0" * (-len(blob) % 4))
        if accessor:
            accessors.append({"bufferView": len(views) - 1, **accessor})
        return len(accessors) - 1

    positions = np.asarray(positions, np.float32)
    add(np.asarray(faces, np.uint32).tobytes(), componentType=5125, count=int(np.size(faces)), type="SCALAR")
    attributes = {"POSITION": add(positions.tobytes(), componentType=5126, count=len(positions), type="VEC3", min=positions.min(0).tolist(), max=positions.max(0).tolist())}
    if normals is not None:
        attributes["NORMAL"] = add(np.asarray(normals, np.float32).tobytes(), componentType=5126, count=len(positions), type="VEC3")
    if uv is not None:
        attributes["TEXCOORD_0"] = add(np.asarray(uv, np.float32).tobytes(), componentType=5126, count=len(positions), type="VEC2")
    material = json.loads(json.dumps(material or {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.5}}))
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes or [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": attributes, "indices": 0, "material": 0}]}], "materials": [material], "accessors": accessors}
    if texture is not None:
        out = io.BytesIO()
        texture.save(out, format="PNG")
        add(out.getvalue())
        doc.update(images=[{"bufferView": len(views) - 1, "mimeType": "image/png"}], textures=[{"source": 0, **({"sampler": 0} if sampler else {})}])
        if sampler:
            doc["samplers"] = [sampler]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    doc.update(bufferViews=views, buffers=[{"byteLength": len(body)}])
    return write_glb(doc, bytes(body))


def axis_box():
    """A 2-unit box with one colour per face and a small black marker box off its (+X, +Y, +Z) corner, outside every
    view's silhouette of the big box: a swapped view shows the wrong colour, a mirrored or upside-down one moves the
    marker to the wrong side."""
    big, big_normals, big_offsets = box_quads((0, 0, 0), (1, 1, 1))
    small, small_normals, small_offsets = box_quads((1.3, 0.7, 1.3), (0.2, 0.2, 0.2))
    cells = np.r_[np.repeat(np.arange(6), 4), np.full(24, 6)]  # atlas cell per corner
    offsets = np.r_[big_offsets, small_offsets]
    # A 4 × 2 atlas of 8 px cells; UVs stay in each cell's inner half, so bilinear filtering never mixes colours.
    uv = np.c_[(cells % 4 + 0.5 + 0.25 * offsets[:, 0]) / 4, (cells // 4 + 0.5 + 0.25 * offsets[:, 1]) / 2]
    atlas = np.zeros((16, 32, 4), np.uint8)
    atlas[..., 3] = 255
    for index, color in enumerate(CELLS):
        atlas[(index // 4) * 8:(index // 4 + 1) * 8, (index % 4) * 8:(index % 4 + 1) * 8, :3] = np.array(color) * 255
    return build_glb(np.r_[big, small], quad_faces(12), normals=np.r_[big_normals, small_normals], uv=uv, texture=Image.fromarray(atlas, "RGBA"))


def pixels(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)


def classify(image):
    """Per pixel: a COLORS name (its channels clearly high or low), "black", or ""."""
    high, low = image > 150, image < 100
    names = np.full(image.shape[:2], "", object)
    for name, pattern in COLORS.items():
        on = np.array(pattern, bool)
        names[high[..., on].all(-1) & low[..., ~on].all(-1)] = name
    names[(image < 90).all(-1)] = "black"
    return names


def dominant(names):
    counts = {c: int((names == c).sum()) for c in COLORS}
    best = max(counts, key=counts.get)
    return best, counts[best] / max(sum(counts.values()), 1)


def foreground(image):
    return np.abs(image - BACKGROUND).max(axis=-1) > 24


def test_each_view_shows_its_own_side_unmirrored():
    files = render_previews(axis_box())
    # colour, and where the marker at (+X, +Y, +Z) must appear: right/left, top/bottom of the picture
    expected = {"front": ("blue", "right", "top"), "back": ("yellow", "left", "top"), "left": ("cyan", "right", "top"),
                "right": ("red", "left", "top"), "top": ("green", "right", "bottom"), "bottom": ("magenta", "right", "top")}
    for view, (color, across, height) in expected.items():
        image = pixels(files[f"model_{view}.png"])
        assert image.shape == (512, 512, 3)
        names = classify(image)
        best, share = dominant(names)
        assert best == color and share > 0.97, view
        rows, columns = np.nonzero(names == "black")
        assert len(rows) > 2000, view  # 0.4 units at 180 px per unit
        assert ("right" if columns.mean() > 256 else "left") == across and abs(columns.mean() - 256) > 100, view
        assert ("top" if rows.mean() < 256 else "bottom") == height and abs(rows.mean() - 256) > 60, view
    sheet = pixels(files["model_preview.png"])
    panel = sheet.shape[1] // 3
    assert sheet.shape[0] == 2 * panel
    for index, view in enumerate(VIEWS):  # the sheet's panels are the same views, in the manifest's order
        cell = sheet[(index // 3) * panel:(index // 3 + 1) * panel, (index % 3) * panel:(index % 3 + 1) * panel]
        assert dominant(classify(cell))[0] == expected[view][0], view


def test_node_transforms_are_baked_and_every_view_is_framed_alike():
    corners, _, _ = box_quads((5, 5, 5), (1, 0.5, 0.25))  # 2 × 1 × 0.5, far from the origin; no NORMAL, no UVs
    turn = [0, np.sin(np.pi / 4), 0, np.cos(np.pi / 4)]  # 90° about +Y: the long x side ends up along -z
    nodes = [{"translation": [10, -4, 3], "scale": [2, 2, 2], "children": [1]}, {"rotation": turn, "mesh": 0}]
    material = {"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.3, 0.1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.6}}
    files = render_previews(build_glb(corners, quad_faces(6), material=material, nodes=nodes))
    scale = 0.88 * 512 / 4  # world extents 1 (x) × 2 (y) × 4 (z): the largest spans 88 % of every panel
    extents = {"front": (1, 2), "back": (1, 2), "left": (4, 2), "right": (4, 2), "top": (1, 4), "bottom": (1, 4)}
    for view, (width, height) in extents.items():
        image = pixels(files[f"model_{view}.png"])
        rows, columns = np.nonzero(foreground(image))
        assert abs(columns.max() - columns.min() + 1 - width * scale) <= 3, view
        assert abs(rows.max() - rows.min() + 1 - height * scale) <= 3, view
        assert abs((columns.max() + columns.min() + 1) / 2 - 256) <= 2 and abs((rows.max() + rows.min() + 1) / 2 - 256) <= 2, view
        red, green, blue = image[foreground(image)].mean(axis=0)
        assert red > green + 40 > blue + 80, view  # the factor's orange: no texture needed


def test_cutouts_and_sidedness_follow_the_material():
    corners, normals, offsets = box_quads((0, 0, 0), (1, 1, 1))
    front = slice(16, 20)  # the +Z quad alone
    uv = np.c_[(offsets[front, 0] + 1) / 2, (1 - offsets[front, 1]) / 2]
    texture = np.zeros((1, 4, 4), np.uint8)
    texture[..., 0] = 255
    texture[:, 2:, 3] = 255  # right half opaque red, left half fully transparent
    clamp = {"wrapS": 33071, "wrapT": 33071}

    def views(alpha_mode, double_sided=False, alpha=None):
        pixels_ = texture.copy()
        if alpha is not None:
            pixels_[..., 3] = alpha
        material = {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}, "alphaMode": alpha_mode, "doubleSided": double_sided}
        files = render_previews(build_glb(corners[front], [[0, 1, 2], [0, 2, 3]], normals=normals[front], uv=uv,
                                          texture=Image.fromarray(pixels_, "RGBA"), material=material, sampler=clamp))
        return pixels(files["model_front.png"]), pixels(files["model_back.png"])

    left, right = (slice(180, 330), slice(150, 220)), (slice(180, 330), slice(300, 370))  # rows, columns
    shown, behind = views("MASK")
    assert not foreground(shown[left]).any() and (classify(shown[right]) == "red").all()
    assert not foreground(behind).any()  # single-sided: nothing from behind
    shown, behind = views("OPAQUE", double_sided=True)
    assert foreground(shown[left]).all()  # alpha is ignored when opaque
    assert foreground(behind[left]).all() and foreground(behind[right]).all()
    shown, _ = views("BLEND", alpha=128)
    red = np.array([255, 0, 0])
    coverage = ((shown[left] - BACKGROUND) @ (red - BACKGROUND)) / ((red - BACKGROUND) @ (red - BACKGROUND))
    assert 0.3 < coverage.mean() < 0.7  # half the surface, the background through the rest




def test_previews_are_written_only_when_declared(tmp_path):
    (tmp_path / "model.glb").write_bytes(axis_box())
    assert write_model_previews(tmp_path, ["model.glb", "model.web.glb"]) is None
    assert write_model_previews(tmp_path, ["model.glb", "model_front.png"]) is None  # the sheet's name switches previews on
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.glb"]
    summary = write_model_previews(tmp_path, ["model.glb", "model_preview.png", "model_views.json"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.glb", "model_preview.png", "model_views.json"]
    assert summary["files"] == ["model_preview.png", "model_views.json"]
    assert [v["file"] for v in json.loads((tmp_path / "model_views.json").read_bytes())["views"]] == [None] * 6  # no undeclared file named


def test_a_model_that_cannot_be_rendered_fails_the_stage(tmp_path):
    (tmp_path / "model.glb").write_bytes(b"glTF\x02\x00\x00\x00" + b"\x00" * 64)
    with pytest.raises(ToolError) as failure:
        write_model_previews(tmp_path, ["model.glb", *PREVIEWS])
    assert failure.value.code == "invalid_output"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.glb"]
