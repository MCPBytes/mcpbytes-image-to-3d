"""Rendered previews of a finished model: six orthographic views and a labeled contact sheet (NumPy + Pillow only).

A stage opts in by declaring `<stem>_preview.png` next to its primary `<stem>.glb`; `write_model_previews` then
renders that GLB's final bytes (after the finishing pass, before validation) and writes whichever of the ten preview
contract's names were declared: `<stem>_preview.png`, `<stem>_<view>.png` for the six views, `<stem>_views.json`.
No tool ids: recipes without the preview name keep their outputs byte for byte.

What is drawn is the file itself: node transforms baked, embedded base-colour and metallic-roughness textures with
their factors (and COLOR_0), the stored normals (welded ones where a primitive has none), front faces only unless the
material is doubleSided, MASK cut-outs, BLEND as stochastic coverage. Views follow glTF's axes (+Y up, front from +Z);
they do not infer a semantic front or fix a tilt. Every view uses one scale: the model's largest bounding-box extent
spans 88 % of a panel (6 % margin per side), so sizes compare across views.

The rasterizer is exact and vectorised: pixel rows per triangle, spans between edge crossings, and a z-buffer holding
one int64 per pixel (quantized depth above the face index) resolved with `np.minimum.at`, on a 2x supersampled grid
that is box-filtered to 512 px. Time and memory follow the covered pixels, not the triangle sizes (a long sliver
enumerates its rows, not its bounding box), and candidates are processed in bounded batches. Lighting is a neutral
studio look fixed to the camera, so every view is lit alike: hemisphere ambient, one key light, a normalized
Blinn-Phong highlight whose width follows roughness; metals lose diffuse and tint the highlight. No shadows, no floor.
"""

import hashlib
import io
import json
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mcpbytes_core.errors import ToolError
from .glb import read_accessor, read_glb
from .normals import welded_vertex_normals
from .transforms import trs
from .validation import MAX_MODEL_FACES, image_info

RENDERER = "mcpbytes-model-preview-1"  # bump when the look or the framing changes: model_views.json records it
SIZE, SUPERSAMPLE = 512, 2
MARGIN = 0.06
BACKGROUND = "#e9eaec"
BACKGROUND_RGB = tuple(int(BACKGROUND[i:i + 2], 16) for i in (1, 3, 5))
PANELS = (384, 320, 256)  # contact-sheet panel sizes, tried in order until the sheet fits MAX_SHEET_BYTES
MAX_SHEET_BYTES = 2 * 1024 * 1024  # the sheet is returned inline to agents (before base64)
COLUMNS = 3
TEXTURE_LIMIT = 4096  # larger textures are box-reduced: a 1024 px supersampled view cannot show more
# (name, toward the camera, screen up). right = up × toward, so no view is mirrored.
VIEWS = (("front", (0, 0, 1), (0, 1, 0)), ("back", (0, 0, -1), (0, 1, 0)), ("left", (-1, 0, 0), (0, 1, 0)),
         ("right", (1, 0, 0), (0, 1, 0)), ("top", (0, 1, 0), (0, 0, -1)), ("bottom", (0, -1, 0), (0, 0, 1)))
# Camera-relative lighting, in linear units where a camera-facing surface shows its albedo (0.46 + 0.62 * 0.76 ≈ 0.93).
SKY, GROUND, KEY, SPECULAR = 0.62, 0.30, 0.62, 0.6
KEY_DIRECTION = np.array([-0.35, 0.55, 0.76]) / np.linalg.norm([-0.35, 0.55, 0.76])  # (right, up, toward): upper left
SHOULDER = 0.8  # tone map: linear below, then an exponential roll-off towards 1 (continuous slope)
FACE_BITS, DEPTH_LEVELS = 24, (1 << 38) - 1  # z-buffer key = depth << 24 | face; MAX_MODEL_FACES < 2^24
EMPTY = np.iinfo(np.int64).max
BATCH = 1 << 20  # rasterizer candidates per batch: bounds its temporaries to ~100 MB
REPEAT, CLAMP, MIRROR = 10497, 33071, 33648

_levels = np.arange(256) / 255
SRGB_TO_LINEAR = np.where(_levels <= 0.04045, _levels / 12.92, ((_levels + 0.055) / 1.055) ** 2.4).astype(np.float32)
# Texel decoding per RGBA channel: base colour is sRGB with linear alpha; metallic-roughness is linear throughout.
BASE_CURVE = np.stack([SRGB_TO_LINEAR] * 3 + [_levels.astype(np.float32)], axis=1)
LINEAR_CURVE = np.stack([_levels.astype(np.float32)] * 4, axis=1)


def write_model_previews(folder, names):
    """Render the declared previews of the primary model among `names` into `folder`; None when none are declared.
    Previews are required outputs: any failure is the stage's `invalid_output` (never a success without them)."""
    models = [n for n in names if n.endswith(".glb") and not n.endswith(".web.glb")]
    if len(models) != 1 or f"{models[0][:-4]}_preview.png" not in names:
        return None
    stem, started = models[0][:-4], time.monotonic()
    try:
        files = render_previews((folder / models[0]).read_bytes(), stem, source_file=models[0], declared=names)
    except MemoryError:
        raise
    except ToolError as exc:
        raise ToolError("invalid_output", f"model preview: {exc}") from None
    except Exception:
        raise ToolError("invalid_output", "model preview could not be rendered") from None
    written = [name for name in names if name in files]
    for name in written:
        (folder / name).write_bytes(files[name])
    return {"files": written, "render_ms": int((time.monotonic() - started) * 1000)}


def render_previews(data, stem="model", source_file=None, declared=None):
    """{name: bytes} for the sheet, the six views and the view manifest of the GLB `data`. `declared` (the stage's
    output names) only decides which view files the manifest may point to; everything is rendered."""
    source_file = source_file or f"{stem}.glb"
    model = load_model(data)
    lo, hi = model.bounds
    center = (lo + hi) / 2
    half = max(float((hi - lo).max()), 1e-9) / 2 / (1 - 2 * MARGIN)
    files, images, entries = {}, [], []
    for name, toward, up in VIEWS:
        toward, up = np.array(toward, float), np.array(up, float)
        right = np.cross(up, toward)
        image = render_view(model, (right, up, toward), center, half)
        images.append(image)
        file = f"{stem}_{name}.png"
        files[file] = _png(image)
        # The frame each view shows: the window across (right, up), the model's own depth along the view axis.
        across = np.abs(right) + np.abs(up)
        low, high = np.where(across > 0, center - half, lo), np.where(across > 0, center + half, hi)
        entries.append({"name": name, "file": file if declared is None or file in declared else None, "width": SIZE, "height": SIZE,
                        "camera": {"position": _round(center + toward * 2 * half), "target": _round(center), "up": _round(up)},
                        "bounds": {"min": _round(low), "max": _round(high)}})
    sheet, width, height = contact_sheet(images, [name for name, _, _ in VIEWS])
    files[f"{stem}_preview.png"] = sheet
    manifest = {"v": 1, "source_file": source_file, "source_sha256": hashlib.sha256(data).hexdigest(), "renderer": RENDERER,
                "up": "+Y", "projection": "orthographic", "background": BACKGROUND,
                "sheet": {"file": f"{stem}_preview.png", "width": width, "height": height, "views": [name for name, _, _ in VIEWS]},
                "views": entries}
    files[f"{stem}_views.json"] = json.dumps(manifest, indent=2).encode()
    return files


def _round(vector):
    return [round(float(v), 6) for v in vector]


def _png(image, optimize=False):
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=optimize)
    return out.getvalue()


def contact_sheet(images, labels):
    """The views in a labeled grid, as (PNG bytes, width, height): the first panel size of PANELS that fits."""
    for panel in PANELS:
        rows = -(-len(images) // COLUMNS)
        sheet = Image.new("RGB", (COLUMNS * panel, rows * panel), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=max(12, panel // 24))
        for index, (image, label) in enumerate(zip(images, labels)):
            x, y = (index % COLUMNS) * panel, (index // COLUMNS) * panel
            sheet.paste(image.resize((panel, panel), Image.LANCZOS), (x, y))
            draw.text((x + 8, y + 6), label, fill="#3c4043", font=font)
        for column in range(1, COLUMNS):  # hairlines, so neighbouring views never read as one picture
            draw.line([(column * panel, 0), (column * panel, sheet.height)], fill="#c4c6ca")
        for row in range(1, rows):
            draw.line([(0, row * panel), (sheet.width, row * panel)], fill="#c4c6ca")
        draw.text((sheet.width - 8 - draw.textlength("+Y up", font=font), 6), "+Y up", fill="#3c4043", font=font)
        data = _png(sheet, optimize=True)
        if len(data) <= MAX_SHEET_BYTES:
            return data, sheet.width, sheet.height
    raise ToolError("invalid_output", "contact sheet exceeds its size limit")


# --- The model: every mesh instance of the default scene, baked into world space and joined. ---

def load_model(data):
    """The GLB as world-space arrays: positions, unit normals, UV sets, vertex colours, faces with their material."""
    doc, binary = read_glb(data)
    materials = [_material(doc, binary, m) for m in doc.get("materials", [])] + [_material(doc, binary, {})]  # last: glTF's default
    positions, normals, uvs, colors, faces, face_material, offset = [], [], ([], []), [], [], [], 0
    has_uv, has_color = [False, False], False
    for world, mesh in _instances(doc):
        linear = world[:3, :3]
        for primitive in mesh["primitives"]:
            if primitive.get("mode", 4) != 4:
                raise ValueError("only triangle primitives are drawn")
            attributes = primitive["attributes"]
            local = read_accessor(doc, binary, attributes["POSITION"])
            count = len(local)
            index = read_accessor(doc, binary, primitive["indices"])[:, 0] if "indices" in primitive else np.arange(count)
            if len(index) % 3 or (len(index) and (index.min() < 0 or index.max() >= count)) or not np.isfinite(local).all():
                raise ValueError("malformed primitive")
            tri = index.reshape(-1, 3)
            if np.linalg.det(linear) < 0:  # a mirroring transform flips the winding (glTF 2.0, 3.7.2.1)
                tri = tri[:, [0, 2, 1]]
            points = local @ linear.T + world[:3, 3]
            if "NORMAL" in attributes:
                normal = read_accessor(doc, binary, attributes["NORMAL"]) @ np.linalg.inv(linear)
                normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            else:
                normal = welded_vertex_normals(points, tri)
            positions.append(points)
            normals.append(normal.astype(np.float32))
            for s in (0, 1):
                name = f"TEXCOORD_{s}"
                has_uv[s] |= name in attributes
                uvs[s].append(read_accessor(doc, binary, attributes[name]).astype(np.float32) if name in attributes else np.zeros((count, 2), np.float32))
            if "COLOR_0" in attributes:
                has_color = True
                color = read_accessor(doc, binary, attributes["COLOR_0"]).astype(np.float32)
                colors.append(np.c_[color, np.ones(count, np.float32)] if color.shape[1] == 3 else color)
            else:
                colors.append(np.ones((count, 4), np.float32))
            faces.append(tri + offset)
            face_material.append(np.full(len(tri), primitive.get("material", len(materials) - 1), np.int32))
            offset += count
            if sum(len(f) for f in faces) > MAX_MODEL_FACES:
                raise ToolError("invalid_output", "too many faces to preview")
    if not faces or not sum(len(f) for f in faces):
        raise ValueError("no triangles in the default scene")
    positions = np.concatenate(positions)
    faces = np.concatenate(faces).astype(np.int32 if offset < 2**31 else np.int64)
    used = np.zeros(len(positions), bool)
    used[faces.ravel()] = True
    edges = np.cross(positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]])
    return SimpleNamespace(positions=positions, normals=np.concatenate(normals), faces=faces, face_normals=edges.astype(np.float32),
                           uvs=[np.concatenate(uvs[s]) if has_uv[s] else None for s in (0, 1)],
                           colors=np.concatenate(colors) if has_color else None, face_material=np.concatenate(face_material),
                           materials=materials, bounds=(positions[used].min(axis=0), positions[used].max(axis=0)))


def _instances(doc):
    """(world matrix, mesh) for every mesh node of the default scene (or of the root nodes when there is no scene)."""
    nodes = doc.get("nodes", [])
    if doc.get("scenes"):
        roots = doc["scenes"][doc.get("scene", 0)].get("nodes", [])
    else:
        children = {c for node in nodes for c in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    stack, seen = [(i, np.eye(4)) for i in roots], set()
    while stack:
        i, parent = stack.pop()
        if i in seen:
            raise ValueError("node graph is not a forest")
        seen.add(i)
        world = parent @ trs(nodes[i])
        if "mesh" in nodes[i]:
            yield world, doc["meshes"][nodes[i]["mesh"]]
        stack.extend((c, world) for c in nodes[i].get("children", []))


def _material(doc, binary, material):
    pbr = material.get("pbrMetallicRoughness", {})
    mode = material.get("alphaMode", "OPAQUE")
    return SimpleNamespace(base_factor=np.array(pbr.get("baseColorFactor", (1, 1, 1, 1)), np.float32), base=_texture(doc, binary, pbr.get("baseColorTexture")),
                           metallic=float(pbr.get("metallicFactor", 1.0)), roughness=float(pbr.get("roughnessFactor", 1.0)),
                           metal_rough=_texture(doc, binary, pbr.get("metallicRoughnessTexture")), double_sided=bool(material.get("doubleSided", False)),
                           alpha_mode=mode if mode in ("MASK", "BLEND") else "OPAQUE", cutoff=float(material.get("alphaCutoff", 0.5)))


def _texture(doc, binary, reference):
    """(RGBA uint8 texels, UV set, (wrapS, wrapT)) of a texture reference, or None."""
    if reference is None:
        return None
    texture = doc["textures"][reference["index"]]
    view = doc["bufferViews"][doc["images"][texture["source"]]["bufferView"]]
    start = view.get("byteOffset", 0)
    image, _ = image_info(binary[start:start + view["byteLength"]])  # the stage's own decoding bounds, RGBA
    if max(image.size) > TEXTURE_LIMIT:
        image = image.reduce(-(-max(image.size) // TEXTURE_LIMIT))
    sampler = doc["samplers"][texture["sampler"]] if "sampler" in texture else {}
    return np.asarray(image), int(reference.get("texCoord", 0)), (sampler.get("wrapS", REPEAT), sampler.get("wrapT", REPEAT))


# --- Rendering one view. ---

def render_view(model, view, center, half, size=SIZE, supersample=SUPERSAMPLE):
    """The model seen along -toward, `half` world units either side of `center`: a size×size RGB image."""
    right, up, toward = view
    res = size * supersample
    relative = model.positions - center
    x = (relative @ right / (2 * half) + 0.5) * res
    y = (0.5 - relative @ up / (2 * half)) * res
    z = -(relative @ toward)  # nearer is smaller
    facing = model.face_normals @ toward.astype(np.float32)
    double = np.array([m.double_sided for m in model.materials])[model.face_material]
    visible = np.flatnonzero((facing > 0) | (double & (facing < 0)))
    test = _cutout_test(model, x, y, visible)
    pixels, rows = rasterize(x, y, z, model.faces[visible], res, test)
    face = visible[rows]
    image = np.empty((res * res, 3), np.float32)
    image[:] = SRGB_TO_LINEAR[list(BACKGROUND_RGB)]
    if len(face):
        weights = barycentric(x, y, model.faces[face], pixels % res + 0.5, pixels // res + 0.5)
        normal = interpolate(model.normals, model.faces[face], weights)
        normal *= np.where(facing[face] < 0, -1, 1).astype(np.float32)[:, None]  # the back of a double-sided face
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        base, metallic, roughness = np.empty((len(face), 4), np.float32), np.empty(len(face), np.float32), np.empty(len(face), np.float32)
        material_of = model.face_material[face]
        for index in np.unique(material_of):
            sel = np.flatnonzero(material_of == index)
            material = model.materials[index]
            base[sel] = base_color(model, material, face[sel], weights[sel])
            metallic[sel], roughness[sel] = material.metallic, material.roughness
            if material.metal_rough is not None:
                texels = sample(model, material.metal_rough, face[sel], weights[sel], LINEAR_CURVE)
                roughness[sel] *= texels[:, 1]
                metallic[sel] *= texels[:, 2]
        image[pixels] = tone_map(shade(normal, base[:, :3], np.clip(metallic, 0, 1), np.clip(roughness, 0, 1), view))
    image = image.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
    encoded = np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(np.maximum(image, 0.0031308), 1 / 2.4) - 0.055)
    return Image.fromarray(np.clip(np.round(encoded * 255), 0, 255).astype(np.uint8), "RGB")


def shade(normal, albedo, metallic, roughness, view):
    """Linear RGB of each pixel: hemisphere ambient + key light, diffuse lost to metalness, a roughness-wide highlight."""
    right, up, toward = (v.astype(np.float32) for v in view)
    light = (KEY_DIRECTION[0] * right + KEY_DIRECTION[1] * up + KEY_DIRECTION[2] * toward).astype(np.float32)
    half_vector = (light + toward) / np.linalg.norm(light + toward)
    ambient = GROUND + (SKY - GROUND) * (0.5 + 0.5 * (normal @ up))
    lit = np.clip(normal @ light, 0, None)
    alpha = np.maximum(roughness * roughness, 0.03)
    shininess = np.clip(2 / (alpha * alpha) - 2, 0, 2048)
    highlight = (shininess + 8) / 8 * np.power(np.clip(normal @ half_vector, 0, 1), shininess)
    m = metallic[:, None]
    f0 = 0.04 * (1 - m) + albedo * m
    return albedo * (1 - m) * (ambient + KEY * lit)[:, None] + f0 * (SPECULAR * KEY * lit * highlight + ambient)[:, None]


def tone_map(color):
    """Linear below SHOULDER, then an exponential roll-off to 1 with a continuous slope: highlights never clip hard."""
    top = 1 - SHOULDER
    return np.where(color < SHOULDER, color, SHOULDER + top * (1 - np.exp(-(color - SHOULDER) / top))).astype(np.float32)


def base_color(model, material, face, weights):
    """Linear RGBA: factor × texture (sRGB decoded) × COLOR_0."""
    color = np.tile(material.base_factor, (len(face), 1))
    if material.base is not None:
        color *= sample(model, material.base, face, weights, BASE_CURVE)
    if model.colors is not None:
        color *= interpolate(model.colors, model.faces[face], weights)
    return color


def sample(model, texture, face, weights, curve):
    """Bilinear RGBA lookup at the interpolated UVs, each texel decoded through `curve` ((256, 4): per level and
    channel) before blending. A missing UV set samples the texel at (0, 0)."""
    texels, uv_set, (wrap_s, wrap_t) = texture
    height, width = texels.shape[:2]
    uvs = model.uvs[uv_set] if uv_set < len(model.uvs) else None
    uv = interpolate(uvs, model.faces[face], weights) if uvs is not None else np.zeros((len(face), 2), np.float32)
    x, y = uv[:, 0].astype(np.float64) * width - 0.5, uv[:, 1].astype(np.float64) * height - 0.5  # glTF: v = 0 is the top row
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0).astype(np.float32)[:, None], (y - y0).astype(np.float32)[:, None]
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    columns, rows = (_wrap(x0, width, wrap_s), _wrap(x0 + 1, width, wrap_s)), (_wrap(y0, height, wrap_t), _wrap(y0 + 1, height, wrap_t))
    channels = np.arange(4)
    out = np.zeros((len(face), 4), np.float32)
    for row, wy in zip(rows, (1 - fy, fy)):
        for column, wx in zip(columns, (1 - fx, fx)):
            out += curve[texels[row, column], channels] * (wy * wx)
    return out


def _wrap(index, size, mode):
    if mode == CLAMP:
        return np.clip(index, 0, size - 1)
    if mode == MIRROR:
        folded = np.mod(index, 2 * size)
        return np.where(folded < size, folded, 2 * size - 1 - folded)
    return np.mod(index, size)


def interpolate(values, tri, weights):
    return values[tri[:, 0]] * weights[:, 0:1] + values[tri[:, 1]] * weights[:, 1:2] + values[tri[:, 2]] * weights[:, 2:3]


def barycentric(x, y, tri, px, py):
    """(k, 3) float32 weights of pixel centres (px, py) in their triangles; clamped to the triangle (a centre on a
    shared edge may sit a rounding error outside)."""
    x0, x1, x2 = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    y0, y1, y2 = y[tri[:, 0]], y[tri[:, 1]], y[tri[:, 2]]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    b1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) / area
    b2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) / area
    weights = np.clip(np.stack([1 - b1 - b2, b1, b2], axis=1), 0, None)
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _cutout_test(model, x, y, visible):
    """The rasterizer's per-candidate alpha test for MASK (alpha ≥ cutoff) and BLEND (alpha above a per-pixel hash:
    stochastic coverage that the supersampling filter turns into partial transparency), or None when all is opaque."""
    if all(m.alpha_mode == "OPAQUE" for m in model.materials):
        return None

    def test(rows, px, py):
        face = visible[rows]
        keep = np.ones(len(rows), bool)
        material_of = model.face_material[face]
        for index in np.unique(material_of):
            material = model.materials[index]
            if material.alpha_mode == "OPAQUE":
                continue
            sel = np.flatnonzero(material_of == index)
            weights = barycentric(x, y, model.faces[face[sel]], px[sel] + 0.5, py[sel] + 0.5)
            alpha = base_color(model, material, face[sel], weights)[:, 3]
            if material.alpha_mode == "MASK":
                keep[sel] = alpha >= material.cutoff
            else:
                noise = ((px[sel] * 73856093) ^ (py[sel] * 19349663) ^ (face[sel].astype(np.int64) * 83492791)) & 0xFFFF
                keep[sel] = alpha * 65536 > noise
        return keep
    return test


def rasterize(x, y, z, faces, res, test=None):
    """(pixel index, row of `faces`) for every covered pixel centre of a res×res grid, nearest face first.

    x, y: vertex positions in pixels (y down); z: depth, smaller nearer; faces: (k, 3) triangles to draw. Coverage is
    exact and crack-free: each triangle's pixel rows, then the span between its edge crossings on each row. Corners are
    sorted top to bottom (ties by x), so an edge two triangles share is evaluated from the same endpoints in the same
    order, bit-identically, and a centre lying on it is claimed by both (the depth test keeps one).
    `test(rows, px, py)` may reject candidates (alpha cut-outs) before the depth test."""
    fx, fy, fz = x[faces], y[faces], z[faces]
    for a, b in ((0, 1), (1, 2), (0, 1)):  # a three-element sorting network on (y, x)
        swap = (fy[:, b] < fy[:, a]) | ((fy[:, b] == fy[:, a]) & (fx[:, b] < fx[:, a]))
        for corners in (fx, fy, fz):
            corners[swap, a], corners[swap, b] = corners[swap, b], corners[swap, a]
    first_row = np.maximum(np.ceil(fy[:, 0] - 0.5), 0)
    rows = np.maximum(np.minimum(np.floor(fy[:, 2] - 0.5), res - 1) - first_row + 1, 0).astype(np.int64)
    # The depth plane z = z0 + gx (x - x0) + gy (y - y0): exact for an orthographic view.
    e1x, e1y, e1z = fx[:, 1] - fx[:, 0], fy[:, 1] - fy[:, 0], fz[:, 1] - fz[:, 0]
    e2x, e2y, e2z = fx[:, 2] - fx[:, 0], fy[:, 2] - fy[:, 0], fz[:, 2] - fz[:, 0]
    det = e1x * e2y - e1y * e2x
    drawn = np.flatnonzero((rows > 0) & (det != 0))
    zbuffer = np.full(res * res, EMPTY, np.int64)
    if len(drawn):
        det = det[drawn]
        gx, gy = (e1z[drawn] * e2y[drawn] - e1y[drawn] * e2z[drawn]) / det, (e1x[drawn] * e2z[drawn] - e1z[drawn] * e2x[drawn]) / det
        fx, fy, fz, rows, first_row = fx[drawn], fy[drawn], fz[drawn], rows[drawn], first_row[drawn]
        near, far = fz.min(axis=1), fz.max(axis=1)
        low = float(near.min())
        scale = DEPTH_LEVELS / max(float(far.max()) - low, 1e-30)
        for start, stop in _batches(rows):
            count = rows[start:stop]
            tri = np.repeat(np.arange(start, stop), count)
            row = first_row[tri] + _ramp(count)
            centre = row + 0.5
            longest = _crossing(fx[tri, 0], fy[tri, 0], fx[tri, 2], fy[tri, 2], centre)
            upper = centre <= fy[tri, 1]
            shorter = _crossing(np.where(upper, fx[tri, 0], fx[tri, 1]), np.where(upper, fy[tri, 0], fy[tri, 1]),
                                np.where(upper, fx[tri, 1], fx[tri, 2]), np.where(upper, fy[tri, 1], fy[tri, 2]), centre)
            first_column = np.maximum(np.ceil(np.minimum(longest, shorter) - 0.5), 0)
            columns = np.maximum(np.minimum(np.floor(np.maximum(longest, shorter) - 0.5), res - 1) - first_column + 1, 0).astype(np.int64)
            for a, b in _batches(columns):
                width = columns[a:b]
                span = np.repeat(np.arange(a, b), width)
                t, py = tri[span], row[span]
                px = first_column[span] + _ramp(width)
                depth = fz[t, 0] + gx[t] * (px + 0.5 - fx[t, 0]) + gy[t] * (py + 0.5 - fy[t, 0])
                depth = np.clip(depth, near[t], far[t])  # a sliver's plane is steep: keep a centre on its edge in range
                key = (np.clip((depth - low) * scale, 0, DEPTH_LEVELS).astype(np.int64) << FACE_BITS) | drawn[t]
                px, py = px.astype(np.int64), py.astype(np.int64)
                if test is not None:
                    keep = test(drawn[t], px, py)
                    key, px, py = key[keep], px[keep], py[keep]
                np.minimum.at(zbuffer, py * res + px, key)
    pixels = np.flatnonzero(zbuffer != EMPTY)
    return pixels, zbuffer[pixels] & ((1 << FACE_BITS) - 1)


def _crossing(ax, ay, bx, by, y):
    """x where the edge a→b (ay ≤ by) crosses the row at `y`; a horizontal edge lies on it, so its far end bounds the span."""
    dy = by - ay
    with np.errstate(divide="ignore", invalid="ignore"):
        x = ax + (y - ay) * ((bx - ax) / dy)
    return np.where(dy > 0, x, bx)


def _ramp(counts):
    """0, 1, … counts[i] - 1 for each i, concatenated."""
    return np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)


def _batches(counts):
    """(start, stop) ranges of `counts` whose sums stay near BATCH (a single item may exceed it)."""
    total = np.cumsum(counts)
    start = 0
    while start < len(counts):
        base = total[start - 1] if start else 0
        stop = max(int(np.searchsorted(total, base + BATCH, side="right")), start + 1)
        yield start, stop
        start = stop
