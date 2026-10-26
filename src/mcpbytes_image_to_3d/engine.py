"""Hunyuan3D 2.1 (Tencent; Tencent Hunyuan 3D 2.1 Community License): one image in, a textured PBR model out.

Shape: a flow-matching DiT conditioned on the image, decoded by marching cubes. Paint: the shape is remeshed to the
detail level's face budget and UV-unwrapped, a multiview diffusion model draws albedo and metallic-roughness views (the
reference image's DINOv2 features steer it), RealESRGAN upscales them, and they are baked and inpainted into the atlas.
Upstream writes an OBJ and converts it to GLB with Blender; this engine writes the GLB itself (glTF metallic-roughness)
from the renderer's mesh and textures. The harness's finishing pass follows.

The only module that imports torch or the vendored code (vendor/hunyuan3d, unchanged; see its README.md).
"""

import importlib.machinery
import os
import sys
import tempfile
import time
import types
from pathlib import Path

from mcpbytes_core.errors import ToolError
from PIL import Image

VENDOR = Path(__file__).parent / "vendor" / "hunyuan3d"
COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
RENDER_SIZE = 2048  # view renders baked into the atlas (upstream default)
TEXTURE_SIZE = 2048  # atlas the views are baked into; the output is resized to texture_resolution
SOLID_ALPHA = 204  # a cut-out without a pixel above 80 % opacity has no subject to rebuild
COMMON = {"texture_resolution": 1024, "shape_steps": 50, "guidance_scale": 5.0, "octree_resolution": 384}
# paint_faces: the remesh budget before painting (upstream 40 000); decimation_agg: fast_simplification's
# aggressiveness (0 keeps the shape best, 10 is fastest; 7 is its default).
DETAIL = {
    "full": {**COMMON, "views": 8, "view_resolution": 768, "paint_faces": 120_000, "decimation_agg": 5},  # hosted economy; 41.2 GB peak on an A40
    "reduced": {**COMMON, "views": 6, "view_resolution": 512, "paint_faces": 40_000, "decimation_agg": 7},  # upstream defaults; 28.2 GB peak
}


def resolve_detail(detail, total_bytes):
    if detail == "auto":
        return "full" if total_bytes >= 44 * 2**30 else "reduced"  # 48 GB cards report about 45 GiB; full peaks at 40.3 GiB
    if detail not in DETAIL:
        raise ToolError("invalid_detail", f"detail must be auto, full or reduced, not {detail!r}")
    return detail


def require_subject(rgba):
    if rgba.getchannel("A").getextrema()[1] <= SOLID_ALPHA:
        raise ToolError("invalid_image", "no clear subject was found in the image", "use a photo or render of one object")
    return rgba


def cut_out(image, remover):
    """The image as RGBA: a cut-out (any alpha < 255) as it is, an opaque image through the background remover."""
    rgba = image.convert("RGBA")
    minimum, maximum = rgba.getchannel("A").getextrema()
    if maximum == 0:
        raise ToolError("invalid_image", "image has no foreground", "use a photo or render of one object")
    if minimum < 255:
        return require_subject(rgba)
    return require_subject(remover(rgba.convert("RGB")))


def square(rgba):
    """The cut-out centred on a transparent square canvas as wide as its longer side, at its own scale. Upstream paint
    resizes the reference to 512x512 whatever its aspect (textureGenPipeline.py `image.resize((512, 512))`), which
    squeezes a non-square upload; a square input with a centred subject comes back as it went in. The shape model crops
    to the subject and pads it square itself, so it sees the same framing either way."""
    left, top, right, bottom = rgba.getchannel("A").getbbox()  # cut_out guarantees a visible subject
    side = max(rgba.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(rgba.crop((left, top, right, bottom)), ((side - (right - left)) // 2, (side - (bottom - top)) // 2))
    return canvas


def stub(name, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)  # importlib.util.find_spec probes need one
    module.__dict__.update(attributes)
    return module


def prepare_vendor(torch):
    """Stand-ins for what the vendored code imports but this engine never runs, and its packages on sys.path.
    Upstream's OBJ-to-GLB converter imports Blender and its remesh step pymeshlab at module import; this engine does
    both itself. hy3dshape's (unused) postprocessors name pymeshlab.MeshSet/Mesh in signatures, evaluated at import.
    The paint UNet's training wrapper (hunyuanpaintpbr/unet/model.py) subclasses pytorch_lightning.LightningModule;
    inference never builds it. RealESRGAN comes from upscale.py instead of realesrgan/basicsr."""
    from . import upscale

    unavailable = type("Unavailable", (), {})
    sys.modules.setdefault("bpy", stub("bpy"))
    sys.modules.setdefault("pymeshlab", stub("pymeshlab", MeshSet=unavailable, Mesh=unavailable))
    sys.modules.setdefault("pytorch_lightning", stub("pytorch_lightning", LightningModule=torch.nn.Module))
    upscale.install()
    for folder in ("hy3dpaint/custom_rasterizer", "hy3dpaint", "hy3dshape"):
        if str(VENDOR / folder) not in sys.path:
            sys.path.insert(0, str(VENDOR / folder))


class Engine:
    def __init__(self, paths, detail="auto", log=print):
        from mcpbytes_core.gpu import require_cuda

        from . import extensions

        torch = require_cuda()
        self.torch, self.log = torch, log
        name = resolve_detail(detail, torch.cuda.get_device_properties(0).total_memory)
        self.settings = {"detail": name, **DETAIL[name]}
        prepare_vendor(torch)
        extensions.build(log)
        # Hunyuan3D's own remover (rembg's U2-Net) finds its pinned file by folder; nothing is downloaded.
        os.environ["U2NET_HOME"] = str(Path(paths["background"]).parent)
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.rembg import BackgroundRemover

        import textureGenPipeline
        from utils import multiview_utils

        main = Path(paths["main"])
        shape = main / "hunyuan3d-dit-v2-1"
        log(f"loading Hunyuan3D 2.1 ({name} detail) on {torch.cuda.get_device_name()}...")
        self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(str(shape / "model.fp16.ckpt"), str(shape / "config.yaml"),
                                                                       device="cuda", dtype=torch.float16)
        # The paint model is resolved by huggingface_hub.snapshot_download(repo id): hand it the verified snapshot instead.
        multiview_utils.huggingface_hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: str(main))
        config = textureGenPipeline.Hunyuan3DPaintConfig(self.settings["views"], self.settings["view_resolution"])
        config.multiview_cfg_path = str(VENDOR / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
        config.dino_ckpt_path = str(paths["dinov2"])
        config.realesrgan_ckpt_path = str(paths["realesrgan"])
        config.render_size, config.texture_size = RENDER_SIZE, TEXTURE_SIZE
        self.paint = textureGenPipeline.Hunyuan3DPaintPipeline(config)
        self.remover = BackgroundRemover()

    def versions(self):
        torch = self.torch
        return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "hunyuan3d": COMMIT}

    def seed_paint(self, seed):
        """Upstream paints every model with seed 0 and has no seed parameter. At the pinned source (82920d6)
        multiviewDiffusionNet.forward_one (hy3dpaint/utils/multiview_utils.py) starts with `self.seed_everything(0)`
        (line 74: Python's, NumPy's and torch's global generators; torch's draws the VAE posterior samples of the reference
        and condition images) and passes `generator=torch.Generator(device=self.pipeline.device).manual_seed(0)` (line 88)
        to the paint pipeline, whose __call__ hands it to `self.denoise(..., **cached_condition)`
        (hunyuanpaintpbr/pipeline.py:329-343) to draw the initial latents (prepare_latents, :597-606). Both are shadowed
        on the instances so the given seed replaces the 0: a new seed gives a new texture, the same seed the same one."""
        net = self.paint.models["multiview_model"]
        diffusion = net.pipeline
        net.seed_everything = lambda _zero: type(net).seed_everything(net, seed)
        diffusion.denoise = lambda *args, generator=None, **kwargs: type(diffusion).denoise(
            diffusion, *args, generator=self.torch.Generator(device=diffusion.device).manual_seed(seed), **kwargs)

    def process(self, image, out, seed):
        """Writes the raw `out / "model.glb"`. `seed` is used as given (the hosted stage seed: stage_seed(job seed, "model"))."""
        import fast_simplification
        import numpy as np
        import trimesh

        settings, torch = self.settings, self.torch
        texture = settings["texture_resolution"]
        started = time.monotonic()

        def phase(name):
            self.log(f"{name} ({time.monotonic() - started:.1f} s)")

        # An opaque photo is cut out by Hunyuan3D's remover; either way a subject must remain for the shape model.
        # Squared before either model sees it, so paint's fixed 512x512 reference keeps the upload's proportions.
        image = square(cut_out(image, self.remover))
        seed = int(seed)
        torch.manual_seed(seed)
        phase("shape")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        mesh = self.shape(image=image, num_inference_steps=settings["shape_steps"], guidance_scale=settings["guidance_scale"],
                          octree_resolution=settings["octree_resolution"], generator=generator)[0]
        shape_faces = int(len(mesh.faces))
        inferred = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hunyuan-") as work:
            # Upstream's remesh: vertices merged, then quadric decimation to the paint budget (the same step, without
            # pymeshlab). More faces keep more of the shape's detail through painting at the same atlas size.
            merged = trimesh.Trimesh(mesh.vertices, mesh.faces)
            points, faces = merged.vertices, merged.faces
            if len(faces) > settings["paint_faces"]:
                points, faces = fast_simplification.simplify(np.asarray(points, np.float32), np.asarray(faces, np.int64),
                                                             target_count=settings["paint_faces"], agg=settings["decimation_agg"])
            shape_path = Path(work) / "shape.obj"
            trimesh.Trimesh(points, faces, process=False).export(shape_path)
            self.seed_paint(seed)
            phase(f"paint ({shape_faces} shape faces, {len(faces)} painted)")
            self.paint(mesh_path=str(shape_path), image_path=image, output_mesh_path=str(Path(work) / "textured.obj"), use_remesh=False, save_glb=False)
        painted = time.monotonic()
        phase("write model.glb")

        render = self.paint.render
        # get_mesh restores the input frame in OBJ conventions (the frame upstream's GLB also has) with OBJ UVs
        # (origin bottom-left), which trimesh flips to glTF's on export. One glTF vertex per (position, UV) corner.
        positions, pos_idx, uv, uv_idx = render.get_mesh(normalize=False)
        corners = np.stack([pos_idx.reshape(-1), uv_idx.reshape(-1)], axis=1)
        unique, inverse = np.unique(corners, axis=0, return_inverse=True)
        albedo = render.get_texture()
        metallic, roughness = render.get_texture_mr()
        mr = np.zeros(albedo.shape[:2] + (3,), np.float32)  # glTF: G roughness, B metallic
        if roughness is not None:
            mr[..., 1], mr[..., 2] = roughness[..., 0], metallic[..., 0]
        else:
            mr[..., 1] = 1.0

        def picture(array):
            pixels = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(pixels).resize((texture, texture), Image.Resampling.LANCZOS)

        material = trimesh.visual.material.PBRMaterial(baseColorTexture=picture(albedo), metallicRoughnessTexture=picture(mr),
                                                       metallicFactor=1.0, roughnessFactor=1.0)
        textured = trimesh.Trimesh(positions[unique[:, 0]], inverse.reshape(-1, 3), process=False,
                                   visual=trimesh.visual.TextureVisuals(uv=uv[unique[:, 1]], material=material))
        textured.export(Path(out) / "model.glb")
        return {"seed": seed, "texture_resolution": texture, "shape_faces": shape_faces, "faces": int(len(textured.faces)),
                "vertices": int(len(textured.vertices)), "views": settings["views"], "view_resolution": settings["view_resolution"],
                "shape_ms": int((inferred - started) * 1000), "paint_ms": int((painted - inferred) * 1000),
                "export_ms": int((time.monotonic() - painted) * 1000)}
