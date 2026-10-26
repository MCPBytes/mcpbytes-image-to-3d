"""The engine's GPU-free parts: framing, background policy, detail levels, the upscaler shim, the vendored imports."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from mcpbytes_core.errors import ToolError
from PIL import Image

from mcpbytes_image_to_3d import engine


def test_square_centres_the_subject_at_its_own_scale():
    rgba = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    rgba.paste((255, 0, 0, 255), (2, 5, 8, 15))  # a 6x10 subject near the left edge
    out = engine.square(rgba)
    assert out.size == (40, 40)
    assert out.getchannel("A").getbbox() == (17, 15, 23, 25)
    assert out.getpixel((20, 20)) == (255, 0, 0, 255)


def test_cut_out_removes_only_opaque_backgrounds():
    calls = []

    def remover(rgb):
        calls.append(rgb.mode)
        cut = rgb.convert("RGBA")
        cut.putalpha(Image.new("L", rgb.size, 0))
        cut.paste((9, 9, 9, 255), (1, 1, 3, 3))
        return cut

    opaque = Image.new("RGB", (8, 8), (200, 200, 200))
    assert engine.cut_out(opaque, remover).getchannel("A").getbbox() == (1, 1, 3, 3) and calls == ["RGB"]
    cutout = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    cutout.paste((1, 2, 3, 255), (2, 2, 4, 4))
    assert engine.cut_out(cutout, remover).tobytes() == cutout.tobytes() and calls == ["RGB"]


@pytest.mark.parametrize("alpha", [0, 120])
def test_cut_out_without_a_solid_subject_is_invalid(alpha):
    with pytest.raises(ToolError) as error:
        engine.cut_out(Image.new("RGBA", (8, 8), (5, 5, 5, alpha)), None)
    assert error.value.code == "invalid_image"


@pytest.mark.parametrize("gib, name", [(96, "full"), (45, "full"), (44, "full"), (40, "reduced"), (32, "reduced"), (16, "reduced")])
def test_auto_detail_follows_vram(gib, name):
    assert engine.resolve_detail("auto", int(gib * 2**30)) == name


def test_explicit_detail_and_tables():
    assert engine.resolve_detail("reduced", 96 * 2**30) == "reduced" and engine.resolve_detail("full", 8 * 2**30) == "full"
    with pytest.raises(ToolError):
        engine.resolve_detail("ultra", 96 * 2**30)
    common = {"texture_resolution": 1024, "shape_steps": 50, "guidance_scale": 5.0, "octree_resolution": 384}
    assert engine.DETAIL["full"] == {**common, "views": 8, "view_resolution": 768, "paint_faces": 120_000, "decimation_agg": 5}
    assert engine.DETAIL["reduced"] == {**common, "views": 6, "view_resolution": 512, "paint_faces": 40_000, "decimation_agg": 7}


def test_rrdbnet_and_enhance_on_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    from mcpbytes_image_to_3d import upscale

    torch.manual_seed(0)
    tiny = {"num_feat": 8, "num_block": 1, "num_grow_ch": 4}
    net = upscale.RRDBNet(3, 3, scale=4, **tiny)
    assert net(torch.rand(1, 3, 5, 7)).shape == (1, 3, 20, 28)
    torch.save({"params_ema": net.state_dict()}, tmp_path / "x4.pth")
    upsampler = upscale.RealESRGANer(4, str(tmp_path / "x4.pth"), model=upscale.RRDBNet(3, 3, **tiny), pre_pad=0, device="cpu")
    image = np.random.default_rng(0).integers(0, 256, (5, 7, 3), np.uint8)
    out, mode = upsampler.enhance(image)
    assert mode == "RGB" and out.shape == (20, 28, 3) and out.dtype == np.uint8
    # the net sees the channels reversed (Real-ESRGAN's BGR handling) and they are reversed back
    x = torch.from_numpy(image[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)
    expected = (net(x)[0].clamp(0, 1).detach().numpy()[::-1].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 1


def run(code):
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout


def test_vendored_code_imports_with_the_shims(tmp_path):
    """Everything Engine imports from vendor/hunyuan3d resolves (compiled extensions stubbed: they need nvcc), and
    Tencent's imageSuperNet runs on upscale.py: the RealESRGAN_x4plus architecture, half precision, x4."""
    pytest.importorskip("torch")
    pytest.importorskip("rembg")
    run(f"""
        import sys, types
        import torch
        from PIL import Image
        from mcpbytes_image_to_3d import engine, upscale

        sys.modules["custom_rasterizer_kernel"] = types.ModuleType("custom_rasterizer_kernel")
        sys.modules["DifferentiableRenderer.mesh_inpaint_processor"] = types.SimpleNamespace(meshVerticeInpaint=None)
        engine.prepare_vendor(torch)
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.rembg import BackgroundRemover
        import custom_rasterizer, textureGenPipeline
        from utils import multiview_utils
        from utils.image_super_utils import imageSuperNet
        from hunyuanpaintpbr.unet.modules import Dino_v2

        assert custom_rasterizer.rasterize and textureGenPipeline.MeshRender.__module__ == "DifferentiableRenderer.MeshRender"
        torch.save({{"params_ema": upscale.RRDBNet(3, 3).state_dict()}}, {str(tmp_path / "x4.pth")!r})
        net = imageSuperNet(types.SimpleNamespace(realesrgan_ckpt_path={str(tmp_path / "x4.pth")!r}))
        assert isinstance(net.upsampler, upscale.RealESRGANer) and next(net.upsampler.model.parameters()).dtype == torch.float16
        assert net(Image.new("RGB", (6, 4), (10, 200, 30))).size == (24, 16)
    """)


def test_harness_imports_do_not_load_torch():
    run("""
        import importlib.util, sys
        import mcpbytes_image_to_3d.finish, mcpbytes_image_to_3d.engine
        if importlib.util.find_spec("mcpbytes_image_to_3d.generate"):
            import mcpbytes_image_to_3d.generate
        assert "torch" not in sys.modules, "torch was imported"
    """)
