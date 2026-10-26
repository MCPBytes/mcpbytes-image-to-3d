"""The model files this tool runs, at pinned revisions: about 19 GB, downloaded on first use.

Hugging Face files go into its standard cache (mcpbytes_core.hub). The upscaler and background-removal checkpoints,
GitHub release assets, go into this tool's cache and are checked against their SHA-256. Nothing is bundled: each file is licensed to you by
its publisher (NOTICE). Revisions are commits, never branches; sizes were read from the Hub for these commits.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

from mcpbytes_core import hub
from mcpbytes_core.errors import ToolError

COMPONENTS = {
    "main": {
        "repository": "tencent/Hunyuan3D-2.1",
        "revision": "0b94677654c57bb9a6b6845cd7b704ccf551d327",
        "files": {
            "hunyuan3d-dit-v2-1/config.yaml": 2078,
            "hunyuan3d-dit-v2-1/model.fp16.ckpt": 7366389768,
            "hunyuan3d-paintpbr-v2-1/model_index.json": 617,
            "hunyuan3d-paintpbr-v2-1/feature_extractor/preprocessor_config.json": 342,
            "hunyuan3d-paintpbr-v2-1/image_encoder/config.json": 554,
            "hunyuan3d-paintpbr-v2-1/image_encoder/model.safetensors": 1264217240,
            "hunyuan3d-paintpbr-v2-1/scheduler/scheduler_config.json": 387,
            "hunyuan3d-paintpbr-v2-1/text_encoder/config.json": 613,
            "hunyuan3d-paintpbr-v2-1/text_encoder/pytorch_model.bin": 1361671895,
            "hunyuan3d-paintpbr-v2-1/tokenizer/merges.txt": 524619,
            "hunyuan3d-paintpbr-v2-1/tokenizer/special_tokens_map.json": 460,
            "hunyuan3d-paintpbr-v2-1/tokenizer/tokenizer_config.json": 807,
            "hunyuan3d-paintpbr-v2-1/tokenizer/vocab.json": 1059962,
            "hunyuan3d-paintpbr-v2-1/unet/attn_processor.py": 35104,
            "hunyuan3d-paintpbr-v2-1/unet/config.json": 911,
            "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin": 3925293863,
            "hunyuan3d-paintpbr-v2-1/unet/model.py": 25514,
            "hunyuan3d-paintpbr-v2-1/unet/modules.py": 45181,
            "hunyuan3d-paintpbr-v2-1/vae/config.json": 553,
            "hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.bin": 334707217,
            "LICENSE": 17915,
            "Notice.txt": 16644,
            "README.md": 3998,
        },
    },
    "dinov2": {
        "repository": "facebook/dinov2-giant",
        "revision": "611a9d42f2335e0f921f1e313ad3c1b7178d206d",
        "files": {"config.json": 548, "model.safetensors": 4546005432, "preprocessor_config.json": 436, "README.md": 3033},
    },
}

ASSETS = {
    "realesrgan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "bytes": 67040989,
    },
    # rembg finds its model by file name in U2NET_HOME, so this one keeps the name u2net.onnx.
    "background": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "sha256": "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        "bytes": 175997641,
    },
}

DOWNLOAD_COMMAND = "mcpbytes-image-to-3d --download"


def cache_dir():
    """This tool's files outside the Hugging Face cache: assets and compiled extensions."""
    root = os.environ.get("MCPBYTES_CACHE_DIR") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mcpbytes"
    return Path(root) / "image-to-3d"


def asset_path(name):
    return cache_dir() / "assets" / Path(ASSETS[name]["url"]).name


def missing_assets():
    return [name for name, spec in ASSETS.items() if not (asset_path(name).is_file() and asset_path(name).stat().st_size == spec["bytes"])]


def pending_bytes():
    return hub.pending_bytes(COMPONENTS) + sum(ASSETS[name]["bytes"] for name in missing_assets())


def _fetch_asset(name):
    spec, target = ASSETS[name], asset_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial, digest = target.with_name(target.name + ".part"), hashlib.sha256()
    with urllib.request.urlopen(spec["url"], timeout=60) as response, partial.open("wb") as file:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            file.write(chunk)
    if digest.hexdigest() != spec["sha256"]:
        partial.unlink()
        raise ToolError("download_corrupt", f"{spec['url']} does not match its pinned SHA-256", "run the download again")
    partial.replace(target)


def fetch():
    """Download whatever is missing: Hugging Face files resume, assets restart."""
    hub.fetch(COMPONENTS)
    for name in missing_assets():
        _fetch_asset(name)


def locate():
    """{name: path} of every model file, without network access; models_missing when a download is needed."""
    if gone := missing_assets():
        raise ToolError("models_missing", f"model files are not downloaded yet ({pending_bytes() / 1e9:.1f} GB)",
                        f"run `{DOWNLOAD_COMMAND}` once; missing: {', '.join(gone)}")
    return {**hub.locate(COMPONENTS, DOWNLOAD_COMMAND), **{name: asset_path(name) for name in ASSETS}}
