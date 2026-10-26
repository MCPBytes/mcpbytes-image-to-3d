"""mcpbytes-image-to-3d: one image in, a textured 3D model (GLB) out, made by Tencent's Hunyuan3D 2.1 on your own GPU
(or MCPBytes' with --cloud).

  mcpbytes-image-to-3d chair.jpg                 write chair-3d/: model.glb, model.web.glb, previews
  mcpbytes-image-to-3d chair.jpg models/chair    write into a new or empty folder of your choice
  mcpbytes-image-to-3d --download                fetch the pinned model files (~19 GB) and stop
  mcpbytes-image-to-3d --mcp                     serve create_3d_model_from_image over MCP (stdio)

Run it with `uvx --torch-backend=auto mcpbytes-image-to-3d ...`: uv installs the PyTorch build for your GPU.
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from mcpbytes_core.errors import ToolError
from mcpbytes_core.gpu import require_cuda

from . import models
from .generate import DETAILS


def log(message):
    print(message, file=sys.stderr, flush=True)


def download():
    """Everything a first run needs: the model files, then Hunyuan3D's native extensions compiled for this GPU."""
    from .extensions import build

    require_cuda()  # without a usable GPU, say so (and suggest --cloud) before 19 GB are downloaded for nothing
    if pending := models.pending_bytes():
        log(f"Downloading {pending / 1e9:.1f} GB of pinned model files (resumable; Hugging Face files go into its cache)")
    models.fetch()
    build(log)
    log("Model files and GPU extensions ready")


def default_output(image):
    """<image name>-3d next to where you run it; a timestamp keeps an earlier result instead of overwriting it."""
    folder = Path(f"{Path(image).stem}-3d")
    return folder if not folder.exists() else Path(f"{folder}-{time.strftime('%Y%m%d-%H%M%S')}")


def generate(args):
    from .generate import run_cloud, run_local

    out = args.output or default_output(args.image)
    if args.cloud:
        return run_cloud(args.image, out, args.seed, idempotency_key=args.idempotency_key, progress=log)
    if models.pending_bytes():
        download()
    return run_local(args.image, out, args.seed, args.detail, progress=log)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mcpbytes-image-to-3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, help="PNG, JPEG or WebP of one object (a transparent background is used as the cut-out)")
    parser.add_argument("output", nargs="?", type=Path, help="new or empty folder for the results (default: <image name>-3d)")
    parser.add_argument("--seed", type=int, help="0-4294967295; the same seed, image and hardware repeat a result")
    parser.add_argument("--detail", choices=DETAILS, default="auto",
                        help="full: 8 views at 768 px, 120k faces (41 GB of GPU memory measured); reduced: 6 views at 512 px, 40k faces (28 GB); auto picks full on 48 GB cards")
    parser.add_argument("--cloud", action="store_true", help="run on MCPBytes GPUs for credits instead (needs MCPBYTES_API_KEY)")
    parser.add_argument("--idempotency-key", help="with --cloud: the key printed by an interrupted run, to resume it without a new charge")
    parser.add_argument("--download", action="store_true", help="download the pinned model files (~19 GB) and exit")
    parser.add_argument("--mcp", action="store_true", help="run the MCP server on stdio (Claude, Cursor, VS Code and other MCP clients)")
    args = parser.parse_args(argv)
    if not (args.image or args.download or args.mcp):
        parser.error("give an image to model (or --download, or --mcp)")
    try:
        if args.download:
            download()
        elif args.mcp:
            from .server import serve

            serve(cloud=args.cloud, detail=args.detail)
        else:
            with contextlib.redirect_stdout(sys.stderr):  # Hunyuan3D prints progress; stdout carries only the JSON summary
                result = generate(args)
            print(json.dumps(result, indent=2))
    except ToolError as e:
        log(f"mcpbytes-image-to-3d: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
