"""MCP server (stdio) exposing create_3d_model_from_image, run on this machine's GPU or, with --cloud, on MCPBytes.

stdout carries the protocol: everything else goes to stderr. The model loads on the first call and stays loaded.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Annotated

import anyio.from_thread
import anyio.to_thread
from pydantic import Field

from mcp.server import MCPServer
from mcp.server.mcpserver import Context, Image
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcpbytes_core.errors import ToolError

from . import models
from .generate import TOOL, run_cloud, run_local

DESCRIPTION = (
    "Turn one image of an object into a textured 3D model on this machine's GPU, with Tencent's Hunyuan3D 2.1. "
    "Takes the absolute path of a PNG, JPEG or WebP; a transparent background is used as the cut-out, otherwise the "
    "background is removed. Writes model.glb (glTF 2.0 with PBR textures, +Y up, standing on y = 0, largest side 1), "
    "model.web.glb (Draco + WebP, for browsers and engines), model_preview.png (the model from six sides), "
    "model_<side>.png, model_views.json, generation.json and provenance.json into output_dir, and returns the preview. "
    "The back and underside are inferred: look at the preview before calling the model usable. Takes a few minutes "
    "per model, plus model loading on the first call."
)


def default_output():
    root = Path(os.environ.get("MCPBYTES_OUTPUT_DIR") or Path.home() / "mcpbytes" / "image-to-3d")
    return root / time.strftime("%Y%m%d-%H%M%S")


def build(cloud=False, detail="auto"):
    server = MCPServer("mcpbytes-image-to-3d", instructions=(
        "create_3d_model_from_image writes a 3D model to a local folder. Pass an absolute output_dir inside the user's "
        "project when they want the files there. Look at the returned preview before calling the model usable."))
    engine, loading = [None], threading.Lock()

    def local_engine():
        with loading:
            if engine[0] is None:
                from .engine import Engine

                engine[0] = Engine(models.locate(), detail)
        return engine[0]

    @server.tool(name=TOOL, title="Create a 3D model from an image", description=DESCRIPTION, structured_output=False)
    async def create_3d_model_from_image(
        image: Annotated[str, Field(description="Absolute path of a PNG, JPEG or WebP of one object you have the right to use")],
        ctx: Context,
        seed: Annotated[int | None, Field(ge=0, le=0xFFFFFFFF, description="Repeat a result with the same image and hardware")] = None,
        output_dir: Annotated[str | None, Field(description="New or empty folder; default ~/mcpbytes/image-to-3d/<timestamp>")] = None,
        idempotency_key: Annotated[str | None, Field(max_length=100, description="Cloud mode: reuse after a lost response to avoid a second charge")] = None,
    ) -> list:
        out = Path(output_dir) if output_dir else default_output()
        steps = [0]

        def progress(message):
            steps[0] += 1
            anyio.from_thread.run(ctx.report_progress, steps[0], None, message)

        def work():
            if cloud:
                return run_cloud(image, out, seed, idempotency_key=idempotency_key, progress=progress)
            if models.pending_bytes():
                # A 19 GB download inside a tool call would outlive any client's timeout.
                raise ToolError("models_missing", "the model files are not downloaded yet",
                                f"ask the user to run `{models.DOWNLOAD_COMMAND}` once")
            if engine[0] is None:
                progress("Loading the model")
            return run_local(image, out, seed, detail, engine=local_engine(), progress=progress)

        try:
            result = await anyio.to_thread.run_sync(work)
        except ToolError as e:
            raise McpToolError(str(e)) from None
        preview = Path(result["output_dir"]) / result["preview"]
        return [json.dumps(result, indent=2), Image(data=preview.read_bytes(), format="png")]

    return server


def serve(cloud=False, detail="auto"):
    build(cloud, detail).run("stdio")
