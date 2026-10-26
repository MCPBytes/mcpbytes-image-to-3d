"""create_3d_model_from_image, run on this machine's GPU or on the hosted API. Both write the same model files into
one folder: model.glb, model.web.glb, model_preview.png, six model_<side>.png views and model_views.json, plus
generation.json (and provenance.json for a local run).

The harness never touches the model directly: `engine.Engine` writes the raw model.glb, then the finishing pass,
previews and file checks here are the same for any model behind it.
"""

import hashlib
import json
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from mcpbytes_core.errors import ToolError
from mcpbytes_core.gpu import job_seed, stage_seed

from .finish import artifact, finish_outputs, write_model_previews
from .models import COMPONENTS

TOOL = "create_3d_model_from_image"
STAGE = "model"  # the hosted stage id: a job seed gives the model the same seed locally and in the cloud
DETAILS = ("auto", "full", "reduced")
NAMES = ["model.glb", "model.web.glb", "model_preview.png",
         *(f"model_{side}.png" for side in ("front", "back", "left", "right", "top", "bottom")), "model_views.json"]
KINDS = {".png": "image/png", ".glb": "model/gltf-binary", ".json": "application/json"}
MAX_PIXELS = 25_000_000  # the hosted tool's decoding bound


def stderr(message):
    print(message, file=sys.stderr, flush=True)


def validate(seed=None, detail="auto"):
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 0xFFFFFFFF):
        raise ToolError("invalid_options", "seed must be an integer from 0 to 4294967295")
    if detail not in DETAILS:
        raise ToolError("invalid_options", f"detail must be one of {', '.join(DETAILS)}")


def load_image(path):
    """(image, SHA-256) of a PNG, JPEG or WebP; a transparent background is kept as the cut-out."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise ToolError("invalid_image", f"cannot read {path}: {e.strerror}") from None
    try:
        with Image.open(Path(path)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP"):
                raise ToolError("invalid_image", "the image must be a PNG, JPEG or WebP")
            if image.width * image.height > MAX_PIXELS:
                raise ToolError("invalid_image", "the image must have at most 25 million pixels")
            image.load()
            picture = image.copy()
    except (OSError, Image.DecompressionBombError) as e:
        raise ToolError("invalid_image", f"cannot read the image: {e}") from None
    return picture, hashlib.sha256(data).hexdigest()


def prepare_output(out):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ToolError("output_exists", f"{out} is not an empty folder", "choose a new output folder; results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_local(image, out, seed=None, detail="auto", engine=None, progress=stderr):
    """Model one image on this machine's GPU. Long-lived callers (the MCP server) pass a loaded `engine` to reuse."""
    validate(seed, detail)
    picture, image_sha256 = load_image(image)
    out = prepare_output(out)
    seed = job_seed(seed)
    if engine is None:
        from .engine import Engine
        from .models import locate

        progress("Loading the model (the first run also compiles its GPU extensions)")
        engine = Engine(locate(), detail, progress)
    started = time.monotonic()
    progress("Modeling")
    summary = engine.process(picture, out, stage_seed(seed, STAGE))
    progress("Finishing")
    if finished := finish_outputs(out, NAMES):
        summary["postprocess"] = finished
    if previews := write_model_previews(out, NAMES):
        summary["previews"] = previews
    if sorted(p.name for p in out.iterdir()) != sorted(NAMES):
        raise ToolError("invalid_output", "the model run did not produce the expected files")
    files = [artifact(out / name, KINDS[Path(name).suffix], compressed=name.endswith(".web.glb")) for name in NAMES]
    if any(item["name"].endswith(".glb") and not item["validation"].get("textured") for item in files):
        raise ToolError("invalid_output", "generated model has no embedded texture")
    (out / "generation.json").write_text(json.dumps({
        "tool": TOOL, "runtime": "local", "options": {"seed": seed, "detail": detail}, "settings": engine.settings,
        "stages": [{"stage_id": STAGE, "label": "Modeling", "files": files, "summary": summary,
                    "timings": {"total_ms": int((time.monotonic() - started) * 1000)}}]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps({
        "runtime": f"mcpbytes-image-to-3d {version('mcpbytes-image-to-3d')}", "input_sha256": image_sha256,
        "model_revisions": {name: spec["revision"] for name, spec in COMPONENTS.items()}, **engine.versions()},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summarize(out, seed=seed, faces=summary["faces"], detail=engine.settings.get("detail", detail))


def run_cloud(image, out, seed=None, cloud=None, idempotency_key=None, progress=stderr):
    """Model one image on the hosted API (economy quality) for credits. Reuse `idempotency_key` after a lost response:
    no second charge."""
    from mcpbytes_core.cloud import Cloud

    validate(seed)
    load_image(image)
    cloud = cloud or Cloud()
    out = prepare_output(out)
    key = idempotency_key or str(uuid.uuid4())
    progress(f"Request key {key}: reuse it with --idempotency-key if this run is interrupted")
    upload_id = cloud.upload(image)
    job = cloud.start(TOOL, {"quality": "economy", **({"seed": seed} if seed is not None else {})}, key, upload_id)
    progress(f"Job {job['id']}")
    job = cloud.wait(job, lambda j: progress((j.get("workflow") or {}).get("stage") or j["status"]))
    cloud.download(job, out)
    return summarize(out, job_id=job["id"], credits_charged=job.get("credits_charged"))


def summarize(out, **extra):
    return {"output_dir": str(out.resolve()), "files": sorted(p.name for p in out.iterdir()), "model": "model.glb",
            "web_model": "model.web.glb", "preview": "model_preview.png",
            "notes": ["Look at model_preview.png (six sides) before using the model: the back and underside are inferred."],
            **extra}
