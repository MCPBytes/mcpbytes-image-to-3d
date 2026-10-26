"""Local and cloud runs with the model replaced: seeds, the finishing pass, file checks, inputs, outputs and the
hosted API exchange."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import trimesh
from PIL import Image

from mcpbytes_core.cloud import Cloud
from mcpbytes_core.errors import ToolError
from mcpbytes_image_to_3d import cli, extensions, generate
from mcpbytes_image_to_3d.generate import NAMES, run_cloud, run_local, validate


def model(textured=True):
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    if textured:
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (64, 64), (200, 120, 40)), metallicRoughnessTexture=Image.new("RGB", (64, 64), (0, 90, 0)),
            metallicFactor=1.0, roughnessFactor=1.0)
        box.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(box.vertices), 2)), material=material)
    return box


class FakeEngine:
    settings = {"detail": "reduced", "views": 6, "view_resolution": 512, "texture_resolution": 1024}

    def __init__(self, textured=True):
        self.textured, self.calls = textured, []

    def process(self, image, out, seed):
        self.calls.append((image.mode, image.size, seed))
        mesh = model(self.textured)
        mesh.export(out / "model.glb")
        return {"seed": seed, "texture_resolution": 1024, "shape_faces": 12, "faces": len(mesh.faces), "vertices": len(mesh.vertices),
                "views": 6, "view_resolution": 512, "shape_ms": 1, "paint_ms": 1, "export_ms": 1}

    def versions(self):
        return {"torch": "test", "source_commit": "test"}


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "chair.png"
    Image.new("RGB", (96, 64), "orange").save(path)
    return path


def test_local_run_finishes_the_model_and_records_seed_settings_and_input(tmp_path, photo):
    engine = FakeEngine()
    result = run_local(photo, tmp_path / "out", seed=42, engine=engine, progress=lambda m: None)
    # The hosted stage id "model": the same job seed gives the model the same seed locally and in the cloud.
    assert engine.calls == [("RGB", (96, 64), int(hashlib.sha256(b"42:model").hexdigest()[:8], 16))]
    assert result["files"] == sorted([*NAMES, "generation.json", "provenance.json"]) and result["seed"] == 42
    generation = json.loads((tmp_path / "out" / "generation.json").read_text(encoding="utf-8"))
    assert generation["options"] == {"seed": 42, "detail": "auto"} and generation["settings"]["views"] == 6
    stage = generation["stages"][0]
    assert stage["summary"]["postprocess"]["applied"] and [f["name"] for f in stage["files"]] == NAMES
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["input_sha256"] == hashlib.sha256(photo.read_bytes()).hexdigest()
    assert provenance["model_revisions"]["main"] and provenance["torch"] == "test"
    finished = trimesh.load(tmp_path / "out" / "model.glb", force="mesh")
    assert max(finished.extents) == pytest.approx(1.0) and finished.bounds[0][1] == pytest.approx(0.0)


def test_an_untextured_model_is_rejected(tmp_path, photo):
    with pytest.raises(ToolError) as error:
        run_local(photo, tmp_path / "out", engine=FakeEngine(textured=False), progress=lambda m: None)
    assert error.value.code == "invalid_output"


def test_results_are_never_written_over_existing_files(tmp_path, photo):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "model.glb").write_bytes(b"mine")
    engine = FakeEngine()
    with pytest.raises(ToolError) as error:
        run_local(photo, tmp_path / "out", engine=engine)
    assert error.value.code == "output_exists" and not engine.calls
    assert (tmp_path / "out" / "model.glb").read_bytes() == b"mine"


def test_inputs_must_be_readable_images_of_bounded_size(tmp_path, monkeypatch):
    Image.new("RGB", (8, 8)).save(tmp_path / "in.gif")
    (tmp_path / "in.png").write_bytes(b"not an image")
    for path, needle in ((tmp_path / "in.gif", "PNG, JPEG or WebP"), (tmp_path / "in.png", "cannot read"), (tmp_path / "gone.png", "cannot read")):
        with pytest.raises(ToolError, match=needle) as error:
            run_local(path, tmp_path / "out", engine=FakeEngine())
        assert error.value.code == "invalid_image"
    monkeypatch.setattr(generate, "MAX_PIXELS", 100)
    Image.new("RGB", (11, 10)).save(tmp_path / "big.png")
    with pytest.raises(ToolError, match="million pixels"):
        run_local(tmp_path / "big.png", tmp_path / "out", engine=FakeEngine())


@pytest.mark.parametrize("seed, detail", [(-1, "auto"), (2**32, "auto"), (True, "auto"), ("7", "auto"), (None, "max")])
def test_options_outside_the_rules_are_rejected(seed, detail):
    with pytest.raises(ToolError) as error:
        validate(seed, detail)
    assert error.value.code == "invalid_options"


@pytest.fixture
def api(tmp_path, photo):
    """A stand-in for api.mcpbytes.com that records requests and serves a finished job's files."""
    published = tmp_path / "published"
    run_local(photo, published, engine=FakeEngine(), progress=lambda m: None)
    files = {path.name: path.read_bytes() for path in published.iterdir()}
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_POST(self):
            payload = json.loads(self.body())
            seen.append(("POST", self.path, dict(self.headers), payload))
            if self.path == "/v1/uploads":
                return self.reply(201, {"upload_id": "up_" + "a" * 26, "method": "PUT", "url": f"{base}/blob/up/1?t=x"})
            self.reply(202, {"id": "j_" + "b" * 26, "status": "queued"})

        def do_PUT(self):
            seen.append(("PUT", self.path, dict(self.headers), self.body()))
            self.reply(200, {})

        def do_GET(self):
            if self.path.startswith("/files/"):
                return self.reply(200, files[self.path[7:]], "application/octet-stream")
            self.reply(200, {"id": "j_" + "b" * 26, "status": "succeeded", "credits_charged": 50, "result": {"files": [
                {"name": name, "url": f"{base}/files/{name}", "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()]}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield Cloud("test-key", base), seen, files
    server.shutdown()


def test_cloud_run_uploads_the_image_asks_for_economy_and_verifies_downloads(api, tmp_path, photo, monkeypatch):
    cloud, seen, files = api
    monkeypatch.setattr("time.sleep", lambda s: None)
    result = run_cloud(photo, tmp_path / "out", seed=7, cloud=cloud, idempotency_key="my-key", progress=lambda m: None)
    upload, put, start = seen[0], seen[1], seen[2]
    assert upload[3]["sha256"] == hashlib.sha256(photo.read_bytes()).hexdigest()
    assert put[3] == photo.read_bytes() and "Authorization" not in put[2]  # the signed URL, never the API key
    assert start[1] == "/v1/tools/create_3d_model_from_image/jobs" and start[2]["Idempotency-Key"] == "my-key"
    assert start[3] == {"options": {"quality": "economy", "seed": 7}, "upload_id": "up_" + "a" * 26}
    assert result["credits_charged"] == 50 and result["job_id"] == "j_" + "b" * 26
    assert all((tmp_path / "out" / name).read_bytes() == data for name, data in files.items())


def test_the_default_output_folder_is_named_after_the_image_and_never_reused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert str(cli.default_output("photos/chair.jpg")) == "chair-3d"
    (tmp_path / "chair-3d").mkdir()
    assert str(cli.default_output("photos/chair.jpg")).startswith("chair-3d-")


def test_download_needs_a_gpu_then_fetches_the_files_and_builds_the_extensions(monkeypatch):
    with pytest.raises(SystemExit) as stop:
        cli.main([])
    assert stop.value.code == 2  # a run needs an image
    steps = []
    monkeypatch.setattr(cli.models, "pending_bytes", lambda: 0)
    monkeypatch.setattr(cli.models, "fetch", lambda: steps.append("fetch"))
    monkeypatch.setattr(extensions, "build", lambda log: steps.append("build"))

    def no_gpu():
        raise ToolError("gpu_unavailable", "this PyTorch installation cannot use a CUDA GPU")

    monkeypatch.setattr(cli, "require_cuda", no_gpu)
    with pytest.raises(SystemExit) as stop:
        cli.main(["--download"])
    assert stop.value.code == 1 and steps == []  # nothing is downloaded for a machine that cannot run it
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    cli.main(["--download"])
    assert steps == ["fetch", "build"]


def test_the_command_prints_only_its_json_summary_on_stdout(tmp_path, photo, monkeypatch, capsys):
    def noisy_run(*args, **kwargs):
        print("Models Loaded.")  # Hunyuan3D's own progress prints
        return {"output_dir": str(tmp_path / "out")}

    monkeypatch.setattr(cli.models, "pending_bytes", lambda: 0)
    monkeypatch.setattr(generate, "run_local", noisy_run)
    cli.main([str(photo), str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"output_dir": str(tmp_path / "out")} and "Models Loaded." in captured.err
