"""Tencent's two native extensions, JIT-compiled from the vendored sources on first run and cached.

Upstream builds them at install time (hy3dpaint/custom_rasterizer/setup.py and
hy3dpaint/DifferentiableRenderer/compile_mesh_painter.sh), which a wheel cannot do for every torch/CUDA/GPU combination;
torch.utils.cpp_extension.load builds the same sources with ninja under a cache folder keyed by Python, torch, CUDA and
the GPU's architecture, so a changed environment gets its own build and an unchanged one loads in a second.
"""

import os
import shutil
import sys
from pathlib import Path

from mcpbytes_core.errors import ToolError
from mcpbytes_core.gpu import require_cuda

from .models import cache_dir

WINDOWS = sys.platform == "win32"
PAINT = Path(__file__).parent / "vendor" / "hunyuan3d" / "hy3dpaint"
# Tencent ships a copy of the rasterizer with portable integer types for MSVC.
KERNEL = PAINT / "custom_rasterizer" / "lib" / ("custom_rasterizer_kernel_for_windows" if WINDOWS else "custom_rasterizer_kernel")
OPTIMIZE = ["/O2"] if WINDOWS else ["-O3"]  # upstream's flags; cpp_extension.load adds none of its own


def build_key(torch, capability):
    python = f"{sys.version_info[0]}{sys.version_info[1]}"
    return f"py{python}-torch{torch.__version__}-cuda{torch.version.cuda}-sm{capability[0]}{capability[1]}".replace("+", "_")


def toolchain_problem(torch):
    """(what is missing, what to install) for this platform's build tools, or None when they are all present."""
    from torch.utils import cpp_extension

    compiler = "cl" if WINDOWS else os.environ.get("CXX", "c++")
    if shutil.which(compiler) is None:
        return (f"no C++ compiler ({compiler}) was found",
                "install Visual Studio Build Tools (Desktop development with C++) and run from an x64 Native Tools prompt" if WINDOWS
                else "install a C++ compiler, e.g. `sudo apt install build-essential`")
    nvcc = cpp_extension.CUDA_HOME and Path(cpp_extension.CUDA_HOME, "bin", "nvcc.exe" if WINDOWS else "nvcc")
    if not nvcc or not nvcc.exists():
        return ("the CUDA compiler (nvcc) was not found",
                f"install the CUDA Toolkit {torch.version.cuda} (matching this PyTorch build) and set CUDA_HOME to it")
    return None


def build(log=print):
    """Build (first run) or load both extensions and register them under the names the vendored code imports.
    Loads no model weights: `--download` calls it to compile ahead of the first job."""
    if "custom_rasterizer_kernel" in sys.modules:
        return
    import ninja
    from torch.utils import cpp_extension

    torch = require_cuda()

    os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")  # the tool env's bin may not be on PATH
    capability = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "%d.%d" % capability)  # this GPU only: a fraction of the build time
    root = cache_dir() / "extensions" / build_key(torch, capability)

    def compile_one(name, sources, cuda):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        if not any(directory.glob(f"{name}.*")):
            log(f"compiling {name} once for this GPU and PyTorch (a few minutes)...")
        try:
            return cpp_extension.load(name, [str(source) for source in sources], extra_cflags=OPTIMIZE,
                                      build_directory=str(directory), with_cuda=cuda, verbose=False)
        except Exception as e:  # ninja, compiler and CUDA errors surface as RuntimeError, OSError or CalledProcessError
            problem = toolchain_problem(torch)
            if problem:
                raise ToolError("extensions_unavailable", f"cannot build Hunyuan3D's {name} extension: {problem[0]}", problem[1]) from None
            raise ToolError("extensions_unavailable", f"building Hunyuan3D's {name} extension failed: {str(e)[-2000:]}",
                            f"delete {directory} to rebuild from scratch; nvcc must support this GPU (sm_{capability[0]}{capability[1]})") from None

    kernel = compile_one("custom_rasterizer_kernel", [KERNEL / "rasterizer.cpp", KERNEL / "grid_neighbor.cpp", KERNEL / "rasterizer_gpu.cu"], True)
    painter = compile_one("mesh_inpaint_processor", [PAINT / "DifferentiableRenderer" / "mesh_inpaint_processor.cpp"], False)
    sys.modules["custom_rasterizer_kernel"] = kernel
    sys.modules["DifferentiableRenderer.mesh_inpaint_processor"] = painter
