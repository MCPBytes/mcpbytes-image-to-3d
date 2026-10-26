# Vendored Hunyuan3D 2.1 source

- Source: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- Commit: `82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- License: Tencent Hunyuan 3D 2.1 Community License Agreement (`LICENSE`, `Notice.txt`, both unchanged).

## Included

The subset MCPBytes' hosted runner builds from (same paths as upstream, relative to the repository root):

- `hy3dshape/hy3dshape/` (shape pipeline)
- `hy3dpaint/hunyuanpaintpbr/` (paint diffusion pipeline, loaded by diffusers as a custom pipeline)
- `hy3dpaint/DifferentiableRenderer/` (renderer, baking, `mesh_inpaint_processor.cpp`)
- `hy3dpaint/utils/`, `hy3dpaint/textureGenPipeline.py`, `hy3dpaint/cfgs/`
- `hy3dpaint/custom_rasterizer/` (CUDA rasterizer; `lib/custom_rasterizer_kernel_for_windows/` is used on Windows)
- `torchvision_fix.py`, `LICENSE`, `Notice.txt`

Everything else in the upstream repository (apps, API server, demos, notebooks, assets, docker files, training code
outside the directories above) is not included. No compiled files were present or added.

## Local changes

None: every file above is byte-identical to the commit. What the harness changes, it changes from its own modules:

- `engine.py` puts `hy3dshape/`, `hy3dpaint/` and `hy3dpaint/custom_rasterizer/` on `sys.path`, stands in for
  `bpy`, `pymeshlab` and `pytorch_lightning` (imported at module load, never run), hands the paint pipeline the
  local model snapshot instead of `huggingface_hub.snapshot_download`, replaces the fixed paint seed 0 with the job's
  seed on the instances, and writes the GLB itself instead of `convert_obj_to_glb` (Blender).
- `extensions.py` JIT-compiles `custom_rasterizer_kernel` and `mesh_inpaint_processor` from these sources with
  `torch.utils.cpp_extension.load` instead of `setup.py` / `compile_mesh_painter.sh`.
- `upscale.py` provides the `realesrgan.RealESRGANer` and `basicsr.archs.rrdbnet_arch.RRDBNet` that
  `hy3dpaint/utils/image_super_utils.py` imports, so `torchvision_fix.py` (a basicsr workaround) is not called.

Any future edit to a file here must carry, after Tencent's header comment,
`# Changed by MCPBytes.com for mcpbytes-image-to-3d (<date>): <what changed>.` and be listed above.
