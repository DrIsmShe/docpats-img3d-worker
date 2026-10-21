# DocPats image → 3D worker

A RunPod Serverless worker that turns one picture (PNG, JPEG or WebP) into a
textured 3D model (GLB with PBR materials) using
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2). It is the GPU side of the
"Image → 3D" feature of [DocPats](https://docpats.com); jobs are taken from the
DocPats queue over HTTP.

## What is changed compared to stock TRELLIS.2

TRELLIS.2 itself is MIT, but two of its dependencies are licensed for
non-commercial or research use only. This worker replaces both:

- **Background removal:** `briaai/RMBG-2.0` (CC BY-NC 4.0) is replaced with
  [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) (MIT).
- **Texture baking:** `nvdiffrast` (NVIDIA Source Code License, research only)
  is not installed. TRELLIS.2 uses it only to rasterise triangles in UV space;
  `uv_raster.py` is a small PyTorch rasteriser with the same interface, and
  `patch_trellis.py` points the two imports at it during the image build.

`selftest.py` runs inside every built image and fails the build if
`nvdiffrast` or RMBG-2.0 come back.

## Runtime configuration (RunPod endpoint variables)

| variable | meaning |
|---|---|
| `IMG3D_API` | queue API base URL |
| `IMG3D_WORKER_SECRET` | shared secret for the queue API |
| `HF_TOKEN` | Hugging Face token of an account that accepted the DINOv3 licence |

Weights (~16 GB) are not baked into the image; they are downloaded on first
start into `/runpod-volume` (if a network volume is attached) or local disk.
A GPU with at least 24 GB of memory is required.

## Third-party components

- TRELLIS.2, CuMesh, FlexGEMM, o-voxel — MIT (Microsoft / J. Xiang).
- DINOv3 image encoder — DINOv3 License (Meta). Built with DINOv3.
- BiRefNet — MIT.

The code in this repository is released under the MIT License.
