# selftest.py
#
# Проверка собранного образа в CI (без GPU): всё ли на месте и не
# вернулись ли детали с некоммерческой лицензией. Не прошла — образ не
# публикуется. Сама генерация (нужна GPU 24 ГБ) здесь не проверяется.

import importlib.util
import os
from json import dumps as json_dumps
import pathlib
import sys

ok = True


def check(name, cond, detail=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)
    ok = ok and bool(cond)


# 1. модули воркера и собранные расширения
for mod in ["img3d_worker", "runpod_handler", "uv_raster", "dino_timm", "timm", "trellis2", "o_voxel", "cumesh", "flex_gemm", "xformers", "utils3d", "transformers", "runpod"]:
    check(f"module {mod}", importlib.util.find_spec(mod) is not None)

# 2. nvdiffrast в образе нет, и TRELLIS.2 его не импортирует там, где зовёт воркер
check("nvdiffrast not installed", importlib.util.find_spec("nvdiffrast") is None)
root = pathlib.Path("/opt/TRELLIS.2")
spec = importlib.util.find_spec("o_voxel")
ovoxel_dir = pathlib.Path(spec.origin).parent if spec and spec.origin else None
targets = [root / "trellis2/pipelines/trellis2_texturing.py"]
if ovoxel_dir:
    targets.append(ovoxel_dir / "postprocess.py")
for p in targets:
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    check(f"patched {p.name}", "import uv_raster as dr" in text and "import nvdiffrast" not in text)

# 2б. воркер берёт задачи ровно на тех поколениях карт, под которые собран образ
import runpod_handler  # noqa: E402

built = {tuple(int(x) for x in a.split(".")) for a in os.environ.get("TORCH_CUDA_ARCH_LIST", "").split(";") if a}
check("handler arch list matches build", built == runpod_handler.BUILT_FOR, f"{sorted(built)}")

# 3. фон вырезает BiRefNet (MIT), а не RMBG-2.0 (CC BY-NC)
import img3d_worker as w  # noqa: E402

check("rembg is BiRefNet MIT", w.REMBG == "ZhengPeng7/BiRefNet")
# DINOv3 — из открытой раздачи timm, не из закрытого facebook/
check("image encoder from timm", w.IMAGE_COND["name"] == "TimmDinoV3FeatureExtractor" and "facebook" not in json_dumps(w.IMAGE_COND))
import dino_timm  # noqa: E402

check("timm encoder class", hasattr(dino_timm, "register") and hasattr(dino_timm.TimmDinoV3FeatureExtractor, "extract_features"))
src = pathlib.Path(w.__file__).read_text(encoding="utf-8")
check("RMBG-2.0 not referenced in worker", "briaai" not in src.replace("briaai/RMBG-2.0 (CC BY-NC)", ""))

# 4. замена растеризатора считает правильно (на процессоре)
import torch  # noqa: E402
import uv_raster as dr  # noqa: E402

uv = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
tri = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)
pos = torch.cat([uv * 2 - 1, torch.zeros(4, 1), torch.ones(4, 1)], -1).unsqueeze(0)
rast, _ = dr.rasterize(dr.RasterizeCudaContext(), pos, tri, resolution=[64, 64])
mask = rast[0, ..., 3] > 0
covered = int(mask.sum())
# центры пикселей 6..57 по обеим осям лежат внутри квадрата 0.1..0.9
check("uv_raster coverage", covered == 52 * 52, f"{covered} px")
attr = torch.cat([uv, torch.zeros(4, 1)], -1)
out, _ = dr.interpolate(attr.unsqueeze(0), rast, tri)
ys, xs = torch.nonzero(mask, as_tuple=True)
want = torch.stack([(xs.float() + 0.5) / 64, (ys.float() + 0.5) / 64], -1)
err = float((out[0][mask][:, :2] - want).abs().max())
check("uv_raster interpolation", err < 1e-5, f"max err {err:.2e}")

print("SELFTEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
