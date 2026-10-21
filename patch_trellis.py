# patch_trellis.py
#
# Сборка образа: убрать nvdiffrast из TRELLIS.2. Он нужен там только для
# растеризации в UV при запекании текстуры — подменяем импорт на свой
# uv_raster (тот же интерфейс). Рендер превью (renderers/) не трогаем: он
# грузится лениво, и воркер его не зовёт.
#
#   python patch_trellis.py <каталог TRELLIS.2>

import pathlib
import sys

OLD = "import nvdiffrast.torch as dr"
NEW = "import uv_raster as dr  # DocPats: nvdiffrast (research-only) replaced"
FILES = ["o-voxel/o_voxel/postprocess.py", "trellis2/pipelines/trellis2_texturing.py"]

root = pathlib.Path(sys.argv[1])
for rel in FILES:
    p = root / rel
    text = p.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        sys.exit(f"{rel}: expected exactly one '{OLD}' — TRELLIS.2 changed, check the patch")
    p.write_bytes(text.replace(OLD, NEW).encode("utf-8"))
    print("patched", rel)
