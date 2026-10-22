# img3d_worker.py
#
# Воркер «Картинка -> 3D». Берёт из очереди DocPats задачу (одна картинка),
# строит по ней 3D-модель с материалами (TRELLIS.2, MIT) и отдаёт GLB.
#
# Что заменено ради лицензий (TRELLIS.2 сам MIT, но тянет чужие детали):
#   - вырезание фона: briaai/RMBG-2.0 (CC BY-NC) -> ZhengPeng7/BiRefNet (MIT);
#   - растеризация UV при запекании текстуры: nvdiffrast (только research)
#     -> uv_raster.py (Dockerfile подменяет импорт в o_voxel и trellis2).
# DINOv3 (кодировщик картинки) — лицензия Meta, коммерция разрешена. Берём
# его из timm (открытая раздача Hugging Face по той же лицензии), а не из
# закрытого facebook/: см. dino_timm.py. HF_TOKEN не нужен.
#
# Веса (~16 ГБ) не в образе, а в кэше: /runpod-volume, если к endpoint
# подключён сетевой диск, иначе диск машины. Первый запуск качает их с
# Hugging Face, дальше FlashBoot держит машину с уже скачанными весами.
#
# Запуск:
#   python img3d_worker.py --local picture.png out_dir   — одна картинка
#   python img3d_worker.py --serve                       — опрашивать сервер
# Для --serve: IMG3D_API (https://backend.docpats.com/api/v1/img3d) и
# IMG3D_WORKER_SECRET (тот же, что в .env сервера).

import argparse
import json
import os
import shutil
import tempfile
import time

WORKER_VERSION = "img3d-1"
IMAGE_TAG = os.environ.get("IMG3D_IMAGE", "local")

# качество: fast — сетка 512 (быстрее, грубее), full — 1024 каскадом
QUALITY = {"fast": "512", "full": "1024_cascade"}
TEXTURE_SIZE = 2048
# столько треугольников остаётся в модели: студия и браузер врача
# крутят её в реальном времени, миллион граней там не нужен
TARGET_FACES = 200000
# до сборки GLB сетку «Подробной» (миллионы граней) упрощаем до стольких:
# перестройка сетки и дерево поиска (CuMesh) иначе не помещаются в 32 ГБ
# RTX 5090 — 22.10.2026 «Подробная» упала «CuMesh … out of memory»
PRE_EXPORT_FACES = 4_000_000

TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
# декодер разреженной структуры TRELLIS.2 берёт из первой TRELLIS (тоже MIT)
SS_DEC = ("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16")
REMBG = "ZhengPeng7/BiRefNet"
# кодировщик картинки — DINOv3 ViT-L из timm (dino_timm.py)
IMAGE_COND = {"name": "TimmDinoV3FeatureExtractor", "args": {"model_name": "vit_large_patch16_dinov3.lvd1689m"}}
# какие веса нужны для fast и full; кодировщики (enc) — только для обучения
NEEDED = [
    "pipeline.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.*",
    "ckpts/shape_dec_next_dc_f16c32_fp16.*",
    "ckpts/tex_dec_next_dc_f16c32_fp16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.*",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cache_root():
    root = "/runpod-volume/img3d" if os.path.isdir("/runpod-volume") else os.environ.get("IMG3D_CACHE", "/models")
    os.makedirs(root, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(root, "hf"))
    return root


def fetch_weights(root):
    """Скачать недостающие веса и записать свой pipeline.json (фон и кодировщик — наши)."""
    from huggingface_hub import snapshot_download

    local = os.path.join(root, "trellis2")
    snapshot_download(TRELLIS_REPO, local_dir=local, allow_patterns=NEEDED)
    snapshot_download(SS_DEC[0], local_dir=os.path.join(local, SS_DEC[0]), allow_patterns=[SS_DEC[1] + ".*"])
    cfg_path = os.path.join(local, "pipeline.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["args"]["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": REMBG}}
    cfg["args"]["image_cond_model"] = IMAGE_COND
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, indent=1)
    return local


_pipeline = None


def pipeline():
    """Конвейер грузится один раз на машину: это ~16 ГБ весов в память GPU."""
    global _pipeline
    if _pipeline is None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        local = fetch_weights(cache_root())
        import attn_guard
        import dino_timm
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        dino_timm.register()
        # на Blackwell ядра xformers падают — внимание через PyTorch (attn_guard.py)
        if attn_guard.install():
            log("attention: torch sdpa (Blackwell)")

        p = Trellis2ImageTo3DPipeline.from_pretrained(local)
        p.cuda()
        _pipeline = p
    return _pipeline


def open_picture(path):
    from PIL import Image, ImageOps

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return img


def free_gpu():
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def is_oom(err):
    return "out of memory" in str(err).lower()


def export_glb(mesh, out_path):
    """GLB с текстурой. Не хватило памяти на перестройку сетки — повторяем
    облегчённо (упрощение без перестройки): модель чуть грубее по топологии,
    зато врач её получает, а не «Не получилось»."""
    try:
        return _export(mesh, out_path, remesh=True)
    except Exception as e:  # noqa: BLE001
        if not is_oom(e):
            raise
        log("export: out of memory with remesh, retrying without")
        free_gpu()
        return _export(mesh, out_path, remesh=False)


def _export(mesh, out_path, remesh):
    import o_voxel
    import trimesh

    tm = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=TARGET_FACES,
        texture_size=TEXTURE_SIZE,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    # узел зовём "model": по этому имени студия находит часть модели
    scene = trimesh.Scene()
    scene.add_geometry(tm, node_name="model", geom_name="model")
    scene.export(out_path, file_type="glb", extension_webp=True)
    ext = tm.bounds[1] - tm.bounds[0]
    return {"faces": int(len(tm.faces)), "size": [round(float(x), 4) for x in ext]}


def no_stage(_name):
    pass


def process(src, out_dir, quality="full", seed=42, stage=no_stage):
    """Картинка -> out_dir/model.glb. Возвращает манифест.

    «Подробной» не хватило памяти видеокарты — строим «Быструю»: врач
    получает модель, а в манифесте видно, что качество снижено."""
    quality = quality if quality in QUALITY else "full"
    try:
        return _process(src, out_dir, quality, seed, stage)
    except Exception as e:  # noqa: BLE001
        if quality == "fast" or not is_oom(e):
            raise
        log("full quality ran out of GPU memory, building fast instead")
        free_gpu()
        manifest = _process(src, out_dir, "fast", seed, stage)
        manifest["fellBack"] = True
        return manifest


def _process(src, out_dir, quality, seed, stage):
    import torch

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    p = pipeline()
    img = open_picture(src)
    stage("generating")
    with torch.no_grad():
        mesh = p.run(img, seed=int(seed), pipeline_type=QUALITY[quality])[0]
    # память нейросети больше не нужна — отдаём её сборке сетки
    stage("finishing")
    free_gpu()
    mesh.simplify(PRE_EXPORT_FACES)
    free_gpu()
    info = export_glb(mesh, os.path.join(out_dir, "model.glb"))
    del mesh
    torch.cuda.empty_cache()
    return {
        "version": WORKER_VERSION,
        "kind": "image",
        "quality": quality,
        "seed": int(seed),
        "textureSize": TEXTURE_SIZE,
        "durationSec": round(time.time() - t0),
        "device": "gpu",
        "parts": [{"id": "model", "label": "model", "faces": info["faces"]}],
        # TRELLIS.2 строит в кубе [-0.5, 0.5]: единицы условные, не миллиметры
        "sizeMm": [round(x * 1000, 1) for x in info["size"]],
    }


# ─── работа с сервером ───────────────────────────────────────────────────


def api(method, path, **kw):
    import requests

    base = os.environ["IMG3D_API"].rstrip("/")
    headers = {"x-img3d-secret": os.environ["IMG3D_WORKER_SECRET"], **kw.pop("headers", {})}
    r = requests.request(method, base + path, headers=headers, timeout=kw.pop("timeout", 120), **kw)
    r.raise_for_status()
    return r


def report_stage(jid):
    """Этап задачи — на страницу врача. Сбой связи работе не мешает."""

    def send(name):
        try:
            api("POST", f"/worker/jobs/{jid}/stage", json={"stage": name}, timeout=15)
        except Exception as e:  # noqa: BLE001
            log("stage not reported:", e)

    return send


def run_job(job):
    """Одна задача: скачать картинку, построить модель, отдать (или отказ)."""
    jid = job["_id"]
    log("job", jid)
    tmp = tempfile.mkdtemp(prefix="img3d_job_")
    try:
        src = os.path.join(tmp, "source" + job.get("sourceExt", ".png"))
        with api("GET", f"/worker/jobs/{jid}/source", stream=True, timeout=300) as r:
            with open(src, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        out = os.path.join(tmp, "out")
        stage = report_stage(jid)
        stage("starting")
        manifest = process(src, out, job.get("quality", "full"), job.get("seed", 42), stage)
        with open(os.path.join(out, "model.glb"), "rb") as glb:
            api(
                "POST",
                f"/worker/jobs/{jid}/result",
                files=[("model", ("model.glb", glb, "model/gltf-binary"))],
                data={"manifest": json.dumps(manifest)},
                timeout=600,
            )
        log("done", jid, manifest["durationSec"], "s")
        return True
    except Exception as e:  # noqa: BLE001 — любая ошибка задачи уходит на сервер
        # пустая маска фона: TRELLIS.2 падает на bbox пустого массива
        code = "NOTHING_FOUND" if "zero-size array" in str(e) else "FAILED"
        log("failed", jid, repr(e))
        try:
            api("POST", f"/worker/jobs/{jid}/fail", json={"code": code, "detail": repr(e)[:500]})
        except Exception as e2:  # noqa: BLE001
            log("report failed:", e2)
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def drain(worker_name=None, info=None):
    """Брать задачи, пока очередь не опустеет. Возвращает число обработанных."""
    name = worker_name or os.environ.get("RUNPOD_POD_ID") or os.environ.get("COMPUTERNAME") or "worker"
    me = {"worker": name, "version": WORKER_VERSION, "image": IMAGE_TAG, **(info or {})}
    done = 0
    while True:
        job = api("POST", "/worker/claim", json=me).json().get("job")
        if not job:
            return done
        run_job(job)
        done += 1


def main():
    ap = argparse.ArgumentParser(description="DocPats image -> 3D worker")
    ap.add_argument("--local", nargs=2, metavar=("PICTURE", "OUT_DIR"))
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--quality", default="full", choices=sorted(QUALITY))
    ap.add_argument("--idle", type=int, default=20, help="пауза между опросами, с")
    a = ap.parse_args()
    if a.local:
        print(json.dumps(process(a.local[0], a.local[1], a.quality), indent=2))
    elif a.serve:
        import requests

        while True:
            try:
                drain()
            except requests.RequestException as e:
                log("claim failed:", e)
            time.sleep(a.idle)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
