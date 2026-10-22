# runpod_handler.py
#
# Воркер «Картинка -> 3D» на RunPod Serverless — так же, как КТ -> 3D:
# сервер DocPats «будит» endpoint пустым запросом, GPU забирает из очереди
# все ждущие картинки по HTTP и гаснет. Платим только за секунды работы.
#
# IMG3D_API и IMG3D_WORKER_SECRET задаются в переменных endpoint'а
# RunPod, а не во входе запроса: вход виден в панели RunPod и в журналах.

import img3d_worker as w


# Поколения карт, под которые собраны расширения (TORCH_CUDA_ARCH_LIST в
# Dockerfile). На другой карте ядра просто не найдутся — задачу врача
# такая машина провалила бы, поэтому она задач не берёт вовсе.
BUILT_FOR = {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)}


def gpu_info():
    """(имя видеокарты, ГБ памяти, (major, minor)); (None, 0, None) — CUDA не работает."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, 0, None
        props = torch.cuda.get_device_properties(0)
        return props.name, round(props.total_memory / 2**30), (props.major, props.minor)
    except Exception:  # noqa: BLE001
        return None, 0, None


def handler(event):
    # Без CUDA или с картой меньше 24 ГБ задачи не берём: TRELLIS.2 на такой
    # не поместится. Ошибка запроса заставит сервер разбудить RunPod снова.
    gpu, mem, arch = gpu_info()
    if not gpu:
        raise RuntimeError("CUDA unavailable on this worker — not taking jobs")
    if arch not in BUILT_FOR:
        raise RuntimeError(f"{gpu} (sm_{arch[0]}{arch[1]}) is not in the image build — not taking jobs")
    if mem < 23:
        raise RuntimeError(f"{gpu} has {mem} GB — TRELLIS.2 needs 24 GB")
    processed = w.drain(info={"gpu": gpu})
    return {"processed": processed, "gpu": gpu, "image": w.IMAGE_TAG}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
