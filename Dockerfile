# Образ воркера «Картинка -> 3D» (TRELLIS.2) для RunPod Serverless (GPU).
# Собирается GitHub Actions (.github/workflows/img3d-image.yml) и кладётся в
# ghcr.io. Секретов внутри нет: адрес API, ключ воркера и HF_TOKEN задаются в
# RunPod. Весов внутри тоже нет (~16 ГБ) — воркер качает их при первом
# запуске в кэш (img3d_worker.py, cache_root).
#
# devel, а не runtime: CuMesh, FlexGEMM и o-voxel собираются из исходников
# (нужен nvcc). GPU для сборки не нужен — архитектуры заданы списком.
#
# CUDA 12.8 и архитектуры до 12.0: у RunPod в бессерверном режиме свободны
# в основном карты Blackwell (RTX 5090, RTX PRO 6000) — 22.10.2026 на
# классических 24–48 ГБ машину не дали за сутки. Внимание — xformers:
# у flash-attn 2 нет надёжной сборки под Blackwell.

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"     ATTN_BACKEND=xformers \
    OPENCV_IO_ENABLE_OPENEXR=1

RUN apt-get update && apt-get install -y --no-install-recommends git libjpeg-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
    && pip install --no-deps "xformers==0.0.31.post1" --index-url https://download.pytorch.org/whl/cu128

# Коды закреплены коммитами: TRELLIS.2 меняется, а патч ниже рассчитан на
# конкретные строки. Все три — MIT.
ARG TRELLIS_SHA=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG CUMESH_SHA=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_SHA=6dd94a859c26ee8246888502eada3dd8ad85532e
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh && cd /tmp/CuMesh && git checkout $CUMESH_SHA \
    && git submodule update --init --recursive && pip install . --no-build-isolation \
    && git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM && cd /tmp/FlexGEMM && git checkout $FLEXGEMM_SHA \
    && git submodule update --init --recursive && pip install . --no-build-isolation \
    && rm -rf /tmp/CuMesh /tmp/FlexGEMM

# nvdiffrast НЕ ставим (лицензия только для исследований): patch_trellis.py
# переводит запекание текстуры на uv_raster.py до установки o-voxel.
COPY uv_raster.py patch_trellis.py ./
RUN git clone https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2 && cd /opt/TRELLIS.2 && git checkout $TRELLIS_SHA \
    && git submodule update --init --recursive \
    && python /app/patch_trellis.py /opt/TRELLIS.2 \
    && pip install /opt/TRELLIS.2/o-voxel --no-build-isolation \
    && rm -rf /opt/TRELLIS.2/assets /opt/TRELLIS.2/data_toolkit /opt/TRELLIS.2/.git
ENV PYTHONPATH=/opt/TRELLIS.2:/app

COPY *.py ./

# Тег образа (короткий коммит) — воркер сообщает его серверу с каждой задачей.
ARG GIT_SHA=local
ENV IMG3D_IMAGE=$GIT_SHA

CMD ["python", "-u", "runpod_handler.py"]
