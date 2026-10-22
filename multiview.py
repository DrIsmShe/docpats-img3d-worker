# multiview.py
#
# Модель по нескольким рисункам одного предмета (спереди, сбоку, сзади).
#
# TRELLIS.2 обучена на одной картинке; отдельных весов «по ракурсам» нет.
# Берём приём из первой TRELLIS (там он был штатным, режим multidiffusion,
# без дообучения): на каждом шаге расчёта нейросеть делает предсказание по
# каждому рисунку, и предсказания усредняются. Модель вынуждена согласовать
# все ракурсы, а не додумывать невидимую сторону.
#
# Подмена — в самом низу цепочки сэмплера (FlowEulerSampler._inference_model):
# выше стоят «classifier-free guidance» и интервал, они получают уже
# усреднённое предсказание и работают как обычно. Пустое (негативное)
# условие одинаково для всех рисунков — его считаем один раз.

import contextlib
import functools

import torch


@contextlib.contextmanager
def views(n):
    """На время блока условие из n картинок усредняется по ракурсам."""
    if n <= 1:
        yield
        return
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler

    base = FlowEulerSampler._inference_model

    def averaged(self, model, x_t, t, cond=None, **kwargs):
        b = x_t.shape[0]
        if not isinstance(cond, torch.Tensor) or cond.shape[0] != n * b:
            return base(self, model, x_t, t, cond, **kwargs)
        if not torch.any(cond):  # негативное условие — нули, у всех ракурсов одно
            return base(self, model, x_t, t, cond[:b], **kwargs)
        preds = [base(self, model, x_t, t, cond[i * b : (i + 1) * b], **kwargs) for i in range(n)]
        return functools.reduce(lambda a, c: a + c, preds) * (1.0 / n)

    FlowEulerSampler._inference_model = averaged
    try:
        yield
    finally:
        FlowEulerSampler._inference_model = base


def run(pipeline, images, **run_kwargs):
    """pipeline.run по нескольким картинкам. Фон каждой вырезается отдельно."""
    if len(images) == 1:
        return pipeline.run(images[0], **run_kwargs)
    prepared = [pipeline.preprocess_image(i) for i in images]
    original = pipeline.get_cond

    # run() зовёт get_cond([одна картинка], …) — отдаём условие по всем
    def get_cond(_image, resolution, include_neg_cond=True):
        return original(prepared, resolution, include_neg_cond)

    pipeline.get_cond = get_cond
    try:
        with views(len(prepared)):
            return pipeline.run(prepared[0], preprocess_image=False, **run_kwargs)
    finally:
        del pipeline.get_cond  # снова метод класса
