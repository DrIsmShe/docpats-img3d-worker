# attn_guard.py
#
# Внимание на картах Blackwell (RTX 5090, RTX PRO 6000) без ядер xformers.
#
# 22.10.2026 на RTX 5090 xformers 0.0.31 выбрал ядро flash-attention для
# Hopper и упал: «CUDA error … flash-attention/hopper/… invalid argument».
# Ошибка не исключение, а аварийный выход процесса — задача врача оставалась
# «в работе», пока сервер не вернул её в очередь.
#
# TRELLIS.2 зовёт внимание только через xformers.ops.memory_efficient_attention
# (плотное и разреженное). На Blackwell подменяем её на
# torch.nn.functional.scaled_dot_product_attention — встроенное внимание
# PyTorch, собранное под эти карты. Разреженные последовательности
# (BlockDiagonalMask) считаются так же, как у xformers: каждая
# последовательность видит только свои ключи.
#
# На старых картах ничего не меняется — там xformers быстрее и работает.

import torch
import torch.nn.functional as F

# больше стольких последовательностей разной длины — дополняем до общей
# длины с маской, меньше — считаем по одной (у полного внимания их 1–2,
# зато длинные: маска на 50 тыс. × 50 тыс. не поместилась бы в память)
LOOP_LIMIT = 8


def _sdpa(q, k, v, scale):
    """q, k, v в раскладке xformers [B, M, H, C] -> [B, M, H, C]."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
    return out.transpose(1, 2)


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, *, op=None, **kwargs):
    if attn_bias is None:
        return _sdpa(query, key, value, scale)
    qs = list(attn_bias.q_seqinfo.seqstart_py)
    ks = list(attn_bias.k_seqinfo.seqstart_py)
    n = len(qs) - 1
    lq = [qs[i + 1] - qs[i] for i in range(n)]
    lk = [ks[i + 1] - ks[i] for i in range(n)]
    q, k, v = query[0], key[0], value[0]  # [total, H, C]
    if len(set(lq)) == 1 and len(set(lk)) == 1:
        # одинаковые длины — просто пачка последовательностей
        out = _sdpa(q.reshape(n, lq[0], *q.shape[1:]), k.reshape(n, lk[0], *k.shape[1:]), v.reshape(n, lk[0], *v.shape[1:]), scale)
        return out.reshape(1, -1, *out.shape[2:])
    if n <= LOOP_LIMIT:
        parts = [
            _sdpa(q[qs[i] : qs[i + 1]][None], k[ks[i] : ks[i + 1]][None], v[ks[i] : ks[i + 1]][None], scale)[0]
            for i in range(n)
        ]
        return torch.cat(parts, 0)[None]
    # много коротких последовательностей разной длины (окна): дополнение и маска
    mq, mk = max(lq), max(lk)
    dev = q.device
    lq_t = torch.tensor(lq, device=dev)
    lk_t = torch.tensor(lk, device=dev)
    qi = torch.arange(mq, device=dev)[None] < lq_t[:, None]  # [n, mq]
    ki = torch.arange(mk, device=dev)[None] < lk_t[:, None]  # [n, mk]
    pq = q.new_zeros(n, mq, *q.shape[1:])
    pk = k.new_zeros(n, mk, *k.shape[1:])
    pv = v.new_zeros(n, mk, *v.shape[1:])
    pq[qi] = q
    pk[ki] = k
    pv[ki] = v
    mask = ki[:, None, None, :].expand(n, 1, mq, mk)
    out = F.scaled_dot_product_attention(pq.transpose(1, 2), pk.transpose(1, 2), pv.transpose(1, 2), attn_mask=mask, scale=scale)
    return out.transpose(1, 2)[qi][None]


def needed():
    """True на картах, где ядра xformers не работают (Blackwell и новее)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10


def install():
    """Подменить xformers.ops.memory_efficient_attention, если карта этого требует."""
    if not needed():
        return False
    import xformers.ops as xops

    xops.memory_efficient_attention = memory_efficient_attention
    return True
