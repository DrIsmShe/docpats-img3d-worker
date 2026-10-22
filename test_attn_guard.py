# CPU check of attn_guard against plain softmax attention, per sequence.
#   python test_attn_guard.py
import random
from types import SimpleNamespace

import torch

from attn_guard import memory_efficient_attention


def mask_for(lq, lk):
    start = lambda ls: [sum(ls[:i]) for i in range(len(ls) + 1)]  # noqa: E731
    return SimpleNamespace(q_seqinfo=SimpleNamespace(seqstart_py=start(lq)), k_seqinfo=SimpleNamespace(seqstart_py=start(lk)))


def reference(q, k, v, lq, lk):
    out, a, b = [], 0, 0
    for x, y in zip(lq, lk):
        qq, kk, vv = q[0, a : a + x], k[0, b : b + y], v[0, b : b + y]  # [L, H, C]
        w = torch.einsum("qhc,khc->hqk", qq, kk) / qq.shape[-1] ** 0.5
        out.append(torch.einsum("hqk,khc->qhc", w.softmax(-1), vv))
        a, b = a + x, b + y
    return torch.cat(out)[None]


def run(lq, lk, h=4, c=16):
    q = torch.randn(1, sum(lq), h, c)
    k = torch.randn(1, sum(lk), h, c)
    v = torch.randn(1, sum(lk), h, c)
    got = memory_efficient_attention(q, k, v, mask_for(lq, lk))
    want = reference(q, k, v, lq, lk)
    return float((got - want).abs().max())


torch.manual_seed(0)
random.seed(0)
cases = {
    "no mask": None,
    "equal lengths": ([64] * 6, [64] * 6),
    "few different": ([50, 70, 33], [50, 70, 33]),
    "cross different": ([40, 10], [25, 60]),
    "many windows": ([random.randint(5, 40) for _ in range(30)],) * 2,
}
ok = True
for name, lens in cases.items():
    if lens is None:
        q, k, v = (torch.randn(2, 77, 4, 16) for _ in range(3))
        got = memory_efficient_attention(q, k, v)
        want = torch.cat([reference(q[i : i + 1], k[i : i + 1], v[i : i + 1], [77], [77]) for i in range(2)])
        err = float((got - want).abs().max())
    else:
        err = run(*lens)
    print(f"{name:16s} max err {err:.2e}")
    ok = ok and err < 1e-4
print("PASSED" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
