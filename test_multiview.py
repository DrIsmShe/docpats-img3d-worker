# CPU check of multiview against a stand-in of the TRELLIS.2 sampler chain.
#   python test_multiview.py
import sys
import types

import torch

# stand-in for trellis2.pipelines.samplers.flow_euler with the same MRO shape:
# guidance mixin -> CFG mixin -> FlowEulerSampler._inference_model (the model call)
calls = []


class FlowEulerSampler:
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        calls.append(cond.clone())
        return model(x_t, cond)


class CfgMixin:
    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, **kwargs):
        pos = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return guidance_strength * pos + (1 - guidance_strength) * neg


class Sampler(CfgMixin, FlowEulerSampler):
    pass


fake = types.ModuleType("trellis2.pipelines.samplers.flow_euler")
fake.FlowEulerSampler = FlowEulerSampler
for name in ["trellis2", "trellis2.pipelines", "trellis2.pipelines.samplers"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["trellis2.pipelines.samplers.flow_euler"] = fake

import multiview  # noqa: E402

# "model": prediction = x + mean of the condition tokens
model = lambda x, c: x + c.mean(dim=(1, 2)).view(-1, 1)  # noqa: E731
x = torch.zeros(1, 1)
conds = torch.stack([torch.full((3, 2), float(v)) for v in (1.0, 2.0, 6.0)])  # 3 views
neg = torch.zeros_like(conds)
s = Sampler()
ok = True


def check(name, cond, detail=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))
    ok = ok and bool(cond)


with multiview.views(3):
    calls.clear()
    out = s._inference_model(model, x, 0.5, conds, neg, guidance_strength=1.0)
check("prediction is the mean over views", torch.allclose(out, torch.tensor([[3.0]])), f"{out.item():.3f}")
check("each view seen once, empty condition once", len(calls) == 4, f"{len(calls)} model calls")
check("empty condition passed as a single row", calls[-1].shape[0] == 1)

with multiview.views(3):
    out = s._inference_model(model, x, 0.5, conds, neg, guidance_strength=3.0)
check("guidance applied on the averaged prediction", torch.allclose(out, torch.tensor([[9.0]])), f"{out.item():.3f}")

calls.clear()
single = s._inference_model(model, x, 0.5, conds[:1], neg[:1], guidance_strength=1.0)
check("sampler restored after the block", FlowEulerSampler._inference_model.__name__ == "_inference_model" and len(calls) == 2)
check("one picture unchanged", torch.allclose(single, torch.tensor([[1.0]])))

print("PASSED" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
