# uv_raster.py
#
# Замена nvdiffrast для TRELLIS.2. Лицензия nvdiffrast разрешает только
# исследования, а TRELLIS.2 зовёт её в одном месте: при запекании текстуры
# треугольники рисуются в пространстве UV (z = 0, w = 1), чтобы узнать, какой
# точке поверхности соответствует каждый пиксель текстуры. Для этого хватает
# простого растеризатора на torch — здесь он, с тем же интерфейсом
# (RasterizeCudaContext, rasterize, interpolate), что и у nvdiffrast.
#
# Ограничение: глубины нет — пиксель получает любой из накрывших его
# треугольников. В UV-развёртке треугольники не пересекаются, так что для
# запекания это не важно; для обычного 3D-рендера этот модуль не годится.
#
# Соглашения как у nvdiffrast: строка 0 — низ (y = -1), центр пикселя на
# (i + 0.5); rast[..., 0:2] — веса вершин 0 и 1 (вес вершины 2 = 1 - u - v),
# rast[..., 3] — номер треугольника + 1 (0 — пусто).

import torch

# сколько пар «треугольник × пиксель-кандидат» считать за раз
BUDGET = 1 << 24


class RasterizeCudaContext:
    def __init__(self, device=None):
        self.device = device


def rasterize(ctx, pos, tri, resolution, ranges=None, grad_db=True):
    h, w = int(resolution[0]), int(resolution[1])
    p = pos[0] if pos.dim() == 3 else pos
    dev = p.device
    tri = tri.long()
    xy = p[:, :2] / p[:, 3:4]
    z = p[:, 2] / p[:, 3]
    # координаты в пикселях: центр пикселя i лежит на целом i
    px = (xy[:, 0] + 1) * (w / 2) - 0.5
    py = (xy[:, 1] + 1) * (h / 2) - 0.5

    out = torch.zeros((h, w, 4), dtype=torch.float32, device=dev)
    if tri.shape[0] == 0:
        return out.unsqueeze(0), None

    tx, ty = px[tri], py[tri]  # (F, 3)
    x0 = tx.min(1).values.ceil().clamp(0, w - 1).long()
    x1 = tx.max(1).values.floor().clamp(0, w - 1).long()
    y0 = ty.min(1).values.ceil().clamp(0, h - 1).long()
    y1 = ty.max(1).values.floor().clamp(0, h - 1).long()
    bw = (x1 - x0 + 1).clamp(min=0)
    bh = (y1 - y0 + 1).clamp(min=0)
    # треугольник целиком за краем или между центрами пикселей
    off = (tx.max(1).values < 0) | (tx.min(1).values > w - 1) | (ty.max(1).values < 0) | (ty.min(1).values > h - 1)
    counts = torch.where(off, torch.zeros_like(bw), bw * bh)

    # пачки треугольников, у которых кандидатов в сумме не больше BUDGET
    ends = torch.cumsum(counts, 0)
    start = 0
    n = tri.shape[0]
    while start < n:
        base = ends[start - 1] if start > 0 else torch.zeros((), dtype=ends.dtype, device=dev)
        stop = int(torch.searchsorted(ends, base + BUDGET, right=True))
        stop = max(stop, start + 1)
        _fill(out, slice(start, stop), counts, x0, y0, bw, tx, ty, z, tri)
        start = stop
    return out.unsqueeze(0), None


def _fill(out, sl, counts, x0, y0, bw, tx, ty, z, tri):
    c = counts[sl]
    total = int(c.sum())
    if total == 0:
        return
    dev = out.device
    local = torch.repeat_interleave(torch.arange(c.shape[0], device=dev), c)
    first = torch.cumsum(c, 0) - c
    k = torch.arange(total, device=dev) - first[local]
    fid = local + sl.start
    wid = bw[fid]
    gx = x0[fid] + k % wid
    gy = y0[fid] + torch.div(k, wid, rounding_mode="floor")
    fx, fy = gx.float(), gy.float()

    ax, bx, cx = tx[fid, 0], tx[fid, 1], tx[fid, 2]
    ay, by, cy = ty[fid, 0], ty[fid, 1], ty[fid, 2]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    ok = area.abs() > 1e-12
    safe = torch.where(ok, area, torch.ones_like(area))
    # веса вершин a и b (вершины 0 и 1), как в nvdiffrast
    u = ((bx - fx) * (cy - fy) - (by - fy) * (cx - fx)) / safe
    v = ((cx - fx) * (ay - fy) - (cy - fy) * (ax - fx)) / safe
    t = 1 - u - v
    eps = -1e-6
    inside = ok & (u >= eps) & (v >= eps) & (t >= eps)
    if not bool(inside.any()):
        return
    fid, gx, gy, u, v = fid[inside], gx[inside], gy[inside], u[inside], v[inside]
    tv = tri[fid]
    zz = u * z[tv[:, 0]] + v * z[tv[:, 1]] + (1 - u - v) * z[tv[:, 2]]
    out[gy, gx] = torch.stack([u, v, zz, (fid + 1).float()], dim=-1)


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    a = attr[0] if attr.dim() == 3 else attr
    r = rast[0]
    tri = tri.long()
    fid = r[..., 3].round().long() - 1
    mask = fid >= 0
    out = torch.zeros((*r.shape[:2], a.shape[-1]), dtype=a.dtype, device=a.device)
    f = fid[mask]
    if f.numel():
        tv = tri[f]
        u = r[..., 0][mask].unsqueeze(-1).to(a.dtype)
        v = r[..., 1][mask].unsqueeze(-1).to(a.dtype)
        out[mask] = u * a[tv[:, 0]] + v * a[tv[:, 1]] + (1 - u - v) * a[tv[:, 2]]
    return out.unsqueeze(0), None
