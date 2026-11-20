"""E30b: the real reconstruction test - sequential layer-wise least squares on teacher targets.

The diagnostics killed the cheap explanations: the error sits on edges (26% of edge pixels above 8
levels) while flat regions read 0.06%, the signed mean is 0.01 levels and a per-channel affine refit
moves nothing (1.573% -> 1.553%). So it is not a DC shift and not a scale mismatch; the cut loses
structure. The remaining question is whether a refit of each layer against the teacher's own activations
recovers it - which is the cheap version of ASVD/SNOWS and the gate before spending a day on the full
method.

Procedure per body layer, in order (so each layer sees the already-refitted student input):

  X = im2col(student activation at this layer's input), subsampled pixels
  Y = teacher pre-activation at this layer, restricted to the kept channels
  W', b' = argmin ||X W' + b' - Y||^2 + lambda ||W' - W0||^2

The tail is fitted to the RESIDUAL the model is built around (`teacher_out - nearest(x)`), because the
whole network is residual and a tail error is what the eye sees.

Usage: ls_refit.py <frame.raw> <W> <H> [--keep 32] [--stride 4] [--lambda 1e-3]
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import cut_model, load, statistic  # noqa: E402


def conv_keys(model):
    keys = [k for k in model.state_dict() if k.endswith(".weight")
            and model.state_dict()[k].ndim == 4]
    return sorted(keys, key=lambda s: int(s.split(".")[1]))


def im2col(x, stride, kernel=3):
    cols = F.unfold(x, kernel, padding=1)          # [1, C*9, H*W]
    return cols[0, :, :: stride].t().float()       # [N, C*9]


def ridge(X, Y, w0, lam):
    """Closed-form ridge towards the previous weights: solve for [W|b] with a bias column."""
    n, d = X.shape
    Xa = torch.cat([X, torch.ones(n, 1, device=X.device)], dim=1)          # [N, d+1]
    A = Xa.t() @ Xa
    reg = torch.eye(d + 1, device=X.device) * lam * n
    reg[d, d] = 0.0                                                        # never shrink the bias
    target = torch.cat([w0.flatten(1), w0.new_zeros(w0.shape[0], 1)], dim=1)  # [out, d+1]
    B = Xa.t() @ Y + lam * n * target.t()
    try:
        sol = torch.linalg.solve(A + reg, B)
    except Exception:
        sol = torch.linalg.lstsq(A + reg, B).solution
    w = sol[:d].t().reshape(w0.shape)
    b = sol[d]
    return w, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
    x = img.permute(2, 0, 1)[None].cuda().half()

    teacher, scale, _ = load(args.model)
    teacher = teacher.cuda().half().eval()
    with torch.no_grad():
        ref = teacher(x).float()
    nearest = F.interpolate(x, scale_factor=scale, mode="nearest").float()

    student, sel = cut_model(teacher, args.keep)
    student = student.cuda().half().eval()
    tkeys = conv_keys(teacher)
    skeys = conv_keys(student)

    # teacher pre-activations, subsampled, for every body conv
    teacher_out = {}

    def capture(name):
        def hook(_mod, _inp, out):
            teacher_out[name] = out.detach()[0].flatten(1)[:, :: args.stride].t().float()
        return hook

    handles = [dict(teacher.named_modules())[k.rsplit(".", 1)[0]].register_forward_hook(capture(k))
               for k in tkeys]
    with torch.no_grad():
        teacher(x)
    for h in handles:
        h.remove()

    modules = dict(student.named_modules())
    for index, (tk, sk) in enumerate(zip(tkeys, skeys)):
        module = modules[sk.rsplit(".", 1)[0]]
        if module.weight.shape[1] == 3:                          # head keeps its input: nothing to fit
            continue
        captured = {}

        def grab(_mod, inp):          # a forward_pre_hook receives (module, input) only
            captured["in"] = inp[0]

        handle = module.register_forward_pre_hook(grab)
        with torch.no_grad():
            student(x)
        handle.remove()

        X = im2col(captured["in"][0:1], args.stride)
        if module.weight.shape[0] == module.weight.shape[1]:     # body: teacher target at kept channels
            Y = teacher_out[tk]
            if sel.get(tk) is not None:
                Y = Y[:, sel[tk]]
        else:                                                    # tail: fit the residual, not the map
            Y = (ref - nearest)[0].flatten(1)[:, :: args.stride].t().float()

        w0 = module.weight.detach().float()
        w, b = ridge(X, Y, w0, args.lam)
        # can this width express the teacher's own output at all? that is the structural question,
        # and it is separable from whether the refit composes through 16 layers
        with torch.no_grad():
            residual = (X @ w.flatten(1).t() + b) - Y
            rel = (residual.pow(2).mean() / Y.pow(2).mean().clamp_min(1e-9)).sqrt().item()
        print(f"  {tk.replace('.weight', ''):<12} fit residual {rel * 100:5.2f}% of the target")
        with torch.no_grad():
            module.weight.copy_(w.half())
            if module.bias is not None:
                module.bias.copy_(b.half())

        with torch.no_grad():
            got = student(x).float()
        share, peak, _ = statistic(ref, got)
        name = tk.replace(".weight", "")
        print(f"after {name:<12} >8: {share:6.3f}%  peak {peak:5.0f}")

    with torch.no_grad():
        got = student(x).float()
    share, peak, median = statistic(ref, got)
    print(f"\nfinal: >8 {share:.3f}% (bar 0.1)  peak {peak:.0f}  median {median:.2f} levels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
