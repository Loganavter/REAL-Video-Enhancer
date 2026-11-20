"""What does the frequency-split correction cost on the GPU, per frame?

The numpy/cv2 prototype runs at ~1 s/frame, which is measurement overhead (three decodes, two DIS flows, float64
numpy at 4K), not the correction. This measures the correction alone, as the pipeline would run it: inputs already
on the GPU, torch ops only, no flow.

  r    = LR_t - LR_ref                     (1080p, 3ch, fp16)
  lf   = box5(r)                           separable, as avg_pool + conv or unfold-free conv
  hf   = r - lf
  keep = (box(|r|) < tau) & (grad < g)     the flat static mask
  out  = HR_ref + nearest2x(lf + (1 + keep*(alpha-1)) * hf)

Measured with CUDA events, 50 iterations after 10 warm-up, at 1080p -> 4K, on the same card the pipeline uses.
"""
import time
import torch
import torch.nn.functional as F

dev = "cuda"
dtype = torch.float16
lr = torch.rand(1, 3, 1080, 1920, device=dev, dtype=dtype)
lr_ref = torch.rand(1, 3, 1080, 1920, device=dev, dtype=dtype)
hr = torch.rand(1, 3, 2160, 3840, device=dev, dtype=dtype)


def box(x, k=5):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2)


def step():
    r = lr - lr_ref
    lf = box(r)
    hf = r - lf
    loc = box(r.abs()).mean(dim=1, keepdim=True)
    g = lr.mean(dim=1, keepdim=True)
    grad = (g.diff(dim=2, prepend=g[:, :, :1]).abs() +
            g.diff(dim=3, prepend=g[:, :, :, :1]).abs())
    keep = ((loc < 3) & (grad < 5)).to(dtype)
    corr = lf + (1.0 + keep * 0.35) * hf
    up = F.interpolate(corr, scale_factor=2, mode="nearest")
    return hr + up


for _ in range(10):
    out = step()
torch.cuda.synchronize()

ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
ev0.record()
for _ in range(50):
    out = step()
ev1.record()
torch.cuda.synchronize()
ms = ev0.elapsed_time(ev1) / 50
print(f"split correction at 1080p->4K: {ms:.3f} ms/frame (fp16, torch, no flow)")

# the parts, one at a time
for name, fn in (("residual + box + hp (1080p)", lambda: (lambda r=lr - lr_ref: r - box(r))()),
                 ("mask (|r| and gradient)", lambda: ((box((lr - lr_ref).abs()).mean(1, keepdim=True) < 3) &
                                                      (lr.mean(1, keepdim=True).abs() < 5))),
                 ("upsample + add at 4K", lambda: hr + F.interpolate(lr, scale_factor=2, mode="nearest"))):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(50):
        fn()
    e1.record()
    torch.cuda.synchronize()
    print(f"  {name}: {e0.elapsed_time(e1)/50:.3f} ms")
