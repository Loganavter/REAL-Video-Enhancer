"""What does the network actually add on top of a pixel repeat?

The model is residual by construction (pixel-unshuffle -> body -> pixel-shuffle on top of a nearest
upsample), so the whole output is `nearest(x) + correction`. If the correction is a few levels, then
approximations with a 10-20% relative error stay inside the 8-level acceptance bar, and a whole class
of less exact algorithms becomes admissible. If it is tens of levels, that class is closed and the
answer is worth having anyway.

Run: python3 residual_scale.py <frame.raw> <W> <H> [model.pth]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

BACKEND = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/backend"
sys.path.insert(0, f"{BACKEND}/src/pytorch")
from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact  # noqa: E402


def main() -> int:
    path, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    model_path = sys.argv[4] if len(sys.argv) > 4 else "/tmp/opencode/models2/futsuu.pth"

    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size == w * h * 3, f"{raw.size} bytes, expected {w * h * 3}"
    x = torch.from_numpy(raw.reshape(h, w, 3).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1)[None].cuda().half()

    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    # the tail's out_channels is 3 * scale^2, so the scale comes from the checkpoint, not from a guess
    tail = [k for k in sd if k.endswith("34.weight")][0]
    scale = int(round((sd[tail].shape[0] / 3) ** 0.5))
    print(f"model {model_path}: scale x{scale}")
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale)
    model.load_state_dict(sd)
    model = model.eval().cuda().half()

    with torch.no_grad():
        out = model(x)
    base = F.interpolate(x, scale_factor=scale, mode="nearest")

    # in 0..255 levels, worst channel - the same reduction the acceptance rule uses
    res = (out.float() - base.float()).abs() * 255.0
    per_px = res.amax(dim=1).flatten()
    sample = per_px[::37]
    q = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.99, 0.999]).cuda())
    print("correction |out - nearest(x)|, worst channel, levels:")
    print("  median %.1f  p90 %.1f  p99 %.1f  p99.9 %.1f  max %.1f"
          % (*q.tolist(), float(sample.max())))
    for thr in (2, 5, 8, 20, 40):
        share = float((per_px < thr).float().mean()) * 100.0
        print(f"  share below {thr:>2} levels: {share:5.2f}%")
    print("output range %.3f..%.3f mean %.3f" % (out.min(), out.max(), out.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
