"""Which of the collection's models the int8 pipeline can actually run, and which opt-ins each gets.

The lane's boundary is narrow and stated: `Int8Body` needs a Compact/SRVGG-style body - sixteen 64-channel
convs between the head and the tail - because the channel count is a `tl.constexpr` in the kernel. Around that
sit four opt-ins with their own preconditions:

  head fused      the head conv is a plain 3 -> 64 convolution (or a PadInputConv wrapping one)
  tail int8       the tail has <= 16 output channels (the kernel pads to 16)
  shuffle folded  the upsampler is `nn.PixelShuffle` with r = 2 AND the tail outputs 3*r^2 = 12 channels
  tail CSC        needs the shuffle fold; then the tail can write the pipe payload itself

This script answers the question with the loader the app uses (its own vendored spandrel), not by guessing from
file names, and it reports WHY a model is refused - a "does not fit" without a reason is not a finding.

Usage: python harness/scripts/model_inventory.py [--root DIR] [--out FILE] [--json FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(APP, "backend", "src", "pytorch"))
sys.path.insert(0, os.path.join(REPO, "backend"))

SUFFIXES = (".pth", ".pt", ".safetensors")


def inspect(path: str) -> dict:
    """Everything the pipeline needs to know about one file, from the model itself."""
    import torch
    from spandrel import ModelLoader
    from src.pytorch.Int8Body import Int8Body

    row = {"file": os.path.basename(path), "dir": os.path.dirname(path), "size_mb": None,
           "arch": None, "scale": None, "channels": None, "body_convs": None,
           "tail_out": None, "up": None, "residual": None,
           "int8_body": False, "head_fused": False, "tail_int8": False, "shuffle": False,
           "tail_csc": False, "verdict": "NO", "why": ""}
    try:
        row["size_mb"] = round(os.path.getsize(path) / 2**20, 2)
        with open(path, "rb") as fh:
            row["sha256_12"] = hashlib.sha256(fh.read()).hexdigest()[:12]
        desc = ModelLoader().load_from_file(path)
        model = desc.model.eval()
        row["arch"] = type(model).__name__
        row["arch_family"] = str(desc.architecture)
        row["scale"] = int(desc.scale)
        row["channels"] = f"{desc.input_channels}->{desc.output_channels}"
    except Exception as error:                                  # noqa: BLE001
        row["why"] = f"loader refused: {type(error).__name__}: {str(error)[:110]}"
        return row

    body = getattr(model, "body", None)
    if body is None:
        row["why"] = (f"no `body` sequence - this is a {row['arch']}, not a Compact/SRVGG body, so the int8 "
                      f"kernel has nothing to replace")
        return row
    convs = [m for m in body if isinstance(m, torch.nn.Conv2d)]
    row["body_convs"] = len(convs) - 2                        # inner convs only: the last one is the tail
    row["tail_out"] = int(convs[-1].out_channels) if convs else None
    up = getattr(model, "upsampler", None)
    row["up"] = type(up).__name__ if up is not None else None
    row["residual"] = True                                    # the int8 body adds interpolate(x) itself

    ok = Int8Body.supports(model)
    row["int8_body"] = bool(ok)
    if not ok:
        feats = {int(c.out_channels) for c in convs[:-1]} or {0}
        w = sorted(feats)
        row["why"] = (f"body is {w[0] if len(w) == 1 else w} channel(s) over {len(convs) - 2} inner convs; "
                      f"the kernel's width is a constexpr fixed at 64, and its tile/launch tuning is measured "
                      f"at that width (the hop COUNT is data since rve-hop-count-general, the WIDTH is not)")
        return row

    head = list(body)[0]
    plain = getattr(head, "conv", head)
    row["head_fused"] = bool(isinstance(plain, torch.nn.Conv2d) and plain.out_channels == 64
                             and plain.in_channels >= 3)
    row["tail_int8"] = bool(convs[-1].out_channels <= 16)
    r = int(getattr(up, "upscale_factor", 0) or 0) if isinstance(up, torch.nn.PixelShuffle) else 0
    row["shuffle"] = bool(row["tail_int8"] and r == 2 and convs[-1].out_channels == 3 * r * r)
    row["tail_csc"] = row["shuffle"]
    if row["tail_csc"]:
        row["verdict"] = "FULL"
        row["why"] = "int8 hops + int8 tail + shuffle fold + the tail writing the pipe payload"
    elif row["tail_int8"]:
        row["verdict"] = "BODY+TAIL"
        row["why"] = (f"int8 hops and int8 tail, but the upsampler is {row['up']} x{row['tail_out']} channels, "
                      f"so the shuffle fold does not apply and the pack converts colour")
    else:
        row["verdict"] = "BODY"
        row["why"] = (f"int8 hops only: the tail has {row['tail_out']} output channels (the int8 tail kernel "
                      f"pads to 16), so the tail runs in fp16 and the pack converts")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/jorj/Загрузки/Upscaling")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    paths = []
    for dirpath, _dirs, files in os.walk(args.root):
        for name in sorted(files):
            if name.lower().endswith(SUFFIXES):
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    print(f"# {len(paths)} model files under {args.root}", file=sys.stderr)

    rows, seen = [], {}
    for path in paths:
        row = inspect(path)
        key = row.get("sha256_12") or row["file"]
        if key in seen:                        # the collection holds duplicates under different folders
            seen[key].setdefault("also", []).append(os.path.relpath(path, args.root))
            continue
        seen[key] = row
        rows.append(row)
        print(f"{row['verdict']:9s} {row['file'][:44]:44s} {str(row['arch'])[:22]:22s} "
              f"x{row['scale']}  {row['why'][:70]}", file=sys.stderr)

    rows.sort(key=lambda r: ({"FULL": 0, "BODY+TAIL": 1, "BODY": 2, "NO": 3}[r["verdict"]],
                             r["file"].lower()))
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("FULL", "BODY+TAIL", "BODY", "NO")}
    print(f"# {counts} of {len(rows)} distinct models", file=sys.stderr)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump({"root": args.root, "rows": rows, "counts": counts}, open(args.json, "w"), indent=1)
        print(f"# -> {args.json}", file=sys.stderr)
    if args.out:
        lines = ["| verdict | model | arch | scale | channels | what the pipeline gets |",
                 "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| **{r['verdict']}** | `{r['file']}` | {r['arch']} | x{r['scale']} | "
                         f"{r['channels']} | {r['why']} |")
        open(os.path.join(REPO, args.out), "w").write("\n".join(lines) + "\n")
        print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
