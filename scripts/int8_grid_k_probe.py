"""Sweep the grid clamp K per model: `s[c] = max(s_in[c], s_t/K)`.

K=1 is the per-tensor grid reached through the folding path (no epilogue multiply), K=inf is per-channel, and
intermediate K bounds how far one channel's grid can fall below the layer maximum. The fold cancels exactly in
the accumulation, so every K costs the same arithmetic - this is a picture axis only.

Why a ladder and not a choice between the two shipped modes: measured on the first six models, an intermediate
K beat BOTH shipped modes on all six (Futsuu 0.0114% -> 0.0002% at K=2, Garfield 0.8763% -> 0.0029% at K=4).
The per-tensor/per-channel dichotomy sits at the two ends of this axis, and each end is dominated.

Usage: python scripts/int8_grid_k_probe.py "model.pth,model2.pth" [--out-file x.json]
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0,'/home/jorj/Загрузки/projects/REAL-Video-Enhancer/backend')
sys.path.insert(0,'/home/jorj/Загрузки/projects/REAL-Video-Enhancer/scripts')
APP="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
sys.path.insert(0, f"{APP}/backend/src/pytorch")
from model_loader import load_compact
from src.pytorch.Int8Body import Int8Body
from src.pytorch.TorchUtils import TorchUtils
from tail_int8_emulation import CALIB, H, W, decode_yuv10

ap = argparse.ArgumentParser()
ap.add_argument("models")
ap.add_argument("--out-file", default="")
ap.add_argument("--frames", default="100,169,250")
args = ap.parse_args()
names = [n.strip() for n in args.models.split(",") if n.strip()]
held_idx = [int(v) for v in args.frames.split(",")]
def find(n):
    for root,_,fs in os.walk('/home/jorj/Загрузки/Upscaling'):
        if n in fs: return os.path.join(root,n)
    raise SystemExit("not found: "+n)
tu=TorchUtils(width=W,height=H,device_type="cuda",hdr_mode=True,gpu_id=0,high_bitdepth=True)
st=torch.cuda.Stream()
frame=lambda i: tu.yuv420p10_to_tensor(decode_yuv10(i), st, torch.device('cuda',0), torch.float16)
calib=[frame(i) for i in CALIB]; held=[frame(i) for i in held_idx]
for name in names:
  path=find(name)
  try:
    model,scale=load_compact(path); model=model.cuda().half()
  except Exception as exc:
    print(f"# {name}  LOAD-FAILED {type(exc).__name__}: {str(exc)[:90]}")
    continue
  print(f"# {name}  feat/scale from the weights: x{scale}")
  results={}
  for label, env in (("per_tensor", {"RVE_INT8_PER_CHANNEL":"0","RVE_INT8_GRID_K":"0"}),
                     ("K=1", {"RVE_INT8_PER_CHANNEL":"1","RVE_INT8_GRID_K":"1"}),
                     ("K=2", {"RVE_INT8_PER_CHANNEL":"1","RVE_INT8_GRID_K":"2"}),
                     ("K=3", {"RVE_INT8_PER_CHANNEL":"1","RVE_INT8_GRID_K":"3"}),
                     ("K=4", {"RVE_INT8_PER_CHANNEL":"1","RVE_INT8_GRID_K":"4"}),
                     ("per_channel", {"RVE_INT8_PER_CHANNEL":"1","RVE_INT8_GRID_K":"0"})):
    for k,v in env.items(): os.environ[k]=v
    try:
      body=Int8Body(model); body.out_yuv=False; body.calibrate(calib); body._build()
      shares=[body.certify(f)[0] for f in held]
      results[label]=round(max(shares),5)
      print(f"  {label:12s} worst {max(shares):8.4f}%   per-frame {[round(s,4) for s in shares]}")
      del body; torch.cuda.empty_cache()
    except Exception as exc:
      print(f"  {label:12s} FAILED {type(exc).__name__}: {str(exc)[:90]}")
  if args.out_file:
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file,"a") as fh:
      fh.write(json.dumps({"file":name,"scale":scale,"shares":results})+"\n")
