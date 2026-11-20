import ctypes, statistics, torch
lib = ctypes.CDLL("/tmp/opencode/smem_bench.so")
H, W, C = 1080, 1920, 64
x = torch.randint(0, 127, (H * W, C), dtype=torch.int8, device="cuda")
out = torch.zeros(H * W, dtype=torch.int32, device="cuda")
lib.run_gather9.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4
lib.run_smem9.argtypes   = [ctypes.c_void_p]*2 + [ctypes.c_int]*5

def timeit(fn, iters=10):
    fn(1); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); fn(iters); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters

res = {"gather9": [], "smem9 BM64": [], "smem9 BM128": [], "smem9 BM256": []}
for r in range(3):
    res["gather9"].append(timeit(lambda n: lib.run_gather9(x.data_ptr(), out.data_ptr(), H, W, C, 256, n)))
    for bm in (64, 128, 256):
        res[f"smem9 BM{bm}"].append(timeit(lambda n, b=bm: lib.run_smem9(x.data_ptr(), out.data_ptr(), H, W, C, b, 256, n)))
print(f"{'arm':>14} {'ms':>9} {'vs gather9':>12}")
base = statistics.median(res["gather9"][1:])
for k, v in res.items():
    m = statistics.median(v[1:])
    print(f"{k:>14} {m:9.3f} {100*(base-m)/base:+11.1f}%")
# sanity: the two must agree on the result (same sums)
lib.run_gather9(x.data_ptr(), out.data_ptr(), H, W, C, 256, 1); torch.cuda.synchronize()
a = out.clone()
lib.run_smem9(x.data_ptr(), out.data_ptr(), H, W, C, 128, 256, 1); torch.cuda.synchronize()
b = out
print("sums identical:", bool((a == b).all()))
import json
json.dump({k: statistics.median(v[1:]) for k, v in res.items() if k != "smem9 BM256"},
          open("/home/jorj/Загрузки/projects/REAL-Video-Enhancer/docs/wsp/verified/logs/hop-smem-theory/bench.json", "w"), indent=1)
