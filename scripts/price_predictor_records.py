"""Records for the curvature-priced K predictor."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/wsp/verified/logs")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-price-predictor.jsonl")
    a = ap.parse_args(argv)
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
    st = Store(session="s-2026-11-19T16:40:00+03:00")
    n = 1
    agree = 0
    total = 0
    for fname in ("price-realesr-animevideov3(1).json", "price-2x_Ani4K_Compact_35000.json",
                  "price-2x-Futsuu-Anime(1).json"):
        path = os.path.join(a.dir, fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        pop = Population.of("models", (d["model"],), f"{d['model']}, three held-out frames")
        for k, v in d["price"].items():
            st.emit(Record(value=float(v), unit="levels", population=pop,
                           instrument="curvature-priced score: sum_l mse_l(K) * c_l, c from a finite-difference "
                                      "sensitivity of the final output to a 5% perturbation of each layer",
                           config=f"{d['model']} at {d['scale']}x, K={k}", session=st.session,
                           statistic="single", tier="measured", id=f"PR{n:03d}",
                           run="scripts/price_predictor_probe.py", artifact=path,
                           extra={"artifact_sha256": sha(path), "k": k,
                                  "predicted_best": d["predicted_best"], "ladder_best": d["ladder_best"]}))
            n += 1
        total += 1
        agree += 1 if d["agree"] else 0
    pop = Population.of("models", tuple(f"m{i}" for i in range(total)), "the three models run")
    st.emit(Record(value=float(agree), unit="count", population=pop,
                   instrument="models where the priced argmin equals the ladder's argmin",
                   config="curvature-priced predictor against the ladder", session=st.session,
                   statistic="single", tier="measured", id=f"PR{n:03d}",
                   run="scripts/price_predictor_probe.py", artifact=os.path.join(a.dir, "price-realesr-animevideov3(1).json"),
                   extra={"artifact_sha256": sha(os.path.join(a.dir, "price-realesr-animevideov3(1).json")),
                          "of": "agreement", "n_models": total}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
