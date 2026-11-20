"""Would a coarse-scale gate reuse more of THIS material - measured on the gate's own population?

The 2026-11-20 finding: the shipped statistic (max over 32 px blocks of the mean |difference|, full resolution)
measures the registration of the scan, not whether the cel changed. A 0.5 px shift moves a 1-2 px black line by
tens of levels. A 2 px low-pass plus a translation estimate qualified 47.8% of consecutive PAIRS at t=120 on the
calm window, against 6.7% for the shipped statistic.

47.8% of pairs is not 47.8% of frames. The gate keeps a reference and compares every later frame against it, up to
``max_chain`` frames deep, so a reused frame is compared against a picture one to four frames old. The share of
reusable FRAMES is what a render gains, so this instrument runs the gate's own state machine (reference keeping,
chain cap, scene cut) over one decoded stream, twice, on the same frames:

  pixel   shipped: max over 32 px blocks of mean |luma_t - luma_ref|
  coarse  proposed: the same statistic on luma after a 2 px Gaussian low-pass, with the estimated translation of
          the reference compensated. It takes the MINIMUM of the warped and the un-warped comparison, so a bad
          registration estimate can never make the statistic worse than the shipped geometry - the estimate can
          only help. The shift it reports is also what a reuse must warp the HR frame by (item (b) of the row).

For every would-be reuse it reports the two costs separately, because they are different objects:

  pixel share    the shipped acceptance statistic (share of pixels > 8 levels in any RGB channel), worst frame -
                 dust dots, scan registration and all. This is what the current bar judges.
  coarse share   the same share after the low-pass, i.e. whether the DRAWING moved.
  dust area      area of the pixels > 8 levels that belong to small connected components (< 64 px): the part of
                 the pixel share that is dots rather than drawing.

Controls, on every run (a broken pair cost the previous investigation a day):
  * identity - a frame against itself reads exactly 0 in both statistics;
  * a synthetic 0.35 px shift of real line art must collapse under the coarse statistic and NOT under the pixel
    one - that is the whole claim, so it is checked before any number is read;
  * a synthetic +12 level change must not collapse (a statistic that accepts everything is not a statistic).

Usage:
  python3 harness/scripts/reuse_structural_probe.py CLIP --start 120 --frames 96 --max-chain 4
  python3 harness/scripts/reuse_structural_probe.py CLIP --start 120 --frames 72 --threshold 4 5
"""

from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import subprocess
import time

import numpy as np

FFMPEG = "bin/ffmpeg"
BLOCK = 32          # the shipped statistic's block side at 1080p
SIGMA = 3.0         # the coarse scale in LR pixels: where the drawing is stable against the scan's registration
DUST_MAX_AREA = 64  # connected components up to this area are dots, not drawing


def _frame_iter(path, width, height, count, start):
    """Decode ``count`` frames at native rate, one rgb24 frame at a time (no pile in RAM)."""
    cmd = [FFMPEG, "-v", "error", "-ss", str(start), "-i", path, "-frames:v", str(count),
           "-vf", f"scale={width}:{height}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                return
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def block_max(x):
    """Max over 32 px blocks of the block mean of ``x`` (an error field, in 8-bit levels)."""
    h, w = x.shape
    gh, gw = -(-h // BLOCK), -(-w // BLOCK)
    pad = np.zeros((gh * BLOCK, gw * BLOCK), np.float32)
    pad[:h, :w] = x
    return float(pad.reshape(gh, BLOCK, gw, BLOCK).mean(axis=(1, 3)).max())


def lowpass(x, sigma=None):
    import cv2
    sigma = SIGMA if sigma is None else sigma
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(np.ascontiguousarray(x, np.float32), (0, 0), sigma,
                            borderType=cv2.BORDER_REPLICATE)


def register(a_gray, b_gray, scale=4, min_response=0.2, max_shift_frac=0.06):
    """Shift that puts ``b`` on top of ``a``, in full-resolution pixels: ``(dx, dy, response, trusted)``.

    Phase correlation on a ``scale``-times downscaled luma. Two guards, both from a real failure: a uniform frame
    has no correlation peak and the estimator answers with half the picture (measured: 120 px on a flat field,
    response 0.02), and a shift beyond a sixteenth of the frame is a cut or a bad estimate, not registration. In
    either case the answer is zero and ``trusted`` is False, so refusals can be counted instead of invented.
    """
    import cv2
    if scale > 1:
        a = cv2.resize(a_gray, (a_gray.shape[1] // scale, a_gray.shape[0] // scale),
                       interpolation=cv2.INTER_AREA)
        b = cv2.resize(b_gray, (b_gray.shape[1] // scale, b_gray.shape[0] // scale),
                       interpolation=cv2.INTER_AREA)
    else:
        a, b = a_gray, b_gray
    (dx, dy), response = cv2.phaseCorrelate(np.ascontiguousarray(a, np.float32),
                                            np.ascontiguousarray(b, np.float32))
    dx, dy, response = float(dx) * scale, float(dy) * scale, float(response)
    limit = max_shift_frac * max(a_gray.shape)
    if response < min_response or abs(dx) > limit or abs(dy) > limit:
        return 0.0, 0.0, response, False
    return dx, dy, response, True


def warp(x, dx, dy):
    import cv2
    return cv2.warpAffine(x, np.float32([[1, 0, dx], [0, 1, dy]]), (x.shape[1], x.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def coarse_delta(ref_blurred, cur_blurred, scale=4):
    """(delta, delta_unwarped, shift, trusted): the coarse statistic and its own geometry.

    The decision takes the smaller of the two comparisons, so the registration estimate can only lower the
    statistic - a biased or refused estimate degrades to the un-warped number, never past it.
    """
    dx, dy, _resp, trusted = register(ref_blurred, cur_blurred, scale=scale)
    warped = warp(ref_blurred, dx, dy) if trusted else ref_blurred
    delta_warped = block_max(np.abs(warped - cur_blurred))
    delta_plain = block_max(np.abs(ref_blurred - cur_blurred))
    return (min(delta_warped, delta_plain), delta_plain,
            (dx, dy) if trusted else (0.0, 0.0), trusted)


def shares(frame, reference, level=8.0):
    """(pixel share, dust area, coarse share) of one would-be reuse, in per cent.

    ``reference`` is already the frame a reuse would emit: the reference, shifted by the measured translation
    when the registration was trusted (an un-warped reuse is the verbatim emission, which is what the renderer's
    corrector exists to fix - so this is an upper bound on the pixel cost, not the realised defect).
    """
    import cv2
    d = np.abs(frame.astype(np.float32) - reference.astype(np.float32)).max(axis=2)
    share = float((d > level).mean()) * 100.0
    dust = 0.0
    mask = (d > level).astype(np.uint8)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        dust = float(areas[areas <= DUST_MAX_AREA].sum()) / d.size * 100.0
    ga = lowpass(frame.astype(np.float32).mean(2))
    gb = lowpass(reference.astype(np.float32).mean(2))
    coarse = float((np.abs(ga - gb) > level).mean()) * 100.0
    return share, dust, coarse


def controls(sample_frame, sigma=SIGMA):
    """Identity zero, a sub-pixel shift that collapses, a real change that does not.

    Run on REAL line art from the clip under test, at a known shift, because a synthetic picture is not the
    population: a periodic pattern already made one version of this control read an 80-level difference between
    identical frames (phase correlation is ambiguous between a pattern's repetitions).
    """
    frame = sample_frame.copy()
    shifted = warp(frame.astype(np.float32), 0.35, -0.2)
    changed = np.clip(frame.astype(np.int16) + 12, 0, 255).astype(np.uint8)

    def pixel(a, b):
        return block_max(np.abs(a.astype(np.float32).mean(2) - b.astype(np.float32).mean(2)))

    def coarse(a, b):
        return coarse_delta(lowpass(a.astype(np.float32).mean(2), sigma),
                            lowpass(b.astype(np.float32).mean(2), sigma))[0]

    ident = (pixel(frame, frame), coarse(frame, frame))
    px_shift, co_shift = pixel(frame, shifted), coarse(frame, shifted)
    co_change = coarse(frame, changed)
    ok = (abs(ident[0]) < 1e-6 and abs(ident[1]) < 1e-6
          and co_shift < 2.0 and px_shift > 4 * co_shift and co_change > 8.0)
    print(f"controls (on a real frame of this clip): identity {ident[0]:.4f}/{ident[1]:.4f} (must be 0); "
          f"0.35 px shift: pixel {px_shift:.2f} vs coarse {co_shift:.2f} "
          f"(coarse must be < 2 and >= 4x smaller); +12 levels coarse {co_change:.2f} (must be > 8) "
          f"-> {'ok' if ok else 'BROKEN'}")
    return ok


def simulate(frames, max_chain, thresholds, coarse_thresholds, scene_cut=32.0, sigma=SIGMA):
    """Run BOTH gate state machines over one decoded stream: one population, two decisions."""
    models = {}
    for name, coarse, cands in (("pixel", False, thresholds), ("coarse", True, coarse_thresholds)):
        models[name] = {"coarse": coarse, "thresholds": list(cands),
                        "ref_luma": None, "ref_blur": None, "ref_rgb": None, "ref_mean": 0.0,
                        "chain": 0, "frames": 0, "kept": 0,
                        "reuse": {t: 0 for t in cands}, "depth": {t: {} for t in cands},
                        "deltas": [], "refusals": 0, "cost": 0.0,
                        "shares": {t: [] for t in cands}, "dust": {t: [] for t in cands},
                        "coarse_share": {t: [] for t in cands}, "shifts": []}
    for index, rgb in enumerate(frames):
        luma = rgb.astype(np.float32).mean(2)
        for name, m in models.items():
            m["frames"] += 1
            t0 = time.perf_counter()
            ref, ref_blur = m["ref_luma"], m["ref_blur"]
            mean = float(luma.mean())
            scene = ref is not None and abs(mean - m["ref_mean"]) > scene_cut
            delta, shift, emit_ref = float("inf"), (0.0, 0.0), None
            if ref is not None and not scene:
                if not m["coarse"]:
                    delta = block_max(np.abs(luma - ref))
                    emit_ref = m["ref_rgb"]
                else:
                    blurred = lowpass(luma, sigma)
                    delta, plain, shift, trusted = coarse_delta(ref_blur, blurred)
                    m["refusals"] += 0 if trusted else 1
                    m["shifts"].append((shift[0], shift[1], plain, delta))
                    # what a reuse would EMIT: the reference picture, moved by the estimate
                    emit_ref = (warp(m["ref_rgb"], shift[0], shift[1]) if trusted
                                else m["ref_rgb"])
            m["cost"] += (time.perf_counter() - t0) * 1000.0
            for cand in m["thresholds"]:
                if emit_ref is None or scene or delta > cand:
                    continue
                m["reuse"][cand] += 1
                m["depth"][cand][m["chain"] + 1] = m["depth"][cand].get(m["chain"] + 1, 0) + 1
                m["deltas"].append(delta)
                share, dust, coarse_share = shares(rgb, emit_ref)
                m["shares"][cand].append(share)
                m["dust"][cand].append(dust)
                m["coarse_share"][cand].append(coarse_share)
            # one state machine per statistic: the widest threshold drives the reference history
            reuse = (emit_ref is not None and not scene and m["chain"] < max_chain
                     and delta <= max(m["thresholds"]))
            if not reuse:
                m["ref_luma"], m["ref_mean"], m["chain"] = luma, mean, 0
                m["kept"] += 1
                m["ref_rgb"] = rgb.astype(np.float32)
                if m["coarse"]:
                    m["ref_blur"] = lowpass(luma, sigma)
            else:
                m["chain"] += 1
    return models


def report(models, n_frames, max_chain, label):
    print(f"\n{label}: {n_frames} frames at native rate, chain cap {max_chain}")
    for name, m in models.items():
        n = max(1, m["frames"])
        print(f"\n[{name}] kept {m['kept']} references, statistic cost {m['cost'] / n:.2f} ms/frame "
              f"(cv2, full resolution, numpy - the torch gate is measured separately)")
        if m["coarse"]:
            print(f"  registration refused on {m['refusals']} comparisons (no peak, or a shift too large to be "
                  f"registration); those compare un-warped")
        for cand in m["thresholds"]:
            share = 100.0 * m["reuse"][cand] / n
            depths = dict(sorted(m["depth"][cand].items()))
            line = f"  threshold {cand:>4.1f}: {share:5.1f}% of frames reusable " \
                   f"({m['reuse'][cand]}/{n}) depths {depths}"
            print(line)
            if m["shares"][cand]:
                print(f"      cost of those reuses: pixel share median {np.median(m['shares'][cand]):.3f}% "
                      f"worst {max(m['shares'][cand]):.3f}% | dust area median "
                      f"{np.median(m['dust'][cand]):.3f}% | coarse share median "
                      f"{np.median(m['coarse_share'][cand]):.3f}% worst {max(m['coarse_share'][cand]):.3f}%")
        if m["coarse"] and m["shifts"]:
            s = np.array(m["shifts"])
            print(f"  shift per comparison: median ({np.median(np.abs(s[:, 0])):.2f}, "
                  f"{np.median(np.abs(s[:, 1])):.2f}) px, max {np.abs(s[:, :2]).max():.2f} px")
            print(f"  delta before/after compensation: median {np.median(s[:, 2]):.2f} -> "
                  f"{np.median(s[:, 3]):.2f} levels")


def render_pair(gated, ungated, width, height, sigma_hr, level=8.0, max_frames=0):
    """The acceptance table of a render pair, at BOTH scales, with the pixel share split into dust.

    The row this instrument serves asks for exactly this: the decision and the acceptance at the coarse scale,
    and the pixel-scale difference reported as what it is. Run on 078 Pink Panther, the same pair reads 12.59%
    worst at the pixel scale and 0.00% at the coarse scale - the difference is dust dots and the scan's own
    registration, which is the thing the coarse statistic was built to ignore.
    """
    import cv2
    a = _frame_iter(gated, width, height, max_frames or 10 ** 9, 0)
    b = _frame_iter(ungated, width, height, max_frames or 10 ** 9, 0)
    rows = []
    for index, (ga, gb) in enumerate(zip(a, b)):
        pa = ga.astype(np.float32)
        pb = gb.astype(np.float32)
        d = np.abs(pa - pb).max(axis=2)
        pixel = float((d > level).mean()) * 100.0
        peak = float(d.max())
        mask = (d > level).astype(np.uint8)
        dust = 0.0
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            dust = float(areas[areas <= DUST_MAX_AREA].sum()) / d.size * 100.0
        coarse = float((np.abs(lowpass(pa.mean(2), sigma_hr)
                              - lowpass(pb.mean(2), sigma_hr)) > level).mean()) * 100.0
        rows.append((index, pixel, coarse, dust, peak))
    if not rows:
        raise SystemExit("no frames decoded from the pair - check the paths and the geometry")
    pixels = np.array([r[1] for r in rows])
    coarses = np.array([r[2] for r in rows])
    dusts = np.array([r[3] for r in rows])
    print(f"render pair over {len(rows)} frames at {width}x{height}, coarse scale {sigma_hr:g} px")
    print(f"  PIXEL scale: worst {pixels.max():8.3f}% (frame {int(pixels.argmax())}), "
          f"median {np.median(pixels):7.3f}%, frames over 0.1%: {int((pixels > 0.1).sum())} of {len(rows)}")
    print(f"    of that, dust dots (components <= {DUST_MAX_AREA} px): median {np.median(dusts):7.3f}%, "
          f"worst {dusts.max():7.3f}%")
    print(f"  COARSE scale: worst {coarses.max():8.3f}% (frame {int(coarses.argmax())}), "
          f"median {np.median(coarses):7.3f}%, frames over 0.1%: {int((coarses > 0.1).sum())} of {len(rows)}")
    differing = [r for r in rows if r[1] > 0.0]
    print(f"  frames that differ at all: {len(differing)}; identical frames: {len(rows) - len(differing)}")
    for index, pixel, coarse, dust, peak in differing[:6]:
        print(f"    frame {index:>3}: pixel {pixel:7.3f}% (dust {dust:6.3f}%)  coarse {coarse:7.3f}%  peak {peak:5.0f}")
    print("  uncovered: no temporal judgement beyond the pair, no perceptual model, and nothing here says the")
    print("  ungated render is right. Dust is counted by area only, not by how visible a dot is.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, default=120.0)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--max-chain", type=int, default=4)
    ap.add_argument("--threshold", type=float, nargs="+", default=[4.0, 5.0],
                    help="pixel statistic thresholds to tabulate")
    ap.add_argument("--coarse-threshold", type=float, nargs="+", default=[2.0, 3.0, 4.0],
                    help="coarse statistic thresholds to tabulate")
    ap.add_argument("--sigma", type=float, default=SIGMA, help="coarse scale, in LR pixels")
    ap.add_argument("--render-pair", nargs=2, metavar=("GATED", "UNGATED"),
                    help="two renders of the same clip: the acceptance table at both scales")
    ap.add_argument("--sigma-hr", type=float, default=6.0,
                    help="coarse scale in HR pixels for --render-pair (3 LR px at a 2x model)")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--label", default="")
    # positional, because that is how the docstring and every recorded invocation call it
    ap.add_argument("clip", nargs="?", help="single clip: report the reuse ceiling of both statistics")
    args = ap.parse_args()
    if args.render_pair:
        render_pair(args.render_pair[0], args.render_pair[1], args.width * 2, args.height * 2,
                    args.sigma_hr)
        return 0
    if not args.clip:
        ap.error("give a clip, or --render-pair GATED UNGATED")
    frames = list(_frame_iter(args.clip, args.width, args.height, args.frames, args.start))
    if len(frames) < 8:
        raise SystemExit(f"only {len(frames)} frames decoded - check the path and --start")
    if not args.no_controls and not controls(frames[len(frames) // 2], args.sigma):
        print("controls failed: fix the instrument before reading a number")
        return 2
    models = simulate(frames, args.max_chain, args.threshold, args.coarse_threshold,
                      sigma=args.sigma)
    report(models, len(frames), args.max_chain, args.label or args.clip)
    print("\nuncovered: these are the LOW-resolution frames the gate sees. The acceptance statistic is judged on"
          "\nthe upscaled output, so the shares here bound what a verbatim reuse would cost and say nothing about the"
          "\ncorrection the renderer applies. Registration is translation only - rotation and scale of the scan are"
          "\nnot modelled, and the estimator's own error is included in the coarse statistic, not subtracted.")
    return 0


if __name__ == "__main__":
    _bsys.exit(main())
