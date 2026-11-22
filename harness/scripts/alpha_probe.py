"""Is the window's fill opaque? Two screenshots over two known backgrounds answer it as a number.

Method: put a solid-colour window behind the app, screenshot the screen, repeat with a different solid colour, then
compare the same pixels. For every pixel the compositor blended, `observed = a*colour + (1-a)*background`, so

    a = 1 - (observed_B - observed_A) / (background_B - background_A)

`a == 1` (a channel difference of 0) means the app's surface is opaque and nothing behind it can bleed. A few units of
difference is the "few tens of alpha" case: the surface carries per-pixel alpha and the compositor is blending.

    python3 harness/scripts/alpha_probe.py compare over-black.png over-magenta.png --bg-a 000000 --bg-b ff00ff
    python3 harness/scripts/alpha_probe.py compare over-black.png over-magenta.png --bg-a 000000 --bg-b ff00ff --rect 1200,200,60,60
    python3 harness/scripts/alpha_probe.py probe browser.png app.png --at 700,230 --plate-from 700,600

Why screenshots and not `QWidget.grab()`: `grab()` repaints the widget tree into a fresh buffer, so it can never see a
surface-level alpha problem - it reports 255 everywhere on a window that bleeds.
"""

from __future__ import annotations

import argparse
import sys


def _parse_colour(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise SystemExit(f"colour must be RRGGBB, got {value!r}")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _load(path: str):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - the harness runs where Pillow is installed
        raise SystemExit("needs Pillow (python3 -m pip install pillow)")
    return Image.open(path).convert("RGB")


def compare(path_a: str, path_b: str, bg_a: tuple[int, int, int], bg_b: tuple[int, int, int],
            rect: tuple[int, int, int, int] | None) -> int:
    import numpy as np

    image_a, image_b = _load(path_a), _load(path_b)
    if image_a.size != image_b.size:
        raise SystemExit(f"screenshots differ in size: {image_a.size} vs {image_b.size}")
    box = (0, 0, *image_a.size) if rect is None else (
        rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
    region_a = np.asarray(image_a.crop(box), dtype=np.float64)
    region_b = np.asarray(image_b.crop(box), dtype=np.float64)

    # Per channel:  observed_B - observed_A = (1 - a) * (background_B - background_A)
    spans = np.array([bg_b[i] - bg_a[i] for i in range(3)], dtype=np.float64)
    delta = np.abs(region_b - region_a) / np.where(spans == 0, 1.0, np.abs(spans))
    delta[:, :, spans == 0] = 0.0
    transparency = delta.max(axis=2)
    bleeding = int((transparency > 0).sum())
    total = transparency.size
    if not bleeding:
        print(f"OPAQUE: {total} px identical over both backgrounds (max channel difference 0)")
        return 0

    median = float(np.median(transparency[transparency > 0]))
    worst = float(transparency.max())
    print(f"SEMI-TRANSPARENT: {bleeding} of {total} px ({100.0 * bleeding / total:.2f}%) changed with the background")
    print(f"  implied transparency: median {100.0 * median:.2f}%  worst {100.0 * worst:.2f}%  "
          f"(alpha {round(255 * (1 - median))} … {round(255 * (1 - worst))})")
    return 1


def probe(bg_path: str, app_path: str, at: tuple[int, int], plate: tuple[int, int, int]) -> int:
    """One point, two shots: what the window shows over a known background.

    The owner's workflow is "here is the browser, here is the app over it". At a screen point the app paints `plate`
    when opaque, so `observed = a*plate + (1-a)*background` and

        a = (observed - background) / (plate - background)

    is the app's opacity AT THAT PIXEL. Channels where the two differ by <= 8 levels carry no information and are
    skipped; `a == 1` means the pixel is the app and nothing of the background survived.
    """
    image_bg, image_app = _load(bg_path), _load(app_path)
    if image_bg.size != image_app.size:
        raise SystemExit(f"screenshots differ in size: {image_bg.size} vs {image_app.size}")
    x, y = at
    background = tuple(image_bg.getpixel((x, y)))
    observed = tuple(image_app.getpixel((x, y)))
    result = 0
    print(f"point ({x},{y})  background {background}  observed {observed}  plate {plate}")
    for channel, name in enumerate("RGB"):
        span = plate[channel] - background[channel]
        if abs(span) <= 8:
            continue
        opacity = (observed[channel] - background[channel]) / span
        verdict = "opaque" if opacity > 0.97 else "SEMI-TRANSPARENT"
        print(f"  {name}: opacity {opacity:.3f} (alpha {round(255 * opacity)}) -> {verdict}")
        if opacity <= 0.97:
            result = 1
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Measure whether a window's fill is opaque, from two screenshots.")
    sub = parser.add_subparsers(dest="command", required=True)
    compare_parser = sub.add_parser("compare", help="two screenshots over two solid backgrounds")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    compare_parser.add_argument("--bg-a", default="000000", help="colour behind the app in the first shot")
    compare_parser.add_argument("--bg-b", default="ff00ff", help="colour behind the app in the second shot")
    compare_parser.add_argument("--rect", default=None, metavar="X,Y,W,H", help="restrict the comparison")
    probe_parser = sub.add_parser("probe", help="one screen point: browser-only shot vs app-over-browser shot")
    probe_parser.add_argument("background_shot")
    probe_parser.add_argument("app_shot")
    probe_parser.add_argument("--at", required=True, metavar="X,Y")
    probe_parser.add_argument("--plate", default=None, metavar="RRGGBB",
                              help="the colour the app paints there when opaque (default: sampled from the app shot)")
    probe_parser.add_argument("--plate-from", default=None, metavar="X,Y",
                              help="sample the plate colour from this point of the app shot")
    args = parser.parse_args(argv)

    if args.command == "probe":
        at = tuple(int(part) for part in args.at.split(","))
        if args.plate:
            plate = _parse_colour(args.plate)
        elif args.plate_from:
            px, py = (int(part) for part in args.plate_from.split(","))
            plate = tuple(_load(args.app_shot).getpixel((px, py)))
        else:
            raise SystemExit("give --plate RRGGBB or --plate-from X,Y (the app's own opaque colour there)")
        return probe(args.background_shot, args.app_shot, at, plate)
    if args.command == "compare":
        rect = tuple(int(part) for part in args.rect.split(",")) if args.rect else None
        return compare(args.first, args.second, _parse_colour(args.bg_a), _parse_colour(args.bg_b), rect)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
