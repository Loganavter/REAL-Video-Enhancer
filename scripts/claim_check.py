"""claim_check: a calculator that audits measurement claims before they reach the ledger.

Why it exists: nine conclusions in this investigation were wrong the same way - a number computed on one
population, scale or column and applied to another. That class is mechanical, so it can be checked
mechanically:

  arithmetic      a claimed ratio must equal the numbers it cites
  units           you cannot divide TOPS by TFLOP/s, or compare a share in percent with a margin in points
  populations     a ratio between numbers measured on different populations is not a ratio
  share sums      two shares that add up to 100% are a partition; three that add up to more are not

Claims live in a plain-text table next to the measurement, one per line:

    @claim  name  value  unit  "population"
    @ratio  numerator  denominator  claimed  [population if both sides share one]

Usage: claim_check.py claims.txt [--quiet]
Exit code 1 if any check fails, so it can gate a ledger edit.
"""
import argparse
import re
import sys

CLAIM = re.compile(r'@claim\s+(\S+)\s+(-?[\d.]+)\s+(\S+)\s+"([^"]*)"')
RATIO = re.compile(r'@ratio\s+(\S+)\s+(\S+)\s+(-?[\d.]+)(?:\s+"([^"]*)")?')
SUM = re.compile(r'@sum\s+([\d.]+)%\s+@of\s+(.*)')
BELOW = re.compile(r'@below\s+(\S+)\s+([\d.]+)\s*(\S*)')

UNIT_FAMILY = {
    "ms": "time", "s": "time",
    "TOPS": "rate", "TFLOP/s": "rate", "GFLOP": "work", "GFLOP/s": "rate",
    "%": "share", "pp": "delta", "levels": "levels", "x": "factor",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    claims: dict[str, tuple[float, str, str]] = {}
    failures: list[str] = []
    checked = 0

    for lineno, line in enumerate(open(args.path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if m := CLAIM.match(line):
            name, value, unit, pop = m.group(1), float(m.group(2)), m.group(3), m.group(4)
            claims[name] = (value, unit, pop)
            checked += 1

        elif m := RATIO.match(line):
            num, den, claimed = m.group(1), m.group(2), float(m.group(3))
            if num not in claims or den not in claims:
                failures.append(f"{lineno}: ratio cites an unknown claim ({num}, {den})")
                continue
            vn, un, pn = claims[num]
            vd, ud, pd = claims[den]
            fn, fd = UNIT_FAMILY.get(un, un), UNIT_FAMILY.get(ud, ud)
            if fn != fd:
                failures.append(f"{lineno}: unit families differ - {num} is {un}, {den} is {ud}")
            if pn != pd:
                failures.append(f"{lineno}: populations differ - {num} on '{pn}', {den} on '{pd}'")
            if m.group(4) and m.group(4) not in (pn, pd):
                failures.append(f"{lineno}: the ratio declares '{m.group(4)}' but the operands are on "
                                f"'{pn}' and '{pd}'")
            if vd == 0:
                failures.append(f"{lineno}: zero denominator")
                continue
            actual = vn / vd
            if abs(actual - claimed) > max(0.005, 0.005 * abs(claimed)):
                failures.append(f"{lineno}: {num}/{den} = {actual:.3f}, claimed {claimed:g}")
            checked += 1

        elif m := BELOW.match(line):
            name, limit = m.group(1), float(m.group(2))
            if name not in claims:
                failures.append(f"{lineno}: @below cites an unknown claim {name}")
            else:
                v, u, pop = claims[name]
                if v >= limit:
                    failures.append(f"{lineno}: {name} = {v:g} {u} is not below {limit:g} - it fails the "
                                    f"bar on '{pop}'")
            checked += 1

        elif m := SUM.match(line):
            total, names = float(m.group(1)), [n.strip() for n in m.group(2).split("+")]
            shares = []
            for n in names:
                if n not in claims:
                    failures.append(f"{lineno}: sum cites an unknown claim {n}")
                    break
                v, u, _ = claims[n]
                if UNIT_FAMILY.get(u) != "share":
                    failures.append(f"{lineno}: {n} is {u}, not a share - it cannot be summed into a "
                                    f"percentage")
                shares.append(v)
            else:
                if abs(sum(shares) - total) > 0.02:
                    failures.append(f"{lineno}: shares sum to {sum(shares):.3f}%, claimed {total:g}%")
            checked += 1

    if not args.quiet:
        print(f"{args.path}: {checked} checks")
        for c in claims.items():
            print(f"  {c[0]:<22} {c[1][0]:>10.3f} {c[1][1]:<9} {c[1][2]}")
    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print("  " + f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
