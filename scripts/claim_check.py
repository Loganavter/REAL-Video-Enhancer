"""claim_check: an auditor for measurement claims, with the classes it cannot check printed.

Nine conclusions in this investigation failed the same way - a number computed on one population, scale or
column and applied to another. Four of those classes are mechanical and are checked here:

  arithmetic    a claimed ratio must equal the numbers it cites
  units         share vs delta, TOPS vs TFLOP/s, ms vs s
  populations   a ratio or a verdict whose operands were measured on different populations
  partitions    shares that are presented as a partition must sum to it

And three more that were semantic, which is why they were the expensive ones:

  tags          every number is measured | derived | assumed; a derived number must say how it was derived,
                and an assumed one may not be compared with a measured one as an equal
  causes        a causal claim must cite two measurements that differ in that mechanism - "write-allocate"
                and "halo on the writes" both failed exactly here: nobody had varied them
  bars          the bar is a claim too; `peak 7 against 8` fails because 8 is the share threshold, not the
                peak bar

What is still NOT checked is printed at the end of every run, because a green table looks like validation and
would otherwise become a ceiling instead of a floor.

    @claim   name value unit "population" [measured|derived|assumed]
    @ratio   numerator denominator claimed ["population"]
    @below   name bar_name
    @sum     X% @of a + b
    @derived name "how"            # the derivation of a claim tagged derived
    @cause   name "mechanism" m1 m2
    @verdict name c1 c2 c3

Usage: claim_check.py claims.txt [--quiet]; exit 1 on any failure.
"""
import argparse
import re
import sys

CLAIM = re.compile(r'@claim\s+(\S+)\s+(-?[\d.]+)\s+(\S+)\s+"([^"]*)"(?:\s+(measured|derived|assumed))?')
RATIO = re.compile(r'@ratio\s+(\S+)\s+(\S+)\s+(-?[\d.]+)(?:\s+"([^"]*)")?')
BELOW = re.compile(r'@below\s+(\S+)\s+(\S+)')
SUM = re.compile(r'@sum\s+([\d.]+)%\s+@of\s+(.*)')
DERIVED = re.compile(r'@derived\s+(\S+)\s+"([^"]*)"')
CAUSE = re.compile(r'@cause\s+(\S+)\s+"([^"]*)"\s+(\S+)\s+(\S+)')
VERDICT = re.compile(r'@verdict\s+(\S+)\s+(.+)')
BAR = re.compile(r'@bar\s+(\S+)\s+(-?[\d.]+)\s+(\S+)')

UNIT_FAMILY = {
    "ms": "time", "s": "time",
    "TOPS": "rate", "TFLOP/s": "rate", "GFLOP": "work", "GFLOP/s": "rate",
    "%": "share", "pp": "delta", "levels": "levels", "x": "factor",
}

NOT_CHECKED = """\
NOT checked (a green table is a floor, not a ceiling):
  - whether a population LABEL is true  (a broken selector still gets called "rendered frames")
  - whether a population is the one the DECISION needs  (mask area vs mask+dilation are both "share of frame")
  - whether a cost model is COMPLETE  (8 conversions x 0.1 ms looked right and was 1.5-2x wrong)
  - whether a cited measurement DISCRIMINATES the mechanism it is used to name
  - whether two correct claims are a correct PAIR"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    claims: dict[str, tuple[float, str, str, str]] = {}
    bars: dict[str, tuple[float, str]] = {}
    derived: dict[str, str] = {}
    failures: list[str] = []
    checked = 0

    for lineno, line in enumerate(open(args.path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if m := BAR.match(line):
            bars[m.group(1)] = (float(m.group(2)), m.group(3))
            checked += 1

        elif m := CLAIM.match(line):
            name, value, unit, pop = m.group(1), float(m.group(2)), m.group(3), m.group(4)
            tag = m.group(5) or "measured"
            claims[name] = (value, unit, pop, tag)
            checked += 1

        elif m := DERIVED.match(line):
            derived[m.group(1)] = m.group(2)
            checked += 1

        elif m := RATIO.match(line):
            num, den, claimed = m.group(1), m.group(2), float(m.group(3))
            if num not in claims or den not in claims:
                failures.append(f"{lineno}: ratio cites an unknown claim ({num}, {den})")
                continue
            vn, un, pn, tn = claims[num]
            vd, ud, pd, td = claims[den]
            if UNIT_FAMILY.get(un, un) != UNIT_FAMILY.get(ud, ud):
                failures.append(f"{lineno}: unit families differ - {num} is {un}, {den} is {ud}")
            if pn != pd:
                failures.append(f"{lineno}: populations differ - {num} on '{pn}', {den} on '{pd}'")
            if "assumed" in (tn, td) and "measured" in (tn, td):
                failures.append(f"{lineno}: an assumed number ({num if tn == 'assumed' else den}) is "
                                f"compared with a measured one as an equal")
            if vd == 0:
                failures.append(f"{lineno}: zero denominator")
                continue
            if abs(vn / vd - claimed) > max(0.005, 0.005 * abs(claimed)):
                failures.append(f"{lineno}: {num}/{den} = {vn / vd:.3f}, claimed {claimed:g}")
            checked += 1

        elif m := BELOW.match(line):
            name, bar = m.group(1), m.group(2)
            if name not in claims:
                failures.append(f"{lineno}: @below cites an unknown claim {name}")
            elif bar not in bars:
                failures.append(f"{lineno}: @below cites an unknown bar '{bar}' - bars are claims too")
            else:
                v, u, pop, _ = claims[name]
                bv, bu = bars[bar]
                if UNIT_FAMILY.get(u) != UNIT_FAMILY.get(bu):
                    failures.append(f"{lineno}: {name} is {u} but bar '{bar}' is {bu} - a peak is not "
                                    f"judged against a share threshold")
                elif v >= bv:
                    failures.append(f"{lineno}: {name} = {v:g} {u} is not below {bv:g} ({bar}) on '{pop}'")
            checked += 1

        elif m := SUM.match(line):
            total, names = float(m.group(1)), [n.strip() for n in m.group(2).split("+")]
            shares = []
            for n in names:
                if n not in claims:
                    failures.append(f"{lineno}: sum cites an unknown claim {n}")
                    break
                v, u, _, _ = claims[n]
                if UNIT_FAMILY.get(u) != "share":
                    failures.append(f"{lineno}: {n} is {u}, not a share - it cannot enter a partition")
                shares.append(v)
            else:
                if abs(sum(shares) - total) > 0.02:
                    failures.append(f"{lineno}: shares sum to {sum(shares):.3f}%, claimed {total:g}%")
            checked += 1

        elif m := CAUSE.match(line):
            name, mech, m1, m2 = m.groups()
            for cited in (m1, m2):
                if cited not in claims:
                    failures.append(f"{lineno}: @cause cites an unknown measurement {cited}")
                elif claims[cited][3] != "measured":
                    failures.append(f"{lineno}: @cause cites {cited}, which is {claims[cited][3]} - a cause "
                                    f"needs a measurement")
            if m1 in claims and m2 in claims and claims[m1][2] == claims[m2][2]:
                failures.append(f"{lineno}: @cause '{mech}' cites two measurements on the same population "
                                f"'{claims[m1][2]}' - the mechanism was never varied")
            checked += 1

        elif m := VERDICT.match(line):
            name, names = m.group(1), m.group(2).split()
            pops = {claims[n][2] for n in names if n in claims}
            missing = [n for n in names if n not in claims]
            if missing:
                failures.append(f"{lineno}: @verdict {name} cites unknown claims {missing}")
            elif len(pops) > 1:
                failures.append(f"{lineno}: @verdict {name} mixes populations {sorted(pops)} - each claim is "
                                f"true and the pair is not")
            checked += 1

    for name, (v, u, pop, tag) in claims.items():
        if tag == "derived" and name not in derived:
            failures.append(f"claim {name} is tagged derived and gives no @derived line")

    if not args.quiet:
        print(f"{args.path}: {checked} checks, {len(claims)} claims, {len(bars)} bars")
        for name, (v, u, pop, tag) in claims.items():
            print(f"  {name:<24} {v:>10.3f} {u:<9} [{tag}] {pop}")
    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print("  " + f)
    else:
        print("PASS")
    print(NOT_CHECKED)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
