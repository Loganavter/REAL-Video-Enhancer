# `c2` read before it was assigned: the NameError became a probe that reported `failed`, and the failure was read
# as a verdict about fp16. A three-state probe catches it at runtime; this catches it before the GPU run.
# provenance: NONE
# provenance-note: authored to the shape named in harness/astcheck.py's docstring; no harness module in this
#   clone's history carries it (the four commits that touch harness/*.py are clean for this rule).
#@check: used before assignment
#@findings: 1

def probe(candidates):
    total = 0
    for index in range(len(candidates)):
        total += c2
        c2 = index
    return total
