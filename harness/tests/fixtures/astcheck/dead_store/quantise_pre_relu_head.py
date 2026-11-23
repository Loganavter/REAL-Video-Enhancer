# The dead store that caused a false kernel refusal, as harness/astcheck.py's docstring states it:
#   buf = quantise(pre_relu_head), later buf = something_else in the same straight-line block. Both variants then
#   measured identical values, a correct fused kernel was rejected, and the rejection was never re-checked.
# provenance: NONE
# provenance-note: the probe script that carried the original store is not in this clone's history (the docstring
#   at 56328bf6 is the only citation); this is the docstring's shape, authored, and says so instead of pretending.
#@check: dead store
#@findings: 1

def measure_head(pre_relu_head, quantise, fused):
    buf = quantise(pre_relu_head)       # this value is never read: the next line overwrites it
    buf = fused(pre_relu_head)
    return buf
