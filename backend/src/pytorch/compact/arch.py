"""The Compact architecture definition, re-exported from the vendored spandrel tree.

`SRVGGNetCompact` is third-party code (spandrel, with its own LICENSE next to it) and it is updated
with spandrel, so it is not moved here: this module is the one import a caller inside the engine
needs, and it keeps the ownership boundary visible - ours is `int8_body`, `int8_kernels`,
`fused_tail`; theirs is the network and the state-dict loader.
"""
from __future__ import annotations

from ..spandrel.architectures.Compact import CompactArch
from ..spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact

__all__ = ["CompactArch", "SRVGGNetCompact"]
