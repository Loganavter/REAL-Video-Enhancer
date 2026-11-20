"""Compatibility shim: the Compact tail fusion now lives in ``pytorch/compact/fused_tail.py``.

The shim aliases the module instead of re-exporting names: ``backend/tests/test_fused_tail.py``
drives private helpers (``_conv``, ``_pixel_shuffle_numpy``, ``_assert_block_diagonal``), and
``import *`` would silently drop them.
"""
import sys

from .compact import fused_tail as _impl

sys.modules[__name__] = _impl
