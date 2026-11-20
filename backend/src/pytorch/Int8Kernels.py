"""Compatibility shim: the Compact int8 Triton kernels now live in ``pytorch/compact/int8_kernels.py``."""
import sys

from .compact import int8_kernels as _impl

sys.modules[__name__] = _impl
