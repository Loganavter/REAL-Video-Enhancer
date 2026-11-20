"""Compatibility shim: the Compact int8 body now lives in ``pytorch/compact/int8_body.py``.

Kept because `src.pytorch.Int8Body`, the probe (``calib_crops``) and the engine
(``FIRST_FRAME_SANITY``) import it by name; the implementation and its docstring are in the package.
The module is aliased, not star-exported, so every name - including the private constants the body
uses - resolves exactly as before.
"""
import sys

from .compact import int8_body as _impl

sys.modules[__name__] = _impl
