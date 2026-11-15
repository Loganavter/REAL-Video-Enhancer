"""The head padding: same function, wider first convolution.

Measured on the compact 2x model (ledger E19): the head is the only 3-channel convolution, and at C=3
TensorRT does not put it into its myelin pattern compiler at all - 57.51 ms per frame with a separate
PReLU pass. Padded to 8 channels the head block becomes one fused kernel, 54.15 ms (-5.8%), picture
unchanged (0.000% of pixels above 8 levels, 78.69 dB).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pytorch.UpscaleModelWrapper import PadInputConv, pad_head_channels  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv_first = torch.nn.Conv2d(channels, 8, 3, padding=1)
        self.act = torch.nn.PReLU(8)
        self.conv_last = torch.nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, x):
        return self.conv_last(self.act(self.conv_first(x)))


def test_the_padded_model_computes_the_same_function(monkeypatch):
    monkeypatch.setenv("RVE_HEAD_PAD", "1")
    torch.manual_seed(0)
    model = Tiny(3).eval()
    x = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        before = model(x).clone()
    padded, extra = pad_head_channels(model)
    assert extra == 5
    head = padded.conv_first
    assert isinstance(head, PadInputConv)
    assert head.in_channels == 3, "the model still sees a 3-channel input"
    assert head.conv.in_channels == 8 and head.conv.weight.shape[1] == 8
    # the five added input channels are zero and their weights are zero
    assert float(head.conv.weight[:, 3:].abs().sum()) == 0.0
    with torch.inference_mode():
        after = padded(x)
    assert torch.allclose(before, after, atol=1e-6), "padding must not change the function"


def test_a_model_that_is_already_aligned_is_left_alone(monkeypatch):
    monkeypatch.setenv("RVE_HEAD_PAD", "1")
    model = Tiny(8).eval()
    same, extra = pad_head_channels(model)
    assert extra == 0 and same is model


def test_the_padding_is_off_by_default_and_opt_in(monkeypatch):
    monkeypatch.delenv("RVE_HEAD_PAD", raising=False)
    assert pad_head_channels(Tiny(3).eval())[1] == 0, "off unless asked"
    monkeypatch.setenv("RVE_HEAD_PAD", "1")
    assert pad_head_channels(Tiny(3).eval())[1] == 5, "on when asked"


def test_a_grouped_or_non_three_channel_head_is_not_touched(monkeypatch):
    monkeypatch.setenv("RVE_HEAD_PAD", "1")
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, groups=1, padding=1))
    assert pad_head_channels(model.model if hasattr(model, "model") else model)[1] == 5
    grouped = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3, groups=2, padding=1))
    assert pad_head_channels(grouped)[1] == 0, "grouped convolutions change meaning under padding"
