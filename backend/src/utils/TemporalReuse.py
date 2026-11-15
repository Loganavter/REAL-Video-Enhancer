import torch
import torch.nn.functional as F


class TemporalReuse:
    """Skip the upscaler when the frame is (structurally) the same as the previous one.

    Comparing raw pixels does not work on encoded video: on a clean 10-bit HEVC source
    there are 0% bit-identical consecutive frames, because dither/codec noise differs in
    every frame (grain is even worse when the studio adds it back). So the decision is
    taken on an 8x-downsampled luma - grain and dither average out, real motion does not -
    and the *worst* 512 px tile decides: if no tile moved, the SR result of this frame is
    a copy of the previous one.

    Measured on 40 s of 1440x1080 23.976 fps film (threshold 0.35/255, chain <= 2):
    42% of frames can be reused, while pans sit at ~3.1/255 and are never reused.

    Reuse is only a *skipping* optimisation: it never invents pixels, it re-emits the
    already rendered high-resolution frame, so a wrong decision shows up as a frozen
    frame, not as a blended/ghosted one. `max_chain` bounds how long that can last.
    """

    def __init__(self, threshold: float = 0.35, max_chain: int = 2,
                 downscale: int = 8, tile: int = 512):
        self.threshold = threshold      # in 8-bit luma levels (0..255)
        self.max_chain = max_chain
        self.downscale = downscale
        self.tile = tile
        self._prev_tiles = None
        self._chain = 0
        self._last_frame = None
        self.processed = 0
        self.reused = 0

    @torch.inference_mode()
    def _tiles(self, tensor: torch.Tensor) -> torch.Tensor:
        # (1, 3, H, W) in [0, 1] -> (1, 1, h, w) tile means in 8-bit units
        luma = tensor.mean(dim=1, keepdim=True)
        small = F.avg_pool2d(luma.float(), self.downscale)
        k = max(1, self.tile // self.downscale)
        return F.avg_pool2d(small, k, ceil_mode=True) * 255.0

    def can_reuse(self) -> bool:
        return self._last_frame is not None

    @torch.inference_mode()
    def should_reuse(self, frame) -> bool:
        """Feeds the low-resolution frame and returns True if the previous HR frame
        may be emitted instead of running the model."""
        self.processed += 1
        tiles = self._tiles(frame.get_frame_tensor())
        reuse = False
        if (self._prev_tiles is not None
                and self._chain < self.max_chain
                and tiles.shape == self._prev_tiles.shape):
            worst_tile = (tiles - self._prev_tiles).abs().max().item()
            reuse = worst_tile <= self.threshold
        self._prev_tiles = tiles
        if reuse:
            self._chain += 1
            self.reused += 1
        else:
            self._chain = 0
        return reuse

    def remember(self, hr_frame) -> None:
        self._last_frame = hr_frame

    def last_frame(self):
        return self._last_frame

    def stats(self) -> str:
        pct = 100.0 * self.reused / self.processed if self.processed else 0.0
        return (f"temporal reuse: {self.reused}/{self.processed} frames reused ({pct:.1f}%), "
                f"threshold={self.threshold}/255, max chain={self.max_chain}")
