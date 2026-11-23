"""The models a job needs, made present before the job is queued.

`getCurrentRenderOptions` in `REAL-Video-Enhancer.py` repeated the same five-line block once per model slot
(interpolate, scene detection, upscale, deblur, denoise, decompress): build a `DownloadModel`, call `downloadModel()`,
and stop the add with a message when it answers False. The block is the same every time, so it lives here as one
function (P3 of the seam plan): the window still decides which slots are on and what the message says, and this decides
what "download the pair" means.

`download_model` is `src.DownloadModels.DownloadModel` in the app - injected, because that module builds a Qt popup and
this one has to stay exercisable without a display.
"""

from __future__ import annotations

from typing import Callable, Iterable


def ensure_models(pairs: Iterable[tuple[str, str]], download_model: Callable[..., object]) -> bool:
    """Download every `(model_file, download_file)` pair; True when all of them are present afterwards.

    False at the FIRST failure, without trying the rest: `downloadModel` returns False for "no network" and for a
    download that did not land, and the caller's answer is to stop the queue add and say so - carrying on would queue a
    job whose model is missing. A downloader that raises propagates, as it did in the window.
    """
    for model_file, download_file in pairs:
        downloader = download_model(modelFile=model_file, downloadModelFile=download_file)
        if not downloader.downloadModel():
            return False
    return True
