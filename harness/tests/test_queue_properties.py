"""The queue's **Properties…** row and the two-tab window it opens, pinned offscreen with a fake probe.

Owner, 2026-11-24: *"Also we could steal the properties window from BlinkTable for a function -- properties in the
render queue via the context menu (so you can see the video's properties)."* Then, while the first cell was not reading
its mail: *"MediaInfo is the preferred probe"*, *"TWO tabs"* (MediaInfo / RVE), and the RVE tab's own facts
(queue position, timestamps, models, coefficients, the run report's numbers).

What is pinned here, with no media, no ffmpeg decode, no GPU and no window:

(a) the queue menu carries a ``queue.properties`` row whose enabled state follows the selection (and does NOT follow
    the render lock: the window is read-only whatever the queue is doing);
(b) the row applies to the FIRST selected row, and the window says which row that is - including how many other rows
    were selected with it;
(c) the two tabs exist, are toolkit ``TopTabHost`` pages, and switch;
(d) MediaInfo's JSON and the ffmpeg fallback fill the SAME rows, and the caption names which probe answered;
(e) a field neither probe answered is an em dash with its reason next to it (never a plausible default);
(f) the RVE tab reports the queue's own record: timestamps (em dashes for an entry that never ran), models by file
    name with absent ones omitted, the coefficients, and the run report's numbers with its file name;
(g) the window is built from toolkit controls only - no stock ``QLabel``/``QTableWidget``/``QTextEdit``, no QSS;
(h) ``RenderQueue.add``/``addToStart`` record when an item entered the queue (``added_wall``/``added_mono``).

The probes are injected in (a)-(g); the parser rules are pinned on real MediaInfo JSON and real ffmpeg header text
captured on this machine, so a change in what the window reports from a real file shows up here as well.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# The app and the backend both live in a `src` namespace package; the suite's conftest puts the backend first. The
# app's `ProcessTab` needs the app's `src.constants`, so the app root goes first for this module (same pattern as
# `test_queue_pane.py` / `test_display_wiring.py`).
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them.
try:  # pragma: no cover - the branch taken depends on the interpreter
    import cpuinfo  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda *args, **kwargs: {}
    sys.modules["cpuinfo"] = _cpuinfo
try:  # pragma: no cover
    import pypresence  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _presence = types.ModuleType("pypresence")
    _presence.Presence = object
    sys.modules["pypresence"] = _presence

pytest.importorskip("PySide6")

from src.ui.ProcessTab import queue_menu_entries, queue_properties_target  # noqa: E402
from src.ui.RenderQueue import JobState, RenderQueue  # noqa: E402
from src.ui.dialogs import queue_properties as properties_module  # noqa: E402

# Leave the suite's path order as it was (the app modules are imported and bound now).
sys.path[:] = _ORIGINAL_SYS_PATH


@pytest.fixture(autouse=True)
def _application(qapp):
    """The suite's one ``QApplication``: a ``QDialog`` cannot be constructed without it (Qt aborts, it does not raise)."""
    return qapp


#: An input header exactly as ffmpeg printed it for a plain SDR mp4 (`-hide_banner -i`, 2026-11-24 on this machine).
SDR_HEADER = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'sdr.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:00:02.00, start: 0.000000, bitrate: 50 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 320x240 [SAR 1:1 DAR 4:3], 45 kb/s, 25 fps, 25 tbr, 12800 tbn (default)
"""

#: The same probe for a 10-bit HDR clip: range and the colours are printed inside the pixel format's parentheses,
#: and the primaries/transfer the file does not declare read ``unknown``.
HDR_HEADER = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'hdr.mp4':
  Duration: 00:00:02.00, start: 0.000000, bitrate: 51 kb/s
  Stream #0:0[0x1](und): Video: h264 (High 10) (avc1 / 0x31637661), yuv420p10le(tv, bt2020nc/unknown/unknown, progressive), 320x240 [SAR 1:1 DAR 4:3], 46 kb/s, 25 fps, 25 tbr, 12800 tbn (default)
"""

#: A header with an audio and a subtitle stream, as ffmpeg prints them for a muxed mp4.
AV_HEADER = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'av.mp4':
  Duration: 00:00:05.00, start: 0.000000, bitrate: 400 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 640x360, 25 fps, 25 tbr, 12800 tbn (default)
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s (default)
  Stream #0:2[0x3](rus): Subtitle: mov_text (tx3g / 0x67337874), 0 kb/s (default)
"""

#: A clip whose header ffmpeg could not read at all.
NO_INPUT_TEXT = "Error opening input file /nope/clip.mp4.\nError opening input files: No such file or directory\n"

#: MediaInfo's JSON for a real 10-bit HDR clip (`mediainfo --Output=JSON`, 2026-11-24, MediaInfoLib 26.10), trimmed
#: to the keys this window reads. Note `ColorSpace: YUV` (the colour MODEL) is not the matrix: the space is
#: `matrix_coefficients`.
MI_DOCUMENT = {
    "creatingLibrary": {"name": "MediaInfoLib", "version": "26.10"},
    "media": {
        "@ref": "hdr.mkv",
        "track": [
            {"@type": "General", "Format": "Matroska", "Duration": "2.000", "FrameRate": "25.000",
             "FrameCount": "50"},
            {"@type": "Video", "Format": "AVC", "Format_Profile": "High 10", "Duration": "2.000",
             "Width": "320", "Height": "240", "FrameRate": "25.000", "FrameCount": "50",
             "ColorSpace": "YUV", "ChromaSubsampling": "4:2:0", "BitDepth": "10",
             "colour_range": "Limited", "colour_primaries": "BT.2020",
             "transfer_characteristics": "PQ", "matrix_coefficients": "BT.2020 non-constant"},
            {"@type": "Audio", "Format": "AAC", "Format_Profile": "LC", "Channels": "2",
             "SamplingRate": "48000", "BitDepth": "32", "Language": "en", "Duration": "2.000",
             "BitRate": "128000"},
            {"@type": "Text", "Format": "UTF-8", "Language": "ru", "Duration": "2.000"},
        ],
    },
}


# --- fakes ------------------------------------------------------------------------------------------


class _Queue:
    """The queue API the menu and the handler read - `RenderQueue` in the app, no widget here."""

    def __init__(self, items=(), selected=(), locked: bool = False) -> None:
        self._items = list(items)
        self._selected = list(selected)
        self._locked = locked

    def selectedRows(self) -> list[int]:
        return list(self._selected)

    def getQueue(self) -> list:
        return list(self._items)

    def is_locked(self) -> bool:
        return self._locked


def _job(path: str, **overrides):
    """A queued job: only the attributes the properties window reads."""
    values = dict(
        inputFile=path,
        outputPath=str(Path(path).with_name("clip_2x.mp4")),
        modelScale=4,
        interpolateTimes=2,
        encoderCommand="-c:v libx264 -crf 18",
        upscaleModelFile="/models/realesrgan-x4plus.pth",
        upscaleModelArch="realesrgan",
        interpolateModelFile="/models/rife-4.25.pth",
        deblurModelFile=None,
        denoiseModelFile=None,
        decompressModelFile=None,
        sceneChangeModelFile=None,
        overrideUpscaleScale=None,
        ensemble=False,
        dyanmicScaleOpticalFlow=True,
        sloMoMode=False,
        benchmarkMode=False,
        tilingEnabled=True,
        tilesize="256",
        backend="tensorrt",
        hdrMode=False,
        startTime=None,
        endTime=None,
        videoFrameCount=240,
        state=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _probe(fields: dict, reasons: dict | None = None, *, audio=(), text=(), source="fake probe"):
    """A probe as the window calls it: the shipped `ProbeResult` shape, no subprocess."""
    reasons = dict(reasons or {})

    def probe(_path: str) -> "properties_module.ProbeResult":
        return properties_module.ProbeResult(
            fields=dict(fields), reasons=reasons, audio=tuple(audio), text=tuple(text), source=source
        )

    return probe


def _no_mediainfo(_path, _binary=None):
    """A MediaInfo reader that answers "not installed": the probe must fall back to ffmpeg."""
    return None, "", "no mediainfo binary to probe with (mediainfo is not on PATH)"


def _tab(properties, title: str):
    return properties.tab(title)


def _rows(tab) -> dict:
    sections = (tab,) if isinstance(tab, properties_module.PropertySection) else tab.sections
    return {row.label: row for section in sections for row in section.rows}


def _section(tab, title: str):
    return next(section for section in tab.sections if section.title == title)


def _dialog(path, *, row=0, row_count=1, selected_count=1, fields=None, reasons=None, job=None):
    fields = _MEASURED if fields is None else fields
    return properties_module.QueuePropertiesDialog(
        job if job is not None else _job(str(path)),
        row=row,
        row_count=row_count,
        selected_count=selected_count,
        probe=_probe(fields, reasons or {"color_range": "the file does not declare a colour range"}),
        blocking=False,
    )


_MEASURED = {
    "container": "mov,mp4,m4a,3gp,3g2,mj2",
    "codec": "h264",
    "width": 1920,
    "height": 1080,
    "pixel_format": "yuv420p10le",
    "chroma_subsampling": "4:2:0",
    "bit_depth": 10,
    "fps": 23.976,
    "frames": 240,
    "duration": 10.01,
    "color_space": "bt709",
    "color_primaries": "bt709",
    "color_transfer": "bt709",
}


# --- (a) the menu row ---------------------------------------------------------------------------------


def test_queue_menu_offers_properties_and_follows_the_selection():
    entries = queue_menu_entries(_Queue(items=[_job("a.mp4")], selected=[0]))
    actions = {entry.action_id: entry for entry in entries if hasattr(entry, "action_id")}
    assert "queue.properties" in actions, "the queue menu must offer a Properties row"
    assert actions["queue.properties"].text == "Properties…"
    assert actions["queue.properties"].enabled is True

    empty = queue_menu_entries(_Queue(items=[_job("a.mp4")], selected=[]))
    empty_actions = {entry.action_id: entry for entry in empty if hasattr(entry, "action_id")}
    assert empty_actions["queue.properties"].enabled is False, "no selection -> nothing to show"


def test_queue_menu_properties_stays_enabled_while_the_queue_is_locked():
    entries = queue_menu_entries(_Queue(items=[_job("a.mp4")], selected=[0], locked=True))
    actions = {entry.action_id: entry for entry in entries if hasattr(entry, "action_id")}
    assert actions["queue.properties"].enabled is True, "reading a job's properties is not a queue mutation"
    assert actions["queue.remove"].enabled is False, "the destructive rows still follow the lock"


# --- (b) which row, and what the window says about it -------------------------------------------------


def test_properties_applies_to_the_first_selected_row():
    first, second, third = _job("a.mp4"), _job("b.mp4"), _job("c.mp4")
    target = queue_properties_target(_Queue(items=[first, second, third], selected=[2, 0]))
    assert target is not None
    job, row, row_count, selected_count = target
    assert job is first, "the first selected row (lowest index) is the one shown"
    assert (row, row_count, selected_count) == (0, 3, 2)


def test_properties_target_is_none_without_a_selection():
    assert queue_properties_target(_Queue(items=[_job("a.mp4")], selected=[])) is None


def test_dialog_names_the_first_selected_row_and_the_rest_of_the_selection(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 1536)
    dialog = _dialog(path, row=2, row_count=7, selected_count=3)
    text = dialog.plain_text()
    assert "Render Queue row 3 of 7" in text, "the window names the row it describes"
    assert "3 rows selected" in text and "showing row 3" in text
    assert dialog.windowTitle() == "Properties"


# --- (c) the two tabs ---------------------------------------------------------------------------------


def test_the_window_has_two_tabs_and_they_switch(tmp_path):
    from sli_ui_toolkit.widgets import TopTabHost

    dialog = _dialog(tmp_path / "clip.mp4")
    assert isinstance(dialog.tabs, TopTabHost), "the tabs are the toolkit's host, never QTabWidget"
    assert dialog.tabs.count() == 2
    assert [dialog.tabs.tabText(index) for index in range(2)] == ["MediaInfo", "RVE"]
    assert dialog.tabs.currentIndex() == 0, "the probe's output opens first"
    dialog.tabs.setCurrentIndex(1)
    assert dialog.tabs.currentIndex() == 1
    assert dialog.tabs.currentWidget() is not None


def test_each_tab_renders_its_own_document(tmp_path):
    dialog = _dialog(tmp_path / "clip.mp4")
    media, rve = dialog.documents["MediaInfo"], dialog.documents["RVE"]
    assert media is not rve
    assert "Codec" in media.text() and "h264" in media.text()
    assert "Added" in rve.text() and "Upscale model" in rve.text()
    assert "Codec" not in rve.text(), "the probe's rows live on the MediaInfo tab only"


# --- (d) the probe: MediaInfo first, ffmpeg as the fallback -------------------------------------------


def test_mediainfo_json_fills_the_video_rows_and_names_the_source(tmp_path):
    result = properties_module.probe_video_file(
        str(tmp_path / "hdr.mkv"), mediainfo_reader=lambda _path, _binary: (MI_DOCUMENT, "26.10", "")
    )
    assert result.source == "mediainfo 26.10"
    assert result.fields["container"] == "Matroska"
    assert result.fields["codec"] == "AVC"
    assert (result.fields["width"], result.fields["height"]) == (320, 240)
    assert result.fields["bit_depth"] == 10
    assert result.fields["chroma_subsampling"] == "4:2:0"
    assert result.fields["fps"] == 25
    assert result.fields["frames"] == 50
    assert result.fields["duration"] == 2.0
    assert result.fields["color_space"] == "BT.2020 non-constant"
    assert result.fields["color_primaries"] == "BT.2020"
    assert result.fields["color_transfer"] == "PQ"
    assert result.fields["color_range"] == "Limited"
    assert result.audio[0]["codec"] == "AAC"
    assert result.audio[0]["channels"] == 2
    assert result.audio[0]["sample_rate"] == 48000
    assert set(result.audio[0]) <= set(properties_module.AUDIO_FIELDS), "a stream dict uses the documented keys"
    assert result.text[0]["language"] == "ru"
    assert set(result.text[0]) <= set(properties_module.TEXT_FIELDS)

    properties = properties_module.build_queue_properties(
        _job(str(tmp_path / "hdr.mkv")), probe=lambda _path: result
    )
    assert properties.probe_caption == "probe: mediainfo 26.10"
    rows = _rows(_section(_tab(properties, "MediaInfo"), "Video"))
    assert rows["Codec"].value == "AVC"
    assert rows["Resolution"].value == "320 x 240"
    assert rows["Bit depth"].value == "10 bit"
    assert rows["Chroma subsampling"].value == "4:2:0"
    assert rows["Frame rate"].value == "25 fps"
    assert rows["Frames"].value == "50"
    assert rows["Colour range"].value == "Limited"


def test_the_ffmpeg_fallback_fills_the_same_rows_and_names_itself(tmp_path):
    result = properties_module.probe_video_file(
        str(tmp_path / "sdr.mp4"),
        mediainfo_reader=_no_mediainfo,
        header_reader=lambda _path, _ffmpeg: (SDR_HEADER, ""),
        backend_probe=lambda _path: {},
    )
    assert result.source == "ffmpeg"
    properties = properties_module.build_queue_properties(
        _job(str(tmp_path / "sdr.mp4")), probe=lambda _path: result
    )
    assert properties.probe_caption == "probe: ffmpeg"
    rows = _rows(_tab(properties, "MediaInfo"))
    for label in ("Codec", "Resolution", "Pixel format", "Bit depth", "Frame rate", "Duration"):
        assert rows[label].value, f"{label} is answered by the ffmpeg path too"
    assert rows["Codec"].value == "h264"
    assert rows["Pixel format"].value == "yuv420p"
    assert rows["Bit depth"].value == "8 bit"


def test_both_probe_shapes_answer_the_same_video_row_labels():
    from_mediainfo = properties_module.parse_mediainfo_json(MI_DOCUMENT)[0]
    from_ffmpeg = properties_module.parse_input_header(SDR_HEADER)
    mediainfo_rows = _rows(_tab(
        properties_module.build_queue_properties(
            _job("x.mkv"), probe=_probe(from_mediainfo, source="mediainfo 26.10")
        ),
        "MediaInfo",
    ))
    ffmpeg_rows = _rows(_tab(
        properties_module.build_queue_properties(_job("x.mp4"), probe=_probe(from_ffmpeg, source="ffmpeg")),
        "MediaInfo",
    ))
    assert set(mediainfo_rows) == set(ffmpeg_rows), "a row that disappears with the probe is a row nobody can compare"
    assert mediainfo_rows["Codec"].value and ffmpeg_rows["Codec"].value


def test_a_field_the_chosen_probe_cannot_answer_is_an_em_dash_with_its_reason(tmp_path):
    result = properties_module.probe_video_file(
        str(tmp_path / "hdr.mkv"), mediainfo_reader=lambda _path, _binary: (MI_DOCUMENT, "26.10", "")
    )
    properties = properties_module.build_queue_properties(
        _job(str(tmp_path / "hdr.mkv")), probe=lambda _path: result
    )
    rows = _rows(_tab(properties, "MediaInfo"))
    assert rows["Pixel format"].value == "", "MediaInfo does not name ffmpeg's pixel format"
    assert rows["Pixel format"].display_value == properties_module.EM_DASH == "—"
    assert "mediainfo" in rows["Pixel format"].reason.lower()


def test_audio_and_subtitle_streams_are_reported_from_mediainfo(tmp_path):
    result = properties_module.probe_video_file(
        str(tmp_path / "hdr.mkv"), mediainfo_reader=lambda _path, _binary: (MI_DOCUMENT, "26.10", "")
    )
    properties = properties_module.build_queue_properties(
        _job(str(tmp_path / "hdr.mkv")), probe=lambda _path: result
    )
    media = _tab(properties, "MediaInfo")
    titles = [section.title for section in media.sections]
    assert "Audio 1" in titles and "Subtitle 1" in titles
    audio = _rows(_section(media, "Audio 1"))
    assert audio["Codec"].value == "AAC"
    assert audio["Channels"].value == "2"
    assert audio["Sample rate"].value == "48000 Hz"
    assert audio["Language"].value == "en"
    assert audio["Bit rate"].value == "128 kb/s", "MediaInfo reports bits/s; the row keeps a readable unit"
    subtitle = _rows(_section(media, "Subtitle 1"))
    assert subtitle["Language"].value == "ru"


def test_audio_and_subtitle_streams_are_reported_from_the_ffmpeg_header(tmp_path):
    result = properties_module.probe_video_file(
        str(tmp_path / "av.mp4"),
        mediainfo_reader=_no_mediainfo,
        header_reader=lambda _path, _ffmpeg: (AV_HEADER, ""),
        backend_probe=lambda _path: {},
    )
    properties = properties_module.build_queue_properties(
        _job(str(tmp_path / "av.mp4")), probe=lambda _path: result
    )
    media = _tab(properties, "MediaInfo")
    audio = _rows(_section(media, "Audio 1"))
    assert audio["Codec"].value == "aac"
    assert audio["Sample rate"].value == "48000 Hz"
    assert audio["Channels"].value == "stereo", "ffmpeg names the layout; the row keeps the probe's own word"
    assert audio["Language"].value == "eng"
    assert audio["Bit rate"].value == "128 kb/s"
    subtitle = _rows(_section(media, "Subtitle 1"))
    assert subtitle["Codec"].value == "mov_text"
    assert subtitle["Language"].value == "rus"


def test_a_file_with_no_audio_says_so_instead_of_omitting_the_group(tmp_path):
    properties = properties_module.build_queue_properties(
        _job(str(tmp_path / "clip.mp4")), probe=_probe(_MEASURED)
    )
    media = _tab(properties, "MediaInfo")
    audio = _section(media, "Audio")
    assert audio.rows[0].display_value == properties_module.EM_DASH
    assert "no audio stream" in audio.rows[0].reason


# --- (e) rows come from the probe, missing fields are em dashes ---------------------------------------


def test_rows_are_filled_from_the_probe_value_for_value(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 1536)

    properties = properties_module.build_queue_properties(
        _job(str(path)), row=0, row_count=1, selected_count=1, probe=_probe(_MEASURED)
    )
    rows = _rows(_tab(properties, "MediaInfo"))

    assert [section.title for section in _tab(properties, "MediaInfo").sections] == [
        "File", "Video", "Audio", "Subtitles"
    ]
    assert rows["Container"].value == "mov,mp4,m4a,3gp,3g2,mj2"
    assert rows["Codec"].value == "h264"
    assert rows["Resolution"].value == "1920 x 1080"
    assert rows["Pixel format"].value == "yuv420p10le"
    assert rows["Chroma subsampling"].value == "4:2:0"
    assert rows["Bit depth"].value == "10 bit"
    assert rows["Frame rate"].value == "23.976 fps"
    assert rows["Frames"].value == "240"
    assert rows["Duration"].value == "00:00:10.010"
    assert rows["Colour space"].value == "bt709"
    assert rows["Colour primaries"].value == "bt709"
    assert rows["Transfer"].value == "bt709"


def test_a_field_the_probe_could_not_answer_is_an_em_dash_with_its_reason(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 1536)
    properties = properties_module.build_queue_properties(
        _job(str(path)),
        row=0,
        row_count=1,
        selected_count=1,
        probe=_probe(_MEASURED, {"color_range": "the file does not declare a colour range"}),
    )
    rows = _rows(_tab(properties, "MediaInfo"))
    assert rows["Colour range"].value == "", "a missing measurement is not a value"
    assert rows["Colour range"].display_value == properties_module.EM_DASH == "—"
    assert rows["Colour range"].reason == "the file does not declare a colour range"


def test_the_handler_builds_the_window_from_the_first_selected_row(tmp_path, monkeypatch):
    from src.ui import ProcessTab as process_tab_module

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    first, second = _job(str(path)), _job(str(tmp_path / "other.mp4"))
    opened = []

    class _FakeDialog:
        def __init__(self, job, **kwargs):
            opened.append((job, kwargs))

        def exec(self):
            opened.append("exec")

    monkeypatch.setattr(process_tab_module, "QueuePropertiesDialog", _FakeDialog)
    tab = object.__new__(process_tab_module.ProcessTab)
    tab.parent = types.SimpleNamespace(renderQueue=_Queue(items=[first, second], selected=[1, 0]))

    tab.openQueueProperties()

    job, kwargs = opened[0]
    assert job is first, "the lowest selected index is the row described"
    assert (kwargs["row"], kwargs["row_count"], kwargs["selected_count"]) == (0, 2, 2)
    assert kwargs["blocking"] is False and opened[-1] == "exec", "the handler owns the blocking call"


def test_the_document_carries_the_same_values_the_model_has(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    dialog = _dialog(path)
    rendered = dialog.document.text()
    assert "Video" in rendered
    assert "1920 x 1080" in rendered
    assert "—" in rendered


def test_a_file_that_is_not_there_still_opens_with_em_dashes(tmp_path):
    missing = tmp_path / "gone.mp4"
    properties = properties_module.build_queue_properties(
        _job(str(missing)), row=0, row_count=1, selected_count=1, probe=_probe({})
    )
    media_rows = _rows(_tab(properties, "MediaInfo"))
    assert media_rows["Codec"].display_value == "—"
    assert media_rows["Codec"].reason, "a missing value always says why"
    rve_rows = _rows(_tab(properties, "RVE"))
    assert rve_rows["Input size"].display_value == "—"
    assert "gone.mp4" in rve_rows["Input size"].reason


# --- (f) the RVE tab: the app's own facts -------------------------------------------------------------


def _ran_state(*, added=1_766_000_000.0, started=1_766_000_010.0, finished=1_766_000_022.5):
    return JobState(
        status="done",
        progress=1.0,
        added_mono=100.0,
        added_wall=added,
        started_mono=110.0,
        started_wall=started,
        finished_mono=122.5,
        finished_wall=finished,
        source_seconds=10.01,
    )


def _wall(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")


def test_the_rve_tab_reports_the_queue_position_and_the_timestamps_of_an_entry_that_ran(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    job = _job(str(path), state=_ran_state())
    properties = properties_module.build_queue_properties(job, row=2, row_count=7, probe=_probe(_MEASURED))
    rve = _tab(properties, "RVE")
    rows = _rows(rve)

    assert rows["Position"].value == "3 of 7", "1-based, and 'n of m'"
    assert rows["Added"].value == _wall(1_766_000_000.0)
    assert rows["Started"].value == _wall(1_766_000_010.0)
    assert rows["Finished"].value == _wall(1_766_000_022.5)
    assert rows["Render duration"].value == "00:00:12.500"
    assert rows["Frames rendered"].value == "240", "a finished job rendered its input's frames"
    assert rows["Per-frame cost"].display_value == "—", "no report beside the output -> not measured"
    assert "report" in rows["Per-frame cost"].reason


def test_an_entry_that_never_ran_reads_em_dashes_not_zeros(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    job = _job(str(path), state=JobState(status="pending", added_mono=100.0, added_wall=1_766_000_000.0))
    properties = properties_module.build_queue_properties(job, probe=_probe(_MEASURED))
    rows = _rows(_tab(properties, "RVE"))

    assert rows["Added"].value == _wall(1_766_000_000.0), "entering the queue is a fact even for a pending item"
    for label in ("Started", "Finished", "Render duration", "Frames rendered", "Per-frame cost"):
        assert rows[label].value == "", f"{label} is not a zero"
        assert rows[label].display_value == "—"
        assert rows[label].reason, f"{label} says why it is empty"


def test_models_are_listed_by_file_name_and_absent_ones_are_omitted(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    job = _job(str(path), deblurModelFile="/models/deblur-v2.pth")
    rows = _rows(_tab(properties_module.build_queue_properties(job, probe=_probe(_MEASURED)), "RVE"))

    assert rows["Upscale model"].value == "realesrgan-x4plus.pth", "the file NAME, not the path"
    assert rows["Upscale model arch"].value == "realesrgan"
    assert rows["Interpolation model"].value == "rife-4.25.pth"
    assert rows["Deblur model"].value == "deblur-v2.pth"
    for label in ("Denoise model", "Decompress model", "Scene-change model"):
        assert label not in rows, "a model this job does not carry is omitted, not invented"


def test_the_coefficients_are_the_entry_s_own_numbers(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    job = _job(
        str(path),
        overrideUpscaleScale=2,
        startTime=1.5,
        endTime=8.25,
        denoiseModelFile="/models/denoise.pth",
    )
    rows = _rows(_tab(properties_module.build_queue_properties(job, probe=_probe(_MEASURED)), "RVE"))

    assert rows["Scale"].value == "×4"
    assert rows["Override upscale scale"].value == "×2", "only shown because it differs from the model scale"
    assert rows["Interpolate times"].value == "×2"
    assert rows["Ensemble"].value == "No", "booleans read Yes/No"
    assert rows["Dynamic scale optical flow"].value == "Yes", "the source spells it `dyanmicScaleOpticalFlow`"
    assert rows["Slow motion mode"].value == "No"
    assert rows["Benchmark mode"].value == "No"
    assert rows["Tiling"].value == "Yes"
    assert rows["Tile size"].value == "256 px"
    assert rows["Backend"].value == "tensorrt"
    assert rows["HDR mode"].value == "No"
    assert rows["Segment start"].value == "00:00:01.500"
    assert rows["Segment end"].value == "00:00:08.250"
    assert rows["Encoder"].value == "-c:v libx264 -crf 18"


def test_an_override_equal_to_the_model_scale_is_not_repeated(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    job = _job(str(path), overrideUpscaleScale=4)
    rows = _rows(_tab(properties_module.build_queue_properties(job, probe=_probe(_MEASURED)), "RVE"))
    assert "Override upscale scale" not in rows


def test_the_run_report_rows_are_labelled_with_the_report_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    output = tmp_path / "clip_2x.mp4"
    report_path = tmp_path / "clip_2x.report.json"
    report_path.write_text(json.dumps({
        "provenance": {"commit": "abc123"},
        "run": {"input": str(path), "output": str(output), "frames": 240},
        "settings": {
            "temporal_reuse_threshold_effective": 3.5,
            "temporal_reuse_max_chain": 2,
            "temporal_reuse_statistic": "median",
            "quality_preset": "best",
            "int8_fit": "fp16",
        },
        "phases_ms_per_frame": {"gate": 0.4, "up_wall": 31.2, "pack": 3.0, "read": 0.5, "frame": 35.1},
    }))
    job = _job(str(path), outputPath=str(output), state=_ran_state())
    properties = properties_module.build_queue_properties(job, probe=_probe(_MEASURED))
    rve = _tab(properties, "RVE")
    rows = _rows(rve)

    assert rows["Per-frame cost"].value == "35.1 ms/frame"
    assert rows["Run report"].value == str(report_path)
    assert "clip_2x.report.json" in _section(rve, "From the run report — clip_2x.report.json").title
    report_rows = _rows(_section(rve, "From the run report — clip_2x.report.json"))
    assert report_rows["Temporal reuse threshold"].value == "3.5"
    assert report_rows["Max chain"].value == "2"
    assert report_rows["Statistic"].value == "median"
    assert report_rows["Quality preset"].value == "best"
    assert report_rows["int8 fit"].value == "fp16"
    assert "frame 35.1" in report_rows["Phases (ms/frame)"].value


def test_without_a_report_the_report_rows_are_em_dashes_with_the_reason(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    job = _job(str(path), state=_ran_state())
    properties = properties_module.build_queue_properties(job, probe=_probe(_MEASURED))
    rve = _tab(properties, "RVE")
    report_section = _section(rve, "From the run report")
    assert "clip_2x.report.json" in report_section.rows[0].reason
    for row in report_section.rows:
        assert row.value == ""
        assert row.display_value == "—"
        assert row.reason, "every empty report row says why"


def test_the_rve_tab_reports_the_input_and_output_paths_and_the_size(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 1536)
    properties = properties_module.build_queue_properties(_job(str(path)), probe=_probe(_MEASURED))
    rows = _rows(_tab(properties, "RVE"))
    assert rows["Input path"].value == str(path)
    assert rows["Input size"].value == "1.5 KB"
    assert rows["Output path"].value == str(path.with_name("clip_2x.mp4"))


# --- (h) the queue records when an item arrived --------------------------------------------------------


class _ListWidget:
    """The two list calls `RenderQueue` makes, no widget."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def addItem(self, value) -> None:
        self.items.append(str(value))

    def insertItem(self, index, value) -> None:
        self.items.insert(index, str(value))

    def clear(self) -> None:
        self.items.clear()


def test_render_queue_add_records_when_the_item_entered_the_queue():
    queue = RenderQueue(_ListWidget())
    job = _job("a.mp4")
    queue.add(job)
    assert job.state.added_wall is not None and job.state.added_mono is not None
    assert job.state.added_wall == pytest.approx(datetime.now().timestamp(), abs=5)


def test_render_queue_add_to_start_records_the_same_timestamp():
    queue = RenderQueue(_ListWidget())
    job = _job("a.mp4")
    queue.addToStart(job)
    assert job.state is not None, "an item inserted at the top needs a state record too"
    assert job.state.added_wall is not None
    assert job.state.started_wall is None and job.state.finished_wall is None


def test_the_queue_does_not_rewrite_an_arrival_it_already_recorded():
    queue = RenderQueue(_ListWidget())
    job = _job("a.mp4", state=JobState(added_wall=1_766_000_000.0, added_mono=1.0))
    queue.add(job)
    assert job.state.added_wall == 1_766_000_000.0


# --- (g) toolkit controls only -----------------------------------------------------------------------


def test_the_window_is_built_from_toolkit_controls_only(tmp_path):
    toolkit = pytest.importorskip("sli_ui_toolkit")
    from PySide6.QtWidgets import (
        QGroupBox,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QTableWidget,
        QTabWidget,
        QTextEdit,
        QTreeWidget,
        QWidget,
    )

    dialog = _dialog(tmp_path / "clip.mp4")
    assert dialog.styleSheet() == "", "no QSS: a custom-painted toolkit widget ignores it anyway"

    children = dialog.findChildren(QWidget)
    banned = (QLabel, QPushButton, QTableWidget, QTreeWidget, QTextEdit, QPlainTextEdit, QGroupBox, QScrollArea,
              QTabWidget)
    # A toolkit widget may legitimately subclass a stock one (`OverlayScrollArea` is a `QScrollArea`) - the rule is
    # "nothing the app did not paint itself", so the class's own module decides.
    stock = [
        type(child).__name__
        for child in children
        if isinstance(child, banned) and not type(child).__module__.startswith("sli_ui_toolkit")
    ]
    assert stock == [], "every widget in the window is a toolkit widget"
    assert not dialog.findChildren(QTabWidget), "the tabs are TopTabHost, never QTabWidget"

    widgets = toolkit.widgets
    assert dialog.findChildren(widgets.Text), "the rows render on the toolkit's painted Text"
    assert dialog.findChildren(widgets.Button), "the actions are toolkit Buttons"
    assert dialog.findChildren(widgets.OverlayScrollArea), "each document scrolls on the toolkit's scroll area"
    assert dialog.findChildren(widgets.TopTabHost), "the two tabs are the toolkit host"


# --- the probe's own rules, pinned on real ffmpeg output ----------------------------------------------


def test_parse_input_header_reads_the_fields_ffmpeg_printed():
    fields = properties_module.parse_input_header(SDR_HEADER)
    assert fields["container"] == "mov,mp4,m4a,3gp,3g2,mj2"
    assert fields["codec"] == "h264"
    assert fields["width"] == 320
    assert fields["height"] == 240
    assert fields["pixel_format"] == "yuv420p"
    assert fields["bit_depth"] == 8
    assert fields["fps"] == 25
    assert fields["duration"] == 2.0
    assert "color_range" not in fields, "ffmpeg printed no range for this file: absent, not 'tv'"
    assert "color_primaries" not in fields


def test_parse_input_header_reads_hdr_range_and_colours_and_refuses_unknown():
    fields = properties_module.parse_input_header(HDR_HEADER)
    assert fields["pixel_format"] == "yuv420p10le"
    assert fields["bit_depth"] == 10
    assert fields["color_range"] == "tv"
    assert fields["color_space"] == "bt2020nc"
    assert "color_primaries" not in fields, "ffmpeg printed 'unknown': that is not a measurement"
    assert "color_transfer" not in fields


def test_parse_input_header_answers_nothing_for_an_unreadable_input():
    assert properties_module.parse_input_header(NO_INPUT_TEXT) == {}


def test_probe_without_an_ffmpeg_binary_says_so_instead_of_guessing(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 10)
    result = properties_module.probe_video_file(
        str(path),
        ffmpeg_path=str(tmp_path / "no-ffmpeg-here"),
        mediainfo_reader=_no_mediainfo,
        backend_probe=lambda _path: {},
    )
    assert result.fields == {}
    assert result.reasons["codec"], "the window needs a reason for every empty field"
    assert "ffmpeg" in result.reasons["codec"]

    rows = _rows(_tab(
        properties_module.build_queue_properties(_job(str(path)), probe=lambda _path: result), "MediaInfo"
    ))
    assert "ffmpeg" in rows["Resolution"].reason, "a pair-of-fields row keeps the probe's own reason"


def test_probe_takes_the_frame_count_from_the_backend_and_the_rest_from_the_header():
    result = properties_module.probe_video_file(
        "clip.mp4",
        mediainfo_reader=_no_mediainfo,
        header_reader=lambda _path, _ffmpeg: (SDR_HEADER, ""),
        backend_probe=lambda _path: {"frames": 50, "width": 640, "height": 480, "fps": 60.0, "duration": 0.83},
    )
    assert result.fields["width"] == 320 and result.fields["height"] == 240, "the file's own header wins"
    assert result.fields["fps"] == 25
    assert result.fields["duration"] == 2.0
    assert result.fields["frames"] == 50, "only the app's own probe knows the frame count"
    assert "frames" not in result.reasons


def test_mediainfo_wins_over_ffmpeg_when_both_could_answer(tmp_path):
    """The preferred probe is asked first; the fallback is not even started when it answers."""
    started = []

    def ffmpeg_that_must_not_run(*_args, **_kwargs):
        started.append(True)
        return "", "should not be called"

    result = properties_module.probe_video_file(
        str(tmp_path / "hdr.mkv"),
        mediainfo_reader=lambda _path, _binary: (MI_DOCUMENT, "26.10", ""),
        header_reader=ffmpeg_that_must_not_run,
    )
    assert started == [], "MediaInfo answered, so ffmpeg was not started"
    assert result.source == "mediainfo 26.10"


def test_the_dialog_tabs_use_the_main_windows_strip_style(tmp_path):
    """The owner asked for the main window's tabs, not the toolkit default.

    `src/ui/widgets/tab_strip.py` exists because the process and settings strips drifted into two looks; a dialog that
    builds its host without `tab_style()`/`TabButtonFactory` is the third look. The default host builds `TopTabButton`;
    the shared factory hands it a plain toolkit `Button` and a flat, borderless style with a gap under the strip.
    """
    dialog = _dialog(tmp_path / "clip.mp4")
    try:
        host = dialog.tabs
        style = host.tab_bar._style
        assert style.flat is True, "the strip must be the flat button strip the pages use"
        assert style.border is False, "the main window's strip has no folder outline"
        assert style.gap == 8, f"the gap under the strip is the shared 8 px, got {style.gap}"
        # `TabItem` carries text/data only (the bar's own docstring says so); the built button lives in the bar's tab
        # list, which is exactly where `TabButtonFactory` records what it built.
        button = host.tab_bar._tabs[0].button
        assert type(button).__name__ == "Button", (
            f"a host-built tab is a toolkit Button; the default host would build {type(button).__name__}")
    finally:
        dialog.close()
