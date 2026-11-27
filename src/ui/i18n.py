"""RVE's interface language: English plus ONE translated language (Russian), on the toolkit's runtime.

The tree was already shaped for this before the feature existed: 192 `QCoreApplication.translate("MainWindow", …)`
call sites and a complete `retranslate()` pass on every page (each one restates every string, which is why a second
call repaints rather than appends). What was missing was anything to drive them. This module is that driver and nothing
more - the expensive part of a language switch was a seam, not a catalogue.

Two bridges, one table:

* **Qt's side.** Every page string is a `QCoreApplication.translate(...)` call, so the cheapest honest binding is a
  `QTranslator` installed on the application: `translate(context, source, …)` looks the ENGLISH SOURCE STRING up in
  `TRANSLATIONS` and returns it, or the source itself on a miss. No call site changes; a miss stays English, on screen,
  which is the honest failure mode (a raw key would be worse).
* **The toolkit's side.** `sli_ui_toolkit.i18n` owns the switch signal (`emit_language_changed`), the translator
  callable (`configure_i18n(translator=…)`, so `sli_ui_toolkit.tr()` and the toolkit's `translatable_*` widgets share
  this table) and the page pass (`install_translator(root, tr_func)` calls `retranslate(lang)` on every object under
  `root` that implements it).

The page pass needs a QObject to walk, and RVE's `retranslate()` methods live on plain builder classes
(`ProcPagePage`, `SettingsPagePage`, …), which no `findChildren()` can see: installing on the real window without
`bind_page()` covers nothing at all (measured 2026-11-24: `install_translator(window, …)` returns, no page is touched).
`bind_page(page, root)` parents a small relay to the page's root widget; the relay is the QObject with a
`retranslate(lang)` method, and it forwards the language to the page. One `install(window)` at the end of
`MainWindowShell.build` then covers every relay in the window (and any child added later, the toolkit installs a
`ChildAdded` watcher).

Language state: `src.ui.settings_store.Settings` owns the `language` key and its validator; this module caches the
current language (the QTranslator is consultative, so a switch needs no reinstall) and re-emits it to the toolkit.

Ordering: `configure_i18n(language=…)` must run BEFORE `MainWindowShell.build`, so the very first paint is already in
the chosen language rather than repainting into it.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QTranslator

from sli_ui_toolkit.i18n import (
    configure_i18n as _configure_toolkit_i18n,
    emit_language_changed as _toolkit_emit_language_changed,
    install_translator,
)

from src.ui.settings_store import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, validate_language

#: The interface languages the app ships. The tuple itself lives in `settings_store` (the settings owner, importable
#: without PySide6); this is the public name the UI reads.
LANGUAGES = SUPPORTED_LANGUAGES

#: Display names are endonyms on purpose: a user looking for Russian should not have to read the current language to
#: find it, and these are names, not translations of names.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "ru": "Русский",
}

#: The table, keyed by the ENGLISH SOURCE STRING - the exact literal a call site passes to
#: `QCoreApplication.translate`, byte for byte (a trailing space is part of the key). One dict per language; a key
#: missing from a language's dict falls back to the source string, and that is a supported state, not a bug: the
#: catalogue is filled in slices.
TRANSLATIONS: dict[str, dict[str, str]] = {
    "ru": {
        # --- Process page chrome (`src/ui/pages/proc_page.py::retranslate`) -------------------------------
        "Start Time (seconds)": "Начало (секунды)",
        "End Time (seconds)": "Конец (секунды)",
        "Render Preview": "Предпросмотр рендера",
        # --- The run card's line fragments (`src/ui/pages/general_settings.py::_meta_line` and the run card) -----
        # Rebuilt on every tick, so also keyed at the write site. These are FRAGMENTS: the writer appends the number
        # first and the fragment after it, so the space belongs to the key (`" elapsed"`, `" left"`) and the frame label
        # is the prefix `"frame "`. The bare words are deliberately not keys - the writer never passes them, and a
        # bare-word entry would answer for nothing. (They used to feed the `progressDetail` line under the transport
        # row as well; that line was removed 2026-11-28, the run card is the only reader.)
        " elapsed": " прошло",
        " left": " осталось",
        "frame ": "кадр ",
        # --- The video-info pane's units (`src/ui/pages/general_settings.py::_meta_line` / tag / fps pill) ---------
        # Fragments written after a number, like the progress-detail ones above. `upscale` is the headline tag's word
        # (`2x upscale`); `fps` and `frames` are the headline's muted second line and the run card's pill.
        "fps": "к/с",
        "frames": "кадров",
        "upscale": "увеличение",
        # The five inner tab labels of the process page (`processSettingsContainer`).
        "General": "Общие",
        "Encoder Options": "Параметры кодирования",
        "Render Queue": "Очередь рендера",
        "Advanced": "Дополнительно",
        "Logs": "Журнал",
        # --- Settings page headings (`src/ui/pages/settings_page.py::retranslate`) -----------------------
        "Output Settings": "Настройки вывода",
        "Render Settings": "Настройки рендера",
        "RVE Settings": "Настройки RVE",
        "Reset Settings": "Сбросить настройки",
        # --- RVE Settings rows (`src/ui/pages/tab_4.py::retranslate`) ------------------------------------
        "Preview Enabled": "Предпросмотр включён",
        "Discord Rich Presence": "Активность в Discord",
        "Open RVE Folder": "Открыть папку RVE",
        "Language": "Язык",
        "- 2.6: Supports older CUDA enabled GPUs": "- 2.6: поддерживает старые GPU с CUDA",
        "- 2.8: Supports all RTX cards, faster and recomended.": "- 2.8: все карты RTX, быстрее и рекомендовано.",
        "- 2.9: Experimental.": "- 2.9: экспериментальная.",
        "- Any encoder without a device label is a CPU encoder.": "- Любой кодировщик без метки устройства — процессорный.",
        "- Auto enables HDR mode if RVE detects that the video that is being rendered is HDR. This will cause slower rendering if the video is HDR.": "- Авто включает HDR, если RVE определит, что рендерится HDR-видео. **На HDR-видео рендер будет медленнее.**",
        "- Auto: Defaults to float16 if the GPU is supported (RTX 20 series and up)": "- Авто: float16, если GPU поддерживает (RTX 20 и новее)",
        "- CUDA: NVIDIA GPUs 10 series and up": "- CUDA: NVIDIA GPU 10 серии и новее",
        "- Changes the video container in the default output file generated container.": "- Меняет контейнер выходного файла.",
        "- Compression algorithm for video, most common is libx264": "- Алгоритм сжатия видео; самый распространённый — libx264",
        "- Custom output scale no matter the model. Will run AI on video frame once, and scale the image to match the multiplier. Useful for 4x models.": "- Свой масштаб вывода независимо от модели. Нейросеть прогоняется по кадру один раз, изображение масштабируется до множителя. Полезно для 4x-моделей.",
        "- Defines how color information is stored for each pixel in a video frame.": "- Определяет, как хранится информация о цвете каждого пикселя кадра.",
        "- Enabled: Attempts to detect when scenes change in a video.": "- Включено: приложение пытается находить смены сцен в видео.",
        "- Enables TensorRT to build for many different resolutions in one engine.": "- Позволяет TensorRT собрать один движок под разные разрешения.",
        "- Float16: Faster inference and less VRAM usage.": "- Float16: быстрее и меньше видеопамяти.",
        "- Float32: Slower inference and more VRAM usage, supported on more GPUS.": "- Float32: медленнее и больше видеопамяти, поддерживается на большем числе GPU.",
        "- HDR is only color tested on x264 and x265 encoders.": "- Цвет HDR проверен только на кодировщиках x264 и x265.",
        "- Higher means more optimization, but slower engine generation time.": "- Больше — сильнее оптимизация, но дольше сборка движка.",
        "- Higher number: lower chance of detecting scene changes, with risk of underdetection.": "- Больше — ниже шанс найти смену сцены, риск **пропусков**.",
        "- Lossless Encoders:": "- Кодировщики без потерь:",
        "- Lossy Encoders:": "- Кодировщики с потерями:",
        "- Lower means less optimization, but faster engine generation time.": "- Меньше — слабее оптимизация, но быстрее сборка движка.",
        "- Lower number: higher chance of detecting scene changes, with risk of overdetection.": "- Меньше — выше шанс найти смену сцены, риск **ложных срабатываний**.",
        "- MPS: Apple Mac M series SOCs only.": "- MPS: только Apple Silicon (M).",
        "- ROCm: AMD (Linux Only) 7000 series(maybe) and up": "- ROCm: AMD (только Linux), 7000 серия (возможно) и новее",
        "- Recommended: 2-3 for real life, 4 for animation.": "- Рекомендуется: 2–3 для живого видео, 4 для анимации.",
        "- Renders a small portion of the video to show the results (most useful for upscaling)": "- Рендерит небольшой фрагмент видео, чтобы показать результат (полезнее всего для увеличения)",
        "- Speed at which the encoder processes the video, Slower processing time means Better results for the file size.": "- Скорость работы кодировщика. **Медленнее** обработка — **лучше результат при том же размере файла**.",
        "- The preview colors may be distorted as the preview itself is not HDR.": "- Цвета превью могут искажаться: само превью не HDR.",
        "- This is the version downloaded": "- Это скачанная версия",
        "- Will automatically remove black bars if they are present throughout a video, can increase performance.": "- Автоматически убирает чёрные полосы, если они есть по всему видео; может ускорить обработку.",
        "- av1 (mkv, mp4, webm)": "- av1 (mkv, mp4, webm)",
        "- ffv1 (mkv, avi)": "- ffv1 (mkv, avi)",
        "- libx264 (mkv, mp4, mov, avi)": "- libx264 (mkv, mp4, mov, avi)",
        "- libx265 (mkv, mp4, mov)": "- libx265 (mkv, mp4, mov)",
        "- mean (segmented): fast, prone to overdetection": "- mean (по сегментам): быстро, склонно к ложным срабатываниям",
        "- mean: fastest, least accurate": "- mean: быстрее всех, наименее точный",
        "- mkv is recomended due to its extensive support for different formats.": "- **mkv** рекомендуется: он поддерживает больше всего форматов.",
        "- prores (mkv, mov)": "- prores (mkv, mov)",
        "- pyscenedetect: slow, accuate (Recommended)": "- pyscenedetect: медленно, точно **(рекомендуется)**",
        "- vp9 (mkv, webm)": "- vp9 (mkv, webm)",
        "- xpu: Intel: idk fam, dunno if this even works lowkey": "- xpu: Intel — если честно, не знаю, работает ли это вообще",
        "- yuv 420/422/444 p10le: 10 bit version of each. (MKV and MP4 containers only) (mkv, mov, mp4)": "- yuv 420/422/444 p10le: 10-битная версия каждого (только MKV и MP4)",
        "- yuv420p: Most common, removes the most information. (mkv, mp4, mov, avi, webm)": "- yuv420p: самый распространённый, теряет больше всего информации (mkv, mp4, mov, avi, webm)",
        "- yuv422p: In between 420p and 444p. (mkv, mov, mp4)": "- yuv422p: между 420p и 444p (mkv, mov, mp4)",
        "- yuv444p: Keeps the most information. (mkv, mov, mp4)": "- yuv444p: сохраняет больше всего информации (mkv, mov, mp4)",
        "Audio and Subtitles will not be transfered to output video.": "Аудио и субтитры не попадут в выходное видео.",
        "Audo Border Cropping:": "Автоматическая обрезка полей:",
        "Encoder Speed:": "Скорость кодека:",
        "Ensembles flow, can produce better results.": "Ансамбль потоков может дать лучший результат.",
        "Helps with anime interpolation.": "Помогает при интерполяции аниме.",
        "Increase video length instead of framerate when interpolating.": "Увеличивать длительность видео, а не частоту кадров при интерполяции.",
        "Lowers VRAM usage, but also slows down render.": "Меньше видеопамяти, но рендер медленнее.",
        "NCNN GPU ID -": "GPU NCNN —",
        "Only compadible with older RIFE models and GMFSS": "Совместимо только со старыми моделями RIFE и GMFSS",
        "Only use when render failes due to VRAM limits.": "Использовать, только если рендер падает из-за нехватки видеопамяти.",
        "Perform processing without outputing new video. This tests the raw performance of the inference.": "Выполнять обработку без записи видео — проверка чистой производительности инференса.",
        "Pixel Format": "Формат пикселей",
        "Please look at the audio and video encoder tooltips for compadibility.": "Совместимость смотрите в подсказках к аудио- и видеокодеку.",
        "Precision:": "Точность:",
        "PyTorch GPU ID -": "GPU PyTorch —",
        "Render Preview:": "Предпросмотр рендера:",
        "Saves VRAM and speeds up inference at resolutions above 1080p by calculating flow at a lower resolution. (GMFSS/GIMM)": "Экономит видеопамять и ускоряет инференс выше 1080p: поток считается в меньшем разрешении (GMFSS/GIMM).",
        "Scales optical flow based on the difference between frames.": "Масштабирует оптический поток по разнице между кадрами.",
        "Scene Change Detection Enabled:": "Определение смены сцен:",
        "Sets what GPU is used for NCNN": "Задаёт GPU для NCNN",
        "Sets what GPU is used for PyTorch": "Задаёт GPU для PyTorch",
        "Show REAL Video Enhancer in discord.": "Показывать REAL Video Enhancer в Discord.",
        "Split up processing upscaled frames into chunks.": "Разбивать обработку увеличенных кадров на части.",
        "TensorRT Dynamic Engine:": "Динамический движок TensorRT:",
        "TensorRT Optimization Level:": "Уровень оптимизации TensorRT:",
        "Upscale Scale:": "Масштаб увеличения:",
        "WARNING: Changing this MAY break encoder compadibility.": "**ВНИМАНИЕ: смена контейнера МОЖЕТ сломать совместимость с кодировщиком.**",
        "When running an interpolation, it will avoide interpolating over these scene changes.": "При интерполяции кадры через такие смены сцен не интерполируются.",
        " Add to Render Queue": " Добавить в очередь",
        "\"Support Me!\"": "«Поддержите меня!»",
        "Encoder command: the encoder settings passed to FFMpeg, can be manually tweaked. The field shows the part that fits; the whole command is in this tooltip.": "Команда кодека: настройки, передаваемые FFMpeg, можно править вручную. Поле показывает влезающую часть; полная команда — в этой подсказке.",
        "Lossless": "Без потерь",
        "Ultra": "Ультра",
        "Very_High": "Очень высокое",
        "High": "Высокое",
        "Medium": "Среднее",
        "Low": "Низкое",
        "fastest": "максимальная",
        "fast": "быстрая",
        "medium": "средняя",
        "slow": "медленная",
        "placebo": "placebo (самая медленная)",
        "Best quality": "Лучшее качество",
        "Fast": "Быстрое",
        "Backend": "Бэкенд",
        "Backends": "Бэкенды",
        "Benchmark Mode": "Режим замера",
        "SloMo Mode": "Режим замедления",
        "General Settings": "Общие настройки",
        "Encoder Settings": "Настройки кодека",
        "Interpolate Settings": "Настройки интерполяции",
        "Video Encoder": "Видеокодек",
        "Video Encoder Speed": "Скорость кодека",
        "Video Quality": "Качество видео",
        "Video Container": "Контейнер",
        "Video Pixel Format": "Формат пикселей",
        "Quality Preset": "Пресет качества",
        "Precision": "Точность",
        "TensorRT Optimization Level": "Уровень оптимизации TensorRT",
        "TensorRT Dynamic Engine": "Динамический движок TensorRT",
        "PyTorch Version": "Версия PyTorch",
        "PyTorch Backend": "Бэкенд PyTorch",
        "PyTorch GPU ID": "GPU PyTorch",
        "NCNN GPU ID": "GPU NCNN",
        "Auto HDR Mode": "Автоматический режим HDR",
        "Auto Border Cropping": "Автоматическая обрезка полей",
        "Output folder directory": "Папка вывода",
        "Scene Change Detection Enabled": "Определение смены сцен",
        "Scene Change Detection Method": "Метод определения смены сцен",
        "Scene Change Detection Threshold": "Порог определения смены сцен",
        "Interpolate": "Интерполяция",
        "Interpolate Model": "Модель интерполяции",
        "Interpolation Multiplier": "Множитель интерполяции",
        "Interpolation UHD Mode": "Режим UHD интерполяции",
        "Decompress": "Декомпрессия",
        "Decompress Model": "Модель декомпрессии",
        "Denoise": "Шумоподавление",
        "Denoise Model": "Модель шумоподавления",
        "Deblur": "Устранение размытия",
        "Deblur Model": "Модель устранения размытия",
        "Upscale": "Увеличение",
        "Upscale Model": "Модель увеличения",
        "Upscale Scale": "Масштаб увеличения",
        "Tiling": "Тайлинг",
        "Tile Size": "Размер тайла",
        "Upscale/Restoration Tile Settings": "Тайлинг увеличения",
        "Ensemble (Pytorch/TensorRT Only)": "Ансамбль (PyTorch/TensorRT)",
        "Dynamic Scaled Flow (Pytorch Only)": "Динамический поток (только PyTorch)",
        "Reuse unchanged frames": "Переиспользовать неизменённые кадры",
        "By cel (ignore grain, dust and scan wobble)": "По целлу (без зерна, пыли и дрожания)",
        "Select Input File": "Выбрать входной файл",
        "Select Output Folder": "Выбрать папку вывода",
        "Open Output Folder": "Открыть папку вывода",
        "Batch Select": "Пакетный выбор",
        "Import Custom Upscale Model": "Импорт модели увеличения",
        "Select Model (NCNN/.bin+.param)": "Выбрать модель (NCNN/.bin+.param)",
        "Select Model (PyTorch/TensorRT/.pth)": "Выбрать модель (PyTorch/TensorRT/.pth)",
        "Install Recommended Dependencies": "Установить зависимости",
        "Uninstall App": "Удалить приложение",
        "Support Me!": "Поддержите меня!",
        "GitHub Page": "Страница GitHub",
        "NCNN Vulkan (All GPUs, slower inference, small download)": "NCNN Vulkan (любые GPU, медленнее, небольшая загрузка)",
        "PyTorch (NVIDIA/CUDA, AMD/ROCm (Linux Only), INTEL/xpu, MacOS/MPS)": "PyTorch (NVIDIA/CUDA, AMD/ROCm только Linux, INTEL/xpu, MacOS/MPS)",
        "TensorRT (Nvidia RTX 20 series and up, fastest inference, largest download)": "TensorRT (NVIDIA RTX 20 и новее, самый быстрый вывод, большая загрузка)",
        "DirectML - NOT IMPLEMENTED YET (All DirectX12 capable GPUs, faster inference, small download, Windows only)": "DirectML — пока НЕ РЕАЛИЗОВАНО (любые GPU с DirectX12, быстрее, небольшая загрузка, только Windows)",
        "Collapse the preview band": "Свернуть полосу превью",
        "Expand the preview band": "Развернуть полосу превью",
        "Paused: the preview holds this frame. Press to follow the render.": "Пауза: превью держит этот кадр. Нажмите, чтобы следовать за рендером.",
        "Your PC does not have enough storage for PyTorch/TensorRT. Please free up at least 15gb.": "Недостаточно места для PyTorch/TensorRT. Освободите хотя бы 15 ГБ.",
        # --- The General tab's video-info pane (`src/ui/pages/general_settings.py`, variant A blocks) ---------
        # The pane's nine row names. The values written after them (`MainWindow.updateVideoGUIText`) are not
        # translated: they are file facts.
        "FPS:": "К/с:",
        "Resolution:": "Разрешение:",
        "Frame Count:": "Количество кадров:",
        "Encoder:": "Кодировщик:",
        "Container:": "Контейнер:",
        "Color Space:": "Цветовое пространство:",
        "Pixel Format:": "Формат пикселей:",
        # Not bare `HDR:` on purpose: `updateVideoGUIText` writes its own `HDR: <value>` row, and a translation
        # that collides with it would make the rewrite ambiguous on the way back to English.
        "Is HDR:": "Поддержка HDR:",
        "Bit Depth:": "Глубина цвета:",
        # --- The pipeline ribbon (`src/ui/pages/general_settings.py::_render_pipeline`) ----------------------------
        # The plan card above the headline: a stage's state word and parameter, and the footer's own labels. The stage
        # NAMES are not repeated here - the ribbon paints "Interpolate"/"Denoise"/... through the keys the General tab's
        # rows already state (`"Interpolate"`, `"Decompress"`, ..., `"Upscale"`), so a stage cannot read two ways.
        "off": "выкл",
        "on": "вкл",
        "scale": "масштаб",
        "by cel": "по целлу",
        "Preset": "Пресет",
        "Reuse": "Переиспользование",
        "No video loaded. Nothing will run until a stage is on and an input file is chosen.": "Видео не загружено. Ничего не запустится, пока не включён хотя бы один этап и не выбран входной файл.",
    },
}

#: The toolkit's loader still runs on every `emit_language_changed` (it rebuilds its own pack cache). This root holds
#: one empty pack per language, so the loader neither warns nor serves anything: the callable above is the only
#: backend (`configure_i18n(translator=…)` takes precedence inside `tr()`). It exists to keep the toolkit's own
#: contract quiet, not as a second catalogue.
PACK_ROOT = Path(__file__).resolve().parent / "i18n_packs"

#: The language in force. Cached here because the QTranslator consults it on every lookup and because
#: `configure_i18n` runs before any widget exists.
_current_language: str = DEFAULT_LANGUAGE

#: The installed `QTranslator`. Module-level so it is not garbage-collected (Qt does not keep a Python reference).
_qt_translator: "_SourceTranslator | None" = None

#: True once the table, the toolkit backend and the Qt translator are in place. The window owner calls
#: `ensure_configured()`; `main()` calls it first so the very first paint is already translated.
_configured: bool = False


def language_name(code: str | None) -> str:
    """The language's own name, or the code when it is not one of ours."""
    normalised = validate_language(code)
    return LANGUAGE_NAMES.get(normalised, normalised)


def current_language() -> str:
    """The language in force (also the toolkit's `language_provider`)."""
    return _current_language


def translate(text: str, language: str | None = None) -> str:
    """`text` in `language` (default: the language in force); the source string when the key is missing.

    A table entry that is empty or missing falls back to `text` - the same rule as a missing language, so a half-filled
    catalogue can never blank a label.
    """
    table = TRANSLATIONS.get(validate_language(language or _current_language))
    if not table:
        return text
    translated = table.get(text)
    return translated if translated else text


def translator(key: str, language: str | None = None) -> str:
    """The `(key, language) -> text` callable the toolkit's `configure_i18n` expects."""
    return translate(key, language)


def tr(text: str) -> str:
    """Translate `text` in the language in force. For code that builds a string at call time (a f-string prefix, a
    message not routed through `QCoreApplication.translate`)."""
    return translate(text)


class _SourceTranslator(QTranslator):
    """Qt's bridge: look the English source string up in `TRANSLATIONS`.

    `context` and `disambiguation` are ignored - the whole tree has one context (`"MainWindow"`) and the table is keyed
    by the source string, which is unique enough at this size. `n` is Qt's plural selector; the table carries no
    plurals, so it is ignored too and falls through to the source.
    """

    def translate(self, context, source_text, disambiguation=None, n=-1):  # noqa: N802 - Qt API
        return translate(source_text)


def configure_i18n(language: str | None = None, *, application=None) -> str:
    """Install the table, the toolkit backend and the Qt translator, and select `language`.

    Safe to call more than once: the `QTranslator` is installed once, later calls only select a language. Returns the
    language actually selected (unknown values are coerced to `en` before anything is installed).
    """
    global _current_language, _qt_translator, _configured

    _current_language = validate_language(language)

    # The toolkit shares the table: `sli_ui_toolkit.tr(key)` and toolkit widgets bound with `translatable_*` resolve
    # through the same callable, and the provider makes the toolkit read OUR language rather than its own tracker.
    _configure_toolkit_i18n(
        i18n_root=PACK_ROOT,
        translator=translator,
        language_provider=current_language,
    )

    app = application if application is not None else QCoreApplication.instance()
    if app is not None and _qt_translator is None:
        _qt_translator = _SourceTranslator()
        app.installTranslator(_qt_translator)

    _configured = True
    return _current_language


def is_configured() -> bool:
    """Whether the translator and the toolkit backend are in place (`ensure_configured`'s guard)."""
    return _configured


def configure_from_settings(application=None) -> str:
    """Configure the language `settings.txt` stores - the window owner's entry point."""
    from src.ui.settings_store import Settings

    return configure_i18n(Settings().settings.get("language"), application=application)


def ensure_configured(application=None) -> str:
    """Configure from `settings.txt` once; later calls keep the language in force.

    Every entry point that builds the window calls this: `main()` first, so the first paint is already translated,
    and the shell itself as the guarantee for owners that never run `main()`. That second call is not decoration -
    the live harness builds `MainWindow` directly, and without it a live instance ignored `settings.txt` and the
    first language switch answered `i18n root not configured` (measured on a lease, 2026-11-24).
    """
    if _configured:
        return _current_language
    return configure_from_settings(application=application)


#: The markup a tooltip is built with. The toolkit paints these labels itself, so the shape is RVE's, not Qt's: a bold
#: title line and one paragraph per bullet. It lives HERE so a tooltip is code + translated parts, never a translated
#: HTML document (a document-as-key breaks on any whitespace edit and silently falls back to English).
_TOOLTIP_TITLE = '<p><span style=" font-weight:700;">{}</span></p>'
_TOOLTIP_LINE = "<p>{}</p>"


def _rich(text: str) -> str:
    """A tooltip line with `**bold**` and `__underline__` markers rendered as spans.

    The markers exist so a translator writes plain text with two punctuation marks instead of HTML; the markup stays in
    this module, where a wrong tag is a code bug and not a silent fallback.
    """
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    escaped = re.sub(r"\*\*(.+?)\*\*", r'<span style=" font-weight:700;">\1</span>', escaped)
    escaped = re.sub(r"__(.+?)__", r'<span style=" text-decoration: underline;">\1</span>', escaped)
    return escaped


def tooltip(title: str, *lines: str) -> str:
    """A tooltip document assembled from catalogue parts: the title and each line are translated separately.

    Replaces the old shape, where the WHOLE `<html>…</html>` string was the catalogue key: 27 such keys could not be
    translated without putting markup in a translator's hands, and any markup edit made the key miss. Call sites pass
    English source strings (the catalogue's keys); the markup is fixed here and a switch rebuilds the document.
    """
    parts = [_TOOLTIP_TITLE.format(_rich(tr(title)))]
    parts.extend(_TOOLTIP_LINE.format(_rich(tr(line))) for line in lines if line)
    return "<html><head/><body>" + "".join(parts) + "</body></html>"


def emit_language_changed(language: str | None) -> str:
    """Switch the interface language now: select it, then let the toolkit fan `retranslate(lang)` out.

    The Qt side needs no reinstall (the translator reads the cached language per lookup); every page that restates its
    strings in `retranslate()` follows the toolkit signal. Persisting the choice is the caller's job
    (`src/ui/settings_store.py` owns the `language` key).
    """
    global _current_language

    _current_language = validate_language(language)
    _toolkit_emit_language_changed(_current_language)
    return _current_language


class _PageRelay(QObject):
    """A QObject child of a page root that forwards `retranslate(lang)` to its builder.

    The relay exists because the toolkit walks QObjects: the pages are plain classes, so without a QObject in the
    subtree the install has nothing to call and a switch silently does nothing.
    """

    def __init__(self, page, parent: QObject):
        super().__init__(parent)
        self._page = page

    def retranslate(self, language: str | None = None) -> None:
        self._page.retranslate(language)


def bind_page(page, root: QObject) -> _PageRelay:
    """Make `page.retranslate(lang)` reachable from the toolkit's subtree walk.

    Call at the end of the page's `build()`, with the widget the page returned. Binding does not install anything by
    itself: `install(root)` on the containing window (or the page root, in a test) covers every relay under it.
    """
    relay = _PageRelay(page, root)
    # Keep the relay reachable from the page too: Qt owns it through its parent, the attribute makes the ownership
    # readable from the page and keeps a stale relay from being re-created on a rebuild.
    page._i18n_relay = relay
    return relay


def install(root: QObject) -> None:
    """Cover `root` and its current/future descendants with the translator.

    One call per window (the shell does it at the end of `build`) - the toolkit then calls `retranslate(lang)` on every
    relay under it on each switch. Tests that build a single page call this on that page's root.
    """
    install_translator(root, translator)
