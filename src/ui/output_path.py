"""The output path's only owner: one string, one writer, one place to change it.

Owner report (2026-11-26): *two places set the output folder, and the hidden one always wipes the choice made last
session.* Measured on the tree before this module:

* the General page's `outputFileText` held a full PATH and was **not persisted at all**;
* the Encoder page's `output_folder_location` held a DIRECTORY, persisted under that key (default: the platform's
  Videos folder);
* `MainWindow.updateVideoGUIDetails` - the handler for every option checkbox - called `setDefaultOutputFile`, whose last
  statement was `outputFileText.setText(...)`. So every option change regenerated the path over whatever the user had
  chosen, and a hand-typed path was gone after a restart because nothing had saved it.

The rules this module enforces:

1. **One string.** The path the field shows and the path a render writes to are the same value - the field's text.
   `show_path` is the only function in the tree that writes that field.
2. **A default fills an EMPTY field only.** `apply_default` derives a path from the input file and the options and
   shows it unless the user has chosen one (`stored_path` is non-empty). This is the reported bug, and the guard is
   the one line that fixes it.
3. **An explicit choice may replace it.** The Select Output Folder button is the user speaking, so it may replace even
   a user's path (`apply_chosen`); it then hands the file NAME back to the derivation, because naming a file is a
   choice the app makes from the options, while naming a DIRECTORY is the choice the user just made.
4. **The old directory key stays read.** `output_folder_location` is no longer a second way to set the path; it is the
   directory fallback an existing `settings.txt` already carries, so a returning user's folder does not vanish. A
   stored path's own directory wins over it.

The keys are `output_file_path` (the path the user chose; empty means "derive one") and `output_folder_location` (the
directory fallback). `harness/tests/test_output_path_owner.py` pins each rule above, red before the change.

Qt-free on purpose: the rules are string arithmetic over a settings dict, so they are testable without a window, and
the window, the settings tab and the field all read the same functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Sequence

#: The path the user chose, persisted. Empty is the honest "nothing chosen yet": the default is then derived from the
#: input file and the options, and is never written here.
KEY_PATH = "output_file_path"

#: The directory, persisted since before this change. Now a FALLBACK for a user with no stored path - it is what keeps
#: an existing installation's output folder after the upgrade.
KEY_DIRECTORY = "output_folder_location"

#: What a batch render's file name placeholder has always been (`{MULTIPLE_FILES}` in the input field).
MULTI_FILE_NAME = " { FILE_NAME } "


@dataclass(frozen=True)
class Derivation:
    """Everything a default output path is derived from, handed over as data.

    The window fills this from its widgets (that is wiring); the arithmetic - and the rule that decides whether it may
    replace what the field shows - is this module's.
    """

    input_file: str
    directory: str
    #: `_<model display name>` per enabled stage, in the order the file name has always carried them.
    tags: Sequence[str]
    container: str
    fps: float
    width: int
    height: int


def is_multi_file(input_file: str) -> bool:
    """The batch input field carries a placeholder, not a path."""
    return (input_file or "").strip().replace(" ", "") == "{MULTIPLE_FILES}"


def stored_path(settings) -> str:
    """The path the user chose, or `""`. Also the answer to "may a default replace the field?"."""
    return str(settings.settings.get(KEY_PATH, "") or "").strip()


def user_set(settings) -> bool:
    """Whether a path the USER chose is stored - the one question the default's guard asks.

    No separate flag: the stored string is the fact. A derived path is never stored, so "non-empty on disk" and "the
    user set it" cannot drift apart (and clearing the field, or resetting the settings, takes the ownership back with no
    latch to forget).
    """
    return bool(stored_path(settings))


def store_user_path(settings, path: str) -> str:
    """Persist a path the user chose (an empty string hands the name back to the derivation)."""
    text = (path or "").strip()
    settings.writeSetting(KEY_PATH, text)
    return text


def store_directory(settings, folder: str) -> str:
    """Persist the DIRECTORY fallback - only an explicit "use this folder" action does this."""
    folder = (folder or "").strip()
    settings.writeSetting(KEY_DIRECTORY, folder)
    return folder


def stored_directory(settings, home: str | None = None) -> str:
    """The directory a render goes into when nothing else names one: the stored path's, else the old key, else home.

    The middle term is the migration: a `settings.txt` written before `output_file_path` existed carries only
    `output_folder_location`, and that folder must survive the upgrade.
    """
    folder = os.path.dirname(stored_path(settings))
    if folder:
        return folder
    legacy = str(settings.settings.get(KEY_DIRECTORY, "") or "").strip()
    return legacy or (home or os.path.expanduser("~"))


def directory_for(path: str, fallback: str) -> str:
    """The directory `path` writes into; `fallback` when it does not name one (an empty field, a bare name)."""
    return os.path.dirname((path or "").strip()) or fallback


def derive(derivation: Derivation, exists: Callable[[str], bool] = os.path.isfile) -> str:
    """The default path for one input: `<name>_<models>_<fps>fps_<wxh>.<container>`, made unique with `_(n)`.

    Byte-identical to what `MainWindow.setDefaultOutputFile` built before the move (same tags, same order, same
    `round(..., 0)` fps form, same collision loop) - a render's file name is not this change's subject.
    """
    source = MULTI_FILE_NAME if is_multi_file(derivation.input_file) else derivation.input_file
    name = os.path.splitext(os.path.basename(source))[0] + "".join(derivation.tags)
    base = f"{name}_{round(derivation.fps, 0)}fps_{derivation.width}x{derivation.height}"
    candidate = os.path.join(derivation.directory, f"{base}.{derivation.container}")
    if not exists(candidate):
        return candidate
    # The suffix is MONOTONIC: the next name continues past the highest `_(n)` already on disk, instead of filling the
    # first gap. Filling gaps looked like "the app forgot", and it made a second run write `_(0)` while `_(2)` sat
    # next to it (owner, 2026-11-28: "почему он не умеет циферку в скобочках сам менять на незанятую" - the `(2)` in
    # the name he quoted belongs to the MODEL file, while the app's own counter is the trailing `_(n)`).
    iteration = _next_suffix(os.path.dirname(candidate), os.path.basename(candidate), base, derivation.container)
    candidate = os.path.join(derivation.directory, f"{base}_({iteration}).{derivation.container}")
    while exists(candidate):
        iteration += 1
        candidate = os.path.join(derivation.directory, f"{base}_({iteration}).{derivation.container}")
    return candidate


def _next_suffix(directory: str, name: str, base: str, container: str) -> int:
    """One past the highest `_(n)` this derivation already used in `directory` (0 when there is none).

    Reads the folder once: the alternative - probing `_(0)`, `_(1)`, ... - answers a gap, which reads as a forgotten
    counter. A folder that cannot be listed falls back to the probing loop's start, never raises.
    """
    prefix, suffix = f"{base}_(", f").{container}"
    highest = -1
    try:
        for entry in os.listdir(directory or "."):
            if entry.startswith(prefix) and entry.endswith(suffix):
                digits = entry[len(prefix):-len(suffix)]
                if digits.isdigit():
                    highest = max(highest, int(digits))
    except OSError:
        return 0
    return highest + 1


def show_path(general, path: str) -> None:
    """Write the path field - the ONLY writer of `outputFileText`'s text in the tree.

    `general` is the General page's handle group; the field is the one widget this module touches.
    """
    general.outputFileText.setText(path)


def apply_default(general, derivation: Derivation, settings) -> str:
    """Show the path this input should render to, and return it.

    The rule the owner asked for: a stored user path is shown and returned untouched; only an empty one lets the
    derived default fill the field. A stored path with an empty field (the first call after a restart) is put back into
    the field, which is the half of the report about "the field comes back wrong".
    """
    field = str(general.outputFileText.text() or "").strip()
    chosen = stored_path(settings)
    if chosen:
        path = field or chosen
        if path != field:
            show_path(general, path)
        return path
    path = derive(derivation)
    show_path(general, path)
    return path


def apply_chosen(general, derivation: Derivation, settings) -> str:
    """An explicit "use this directory": persist it, replace the path, and derive the NAME in it.

    Allowed to overwrite a user's path - the user just asked. The FILE NAME is not persisted: it is derived from the
    input and the options, so the next option change re-derives it in the directory the user chose.
    """
    store_directory(settings, derivation.directory)
    store_user_path(settings, "")
    path = derive(derivation)
    show_path(general, path)
    return path


def reset_path(general, settings) -> None:
    """The settings were reset: the path goes back to the empty field and to the derivation."""
    store_user_path(settings, "")
    show_path(general, "")


def mirror_directory(general, mirror, settings) -> str:
    """Write the READ-ONLY mirror (the Encoder page's row) with the directory a render writes into.

    The mirror is one fact with one owner: it is the path field's directory, refreshed whenever the field changes. It
    cannot be edited into a second answer, which is what it was before this module.
    """
    folder = directory_for(general.outputFileText.text(), stored_directory(settings))
    mirror.setText(folder)
    return folder
