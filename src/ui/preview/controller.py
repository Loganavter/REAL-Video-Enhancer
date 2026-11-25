"""The preview's command/state machine: the only writer of `PreviewState`.

The view (`PreviewControls`) never calls a backend and never writes a widget from a signal: it emits a command.
`PreviewController.dispatch` validates the command against the current phase, talks to the selected `PlaybackBackend`
or the page's surface port, updates the state once, and emits `stateChanged`. A command that cannot work in this phase
answers with ONE `status` line - never an exception, never silence.

`observe(**facts)` is the other half: the page reports what it sees (the playhead a frame carried, the produced share,
the probe's source-frame count, whether a painter owns the pane). The controller merges those facts and re-derives the
phase (`PLAYING -> BUFFERING` when a live tail reaches the produced edge, back when new frames land). Both entry points
end in the same single `stateChanged`.

Every dispatch writes one line (`preview command id=... value=... phase=...`) so "the button did nothing" can be
answered from log.txt alone.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from .backends import NoBackend, PlaybackBackend
from .state import (
    MAX_ZOOM,
    MIN_ZOOM,
    VIEW_MODES,
    PreviewBackendKind,
    PreviewCommand,
    PreviewPhase,
    PreviewState,
)

logger = logging.getLogger(__name__)

#: One wheel/keyboard step of the zoom buttons (the wheel's own step is the pane's rule, not the buttons').
ZOOM_STEP = 1.25
#: A target this far past the reachable ceiling is a refusal, not a round-off (half a millisecond).
SEEK_SLACK = 5e-4


class PreviewSurfacePort:
    """What the page must answer for the picture itself (not the clock).

    Kept as a tiny protocol-like object so the controller can be tested with a dictionary recorder and the page can
    implement it on the pane's public API (`set_view_mode`, `set_zoom`, `reset_view`, `set_original`).
    """

    def set_view_mode(self, mode: str) -> None:
        raise NotImplementedError

    def set_zoom(self, zoom: float, *, fit: bool) -> None:
        raise NotImplementedError

    def set_compare(self, held: bool) -> None:
        raise NotImplementedError

    def set_fullscreen(self, fullscreen: bool) -> None:
        raise NotImplementedError


class PreviewController(QObject):
    stateChanged = Signal(object)      # PreviewState
    status = Signal(str)               # one human line for the log / toast

    def __init__(self, surface: PreviewSurfacePort | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = PreviewState()
        self._surface = surface
        self._backends: dict[PreviewBackendKind, PlaybackBackend] = {PreviewBackendKind.NONE: NoBackend()}
        self._backend: PlaybackBackend = self._backends[PreviewBackendKind.NONE]
        self._playing = False
        self._has_picture = False
        self._logger = logger.info
        self._last_emitted: PreviewState | None = None

    # -- wiring ---------------------------------------------------------------------------------

    def set_surface(self, surface: PreviewSurfacePort | None) -> None:
        self._surface = surface

    def set_logger(self, log: object) -> None:
        """Point the command log at the page's `log()` (log.txt); the module logger is the fallback."""
        if callable(log):
            self._logger = log

    def register_backend(self, kind: PreviewBackendKind, backend: PlaybackBackend) -> None:
        self._backends[kind] = backend

    def set_backend(self, kind: PreviewBackendKind, backend: PlaybackBackend | None = None) -> None:
        """Select the strategy for `kind` (the page registers fresh adapters as its owners change)."""
        if backend is not None:
            self._backends[kind] = backend
        self._backend = self._backends.get(kind, NoBackend())
        if self._state.backend is not kind:
            self._state = self._state.with_(backend=kind)

    def backend(self) -> PlaybackBackend:
        return self._backend

    def state(self) -> PreviewState:
        return self._state

    # -- facts from the page -----------------------------------------------------------------------

    def observe(self, *, phase: PreviewPhase | None = None, is_playing: bool | None = None,
                has_picture: bool | None = None, backend: PreviewBackendKind | None = None, **fields) -> None:
        """Merge what the page sees and emit one `stateChanged` when the state actually changed.

        `fields` are the `PreviewState` numbers (`playhead`, `produced`, `duration`, `source_frames`, `frames_done`,
        `output_frames`, `fps`, `seek_ceiling`, `mode`, `zoom`, `zoom_fit`, `compare`, `fullscreen`). Unknown names
        raise - a typo must not look like a control that never updates.
        """
        if backend is not None and backend is not self._state.backend:
            self.set_backend(backend)
        if is_playing is not None:
            self._playing = bool(is_playing)
        if has_picture is not None:
            self._has_picture = bool(has_picture)
        if fields:
            self._state = self._state.with_(**fields)
        self._state = self._state.with_(phase=self._derived_phase() if phase is None else phase)
        self._emit()

    def refresh(self) -> None:
        """Re-derive the phase and re-emit the current state (the page changed an owner, the widgets must re-read)."""
        self._state = self._state.with_(phase=self._derived_phase())
        self._emit()

    def close(self) -> None:
        self._state = self._state.with_(phase=PreviewPhase.CLOSED)
        self._emit()

    # -- commands ----------------------------------------------------------------------------------

    def dispatch(self, command: PreviewCommand | str, value: object = None) -> None:
        try:
            command = PreviewCommand(command)
        except ValueError:
            self.status.emit(f"preview: unknown command {command!r}")
            return
        self._log(command, value)
        state = self._state
        if state.phase in (PreviewPhase.IDLE, PreviewPhase.CLOSED) and command is not PreviewCommand.TOGGLE_FULLSCREEN:
            self._refuse(command, "no picture on the pane")
            return
        handler = self._HANDLERS.get(command)
        if handler is None:
            self._refuse(command, "not implemented")
            return
        handler(self, state, value)

    # -- handlers ----------------------------------------------------------------------------------

    def _on_play_pause(self, state: PreviewState, value: object) -> None:
        if isinstance(self._backend, NoBackend):
            self._refuse(PreviewCommand.PLAY_PAUSE, "nothing can play this pane")
            return
        target = not state.playing()
        if target:
            self._backend.play()
        else:
            self._backend.pause()
        self._playing = target
        self.refresh()

    def _on_step(self, state: PreviewState, frames: int) -> None:
        if isinstance(self._backend, NoBackend):
            self._refuse(PreviewCommand.STEP_FORWARD if frames > 0 else PreviewCommand.STEP_BACK, "nothing to step")
            return
        if not self._backend.step(frames):
            self._refuse(PreviewCommand.STEP_FORWARD if frames > 0 else PreviewCommand.STEP_BACK,
                         "the frame is not available")
            return
        self._state = self._state.with_(playhead=self._backend.position)
        self.refresh()

    def _on_step_back(self, state: PreviewState, value: object) -> None:
        self._on_step(state, -1)

    def _on_step_forward(self, state: PreviewState, value: object) -> None:
        self._on_step(state, 1)

    def _on_seek_to(self, state: PreviewState, value: object) -> None:
        try:
            target = float(value)
        except (TypeError, ValueError):
            self._refuse(PreviewCommand.SEEK_TO, f"not a time: {value!r}")
            return
        self._seek(state, target)

    def _on_seek_fraction(self, state: PreviewState, value: object) -> None:
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            self._refuse(PreviewCommand.SEEK_FRACTION, f"not a fraction: {value!r}")
            return
        if state.duration <= 0.0:
            self._refuse(PreviewCommand.SEEK_FRACTION, "the clip length is unknown")
            return
        self._seek(state, max(0.0, min(1.0, fraction)) * state.duration)

    def _seek(self, state: PreviewState, target: float) -> None:
        target = max(0.0, float(target))
        ceiling = state.seekable_ceiling()
        if target > ceiling + SEEK_SLACK:
            # ONE line, naming both numbers: a press past the produced edge is the refusal the owner must be able to
            # read back ("it did nothing" and "it clamped somewhere else" are otherwise the same silence).
            self.status.emit(
                f"seek {target:.3f} s refused: the pane can show up to {ceiling:.3f} s "
                f"({state.produced:.1%} produced)")
            return
        if isinstance(self._backend, NoBackend) or not self._backend.seek(target, exact=True):
            self._refuse(PreviewCommand.SEEK_TO, f"nothing can show {target:.3f} s")
            return
        self._state = self._state.with_(playhead=min(target, ceiling))
        self.refresh()

    def _on_set_mode(self, state: PreviewState, value: object) -> None:
        mode = str(value or "")
        if mode not in VIEW_MODES:
            self._refuse(PreviewCommand.SET_MODE, f"unknown view mode {value!r}")
            return
        if mode == state.mode:
            return
        surface = self._surface
        if surface is not None:
            surface.set_view_mode(mode)
        compare = state.compare and mode == "single"
        if compare != state.compare and surface is not None:
            surface.set_compare(compare)
        self._state = self._state.with_(mode=mode, compare=compare)
        self.refresh()

    def _on_zoom_in(self, state: PreviewState, value: object) -> None:
        self._apply_zoom(state, min(MAX_ZOOM, max(MIN_ZOOM, state.zoom) * ZOOM_STEP), fit=False)

    def _on_zoom_out(self, state: PreviewState, value: object) -> None:
        self._apply_zoom(state, max(MIN_ZOOM, state.zoom / ZOOM_STEP), fit=False)

    def _on_zoom_fit(self, state: PreviewState, value: object) -> None:
        self._apply_zoom(state, state.zoom, fit=True)

    def _on_zoom_one(self, state: PreviewState, value: object) -> None:
        self._apply_zoom(state, 1.0, fit=False)

    def _apply_zoom(self, state: PreviewState, zoom: float, *, fit: bool) -> None:
        if self._surface is not None:
            self._surface.set_zoom(float(zoom), fit=bool(fit))
        self._state = self._state.with_(zoom=float(zoom), zoom_fit=bool(fit))
        self.refresh()

    def _on_compare_down(self, state: PreviewState, value: object) -> None:
        if state.mode != "single":
            self._refuse(PreviewCommand.COMPARE_DOWN, "the compare modes already show the source")
            return
        if not state.has_picture():
            self._refuse(PreviewCommand.COMPARE_DOWN, "no picture to compare")
            return
        if self._surface is not None:
            self._surface.set_compare(True)
        self._state = self._state.with_(compare=True)
        self.refresh()

    def _on_compare_up(self, state: PreviewState, value: object) -> None:
        if not state.compare:
            return
        if self._surface is not None:
            self._surface.set_compare(False)
        self._state = self._state.with_(compare=False)
        self.refresh()

    def _on_toggle_fullscreen(self, state: PreviewState, value: object) -> None:
        target = not state.fullscreen
        if self._surface is not None:
            self._surface.set_fullscreen(target)
        self._state = self._state.with_(fullscreen=target)
        self.refresh()

    _HANDLERS = {
        PreviewCommand.PLAY_PAUSE: _on_play_pause,
        PreviewCommand.STEP_BACK: _on_step_back,
        PreviewCommand.STEP_FORWARD: _on_step_forward,
        PreviewCommand.SEEK_TO: _on_seek_to,
        PreviewCommand.SEEK_FRACTION: _on_seek_fraction,
        PreviewCommand.SET_MODE: _on_set_mode,
        PreviewCommand.ZOOM_IN: _on_zoom_in,
        PreviewCommand.ZOOM_OUT: _on_zoom_out,
        PreviewCommand.ZOOM_FIT: _on_zoom_fit,
        PreviewCommand.ZOOM_ONE: _on_zoom_one,
        PreviewCommand.COMPARE_DOWN: _on_compare_down,
        PreviewCommand.COMPARE_UP: _on_compare_up,
        PreviewCommand.TOGGLE_FULLSCREEN: _on_toggle_fullscreen,
    }

    # -- internals ---------------------------------------------------------------------------------

    def _derived_phase(self) -> PreviewPhase:
        if self._state.phase is PreviewPhase.CLOSED:
            return PreviewPhase.CLOSED
        if not self._has_picture:
            return PreviewPhase.IDLE
        if not self._playing:
            return PreviewPhase.READY
        state = self._state
        if (state.is_growing() and state.duration > 0.0
                and state.produced < state.duration - SEEK_SLACK
                and state.at_produced_edge()):
            # Playing a growing file AT its produced edge: the clock is honest, the frames are not all written yet.
            return PreviewPhase.BUFFERING
        return PreviewPhase.PLAYING

    def _emit(self) -> None:
        # ONE signal per real change: the page pushes its facts ten times a second, and a repaint of both bars for an
        # unchanged state is exactly the per-tick churn the old strip's `set_progress` guard existed to stop.
        if self._state == self._last_emitted:
            return
        self._last_emitted = self._state
        self.stateChanged.emit(self._state)

    def _refuse(self, command: PreviewCommand, why: str) -> None:
        self.status.emit(f"preview: {command.value} refused - {why}")

    def _log(self, command: PreviewCommand, value: object) -> None:
        try:
            self._logger(f"preview command id={command.value} value={value!r} phase={self._state.phase.value}")
        except Exception:
            logger.info("preview command id=%s value=%r phase=%s",
                        command.value, value, self._state.phase.value)
