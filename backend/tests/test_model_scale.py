"""One scale, derived from the model rather than its name.

The defect (measured live 2026-11-23): `4xNomos8k_span_otf_weak_no_update_params.pth` reported
`Model Scale: 4` - the number in its descriptor/name - while a 160x120 input came back 320x240. The
writer sized the output from 4 and the frames arrived at 2, so the default path carried the same lie
the `--override_upscale_scale 2` crash exposed. In the code, `UpscaleModelWrapper.load_model`'s
TemporalSPAN fallback hardcoded `__scale = 2` while the descriptor path reported whatever the file
claimed, and nothing compared the two.

What this file pins:

* `resolve_scale` - agreed scales pass through, a disagreement raises a named `ValueError` naming both
  numbers and the model path, and it needs neither torch nor any module state to run;
* `resolve_loaded_scale` - the loader's policy: a measurement is used and says so, and with no
  measurement the scale is logged as DECLARED, not observed (never silent);
* `tspan_scale_from_state_dict` - the TemporalSPAN family's stride read from its own upsampler weights
  (12 = 3 * 2**2), replacing the hardcoded 2, and the loader wiring that uses both.

This interpreter has no torch (the backend's venv has it), so the wrapper module - whose *classes* are
torch modules - is loaded with a minimal torch stand-in. Nothing is registered under the module's real
name, so a later `import src.pytorch.UpscaleModelWrapper` in the app's interpreter still constructs the
real thing; the functions under test touch none of it.
"""

import importlib.util
import inspect
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

WRAPPER_PATH = BACKEND_DIR / "src" / "pytorch" / "UpscaleModelWrapper.py"
MODEL = "models/4xNomos8k_span_otf_weak_no_update_params.pth"


class _FakeTorch(types.ModuleType):
    """Enough torch for the wrapper's class statements (bases, annotations, decorators).

    The module under test is imported, not executed, so only what runs at import time has to exist:
    `torch.nn.Module` as a real base class, `torch.inference_mode()` as a decorator, and anything else
    as an attribute that survives being looked at. A function that actually needed torch at import
    time would fail here, which is one of the things this file is checking.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()


class _Anything:
    def __call__(self, *args, **kwargs):
        return _Anything()

    def __getattr__(self, name):
        return _Anything()


def _fake_modules():
    class _Module:  # stands in for torch.nn.Module as a base class
        pass

    torch = _FakeTorch("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = _Module
    functional = types.ModuleType("torch.nn.functional")

    def _inference_mode(*args, **kwargs):
        return lambda function: function

    torch.nn = nn
    torch.nn.functional = functional
    torch.inference_mode = _inference_mode

    stubs = {
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        # every one of these imports torch itself; none of their symbols is under test
        "src.utils.Util": types.SimpleNamespace(log=lambda *a, **k: None),
        "src.pytorch.TorchUtils": SimpleNamespace(TorchUtils=object),
        "src.pytorch.VSRArchs.AnimeSR.vsr_inference_helper":
            SimpleNamespace(AnimeSRInferenceHelper=object),
        "src.pytorch.VSRArchs.AnimeSR.animesr_arch": SimpleNamespace(AnimeSR=object),
        "src.pytorch.VSRArchs.TSPAN.vsr_inference_helper":
            SimpleNamespace(TemporalSPANInferenceHelper=object),
        "src.pytorch.VSRArchs.TSPAN.tspan": SimpleNamespace(TemporalSPAN=object),
    }
    return stubs


@pytest.fixture(scope="module")
def scale():
    """The real `UpscaleModelWrapper` module, loaded without torch and without shadowing its name."""
    import unittest.mock

    stubs = _fake_modules()
    with unittest.mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "src.pytorch._model_scale_under_test", WRAPPER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == WRAPPER_PATH.resolve()
    return module


# --- (a) an agreed scale passes through ----------------------------------------------------------


@pytest.mark.parametrize("value", [1, 2, 3, 4, 8])
def test_an_agreed_scale_passes_through(scale, value):
    assert scale.resolve_scale(value, value, MODEL) == value


def test_a_descriptor_alone_is_returned_as_the_scale(scale):
    """No probe available: the number is still usable, it is just a claim (logged as such)."""
    assert scale.resolve_scale(4, None, MODEL) == 4


def test_a_probe_alone_is_returned_as_the_scale(scale):
    assert scale.resolve_scale(None, 2, MODEL) == 2


def test_a_scale_below_one_is_refused_as_named(scale):
    with pytest.raises(ValueError) as excinfo:
        scale.resolve_scale(0, 0, MODEL)
    assert isinstance(excinfo.value, scale.ScaleMismatch)
    assert MODEL in str(excinfo.value)


def test_no_number_at_all_is_refused_as_named(scale):
    with pytest.raises(ValueError) as excinfo:
        scale.resolve_scale(None, None, MODEL)
    assert isinstance(excinfo.value, scale.ScaleMismatch)
    assert MODEL in str(excinfo.value)


# --- (b) a disagreement is a named refusal, never a silent pick ----------------------------------


def test_the_live_disagreement_names_both_numbers_and_the_model_path(scale):
    """The measured 2026-11-23 pair: descriptor 4x, frames 2x."""
    with pytest.raises(ValueError) as excinfo:
        scale.resolve_scale(4, 2, MODEL)

    assert isinstance(excinfo.value, scale.ScaleMismatch), "a caller must be able to tell this apart"
    message = str(excinfo.value)
    assert "declares 4x" in message, "the claimed number must be named"
    assert "emits 2x" in message, "the measured number must be named"
    assert MODEL in message, "a refusal that does not name the file is not actionable"
    assert "Refusing" in message, "the message must say that no number was chosen"


def test_the_refusal_is_a_value_error_so_existing_handlers_still_catch_it(scale):
    assert issubclass(scale.ScaleMismatch, ValueError)


# --- (c) no probe: the descriptor is used and the line says DECLARED -----------------------------


def test_a_missing_probe_falls_back_to_the_descriptor_and_says_so(scale):
    lines = []

    result = scale.resolve_loaded_scale(4, None, MODEL,
                                        probe_error="RuntimeError: no weights for a probe",
                                        emit=lines.append)

    assert result == 4, "no probe must not mean no render"
    assert len(lines) == 1, lines
    line = lines[0]
    assert "DECLARED" in line and "not observed" in line, line
    assert "4x" in line
    assert "no weights for a probe" in line, "the reason must be named, not just the state"


def test_the_disabled_probe_reports_the_switch_that_disabled_it(scale):
    """`RVE_MODEL_SCALE_PROBE=0` is the control arm: the scale is declared, and the log says why."""
    lines = []

    assert scale.resolve_loaded_scale(2, None, MODEL, probe_error="RVE_MODEL_SCALE_PROBE=0",
                                      emit=lines.append) == 2
    assert "RVE_MODEL_SCALE_PROBE=0" in lines[0]


def test_an_agreeing_probe_is_reported_as_observed(scale):
    lines = []

    assert scale.resolve_loaded_scale(4, 4, MODEL, emit=lines.append) == 4
    assert "agree" in lines[0], lines


def test_a_descriptor_of_4_over_frames_of_2_resolves_to_2_loudly(scale):
    """The loader's policy: a forward pass is what the frames do; a name is a claim about them."""
    lines = []

    result = scale.resolve_loaded_scale(4, 2, MODEL, emit=lines.append)

    assert result == 2, "the measured stride is the one the writer can size an output from"
    logged = "\n".join(lines)
    assert "declares 4x" in logged and "emits 2x" in logged, "both numbers stay on the record"
    assert "MEASURED" in logged, "the choice must be visible in the log, not just in the return value"


def test_the_measured_scale_wins_even_when_it_is_the_larger_one(scale):
    """Not a one-way rule: an under-claimed descriptor is the same defect mirrored."""
    lines = []

    assert scale.resolve_loaded_scale(2, 4, MODEL, emit=lines.append) == 4
    assert "MEASURED 4x" in "\n".join(lines)


# --- (d) the TemporalSPAN family: the stride comes from its own weights ---------------------------


def test_the_tspan_upsampler_weights_carry_a_2x_stride(scale):
    """`pixelshuffle_block`: conv(feature -> 3 * scale**2) then PixelShuffle(scale); 12 = 3 * 2**2."""
    payload = {"upsampler.0.weight": SimpleNamespace(shape=(12, 48, 3, 3))}

    assert scale.tspan_scale_from_state_dict(payload) == 2


def test_the_tspan_upsampler_weights_carry_a_4x_stride_when_they_do(scale):
    payload = {"upsampler.0.weight": SimpleNamespace(shape=(48, 48, 3, 3))}

    assert scale.tspan_scale_from_state_dict(payload) == 4, "the hardcoded 2 must not survive a 4x file"


@pytest.mark.parametrize("payload", [{}, {"upsampler.0.weight": SimpleNamespace(shape=(5, 48, 3, 3))},
                                     {"upsampler.0.weight": SimpleNamespace(shape=(0,))},
                                     "not a state dict"])
def test_an_underivable_checkpoint_yields_no_scale(scale, payload):
    assert scale.tspan_scale_from_state_dict(payload) is None


def test_the_tspan_case_resolves_to_the_scale_the_architecture_emits(scale):
    """Derive the stride from the checkpoint, then confirm it with the (stubbed) probe."""
    payload = {"upsampler.0.weight": SimpleNamespace(shape=(12, 48, 3, 3))}
    declared = scale.tspan_scale_from_state_dict(payload)
    probed = 2  # what a forward pass of the built TemporalSPAN(upscale=declared) returns for 2x
    lines = []

    assert scale.resolve_loaded_scale(declared, probed, "tspan_2x.pth", emit=lines.append) == 2
    assert "agree" in "\n".join(lines)


def test_the_tspan_case_still_prefers_the_probe_when_the_weights_disagree(scale):
    """A checkpoint whose upsampler says 4 but whose built model emits 2 is the same lie, not a pass."""
    declared = scale.tspan_scale_from_state_dict(
        {"upsampler.0.weight": SimpleNamespace(shape=(48, 48, 3, 3))})
    lines = []

    assert scale.resolve_loaded_scale(declared, 2, "tspan_odd.pth", emit=lines.append) == 2
    assert "MEASURED 2x" in "\n".join(lines)


# --- (e) the resolver needs no torch, and the loader is actually wired to it ----------------------


def test_the_resolver_runs_with_nothing_but_its_own_arguments(scale):
    """Exec'd in an empty namespace: no torch, no module state, only `int` and its named exception.

    This is the "importable and testable without torch" property stated as a test rather than trusted:
    a function that reached for a global would `NameError` here.
    """
    namespace = {"ScaleMismatch": scale.ScaleMismatch}
    exec(compile(textwrap.dedent(inspect.getsource(scale.resolve_scale)),
                 "<resolve_scale>", "exec"), namespace)
    resolver = namespace["resolve_scale"]

    assert resolver(4, 4, MODEL) == 4
    assert resolver(4, None, MODEL) == 4
    with pytest.raises(namespace["ScaleMismatch"]):
        resolver(4, 2, MODEL)


def test_the_resolver_names_no_torch_global(scale):
    assert "torch" not in scale.resolve_scale.__code__.co_names


def test_the_loader_asks_the_resolver_in_every_branch_and_hardcodes_nothing(scale):
    source = inspect.getsource(scale.UpscaleModelWrapper.load_model)

    assert source.count("self.__resolve_scale(self.__scale)") == 3, (
        "spandrel, animesr and tspan must each go through the resolver", source)
    assert "__scale = 2" not in source, "the TemporalSPAN hardcode must be gone"
    assert "tspan_scale_from_state_dict" in source, "the tspan stride must come from the checkpoint"


def test_the_probe_is_measured_through_a_helper_of_its_own(scale):
    """The render's own helper must not be the probe's: the temporal helpers cache frames."""
    probe = inspect.getsource(scale.UpscaleModelWrapper._UpscaleModelWrapper__probe_scale)
    assert "__make_inference_helper" in probe
    helper = inspect.getsource(scale.UpscaleModelWrapper._UpscaleModelWrapper__make_inference_helper)
    assert "TemporalSPANInferenceHelper" in helper and "AnimeSRInferenceHelper" in helper


# --- (f) the writer-side guard: a tensor that contradicts the scale cannot reach the writer -------


def test_the_render_path_refuses_a_tensor_that_disagrees_with_the_scale():
    """`UpscaleTorch` cannot be imported here (it imports torch at module scope), so this pins the
    guard in its source: the frame is built at `videoWidth * scale`, and a model that returns another
    geometry must raise before that Frame exists rather than become a crop or a reshape error."""
    source = (BACKEND_DIR / "src" / "pytorch" / "UpscaleTorch.py").read_text()

    assert "def _assert_model_geometry" in source
    assert source.count("self._assert_model_geometry(") == 2, (
        "the whole-frame path and the tiled path both crop against `scale`", source)
    assert 'path="UpscaleTorch._call_single"' in source
    assert 'path="UpscaleTorch.renderTiledImage"' in source
