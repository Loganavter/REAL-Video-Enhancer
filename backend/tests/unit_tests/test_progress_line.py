import io
import logging
import os
import sys

import pytest


BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
	sys.path.insert(0, BACKEND_DIR)

PROGRESS = "FPS: 29 Current Frame: 120 ETA: 0:03:19"


@pytest.fixture()
def util():
	module = pytest.importorskip("src.utils.Util")
	yield module
	module.set_inplace_line(False)


def _emit(util, message):
	"""Send a record through the installed handler's filter and formatting."""
	record = logging.LogRecord("RVE", logging.WARNING, __file__, 1, message, None, None)
	handler = logging.StreamHandler(io.StringIO())
	handler.addFilter(util._EndInplaceLineFilter())
	handler.handle(record)
	return handler.stream.getvalue()


def _redirect(monkeypatch):
	# inside the test body: pytest re-installs its own capture during fixture setup
	out, err = io.StringIO(), io.StringIO()
	monkeypatch.setattr(sys, "stdout", out)
	monkeypatch.setattr(sys, "stderr", err)
	return out, err


def test_a_warning_never_lands_on_the_progress_line(util, monkeypatch):
	out, _ = _redirect(monkeypatch)
	util.set_inplace_line(True)
	sys.stdout.write("\r" + PROGRESS)

	assert _emit(util, "host timeline: upload 0.12") == "host timeline: upload 0.12\n"
	# the record ended the in-place line: the front end reads "ETA: 0:03:19" as one whole line
	assert out.getvalue() == "\r" + PROGRESS + "\n"


def test_the_installed_rve_handler_is_line_aware(util):
	logger = logging.getLogger("RVE")
	aware = [h for h in logger.handlers if getattr(h, "_rve_inplace_aware", False)]
	assert len(aware) == 1
	assert any(isinstance(f, util._EndInplaceLineFilter) for f in aware[0].filters)
	assert logger.propagate is False


def test_log_ends_the_progress_line(util, monkeypatch):
	out, err = _redirect(monkeypatch)
	util.set_inplace_line(True)
	out.write("\r" + PROGRESS)

	util.log("something")

	assert out.getvalue() == "\r" + PROGRESS + "\n"
	assert err.getvalue() == "BACKEND: something\n"


def test_nothing_is_written_when_no_line_is_open(util, monkeypatch):
	out, _ = _redirect(monkeypatch)
	util.set_inplace_line(False)
	_emit(util, "int8: start")
	util.log("plain")
	assert out.getvalue() == ""
