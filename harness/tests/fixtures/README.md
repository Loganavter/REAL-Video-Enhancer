# Gate fixtures: one real violation per rule, cut from the history that carried it

`harness/tests/test_gate_selftest.py` runs every gate over the fixtures here and requires a **non-zero exit**, then
runs the same gate over the current tree and requires **zero**. A gate that cannot be made red is not proven by this
directory; it is reported as missing.

## The header contract

Every `.py` fixture starts with a comment header, a blank line, then the body:

```
# why this shape is the violation the gate names
# provenance: src/ui/pages/proc_page.py:345-354 @ 9bbcd776ecd9
# provenance-note: which commit fixed it, or why the history is not in this clone
#@rule: stock-widget
#@findings: 1
```

* `provenance` - either `<repo-relative path>:<first>-<last> @ <sha>` or the literal `NONE`.
  * A **history** claim is verified, not trusted: the selftest runs `git show <sha>:<path>`, takes lines
    `<first>..<last>`, un-indents them by the first line's own indent, and requires the fixture body to equal that
    slice exactly. An edit to a historical fixture, or an authored body that cites a commit, fails the selftest
    (`test_an_authored_fixture_claiming_history_is_refused`). A cut is a *copy*, never a retelling.
  * `NONE` is a first-class answer: the fixture is authored to the gate's shape and the header says so. It must carry
    a `provenance-note` explaining why no history was available. The selftest accepts that form **without** pretending
    it is history - the two branches are separate.
* `provenance-note` - where the bug came from and which commit retired it, or why the historical lines are out of
  reach (a toolkit checkout outside this cell's jail, a probe script that never landed).
* `@rule` - one of `harness/scripts/ui_gates.py`'s `SCOPES`/`CHECKS` keys. The directory name is
  `rule.replace("-", "_")` and the selftest asserts the two agree.
* `@check` - for `harness/astcheck.py`: `dead store` or `used before assignment`.
* `@findings` - how many findings of **that** check the fixture must produce. The selftest asserts the exact count, so
  a fixture that stops being red, or a rule that silently widens, both fail.

## What each fixture is

| fixture | gate | provenance |
|---|---|---|
| `ui_gates/qss_on_adapter/download_page_low_storage_label.py` | `qss-on-adapter` | `download_page.py:157-165` @ `542b55379cc3` |
| `ui_gates/raw_chrome_px/general_settings_combo_width.py` | `raw-chrome-px` | `general_settings.py:236-245` @ `dfcf06e56e17` |
| `ui_gates/raw_chrome_px/scrollbar_width_six_px.py` | `raw-chrome-px` | `NONE` (the 8-vs-6 bar belongs to the toolkit, outside the jail) |
| `ui_gates/theme_read_at_build/module_level_colour.py` | `theme-read-at-build` | `NONE` (rule started green) |
| `ui_gates/stock_widget/proc_page_tab_strip_qtabs.py` | `stock-widget` | `proc_page.py:345-354` @ `9bbcd776ecd9` (the tab-strip overlay bar) |
| `ui_gates/empty_stylesheet/download_page_empty_reset.py` | `empty-stylesheet` | `download_page.py:126-130` @ `9bbcd776ecd9` |
| `ui_gates/silent_swallow/toolkit_dialogs_close_swallow.py` | `silent-swallow` | `toolkit_dialogs.py:381-389` @ `79520cbdf5b3` |
| `ui_gates/backend_imports_ui/backend_module_importing_the_ui.py` | `backend-imports-ui` | `NONE` (green for the whole history) |
| `ui_gates/page_process_http/home_tab_requests.py` | `page-process-http` | `HomeTab.py:1-3` @ `1edc7c8d3d2a` |
| `astcheck/dead_store/calib_fade_records_log.py` | dead store | `calib_fade_records.py:34-46` @ `f0c47ca5744d` |
| `astcheck/dead_store/quantise_pre_relu_head.py` | dead store | `NONE` (the founding probe is not in this object store) |
| `astcheck/used_before_assignment/c2_probe.py` | used before assignment | `NONE` (authored to the rule's docstring) |

`ui_diff/page_budget_growth_2026_11_21.json` is the one non-Python fixture: the real per-page numbers of the
2026-11-21 run (`ui-page-diff-2026-11-21.json` @ `9ddb7322`), where a flat 0.05 budget failed six pages. It carries its
provenance as JSON fields, and the selftest asserts the fact the per-page ratchet exists for: the two Process
rows (0.393) now pass inside their measured 6.0, while the four 0.168-3.137 pages are still refused.
