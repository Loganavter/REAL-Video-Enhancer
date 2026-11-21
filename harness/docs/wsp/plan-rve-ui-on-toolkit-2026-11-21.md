# Plan: RVE's interface onto `sli-ui-toolkit`

Status: draft — inventory measured, no code changed yet. Board row: `rve-ui-on-toolkit-2026-11-21`.
Task as stated (owner, 2026-11-21): **refactor the interface onto sli-ui-toolkit**. This document exists because the
first attempt at that sentence refactored the toolkit's own API instead of moving a host onto it — the rows
`tk-*-2026-11-21` in `projects/sli-ui-toolkit/docs/backlog/queue.yaml` are that misread request; they are real
improvements to the library, but they are not this task.

## Why RVE is the host

| | binding | verdict |
|---|---|---|
| RVE (`src/ui`, `mainwindow.py`) | **PySide6**, 0 references to the toolkit | compatible, and it is the app this session is about |
| `Improve-ImgSLI-9.0.0` (only app checkout on this machine) | PyQt6, 140 files, vendors a PyQt6-era toolkit copy | cannot use the current toolkit at all |

So "the interface onto the toolkit" can only mean a PySide6 host, and the only one with code here is RVE. If that is
wrong, the inventory below is still the thing to redo, not the conclusion.

## What has to move (measured 2026-11-21, `perf/inference` @ `a565077b`)

| layer | today | size | toolkit counterpart |
|---|---|---|---|
| shell | `testRVEInterface.ui` (Qt Designer) → `mainwindow.py` generated | 6 579 / 3 749 lines | `WindowChrome`, `BaseDialog`, plain `QWidget` composition |
| tabs | `QTabWidget` + `QStackedWidget`, tab classes as plain Python (`ProcessTab` 696, `SettingsTab` 668) | ~2 900 lines in `src/ui/` | `TopTabBar`/`TopTabHost`, `AdaptiveTabStrip`, `sidebar_nav_list` |
| bespoke popups | `NotificationOverlay`, `DownloadProgressPopup`, `SettingUpBackendPopup`, `DisplayCommandOutputPopup`, `TextOutputPopup`, `RegularQTPopup`, `IndependentQTPopup` | `QTcustom.py` 866 lines | `Toast`/`ToastManager`, `BaseDialog` + `dialog_shell`, `BaseFlyout`/`IconActionFlyout`/`SimpleOptionsFlyout` |
| theming | `src/ui/QTstyle.py` 119 lines + inline QSS in `mainwindow.py` | 2 places | `theme` tokens, `WidgetPolicyProxyStyle`, `UiScale` |
| primitives | raw `QPushButton`/`QLabel`/`QLineEdit`/`QComboBox` | throughout `.ui` | `Button`, `Text`, `CustomLineEdit`, `FluentComboBox`, `Checkbox` |
| console / help / queue | custom widgets | `RenderQueue.py` 348, `Updater.py` 204 | `LogConsoleWidget`/`console_view`, `markdown_help_dialog`, `list_panel`, `LoadingSpinner` |

What must stay in RVE (toolkit hard rules): icons, translations, business logic, worker threads, the render queue's
model, the updater's protocol. The toolkit takes widgets and hooks, never app knowledge.

## The fork in the road, decided first

RVE's shell is **Designer-driven** (`.ui` + generated `mainwindow.py`); the toolkit is **code-first** (`Button`,
`Text`, flyouts are constructed in Python). Two ways, and the choice decides every later wave:

* **A. Keep the `.ui`.** Toolkit widgets are `QWidget` subclasses, so they can be dropped into Designer as promoted
  widgets. Cheapest, keeps the existing layout knowledge, but every toolkit widget that wants code (flyouts, toast,
  tab bar) still has to be wired by hand; the `.ui` keeps its inline QSS, so theming stays split.
* **B. Move the shell to code.** Exactly the toolkit's own idiom (and its docs: one class + configuration). A real
  rewrite of the layout layer, and the only path that removes the split theming; highest cost.

The plan must decide this on evidence, not taste: a spike that puts **one toolkit widget into the existing `.ui`**
(a `Button` via promote) and **one toast** into the current window, then a look at what breaks. If the `.ui` interop
is clean, A is the cheap first half and B becomes optional per tab.

## Phases

| phase | scope | done when |
|---|---|---|
| P0 | interop spike: toolkit `Button` promoted in `testRVEInterface.ui`; `Toast` shown from `NotificationOverlay`'s call sites | the app starts, both work, and the spike says A or B |
| P1 | theming + primitives in **one tab** (`SettingsTab`): `Button`/`Text`/`CustomLineEdit`/`Combobox`, QSS for that tab deleted | the tab looks the same and RVE's suites are green |
| P2 | popups: seven bespoke classes → `Toast` / `BaseDialog` / flyouts, one class per commit | no `QProgressDialog`/ad-hoc `QDialog` left in `QTcustom.py` |
| P3 | navigation: `QTabWidget`/`QStackedWidget` → `TopTabBar`/`AdaptiveTabStrip` | tab switching, shortcuts and the render-queue badge behave as before |
| P4 | the shell decision from P0 applied; dead QSS removed | `QTstyle.py` gone or reduced to tokens RVE still owns |

Each phase is a separate commit series on a single branch, one tab/area at a time, and **one writer per area** (the
toolkit board's `owns_files` discipline does not apply here; this is one repo, one host).

## Verification (per phase, the same shape)

1. RVE suites: `PYTHONDONTWRITEBYTECODE=1 <runtime python> -m pytest harness/tests backend/tests -q -o addopts=""`.
2. `python3 -m harness.astcheck` over the touched files.
3. **A live look.** Every one of these phases changes what the user sees; offscreen green is not evidence for a
   layout. The app is launched on a leased instance and the touched screen is compared against the previous build.
4. A render still runs end to end (the GUI is the entry point for a render; a broken tab that cannot queue a job is a
   regression regardless of how it paints).

## Risks

- **The `.ui` is 6 579 lines**: a promoted toolkit widget that misbehaves in Designer costs more than the tab it
  replaces. Mitigation: the P0 spike decides before any tab is touched.
- **Theming split**: as long as QSS lives in both the `.ui` and `QTstyle.py`, every migrated widget has two sources
  of colour. Mitigation: one tab first, with its QSS deleted in the same change.
- **Behaviour creep**: this is a layout refactor; the render pipeline, queue semantics and updater must behave
  identically, and the existing tests are the guard.
- **Scope**: five phases is weeks, not a session. The board row is filed as `todo` with the phases named; nothing is
  started until P0 is agreed.

## Not covered

- The toolkit is not modified by this plan. If a gap appears (a widget RVE needs that the toolkit lacks), it becomes
  its own row on the toolkit board rather than an ad-hoc addition here.
- No visual redesign: same layout, same information, different implementation. A design change is a separate task.
