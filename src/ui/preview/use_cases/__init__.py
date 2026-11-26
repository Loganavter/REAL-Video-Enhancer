"""The tab's use cases: plain functions whose first argument is the tab (the owner).

`ProcessTab` keeps the wiring, the job thread and its instance state, plus one thin delegator per entry point below
under the name it already has in the tab. The logic of each concern lives here so it can be driven by a
`SimpleNamespace` owner in a test, with no window and no tab class.

* `present` - the pane's own moment -> the register's bounded answer -> the pane's `set_source`.
* `supply` - warm the compare decoder, ask for the decode lead, one ask in flight, the pause policy.
* `modes` - `single` / `sbs` / `split`: which modes exist, which require a pair, one place a mode is set.

P2's other half (`seek`, `transport`, `job`) is another owner's files, not re-exported here.

Nothing is imported here on purpose: a re-export would make `import src.ui.preview.use_cases.present` pull `modes` (and
with it the app's `Util`) as a side effect. Callers import the module they need.
"""

from __future__ import annotations
