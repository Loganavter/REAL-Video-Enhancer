# Fira Mono (bundled)

RVE's 25 pt display size (`src/ui/typography.py::DISPLAY_PT`) is used by the Home page's heading, which explicitly
asks for the family `Fira Mono` (the Download page's heading uses the same 25 pt size on the UI face). On a host that
only has **Fira Code** installed, `fc-match "Fira Mono"` answers `NotoSans-Regular` and the Home heading silently
renders in the wrong face. The owner's decision (2026-11-24) is to ship the face with the application instead of
accepting the fallback or removing the family from the type scale.

Files, all from the upstream [mozilla/Fira](https://github.com/mozilla/Fira) release (Fira Mono 3.206):

| file | style | bytes |
|---|---|---|
| `FiraMono-Regular.ttf` | Regular (400) | 174,632 |
| `FiraMono-Medium.ttf` | Medium (500) | 173,516 |
| `FiraMono-Bold.ttf` | Bold (700) | 206,548 |

Total bundled cost: **559,175 bytes** (546 KiB, TTFs + license).

`OFL.txt` is the upstream license, SIL Open Font License 1.1. It ships next to the fonts so a packager cannot miss it:
the OFL requires the license to travel with the font files, and it permits bundling and redistribution inside a larger
work as long as the fonts themselves are not sold on their own and any derivative keeps the reserved names.

They are registered at runtime by `src/ui/bundled_fonts.py::ensure()` (called from `src/ui/toolkit_boot.py::ensure()`
before the theme and before any page is built); nothing is installed system-wide, so an uninstalled copy of the app
still gets the face.
