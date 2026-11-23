#!/usr/bin/env bash
# Rebuild src/ui/widgets/shaders/*.qsb from their GLSL sources.
#
# Why a script: the app ships the baked .qsb (no shader compiler at runtime - PySide6's wheel has no `qsb` at all), and
# the bake must be reproducible for review. qsb defaults to SPIR-V ONLY, which a desktop OpenGL context cannot load
# ("No GLSL shader code found ... in baked shader"); the GLSL target list below is what makes the same bake work on the
# OpenGL path the app uses, including llvmpipe on a headless host.
#
#   qsb --glsl "100 es,120,150,330,440" -o frame.vert.qsb frame.vert
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHADERS="$ROOT/src/ui/widgets/shaders"
QSB="${QSB:-/usr/lib/qt6/bin/qsb}"
GLSL_VERSIONS="${GLSL_VERSIONS:-100 es,120,150,330,440}"

[ -x "$QSB" ] || { echo "build_shaders: qsb not found at $QSB (set QSB=...)" >&2; exit 1; }

built=0
for source in "$SHADERS"/*.vert "$SHADERS"/*.frag; do
    [ -e "$source" ] || continue
    "$QSB" --glsl "$GLSL_VERSIONS" -o "$source.qsb" "$source"
    built=$((built + 1))
done
echo "build_shaders: $built shader(s) baked with GLSL $GLSL_VERSIONS"
