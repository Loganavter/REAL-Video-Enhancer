#!/usr/bin/env bash
# Launch THIS tree: our front end and our backend, on the portable runtime's interpreter.
#
# Why a script and not just `python3 REAL-Video-Enhancer.py`: the front end needs PySide6 and the backend needs the
# bundled interpreter (torch, TensorRT, the models). Those live in the portable runtime, not in this repository, and
# `src/constants.py` finds the backend next to this launcher — so the three things the tree needs (python/, bin/,
# custom_models, plus the generated resources_rc.py) are linked or generated here, once, instead of
# being remembered.
#
#   RVE_RUNTIME=/path/to/portable/app ./run.sh          # a different runtime
#   RVE_FRONT_PYTHON=/path/to/python ./run.sh          # a different front-end interpreter
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
RUNTIME=${RVE_RUNTIME:-/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2}
FRONT=${RVE_FRONT_PYTHON:-"$RUNTIME/venv/bin/python"}

if [ ! -x "$FRONT" ]; then
    echo "run.sh: no front-end interpreter at $FRONT" >&2
    echo "        fix: RVE_RUNTIME=/path/to/portable/app ./run.sh  (the app's own venv lives in <runtime>/venv)" >&2
    exit 1
fi

for part in python bin custom_models; do
    if [ ! -e "$ROOT/$part" ]; then
        if [ ! -e "$RUNTIME/$part" ]; then
            echo "run.sh: $RUNTIME has no $part/ - that directory is not a portable runtime" >&2
            exit 1
        fi
        ln -s "$RUNTIME/$part" "$ROOT/$part"
        echo "run.sh: linked $part -> $RUNTIME/$part"
    fi
done

# generated assets: only the icons/resources are compiled now. The interface used to be compiled from
# `testRVEInterface.ui` here; since W3 of plan-rve-ui-on-toolkit it is code (`src/ui/pages`, `src/ui/shell`).
if [ ! -f "$ROOT/resources_rc.py" ] && [ -x "$RUNTIME/venv/bin/pyside6-rcc" ]; then
    "$RUNTIME/venv/bin/pyside6-rcc" -g python "$ROOT/resources.qrc" > "$ROOT/resources_rc.py"
    echo "run.sh: generated resources_rc.py"
fi

cd "$ROOT"
exec "$FRONT" REAL-Video-Enhancer.py "$@"
