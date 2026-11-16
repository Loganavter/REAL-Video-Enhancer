"""Static checks for the two bugs that cost the most and are not measurements at all.

``dead_stores``
    The one that caused the false refusals: ``buf = quantise(pre_relu_head)`` followed later by
    ``buf = something_else`` in the same straight-line block. Both variants then measured identical
    values, a correct fused kernel was rejected, and the rejection was never re-checked.

``used_before_assignment``
    ``c2`` read before it was assigned -> NameError -> the probe reported "failed" and that was read
    as a verdict about fp16. A three-state probe catches it at runtime; this catches it before the
    GPU run, which is where the budget is.

Both are conservative on purpose: a warning you learn to ignore is worse than no warning. Anything
marked ``# noqa: dead-store`` or ``# noqa: used-before-assignment`` is skipped.

Scope is one function's own straight-line code. Reads inside a nested def/lambda are *closure* captures
that execute at call time, so they are excluded - two of the repo's probes fill a dict after defining the
function that reads it, which is correct Python and would otherwise be reported on every run.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NOQA = re_noqa = ("# noqa: dead-store", "# noqa: used-before-assignment", "# noqa")


@dataclass(frozen=True)
class Finding:
    kind: str
    name: str
    first_line: int
    second_line: int
    file: str = ""

    def __str__(self) -> str:
        where = f"{self.file}:" if self.file else ""
        return f"{where}{self.second_line}: {self.kind}: '{self.name}' (first at line {self.first_line})"


def _simple_stores(stmt: ast.stmt) -> list[str]:
    out: list[str] = []
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            if isinstance(t, ast.Name):
                out.append(t.id)
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
        out.append(stmt.target.id)
    return out


def _loads(stmt: ast.stmt) -> set[str]:
    """Loads in this statement. An augmented assignment (`x += 1`) reads its target, so it counts as a
    load - otherwise `x = 1; x += 1` looks like a dead store followed by an invisible read."""
    out = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        out.add(stmt.target.id)
    return out


class _NestedScope(ast.NodeVisitor):
    """Names bound inside nested functions/lambdas: their parameters and their locals."""

    def __init__(self) -> None:
        self.bound: set[str] = set()      # names the nested scopes bind
        self.nodes: set[int] = set()      # every node inside a nested scope

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._bind(node)

    def _bind(self, node: ast.AST) -> None:
        for a in ast.walk(node):
            self.nodes.add(id(a))
            if isinstance(a, ast.arg):
                self.bound.add(a.arg)
            elif isinstance(a, ast.Name) and isinstance(a.ctx, ast.Store):
                self.bound.add(a.id)


def _noqa_lines(src: str) -> set[int]:
    """Only real COMMENT tokens count: the marker inside a docstring must not whitelist a line."""
    out: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and any(m in tok.string for m in NOQA):
                out.add(tok.start[0])
    except (tokenize.TokenError, IndentationError):
        return {i for i, line in enumerate(src.splitlines(), 1) if any(tok in line for tok in NOQA)}
    return out


def _blocks(tree: ast.AST):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            yield body


def dead_stores(source: str, filename: str = "") -> list[Finding]:
    tree = ast.parse(source)
    skip = _noqa_lines(source)
    findings: list[Finding] = []
    for body in _blocks(tree):
        pending: dict[str, int] = {}
        for stmt in body:
            loads = _loads(stmt)
            for name in list(pending):
                if name in loads:                 # the previous value was used: not a dead store
                    pending.pop(name)
            for name in _simple_stores(stmt):
                if name in pending and stmt.lineno not in skip:
                    findings.append(Finding("dead store", name, pending[name], stmt.lineno, filename))
                pending[name] = stmt.lineno
    return sorted(findings, key=lambda f: f.second_line)


def used_before_assignment(source: str, filename: str = "") -> list[Finding]:
    tree = ast.parse(source)
    skip = _noqa_lines(source)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
        declared = {n for s in ast.walk(node) if isinstance(s, (ast.Global, ast.Nonlocal))
                    for n in s.names}
        # Loop and comprehension variables are bound where they are used; they are not "assigned
        # later". Excluding them is what keeps the c2 case (a plain assignment) detectable.
        loop_bound = {n.id for f in ast.walk(node) if isinstance(f, (ast.For, ast.AsyncFor, ast.comprehension))
                      for n in ast.walk(f.target) if isinstance(n, ast.Name)}
        loads: dict[str, int] = {}
        stores: dict[str, int] = {}
        # A name bound by a nested def/lambda is not this function's assignment. Without this,
        # `lambda _m, _i, out: ...` inside a function that later assigns `out` reads as a
        # use-before-assignment, and the tool cries wolf on 7 of the repo's own probes.
        nested = _NestedScope()
        for stmt in node.body:            # the node itself must NOT be bound: that would silence it
            nested.visit(stmt)
        for n in ast.walk(node):
            if not isinstance(n, ast.Name) or n.id in nested.bound or id(n) in nested.nodes:
                continue
            if isinstance(n.ctx, ast.Load):
                loads[n.id] = min(loads.get(n.id, n.lineno), n.lineno)
            elif isinstance(n.ctx, ast.Store):
                stores[n.id] = min(stores.get(n.id, n.lineno), n.lineno)
        for name, load_line in loads.items():
            if name in params or name in declared or name in loop_bound:
                continue
            store_line = stores.get(name)
            if store_line is not None and load_line < store_line and load_line not in skip:
                findings.append(Finding("used before assignment", name, load_line, store_line, filename))
    return sorted(findings, key=lambda f: f.first_line)


def check_paths(paths: Iterable[str]) -> list[Finding]:
    out: list[Finding] = []
    for p in paths:
        src = Path(p).read_text()
        out += dead_stores(src, p)
        out += used_before_assignment(src, p)
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m harness.astcheck <file.py> [...]")
        return 2
    findings = check_paths(args)
    for f in findings:
        print(f)
    if not findings:
        print(f"{len(args)} file(s): no dead stores, no use-before-assignment")
    print("NOT checked: whether the used value is the RIGHT one; control flow across branches; "
          "closure captures; aliasing through containers")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
