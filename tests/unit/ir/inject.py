"""Feed a chosen TTIR text to a traced launch's IR clients.

``inject_ttir(traced, text_or_fn)`` wraps the trace's host compiler (the
ClientManager's ``compiler``): each call is still bound, specialized,
compiled and delivered by the product (bind errors, dedup, the target and
Launch.grid included); only the compiled kernel's ``asm["ttir"]`` is
replaced, by ``text_or_fn`` or by ``text_or_fn(product_ttir)``. The
specialization becomes a hash of the product's and the injected text, so
two injected texts never share one.

The injected function must keep every argument the product's has, with the
same types (it may add arguments the launch binds); a mismatch raises
AssertionError when the context exits (an error raised inside the compile
would be delivered as a compile failure instead).

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

from tilelens.ir.capture import content_key

_FUNC = re.compile(r"tt\.func\s+(?:\w+\s+)?@(\w+)\((.*?)\)\s*(?:->|attributes|\{)")
_ARG = re.compile(r"%([\w.$-]+)\s*:\s*([^\s,)]+)")


def func_args(text: str) -> tuple[str, dict[str, str]]:
    """The kernel's name and its arguments' types, by name."""
    match = _FUNC.search(text)
    assert match is not None, "no tt.func in the TTIR text"
    return match.group(1), dict(_ARG.findall(match.group(2)))


@contextmanager
def inject_ttir(traced, text_or_fn: str | Callable[[str], str]) -> Iterator[None]:
    compiler = traced.client_manager.compiler
    compile_ = compiler.compile
    errors: list[str] = []

    def compile(jit_fn, args, kwargs, *, target, **options):
        kernel = compile_(jit_fn, args, kwargs, target=target, **options)
        product = kernel.asm["ttir"]
        text = text_or_fn(product) if callable(text_or_fn) else text_or_fn
        name, wanted = func_args(product)
        got_name, got = func_args(text)
        if got_name != name:
            errors.append(f"the injected function is @{got_name}, not @{name}")
        for arg, ty in wanted.items():
            if got.get(arg) != ty:
                errors.append(f"argument %{arg}: {ty} injected as {got.get(arg)}")
        asm = dict(kernel.asm, ttir=text)
        return replace(
            kernel,
            hash=content_key(f"{kernel.hash}\0{text}"),
            asm=MappingProxyType(asm),
        )

    compiler.compile = compile
    try:
        yield
    finally:
        del compiler.compile
    assert not errors, "; ".join(errors)
