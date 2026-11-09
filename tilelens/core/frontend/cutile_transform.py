"""Rewrite cuTile kernel and helper for-loops to report loop iterations."""

import ast
import inspect
import os
from pathlib import Path
import sysconfig
import textwrap
import types
from collections.abc import Callable
from typing import Any, cast
from weakref import WeakKeyDictionary

LOOP_WRAPPER_ARG = "_tilelens_loop_iter_wrapper"
_FACTORY_NAME = "_tilelens_loop_factory"


class CuTileLoopTransformer(ast.NodeTransformer):
    """
    AST transformer for cuTile loop tracing, matching Triton's loop rewrite.

    Transformations:
    - `for x in f(*args, **kwargs)` -> `for x in wrapper(f, args, kwargs, lineno, range_type)`
    - `for x in expr` -> `for x in wrapper(lambda: expr, (), {}, lineno, range_type)`
    - adds the wrapper as a hidden keyword-only argument of every function
    """

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)

        # Preserve which spelling the user wrote so clients can tell loop kinds apart.
        range_type = "unknown"
        if isinstance(node.iter, ast.Call):
            func = node.iter.func
            if isinstance(func, ast.Name) and func.id == "range":
                range_type = "python_range"

        # Pass the callable and its arguments separately so range-wrapper
        # clients can replace the iterable before it is created.
        if isinstance(node.iter, ast.Call):
            iter_callable = node.iter.func
            iter_args = ast.Tuple(elts=node.iter.args, ctx=ast.Load())
            kw_keys: list[ast.expr | None] = []
            kw_vals: list[ast.expr] = []
            for kw in node.iter.keywords:
                if kw.arg is None:
                    continue
                kw_keys.append(ast.Constant(value=kw.arg))
                kw_vals.append(kw.value)
            iter_kwargs = ast.Dict(keys=kw_keys, values=kw_vals)
        else:
            iter_callable = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=node.iter,
            )
            iter_args = ast.Tuple(elts=[], ctx=ast.Load())
            iter_kwargs = ast.Dict(keys=[], values=[])

        loop = ast.For(
            target=node.target,
            iter=ast.Call(
                func=ast.Name(id=LOOP_WRAPPER_ARG, ctx=ast.Load()),
                args=[
                    iter_callable,
                    iter_args,
                    iter_kwargs,
                    ast.Constant(value=node.lineno),
                    ast.Constant(value=range_type),
                ],
                keywords=[],
            ),
            body=node.body,
            orelse=node.orelse,
            type_comment=node.type_comment,
        )
        return ast.fix_missing_locations(ast.copy_location(loop, node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = cast(ast.FunctionDef, self.generic_visit(node))
        # A defaulted keyword-only argument lets rewritten loops call the
        # wrapper without adding a name to the user's module globals.
        if LOOP_WRAPPER_ARG in {
            arg.arg
            for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
        }:
            return node
        node.args.kwonlyargs.append(ast.arg(arg=LOOP_WRAPPER_ARG))
        node.args.kw_defaults.append(ast.Name(id=LOOP_WRAPPER_ARG, ctx=ast.Load()))
        return ast.fix_missing_locations(node)


# Like Triton rewriting only @jit functions, leave library code untouched.
_LIBRARY_DIRS = tuple(
    os.path.join(os.path.realpath(path), "")
    for key in ("stdlib", "platstdlib", "purelib", "platlib")
    if (path := sysconfig.get_paths().get(key))
) + (os.path.join(str(Path(__file__).resolve().parents[2]), ""),)


def _is_library_code(fn: types.FunctionType) -> bool:
    return os.path.realpath(fn.__code__.co_filename).startswith(_LIBRARY_DIRS)


# Rewritten functions keyed by original function; None marks functions that
# are left unchanged (no for-loops or no available source).
_REWRITTEN: WeakKeyDictionary[Callable, Callable | None] = WeakKeyDictionary()


def _function_node(fn: types.FunctionType) -> ast.FunctionDef | None:
    if fn.__code__.co_name == "<lambda>":
        return None
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    node = tree.body[0]
    if node.name != fn.__name__:
        return None
    if not any(isinstance(child, ast.For) for child in ast.walk(node)):
        return None
    # Line 1 of the source is co_firstlineno, the first decorator if any.
    # Keep the original file line numbers in reports and tracebacks.
    ast.increment_lineno(node, fn.__code__.co_firstlineno - 1)
    return node


def _compile_rewritten(
    fn: types.FunctionType, wrapper: Callable
) -> types.FunctionType | None:
    node = _function_node(fn)
    if node is None:
        return None
    node = cast(ast.FunctionDef, CuTileLoopTransformer().visit(node))

    # Compile inside a factory whose parameters are the original free
    # variables, so closure references stay closure references.
    freevars = fn.__code__.co_freevars
    factory = ast.FunctionDef(
        name=_FACTORY_NAME,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in freevars],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[node, ast.Return(value=ast.Name(id=node.name, ctx=ast.Load()))],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.Module(body=[ast.copy_location(factory, node)], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        module_code = compile(module, fn.__code__.co_filename, "exec")
    except SyntaxError:
        return None

    factory_code = next(
        const
        for const in module_code.co_consts
        if isinstance(const, types.CodeType) and const.co_name == _FACTORY_NAME
    )
    code = next(
        (
            const
            for const in factory_code.co_consts
            if isinstance(const, types.CodeType) and const.co_name == fn.__name__
        ),
        None,
    )
    if code is None or set(code.co_freevars) - set(freevars):
        return None
    # The qualified name identifies user frames in tracebacks and the code view.
    code = code.replace(co_qualname=fn.__code__.co_qualname)
    cells = dict(zip(freevars, fn.__closure__ or ()))
    rewritten = types.FunctionType(
        code,
        fn.__globals__,
        fn.__name__,
        fn.__defaults__,
        tuple(cells[name] for name in code.co_freevars) or None,
    )
    rewritten.__kwdefaults__ = {**(fn.__kwdefaults__ or {}), LOOP_WRAPPER_ARG: wrapper}
    rewritten.__qualname__ = fn.__qualname__
    rewritten.__module__ = fn.__module__
    rewritten.__doc__ = fn.__doc__
    rewritten.__dict__.update(fn.__dict__)
    return rewritten


def rewrite_loops(fn: Any, wrapper: Callable) -> Any:
    """Return fn with for-loops routed through wrapper, or fn unchanged."""
    if not isinstance(fn, types.FunctionType) or _is_library_code(fn):
        return fn
    if fn in _REWRITTEN:
        return _REWRITTEN[fn] or fn
    rewritten = _compile_rewritten(fn, wrapper)
    _REWRITTEN[fn] = rewritten
    return rewritten or fn
