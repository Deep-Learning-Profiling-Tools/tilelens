"""The program-id and grid variables every symbolic client names alike.

One place defines how a program id and a grid size along an axis are
spelled as Z3 variables, so the eager engine (``SymbolicExpr.PID0-2``), the
compiled sanitizer and the race detector's compiled leaves speak one
vocabulary. Only the spelling is shared, not the context: ``pid_var`` and
``grid_var`` take the client's ``ctx`` (None: Z3's main context, where the
race detector's solver renames pids per program copy). How a client renames
or bounds these variables is its own policy.
"""

from __future__ import annotations

from z3 import ArithRef, Context, Int

PID_AXES = (0, 1, 2)


def pid_name(axis: int) -> str:
    """The name of the program id along ``axis``."""
    return f"pid_{axis}"


def pid_var(axis: int, ctx: Context | None = None) -> ArithRef:
    """The program id along ``axis`` as a Z3 Int in ``ctx``."""
    return Int(pid_name(axis), ctx)


def grid_name(axis: int) -> str:
    """The name of the grid size along ``axis``."""
    return f"grid_{axis}"


def grid_var(axis: int, ctx: Context | None = None) -> ArithRef:
    """The grid size along ``axis`` as a Z3 Int in ``ctx``."""
    return Int(grid_name(axis), ctx)


def __getattr__(name: str) -> ArithRef:
    """``PID0``, ``PID1``, ``PID2``: the program ids in Z3's main context,
    made on first use (importing this module creates no Z3 term, so a
    client in its own contexts sees Z3 exactly as before)."""
    if name in ("PID0", "PID1", "PID2"):
        value = globals()[name] = pid_var(int(name[-1]))
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
