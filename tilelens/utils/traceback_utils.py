import functools
import linecache
import os
import sys
from dataclasses import dataclass
from types import FrameType
from collections.abc import Callable

# IDs all tilelens-traced functions to display its code within clients
CODE_KEYS: set[tuple[str, str]] = set()


@dataclass
class TracebackInfo:
    filename: str
    lineno: int
    func_name: str
    line_of_code: str


# ---------------------------------------------------------------------------
# Path resolution cache (avoids repeated stat/readlink syscalls)
# ---------------------------------------------------------------------------

_realpath = functools.lru_cache(maxsize=256)(os.path.realpath)


# ---------------------------------------------------------------------------
# Framework-frame detection (lazy-initialized to avoid circular imports)
# ---------------------------------------------------------------------------

_FRAMEWORK_ROOTS: tuple[str, ...] | None = None


def _get_framework_roots() -> tuple[str, ...]:
    global _FRAMEWORK_ROOTS
    if _FRAMEWORK_ROOTS is None:
        import triton as _triton
        import tilelens as _tilelens

        _FRAMEWORK_ROOTS = tuple(
            {
                _realpath(os.path.dirname(_triton.__file__)),
                _realpath(os.path.dirname(_tilelens.__file__)),
            }
        )
    return _FRAMEWORK_ROOTS


def _is_framework_frame(frame: FrameType) -> bool:
    fn = frame.f_code.co_filename
    if fn.startswith("<"):
        return True
    resolved = _realpath(fn)
    return any(resolved.startswith(root) for root in _get_framework_roots())


# ---------------------------------------------------------------------------
# Truthiness-initiator classification (package-boundary walks)
# ---------------------------------------------------------------------------

# Triton's own frontend does truthiness on tensors as None-guard plumbing
# (e.g. semantic.py's ``if mask and mask.type.is_block():``), which must not
# be confused with value-level control flow like ``if pid == 0:``.
# These walks run on hot paths (every tensor truthiness, every arange), so
# unlike _get_framework_roots they compare raw co_filename prefixes instead of
# realpath-ing each frame; that holds because co_filename and the packages'
# __file__ come from the same import path.
_TILELENS_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_TRITON_FRAME_DIRS: tuple[str, str, frozenset[str]] | None = None


def _triton_frame_dirs() -> tuple[str, str, frozenset[str]]:
    """(triton package dir, triton interpreter file, frontend plumbing
    files), resolved lazily.

    The plumbing files are the frontend's canonicalization layer — the
    modules that in compiled Triton execute at compile time, where
    ``bool(tensor)`` is plain object truthiness. @jit modules that also
    live under the triton package (language/standard.py, language/random.py,
    tools/...) are deliberately NOT in this set: their ``if`` statements
    compile to control flow on the value.
    """
    global _TRITON_FRAME_DIRS
    if _TRITON_FRAME_DIRS is None:
        import triton

        pkg_dir = os.path.dirname(os.path.abspath(triton.__file__)) + os.sep
        _TRITON_FRAME_DIRS = (
            pkg_dir,
            os.path.join(pkg_dir, "runtime", "interpreter.py"),
            frozenset(
                (
                    os.path.join(pkg_dir, "language", "semantic.py"),
                    os.path.join(pkg_dir, "language", "core.py"),
                )
            ),
        )
    return _TRITON_FRAME_DIRS


def innermost_user_site() -> tuple[str, int] | None:
    """(filename, lineno) of the nearest frame outside triton/tilelens.

    Unlike the CODE_KEYS-based extraction below, this is a plain
    package-boundary walk, so it identifies the user call line regardless of
    how the executing kernel's code object was recompiled (a kernel defined
    inside a function executes with qualname ``kernel``, not
    ``outer.<locals>.kernel``, which defeats code-key matching). Used for
    stable per-callsite identities (e.g. arange interning), not for
    user-facing tracebacks.
    """
    triton_pkg_dir, _, _ = _triton_frame_dirs()
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.startswith(_TILELENS_PKG_DIR) or filename.startswith(
            triton_pkg_dir
        ):
            frame = frame.f_back
            continue
        return (filename, frame.f_lineno)
    return None


def truthiness_initiator() -> tuple[str, int, str] | None:
    """``(filename, lineno, func_name)`` of the kernel-level code that
    initiated the in-flight tensor truthiness, or None when Triton's
    frontend plumbing initiated it.

    Walk outward from the caller, skipping tilelens frames (wrapper and
    client mechanics) and triton's interpreter (pure truthiness plumbing:
    ``_get_bool`` and its lambdas sit between any initiator and
    ``__bool__``). The first remaining frame is the initiator. Only the
    frontend's canonicalization modules (see ``_triton_frame_dirs``) count
    as internal: there ``if mask and ...`` None-guards must see "present"
    — compiled Triton runs them at compile time with object truthiness.
    Everything else, INCLUDING @jit code that happens to live under the
    triton package tree, is kernel code whose branches compile to control
    flow on the value.
    """
    _, triton_interpreter_file, plumbing_files = _triton_frame_dirs()
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        filename = code.co_filename
        if (
            filename.startswith(_TILELENS_PKG_DIR)
            or filename == triton_interpreter_file
        ):
            frame = frame.f_back
            continue
        if filename in plumbing_files:
            return None
        return (filename, frame.f_lineno, code.co_name)
    return None


def scalar_truthiness_from_user_code() -> bool:
    """True when the in-flight scalar truthiness/read was initiated by
    kernel-level code rather than the triton frontend's plumbing (see
    ``truthiness_initiator``)."""
    return truthiness_initiator() is not None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def get_code_key(obj: FrameType | Callable) -> tuple[str, str]:
    """
    Given an object that points to a function (a call frame/function itself),
    inspect the function's code and create a key that tells tilelens to display
    the code for displaying in clients (i.e. Code View in visualizer).
    """
    code = obj.f_code if isinstance(obj, FrameType) else obj.__code__
    return (_realpath(code.co_filename), code.co_qualname)


def extract_user_frames(num_frames: int = 0) -> list[TracebackInfo]:
    """
    Walk the call stack from inner frame outward, collecting non-framework
    frames until hitting a CODE_KEYS boundary (the @trace-decorated kernel).

    The boundary frame itself is included. Framework frames (triton, tilelens
    internals) are skipped. ``num_frames`` trims to the innermost *N* frames
    after collection (0 = return all).
    """
    collected: list[TracebackInfo] = []
    frame: FrameType | None = sys._getframe(1)
    while frame:
        if _is_framework_frame(frame):
            frame = frame.f_back
            continue

        code = frame.f_code
        resolved = _realpath(code.co_filename)
        code_key = (resolved, code.co_qualname)
        is_boundary = code_key in CODE_KEYS

        # For num_frames=1 we only need the innermost frame: skip until
        # we are about to hit the boundary, then grab just the previous one.
        if num_frames == 1 and not is_boundary:
            # Keep only the latest non-boundary frame (overwrite previous)
            collected = [
                TracebackInfo(
                    resolved,
                    frame.f_lineno,
                    code.co_qualname,
                    linecache.getline(resolved, frame.f_lineno).rstrip(),
                )
            ]
            frame = frame.f_back
            continue

        lineno = frame.f_lineno
        line_of_code = linecache.getline(resolved, lineno).rstrip()
        collected.append(
            TracebackInfo(resolved, lineno, code.co_qualname, line_of_code)
        )

        if is_boundary:
            break

        frame = frame.f_back

    # collected is inner-first; reverse to outer-first (kernel → helper → op)
    collected.reverse()

    if num_frames and len(collected) > num_frames:
        collected = collected[-num_frames:]

    return collected


# ---------------------------------------------------------------------------
# Sanitizer-oriented helpers (originally in report.py)
# ---------------------------------------------------------------------------


def capture_current_source_location() -> tuple[str, int, str] | None:
    """Return ``(filename, lineno, func_name)`` of the innermost user frame.

    Shared helper for symbolic clients that need to pin a deferred check /
    recorded event back to its user-code origin. Returns ``None`` if no
    non-framework frame is available.
    """
    frames = extract_user_frames(num_frames=1)
    frame = frames[-1] if frames else None
    if frame is None:
        return None
    return (frame.filename, frame.lineno, frame.func_name)


def location_to_traceback_info(
    source_location: tuple[str, int, str],
) -> TracebackInfo:
    """Convert a ``(filename, lineno, func_name)`` tuple to ``TracebackInfo``."""
    filename, lineno, func_name = source_location
    line_of_code = linecache.getline(filename, lineno).rstrip()
    return TracebackInfo(
        filename=filename,
        lineno=lineno,
        func_name=func_name,
        line_of_code=line_of_code,
    )


# ---------------------------------------------------------------------------
# Source code reading utilities (originally in interface.py)
# ---------------------------------------------------------------------------


def read_source_segment(
    filename: str,
    lineno: int,
    context: int = 8,
) -> dict | None:
    """Read a segment of a source file around *lineno*.

    Returns a dict with ``filename``, ``lineno``, ``start``, ``end``,
    ``highlight``, and ``lines`` keys, or ``None`` on failure.
    """
    try:
        path = _realpath(filename)
        start = max(1, lineno - context)
        end = lineno + context
        lines: list[dict] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if i > end:
                    break
                lines.append({"no": i, "text": line.rstrip("\n")})
        return {
            "filename": path,
            "lineno": lineno,
            "start": start,
            "end": end,
            "highlight": lineno,
            "lines": lines,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Statement extraction (originally in clients/utils.py)
# ---------------------------------------------------------------------------


def extract_complete_statement_from_line(filename: str, lineno: int | None) -> str:
    """Extract the complete Python statement containing the specified line.

    Handles multi-line statements (e.g. multi-line function calls) by tracking
    bracket balance.  Returns the statement with excess indentation removed, or
    an empty string on failure.
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        if lineno is None or lineno < 1 or lineno > len(lines):
            return ""

        target_idx = lineno - 1
        start_idx = target_idx

        if target_idx > 0:
            prev_line = lines[target_idx - 1].rstrip()
            prev_stripped = prev_line.strip()

            paren_count = prev_line.count("(") - prev_line.count(")")
            bracket_count = prev_line.count("[") - prev_line.count("]")
            brace_count = prev_line.count("{") - prev_line.count("}")

            if (
                paren_count == 0
                and bracket_count == 0
                and brace_count == 0
                and prev_stripped
                and not prev_stripped.endswith(("\\", ",", "(", "[", "{"))
            ):
                start_idx = target_idx
            else:
                paren_depth = 0
                bracket_depth = 0
                brace_depth = 0

                for i in range(target_idx - 1, -1, -1):
                    line = lines[i]
                    for char in reversed(line):
                        if char == ")":
                            paren_depth += 1
                        elif char == "(":
                            paren_depth -= 1
                        elif char == "]":
                            bracket_depth += 1
                        elif char == "[":
                            bracket_depth -= 1
                        elif char == "}":
                            brace_depth += 1
                        elif char == "{":
                            brace_depth -= 1

                    if paren_depth < 0 or bracket_depth < 0 or brace_depth < 0:
                        start_idx = i
                    elif paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                        stripped = line.rstrip()
                        if not stripped.endswith(("\\", ",", "(", "[", "{")):
                            start_idx = i + 1
                            break

                    if i == 0:
                        start_idx = 0

        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        statement_lines: list[str] = []

        i = start_idx
        while i < len(lines):
            line = lines[i].rstrip()
            statement_lines.append(line)

            for char in line:
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1

            if (
                i >= target_idx
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                stripped = line.strip()
                if not stripped.endswith((",", "\\")):
                    break

            i += 1
            if i - start_idx > 20:
                break

        full_statement = "\n".join(statement_lines)
        if statement_lines:
            non_empty = [sl for sl in statement_lines if sl.strip()]
            if non_empty:
                min_indent = min(len(sl) - len(sl.lstrip()) for sl in non_empty)
                full_statement = "\n".join(
                    sl[min_indent:] if len(sl) > min_indent else sl
                    for sl in statement_lines
                )

        return full_statement.strip()
    except Exception:
        return ""
