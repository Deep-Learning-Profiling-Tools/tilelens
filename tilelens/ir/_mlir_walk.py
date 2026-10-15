"""Walk a printed TTIR module: MLIR bindings first, the text for lines and locs.

Private to ``tilelens.ir``; only ``ttir_reader.py`` imports it. This is the one
module that touches Triton's private MLIR bindings (``triton._C.libtriton.ir``),
and every lifetime hazard of those bindings stays here.

``walk_module(text)`` reads the same bytes twice:

1. The bindings parse the text (``ir.parse_mlir_module``, TTIR dialects only)
   and give the module: op names, operand / result values, full type
   strings, region / block nesting, block arguments, value locs, and the
   attributes the reader reads (``_ATTRS``: ``get_int_attr`` for the integer
   and enum attributes, ``get_constant_value`` for integer constants, the
   string / bool / symbol getters for the rest).
2. A line scan of the text rebuilds the op tree for what the bindings cannot
   reach: each op's line, the loc of an op without results (the bindings read
   locs off values only), a region op's closing line, block labels,
   ``scf.for``'s ``unsigned`` keyword (a unit attribute no getter reads) and
   a generic-form ``tt.load``'s operand segments.

The two trees are zipped in pre-order and must agree on every op: name,
result / region / block / op counts, and the printed loc of every op with
results equals the loc the bindings report for it (the anchor that ties
each text line to its op). A module that prints locs must
print one on every op; an op line holding a comment, and an attribute alias
other than ``#loc`` (TTGIR), are refused. The only tolerated differences
are the printer's own elisions: a trailing zero-operand ``scf.yield`` of an
``scf.for`` / ``scf.if`` block (also when that yield is the block's only op
and the region prints as ``{ }``) and unprinted empty trailing regions.
Anything else raises ``MisalignedModule``; a text the MLIR parser rejects
raises ``ModuleParseError`` with the parser's own diagnostic.

The result is frozen pure-Python data: ``Module`` holds ``Op`` / ``Block`` /
``Value`` / ``Func`` records, which compare by value, hash, pickle and
deep-copy. Values are dense local ints (indices into ``Module.values``),
never the bindings' ``value.id()`` pointers. No binding object ever leaves
``_bind_walk``.

Binding lifetime (dropping the context first crashes the process): the
context is pinned on the module (``mod.context = ctx``), everything is
extracted inside one function, the module's body block is erased, the module
is dropped before the context, and nothing derived from either is returned.
No binding erases the module op itself, so each parse still leaks that
(empty) op; results are cached by sha256 of the text, so each text leaks
once. The parse window redirects the process's fd 2 under a lock; a forked
child gets both back. A failed assertion inside the bindings aborts the
process too (Triton's LLVM keeps assertions): a block-pointer type never
reaches the parser (``_screen``), an integer constant wider than 64 bits is
never handed to ``get_constant_value`` (``_constant_value``), and
``get_int_attr`` is asked only for attributes whose signless integer type
the op's verifier enforces.

The getters and the enum numbering are Triton 3.8's; on any other Triton
release ``walk_module`` refuses (tilelens.core.config.IR_TRITON_RELEASE).
"""

from __future__ import annotations

import collections
import hashlib
import os
import re
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

# ─────────────────────────── records ───────────────────────────


class _FrozenMap(Mapping[str, Any]):
    """The read-only mapping behind the records' mapping fields. Unlike
    ``MappingProxyType`` it hashes (its values are plain hashable data) and
    pickles, so the frozen records hash, pickle and deep-copy too."""

    __slots__ = ("_d",)

    def __init__(self, items: Mapping[str, Any]) -> None:
        self._d = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __hash__(self) -> int:
        return hash(frozenset(self._d.items()))

    def __repr__(self) -> str:
        return repr(self._d)

    def __reduce__(self) -> tuple[Any, ...]:
        return (_FrozenMap, (self._d,))


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int


@dataclass(frozen=True)
class Value:
    index: int  # position in Module.values
    type: str  # full type text: "tensor<64x!tt.ptr<f32>>", "i32", ...
    op: int | None  # defining op index (an op result), else None
    block: int | None  # owning block index (a block argument), else None
    position: int  # result number, or argument number
    # NameLoc name of the value's own loc (the Python variable / parameter
    # name), None when unnamed. Printed SSA names are never exposed: the
    # printer sanitises and uniquifies them.
    name: str | None


@dataclass(frozen=True)
class Block:
    index: int  # position in Module.blocks
    op: int  # owning op index
    region: int  # region number within the owning op
    position: int  # block number within the region
    label: str | None  # printed label ("^bb1"); None for an unlabeled entry block
    args: tuple[int, ...]  # value indices
    arg_types: tuple[str, ...]
    arg_names: tuple[str | None, ...]  # NameLoc names (see Value.name)
    ops: tuple[int, ...]  # op indices in program order


@dataclass(frozen=True)
class Op:
    index: int  # pre-order position in Module.ops (the text order)
    name: str  # "tt.load", "scf.for", "builtin.module", ...
    operands: tuple[int, ...]  # value indices, in ODS operand order
    operand_types: tuple[str, ...]
    results: tuple[int, ...]  # value indices
    result_types: tuple[str, ...]
    # What the reader reads (read-only, see _ATTRS): integers are ints, enums
    # their printed keyword ("slt"), program-id axes 0-2, a barrier's
    # address spaces their bits;
    # arith.constant "value" is an int (sign-extended) / bool for i1, else
    # ("float", None) / ("dense", None); scf.for "unsignedCmp" is a bool.
    attrs: Mapping[str, Any]
    regions: tuple[tuple[int, ...], ...]  # block indices, per walked region
    # block indices from the module body down to the parent block; () for
    # the module op itself
    path: tuple[int, ...]
    position: int  # position within the parent block
    line_no: int | None  # header line (1-based); None for an elided terminator
    end_line: int | None  # closing line of a region op, else == line_no
    loc: SourceLoc | None  # the op's own site (callee frame of a callsite loc)
    callers: tuple[SourceLoc, ...]  # callsite chain, innermost caller first
    loc_name: str | None  # NameLoc label of the op's loc, if any
    implicit: bool = False  # a terminator the printer elided

    @property
    def block(self) -> int | None:
        return self.path[-1] if self.path else None


@dataclass(frozen=True)
class FuncArg:
    index: int
    value: int  # value index of the entry-block argument
    type: str
    name: str | None  # NameLoc name: the Python parameter name


@dataclass(frozen=True)
class Func:
    op: int  # the tt.func op index
    sym_name: str
    visibility: str
    args: tuple[FuncArg, ...]  # () for a body-less declaration


@dataclass(frozen=True)
class Module:
    ops: tuple[Op, ...]
    blocks: tuple[Block, ...]
    values: tuple[Value, ...]
    funcs: tuple[Func, ...]  # tt.func ops in text order
    stats: Mapping[str, int]  # what the alignment checked (tests)


class MisalignedModule(Exception):
    """The text scan and the bindings disagree (or the text holds a construct
    the text layer cannot read faithfully). ``problems`` lists every mismatch
    found, ``line_no`` is the first text line involved (None if unknown)."""

    def __init__(self, problems: Sequence[str], line_no: int | None = None) -> None:
        self.problems = tuple(problems) or ("misaligned module",)
        self.line_no = line_no
        super().__init__(self.problems[0])


class ModuleParseError(Exception):
    """The MLIR parser rejected the text (``diagnostic`` is its own message,
    with the temporary parse path replaced by ``<ttir>``), the parse input
    could not be created, the text holds a construct the parser cannot be
    handed (a block-pointer type, see _screen), or the installed Triton is
    not the release the walk reads."""

    def __init__(self, diagnostic: str, line_no: int | None = None) -> None:
        self.diagnostic = diagnostic
        self.line_no = line_no
        super().__init__(diagnostic)


# ─────────────────────────── types ───────────────────────────


@dataclass(frozen=True)
class TypeInfo:
    """A printed TTIR type split into shape and element, with its bit widths."""

    text: str
    shape: tuple[int, ...]  # () for scalars
    elem: str  # element type text: "i32", "f16", "!tt.ptr<f32>", ...
    int_bits: int | None  # iN element -> N (signless); index -> 64
    float_bits: int | None
    pointee: str | None  # element pointer -> pointee type text
    pointee_bits: int | None


_FLOAT_BITS = {
    "f64": 64, "f32": 32, "f16": 16, "bf16": 16, "tf32": 32,
    "f8E4M3FN": 8, "f8E5M2": 8, "f8E4M3FNUZ": 8, "f8E5M2FNUZ": 8,
    "f8E4M3B11FNUZ": 8, "f8E8M0FNU": 8, "f4E2M1FN": 4,
}  # fmt: skip
_RE_TENSOR = re.compile(r"^tensor<((?:\d+x)*)(.*)>$")
_RE_INT = re.compile(r"^i(\d+)$")
_RE_PTR = re.compile(r"^!tt\.ptr<(.*?)(?:, \d+)?>$")


def _scalar_bits(t: str) -> int | None:
    m = _RE_INT.match(t)
    if m:
        return int(m.group(1))
    return _FLOAT_BITS.get(t)


def parse_type(text: str) -> TypeInfo:
    shape: tuple[int, ...] = ()
    elem = text
    m = _RE_TENSOR.match(text)
    if m:
        shape = tuple(int(d) for d in m.group(1).split("x") if d)
        elem = m.group(2)
    im = _RE_INT.match(elem)
    pm = _RE_PTR.match(elem)
    pointee = pm.group(1) if pm else None
    int_bits = int(im.group(1)) if im else (64 if elem == "index" else None)
    return TypeInfo(
        text=text,
        shape=shape,
        elem=elem,
        int_bits=int_bits,
        float_bits=_FLOAT_BITS.get(elem),
        pointee=pointee,
        pointee_bits=_scalar_bits(pointee) if pointee else None,
    )


# ─────────────────────────── strings and brackets ───────────────────────────


def _string_end(s: str, start: int) -> int:
    """Index of the quote closing the string literal that opens at ``start``."""
    i = start + 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i
        i += 1
    raise ValueError("unterminated string literal")


_HEX = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09}


def _unescape(body: str) -> str:
    """Decode an MLIR string literal body. The printer escapes every
    non-printable or non-ASCII byte as ``\\XX``; the escapes of one character
    are the bytes of its UTF-8 encoding, so collect bytes and decode once."""
    out = bytearray()
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out += c.encode("utf-8")
            i += 1
            continue
        nxt = body[i + 1 : i + 2]
        if nxt in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[nxt])
            i += 2
        elif len(body) >= i + 3 and body[i + 1] in _HEX and body[i + 2] in _HEX:
            out.append(int(body[i + 1 : i + 3], 16))
            i += 3
        else:
            raise ValueError(f"bad string escape {body[i : i + 3]!r}")
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"string literal is not UTF-8: {e}") from None


def _mask(line: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Replace every string literal's body by '_' (same length, so indices into
    the masked line index the raw line too). Returns the masked line and the
    literals as (open quote index, close quote index, decoded text)."""
    out: list[str] = []
    lits: list[tuple[int, int, str]] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            j = _string_end(line, i)
            lits.append((i, j, _unescape(line[i + 1 : j])))
            out.append('"' + "_" * (j - i - 1) + '"')
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out), lits


_OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}


def _match_close(s: str, i: int) -> int:
    """Index of the bracket closing ``s[i]`` in masked text ('->' is no '>')."""
    stack = [_OPEN[s[i]]]
    j = i + 1
    while j < len(s):
        c = s[j]
        if c in _OPEN:
            stack.append(_OPEN[c])
        elif c == ">" and s[j - 1] == "-":
            pass
        elif c in ")]}>":
            if c != stack[-1]:
                raise ValueError(f"bracket mismatch at column {j + 1}")
            stack.pop()
            if not stack:
                return j
        j += 1
    raise ValueError("unclosed bracket")


def _trailing_loc(masked: str) -> tuple[int, int] | None:
    """(start, end) of the ``loc(...)`` that ends the masked text, if any."""
    s = masked.rstrip()
    if not s.endswith(")"):
        return None
    depth = 0
    for j in range(len(s) - 1, -1, -1):
        c = s[j]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
            if depth == 0:
                if s[max(0, j - 3) : j] == "loc" and (
                    j < 4 or not (s[j - 4].isalnum() or s[j - 4] in "_.$")
                ):
                    return j - 3, len(s)
                return None
    return None


# ─────────────────────────── locs ───────────────────────────
# One grammar for both sources: the text's `loc(#locN)` trailers resolved
# through the `#locN = loc(...)` table, and the bindings' fully inlined
# `str(value.get_loc())`. Both normalise to nested tuples compared with ==.

_RE_LOC_ALIAS_DEF = re.compile(r"^(#loc\d*)\s*=\s*loc\((.*)\)\s*$")
_RE_LOC_ALIAS = re.compile(r"#loc\d*")
_RE_FILE_POS = re.compile(r":(\d+):(\d+)")
_LOC_DEPTH_LIMIT = 128


class _LocParser:
    def __init__(self, aliases: Mapping[str, str], parse_path: str) -> None:
        self._aliases = aliases  # "#loc12" -> inner text of loc(...)
        self._parse_path = parse_path
        self._alias_memo: dict[str, tuple] = {}
        self._text_memo: dict[str, tuple | None] = {}

    def parse(self, loc_text: str) -> tuple | None:
        """Normalised tree of a ``loc(...)`` text; None for a loc the parser
        invented (it names the temporary parse path: the text printed none)."""
        got = self._text_memo.get(loc_text, _MISSING)
        if got is not _MISSING:
            return got  # type: ignore[return-value]
        s = loc_text.strip()
        if not (s.startswith("loc(") and s.endswith(")")):
            raise ValueError(f"not a loc: {loc_text[:80]!r}")
        tree: tuple | None = self._full(s[4:-1], 0)
        if _names_path(tree, self._parse_path):  # type: ignore[arg-type]
            tree = None
        self._text_memo[loc_text] = tree
        return tree

    def _full(self, s: str, depth: int) -> tuple:
        tree, rest = self._expr(s.strip(), depth)
        if rest.strip():
            raise ValueError(f"trailing loc text: {rest[:60]!r}")
        return tree

    def _expr(self, s: str, depth: int) -> tuple[tuple, str]:
        if depth > _LOC_DEPTH_LIMIT:
            raise ValueError("loc nesting too deep")
        if s.startswith("#loc"):
            m = _RE_LOC_ALIAS.match(s)
            assert m is not None
            name = m.group(0)
            if name not in self._alias_memo:
                if name not in self._aliases:
                    raise ValueError(f"undefined loc alias {name}")
                self._alias_memo[name] = ("pending",)
                self._alias_memo[name] = self._full(self._aliases[name], depth + 1)
            elif self._alias_memo[name] == ("pending",):
                raise ValueError(f"cyclic loc alias {name}")
            return self._alias_memo[name], s[m.end() :]
        if s.startswith("unknown"):
            return ("unknown",), s[len("unknown") :]
        if s.startswith("callsite("):
            callee, rest = self._expr(s[len("callsite(") :].lstrip(), depth + 1)
            rest = rest.lstrip()
            if not rest.startswith("at "):
                raise ValueError("callsite loc without 'at'")
            caller, rest = self._expr(rest[3:].lstrip(), depth + 1)
            rest = rest.lstrip()
            if not rest.startswith(")"):
                raise ValueError("unclosed callsite loc")
            return ("callsite", callee, caller), rest[1:]
        if s.startswith("fused"):
            rest = s[len("fused") :]
            meta: str | None = None
            if rest.startswith("<"):
                close = _match_close(_mask(rest)[0], 0)
                meta = rest[1:close]
                rest = rest[close + 1 :]
            if not rest.startswith("["):
                raise ValueError("bad fused loc")
            parts = []
            rest = rest[1:].lstrip()
            while not rest.startswith("]"):
                p, rest = self._expr(rest, depth + 1)
                parts.append(p)
                rest = rest.lstrip()
                if rest.startswith(","):
                    rest = rest[1:].lstrip()
                elif not rest.startswith("]"):
                    raise ValueError("bad fused loc list")
            return ("fused", meta, tuple(parts)), rest[1:]
        if s.startswith('"'):
            end = _string_end(s, 0)
            text = _unescape(s[1:end])
            rest = s[end + 1 :]
            m = _RE_FILE_POS.match(rest)
            if m:
                rest = rest[m.end() :]
                if rest.lstrip().startswith("to"):
                    raise ValueError("file range locs are not supported")
                return ("file", text, int(m.group(1)), int(m.group(2))), rest
            if rest.startswith("("):
                child, rest = self._expr(rest[1:], depth + 1)
                rest = rest.lstrip()
                if not rest.startswith(")"):
                    raise ValueError("unclosed name loc")
                return ("name", text, child), rest[1:]
            return ("name", text, None), rest
        raise ValueError(f"unrecognized loc: {s[:60]!r}")


_MISSING = object()


def _names_path(tree: tuple, path: str) -> bool:
    """Does any file loc inside ``tree`` name ``path``? (iterative)"""
    stack = [tree]
    while stack:
        t = stack.pop()
        if t is None:
            continue
        kind = t[0]
        if kind == "file":
            if t[1] == path:
                return True
        elif kind == "name":
            stack.append(t[2])
        elif kind == "callsite":
            stack += [t[1], t[2]]
        elif kind == "fused":
            stack += list(t[2])
    return False


def _loc_site(
    tree: tuple | None,
) -> tuple[SourceLoc | None, tuple[SourceLoc, ...], str | None]:
    """(site, callers, name) of a normalised loc. The callee frame of a
    callsite is the op's site (a memory op belongs to the callee); the
    caller chain follows, innermost first. Recursion depth is bounded by the
    parser's ``_LOC_DEPTH_LIMIT``."""
    if tree is None:
        return None, (), None
    kind = tree[0]
    if kind == "file":
        return SourceLoc(tree[1], tree[2], tree[3]), (), None
    if kind == "name":
        site, callers, _ = _loc_site(tree[2])
        return site, callers, tree[1]
    if kind == "callsite":
        site, callers, name = _loc_site(tree[1])
        csite, ccallers, _ = _loc_site(tree[2])
        return site, callers + ((csite,) if csite else ()) + ccallers, name
    if kind == "fused":
        for part in tree[2]:
            got = _loc_site(part)
            if got[0] is not None:
                return got
    return None, (), None


# ─────────────────────────── text scan ───────────────────────────

_RE_RESULTS = re.compile(
    r"^((?:%[-\w.$]+(?::\d+)?)(?:\s*,\s*%[-\w.$]+(?::\d+)?)*)\s*=\s*"
)
_RE_OPNAME = re.compile(r"^[A-Za-z_][\w$.]*")
_RE_LABEL = re.compile(r"^\^[-\w.$]+")


class _TextError(Exception):
    def __init__(self, line_no: int | None, msg: str) -> None:
        super().__init__(msg)
        self.line_no = line_no
        self.msg = msg


@dataclass
class _TBlock:
    label: str | None
    ops: list["_TOp"]
    line_no: int


@dataclass
class _TOp:
    name: str
    line_no: int
    raw: str  # stripped raw header line (comment removed)
    masked: str  # masked header line (same length)
    name_end: int  # index just past the op name
    n_results: int
    generic: bool  # printed in the generic form: "tt.reduce"(...)
    opens_region: bool
    regions: list[list[_TBlock]]
    close_line: int | None = None
    close_raw: str = ""
    close_masked: str = ""


@dataclass
class _TextTree:
    root: _TOp
    aliases: dict[str, str]


def _n_results(prefix: str) -> int:
    """``%a, %b:2`` -> 3."""
    n = 0
    for tok in prefix.split(","):
        _, _, k = tok.strip().partition(":")
        n += int(k) if k else 1
    return n


def _parse_op_line(
    ln: int, raw: str, masked: str, lits: list[tuple[int, int, str]]
) -> _TOp:
    rm = _RE_RESULTS.match(masked)
    start = rm.end() if rm else 0
    body = masked[start:]
    generic = body.startswith('"')
    if generic:
        name_end = _string_end(raw, start) + 1
        name = next((t for a, _b, t in lits if a == start), "")
    else:
        nm = _RE_OPNAME.match(body)
        if nm is None:
            raise _TextError(ln, f"cannot read an op name: {raw[:80]!r}")
        name = nm.group(0)
        name_end = start + nm.end()
        if "." not in name:
            name = f"builtin.{name}"
    n_results = _n_results(rm.group(1)) if rm else 0
    opens = masked.endswith("{")
    return _TOp(name, ln, raw, masked, name_end, n_results, generic, opens, [])


def _scan_text(text: str) -> _TextTree:
    """Rebuild the op tree from printed lines. Returns a synthetic file-level
    op whose single region holds the top-level ops, and the ``#loc`` table."""
    root = _TOp("<file>", 0, "", "", 0, 0, False, True, [[]])
    stack = [root]
    aliases: dict[str, str] = {}
    for ln, raw_line in enumerate(text.splitlines(), start=1):
        raw = raw_line.strip()
        if not raw or raw.startswith("//"):
            continue
        try:
            masked, lits = _mask(raw)
        except ValueError as e:
            raise _TextError(ln, str(e)) from None
        cut = masked.find("//")
        if cut >= 0:  # a block label's predecessor comment
            raw, masked = raw[:cut].rstrip(), masked[:cut].rstrip()
            lits = [x for x in lits if x[1] < cut]
        top = stack[-1]
        if masked.startswith("#"):
            m = _RE_LOC_ALIAS_DEF.match(raw)
            if len(stack) != 1 or m is None:
                raise _TextError(
                    ln,
                    f"attribute alias {raw[:60]!r}: only #loc aliases are TTIR (TTGIR input?)",
                )
            aliases[m.group(1)] = m.group(2)
            continue
        if masked.startswith("^"):  # its arguments come from the bindings
            lm = _RE_LABEL.match(masked)
            if top is root or lm is None:
                raise _TextError(ln, f"bad block label {raw[:60]!r}")
            after = lm.end()
            try:
                if masked.startswith("(", after):
                    after = _match_close(masked, after) + 1
            except ValueError as e:
                raise _TextError(ln, str(e)) from None
            if masked[after:].strip() != ":":
                raise _TextError(ln, f"bad block label {raw[:60]!r}")
            top.regions[-1].append(_TBlock(lm.group(0), [], ln))
            continue
        if cut >= 0:
            comment = raw_line.strip()[_mask(raw_line.strip())[0].find("//") + 2 :]
            raise _TextError(ln, f"unexpected comment {comment.strip()[:60]!r}")
        if masked.startswith("}"):
            if top is root:
                raise _TextError(ln, "unbalanced '}'")
            rest = masked[1:].lstrip()
            if rest.endswith("{"):  # "} else {", "} do {", "}, {"
                top.regions.append([])
                continue
            if rest.startswith(")"):  # generic op closer "}) ..."
                rest = rest[1:].lstrip()
            off = len(masked) - len(rest)
            top.close_line, top.close_raw, top.close_masked = ln, raw[off:], rest
            stack.pop()
            continue
        try:
            op = _parse_op_line(ln, raw, masked, lits)
        except ValueError as e:
            raise _TextError(ln, str(e)) from None
        region = top.regions[-1]
        if not region:
            region.append(_TBlock(None, [], ln))
        region[-1].ops.append(op)
        if op.opens_region:
            op.regions = [[]]
            stack.append(op)
    if len(stack) != 1:
        raise _TextError(
            stack[-1].line_no,
            f"region opened at line {stack[-1].line_no} is never closed",
        )
    return _TextTree(root, aliases)


# what the text alone says: scf.for's `unsigned` keyword (unsignedCmp, a
# unit attribute: it changes the trip count), and the operand segments of a
# generic-form tt.load (its mask and other are both optional)
_RE_FOR_HEAD = re.compile(r"\s+(unsigned\s+)?%")
_RE_SEGMENTS = re.compile(r"\boperandSegmentSizes\s*=\s*array<i32:([\d\s,]*)>")


def _text_attrs(t: _TOp, name: str) -> dict[str, Any]:
    if name == "scf.for":
        m = None if t.generic else _RE_FOR_HEAD.match(t.masked, t.name_end)
        if m is None:
            raise ValueError("unrecognized scf.for header")
        return {"unsignedCmp": m.group(1) is not None}
    if name == "tt.load" and t.generic:
        m = _RE_SEGMENTS.search(t.masked, t.name_end)
        if m is None:
            raise ValueError("generic tt.load without operandSegmentSizes")
        return {"operandSegmentSizes": tuple(int(x) for x in m.group(1).split(","))}
    return {}


# A block-pointer type anywhere in a text: `ptr<tensor` in any spelling the
# parser accepts (`!tt.ptr<tensor<..>>`, `!tt<ptr<tensor<..>>>`, whitespace,
# newlines or `//` comments in between), and any type alias definition, which
# could hide the pointee. The _GAP forms read the raw text (a comment runs to
# the end of its line), the others the code view of _screen_view.
_GAP = r"(?:\s|//[^\n]*(?![^\n]))*"
_RE_BLOCK_PTR_TYPE = re.compile(r"\bptr\s*<\s*tensor\b")
_RE_BLOCK_PTR_TYPE_GAP = re.compile(rf"\bptr{_GAP}<{_GAP}tensor\b")
_RE_TYPE_ALIAS_DEF = re.compile(r"^\s*(![-\w.$]+)\s*=", re.M)
_RE_TYPE_ALIAS_DEF_GAP = re.compile(rf"^\s*![-\w.$]+{_GAP}=", re.M)


def _code_line(line: str) -> str:
    """``line`` as the parser's tokens see it, same length: string literal
    bodies masked ('_'), a ``//`` comment outside strings blanked to the end
    of the line. After an unterminated string the rest stays raw (the
    parser stops there; reading it can only add a refusal)."""
    out: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            try:
                j = _string_end(line, i)
            except ValueError:
                out.append(line[i:])
                break
            out.append('"' + "_" * (j - i - 1) + '"')
            i = j + 1
            continue
        if line.startswith("//", i):
            out.append(" " * (len(line) - i))
            break
        out.append(c)
        i += 1
    return "".join(out)


def _screen_view(text: str) -> str:
    """The code view of ``text`` (``_code_line`` per line), offsets kept, so
    a match's line is its count of newlines before it plus one."""
    return "\n".join(_code_line(line) for line in text.split("\n"))


def _screen(text: str) -> None:
    """Refuse, before the bindings see it, a text the parser cannot be
    handed: without block-pointer types the parser aborts the whole process
    on one (an assertion in PointerType::get). The whole text is searched,
    so a type split over lines or around a comment is found too; string
    literals are not read as types, comments are skipped."""
    if not (_RE_BLOCK_PTR_TYPE_GAP.search(text) or _RE_TYPE_ALIAS_DEF_GAP.search(text)):
        return
    view = _screen_view(text)
    found = []
    m = _RE_BLOCK_PTR_TYPE.search(view)
    if m is not None:
        found.append((m.start(), "a block-pointer type (!tt.ptr<tensor<...>>)"))
    m = _RE_TYPE_ALIAS_DEF.search(view)
    if m is not None:
        found.append(
            (m.start(1), "a type alias definition (it may name a block-pointer type)")
        )
    if not found:
        return
    at, what = min(found)
    ln = view.count("\n", 0, at) + 1
    raise ModuleParseError(
        f"line {ln}: {what}: Triton's TTIR has no "
        "block pointers, and its parser aborts the process on one; the "
        "text is refused before parsing",
        ln,
    )


# ─────────────────────────── attributes (bindings) ───────────────────────────
# What the reader reads off an op, per op name: (key in Op.attrs, getter,
# attribute name, names of an enum's integers or None, required). An enum
# reads as its integer and is named by the keyword the printer prints for it
# (arith::CmpIPredicate, tt::RMWOp: "exch"); a program-id axis stays 0-2 and
# a bit enum its bits. Every int attribute here is an inherent one the op's
# verifier types as a signless integer, so get_int_attr never meets the type
# its getInt() asserts on.

_CMPI = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
_AXIS = (0, 1, 2)
# tt::RMWOp, tt::MemSemantic and tt::MemSyncScope count from 1
_RMW_OP = (None, "and", "or", "xor", "add", "fadd", "max", "min", "umax", "umin", "exch")  # fmt: skip
_SEM = (None, "relaxed", "acquire", "release", "acq_rel")
_SCOPE = (None, "gpu", "cta", "sys")
# ttg::AddrSpace is a bit enum, kept as its bits: local 1, global_read 2,
# global_write 4, tensor_read 8, tensor_write 16 (`all` is 31)
_ADDR_SPACE = tuple(range(32))
_ATTRS: Mapping[str, tuple[tuple[str, str, str, tuple | None, bool], ...]] = {
    "tt.atomic_rmw": (
        ("rmw_op", "int", "atomic_rmw_op", _RMW_OP, True),
        ("sem", "int", "sem", _SEM, True),
        ("scope", "int", "scope", _SCOPE, True),
    ),
    "tt.atomic_cas": (
        ("sem", "int", "sem", _SEM, True),
        ("scope", "int", "scope", _SCOPE, True),
    ),
    "ttg.barrier": (("addr_space", "int", "addrSpace", _ADDR_SPACE, True),),
    "tt.get_program_id": (("axis", "int", "axis", _AXIS, True),),
    "tt.get_num_programs": (("axis", "int", "axis", _AXIS, True),),
    "tt.make_range": (
        ("start", "int", "start", None, True),
        ("end", "int", "end", None, True),
    ),
    "tt.expand_dims": (("axis", "int", "axis", None, True),),
    "arith.cmpi": (("predicate", "int", "predicate", _CMPI, True),),
    "tt.elementwise_inline_asm": (("pure", "bool", "pure", None, False),),
    "tt.extern_elementwise": (
        ("pure", "bool", "pure", None, False),
        ("symbol", "str", "symbol", None, False),
    ),
    "tt.call": (("callee", "sym", "callee", None, False),),
    "tt.func": (
        ("sym_name", "str", "sym_name", None, True),
        ("visibility", "str", "sym_visibility", None, False),
    ),
}
_GETTERS: Mapping[str, tuple[str, type]] = {
    "int": ("get_int_attr", int),
    "bool": ("get_bool_attr", bool),
    "str": ("get_str_attr", str),
    "sym": ("get_flat_symbol_ref_attr", str),
}


def _is_exactly(value: Any, want: type) -> bool:
    """``isinstance`` without bool passing as int."""
    return isinstance(value, bool) == (want is bool) and isinstance(value, want)


def _read_attrs(op, name: str, result_types: Sequence[str]) -> tuple[dict, list]:
    """(attrs, problems) of one binding op; called inside the parse."""
    attrs: dict[str, Any] = {}
    problems: list[str] = []
    for key, kind, aname, names, required in _ATTRS.get(name, ()):
        getter, want = _GETTERS[kind]
        v = getattr(op, getter)(aname)
        if v is None:
            if required:
                problems.append(f"attribute {aname!r} not recovered")
        elif not _is_exactly(v, want):
            problems.append(f"{aname}: {getter} returned {type(v).__name__}")
        elif names is not None and not (0 <= v < len(names) and names[v] is not None):
            problems.append(f"{aname} {v} outside the closed vocabulary")
        else:
            attrs[key] = v if names is None else names[v]
    if name == "arith.constant":
        try:
            attrs["value"] = _constant_value(op, result_types)
        except ValueError as e:
            problems.append(str(e))
    return attrs, problems


def _constant_value(op, result_types: Sequence[str]) -> Any:
    """arith.constant's value: get_constant_value reads an integer scalar
    or splat as its sign-extended int (the printer's signed reading; i1 true
    reads -1, kept as a bool) and anything else as None: a float or a
    non-splat dense is ("float", None) / ("dense", None), as is an integer
    wider than 64 bits, which is never handed to the getter (getSExtValue
    asserts, and Triton's LLVM keeps assertions: the process would abort)."""
    if len(result_types) != 1:
        raise ValueError("arith.constant without one result")
    ty = parse_type(result_types[0])
    elem = parse_type(ty.elem)
    bits = elem.int_bits
    if bits is None:
        if elem.float_bits is None:
            raise ValueError(f"constant of unsupported element type {elem.elem}")
        return ("float", None)
    if bits > 64:
        return ("dense", None)
    v = op.get_constant_value()
    if v is None:
        if not ty.shape:
            raise ValueError("integer constant without a value")
        return ("dense", None)
    if not _is_exactly(v, int):
        raise ValueError(f"get_constant_value returned {type(v).__name__}")
    if bits == 1:
        if v not in (0, -1):
            raise ValueError(f"i1 constant {v}")
        return v == -1
    return v


# ─────────────────────────── bindings side ───────────────────────────


@dataclass
class _BOp:
    name: str
    operands: tuple[int, ...]  # raw value ids (valid within one parse only)
    operand_types: tuple[str, ...]
    results: tuple[int, ...]
    result_types: tuple[str, ...]
    result_locs: tuple[str, ...]
    region_ids: tuple[int, ...]
    region_sizes: tuple[int, ...]
    block_id: int | None
    attrs: dict[str, Any]
    problems: list[str]  # attributes _read_attrs could not read


@dataclass
class _BBlock:
    region_id: int
    args: tuple[int, ...]
    arg_types: tuple[str, ...]
    arg_locs: tuple[str, ...]
    ops: list[int]  # indices into the walk list


@dataclass
class _BindWalk:
    ops: list[_BOp]
    blocks: dict[int, _BBlock]
    region_blocks: dict[int, list[int]]  # raw region id -> raw block ids in order
    path: str  # the parse path (parser-invented locs name it)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


class _ParseInput:
    """The text as a file path for ``parse_mlir_module``: an anonymous memfd
    via ``/proc/self/fd`` when available, else a temporary file."""

    def __init__(self, data: bytes) -> None:
        self.path: str | None = None
        self._fd: int | None = None
        self._tmp: str | None = None
        memfd_create = getattr(os, "memfd_create", None)
        if memfd_create is not None:
            try:
                fd = memfd_create("tilelens-ttir", getattr(os, "MFD_CLOEXEC", 0))
            except OSError:
                fd = None
            if fd is not None:
                path = f"/proc/self/fd/{fd}"
                try:
                    _write_all(fd, data)
                    ok = os.path.exists(path)
                except OSError:
                    ok = False
                if ok:
                    self._fd, self.path = fd, path
                    return
                os.close(fd)
        try:
            fd, tmp = tempfile.mkstemp(prefix="tilelens-", suffix=".ttir")
        except OSError as e:
            raise ModuleParseError(
                f"cannot create a temporary file for the MLIR parser: {e}"
            ) from None
        try:
            try:
                _write_all(fd, data)
            finally:
                os.close(fd)
        except OSError as e:
            os.unlink(tmp)
            raise ModuleParseError(f"cannot write the MLIR parser input: {e}") from None
        self._tmp = self.path = tmp

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._tmp is not None:
            try:
                os.unlink(self._tmp)
            except OSError:
                pass
            self._tmp = None


# (saved fd 2, capture buffer) while a capture redirects fd 2; set and
# cleared under _PARSE_LOCK, read by the fork handler (_after_fork_in_child)
_REDIRECT: tuple[int, int] | None = None


class _StderrCapture:
    """Redirect fd 2 (where the C++ parser prints its diagnostic) into an
    anonymous file for the duration of a ``with`` block; only used under
    ``_PARSE_LOCK``. Everything written to fd 2 in that window lands in
    ``data``: on a failed parse that is the diagnostic, on a successful one
    it belongs to someone else (parser warnings, other threads), so
    ``replay()`` passes it on to the real fd 2."""

    def __init__(self) -> None:
        self._buf: int | None = None
        self._saved: int | None = None
        self.data = b""
        self.text = ""

    def __enter__(self) -> "_StderrCapture":
        global _REDIRECT
        try:
            memfd_create = getattr(os, "memfd_create", None)
            if memfd_create is not None:
                buf = memfd_create("tilelens-diag", getattr(os, "MFD_CLOEXEC", 0))
            else:
                with tempfile.TemporaryFile() as f:
                    buf = os.dup(f.fileno())
        except OSError:
            return self  # no capture: the diagnostic stays on stderr
        try:
            sys.stderr.flush()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            saved = os.dup(2)
        except OSError:
            os.close(buf)
            return self
        # published before fd 2 moves (and cleared after it is back), so a
        # fork at any point of the window can restore fd 2 in the child
        _REDIRECT = (saved, buf)
        try:
            os.dup2(buf, 2)
        except OSError:
            _REDIRECT = None
            os.close(saved)
            os.close(buf)
            return self
        self._buf, self._saved = buf, saved
        return self

    def __exit__(self, *exc: object) -> None:
        global _REDIRECT
        if self._saved is not None:
            os.dup2(self._saved, 2)
            _REDIRECT = None
            os.close(self._saved)
            self._saved = None
        if self._buf is not None:
            try:
                os.lseek(self._buf, 0, os.SEEK_SET)
                chunks = []
                while chunk := os.read(self._buf, 1 << 16):
                    chunks.append(chunk)
                self.data = b"".join(chunks)
                self.text = self.data.decode("utf-8", "replace")
            finally:
                os.close(self._buf)
                self._buf = None

    def replay(self) -> None:
        if self.data:
            try:
                _write_all(2, self.data)
            except OSError:
                pass


_PARSE_LOCK = threading.Lock()  # fd-2 redirection is process-wide


def _bind_walk(data: bytes) -> _BindWalk:
    """Parse ``data`` with the bindings and flatten the module into
    pure-Python records. The context is pinned on the module for the
    module's whole life, the module is dropped before the context, and no
    binding object survives the call (dropping the context first
    segfaults). After a successful parse, whatever else reached fd 2 in the
    window is passed on."""
    from triton._C.libtriton import ir  # TTIR dialects only: no backend (TTGIR) loading

    source = _ParseInput(data)
    try:
        path = source.path
        assert path is not None
        ctx = ir.context()
        try:
            ir.load_dialects(ctx)
            capture = _StderrCapture()
            mod = None
            with _PARSE_LOCK, capture:
                try:
                    mod = ir.parse_mlir_module(path, ctx)
                except RuntimeError:
                    pass
            if mod is None:
                raise _parse_error(capture.text, path)
            capture.replay()
            try:
                walk, failure = _pinned_extract(mod, ctx)
            finally:
                del mod  # the module before its context
        finally:
            del ctx
    finally:
        source.close()
    if walk is None:
        raise MisalignedModule([f"bindings walk failed: {failure}"])
    ops, blocks, region_blocks = walk
    return _BindWalk(ops, blocks, region_blocks, path)


def _pinned_extract(mod, ctx) -> tuple[Any, str | None]:
    """Pin ``ctx`` on ``mod`` (proton's pattern: the module keeps its context
    alive), then extract. Returns (records, None) or (None, reason); an
    exception's traceback, whose frames hold binding objects, dies here while
    the module and its context are still alive."""
    try:
        mod.context = ctx
    except Exception as e:  # noqa: BLE001
        return (
            None,
            f"cannot pin the MLIR context on the module: {type(e).__name__}: {e}",
        )
    body: list[Any] = []
    try:
        walk = _extract(mod, body)
    except Exception as e:  # noqa: BLE001  (bindings drift: an unexpected getter result)
        return None, f"{type(e).__name__}: {e}"
    # No binding erases a module, so each parse would leak its whole op tree
    # (~20 KiB for a 12 KiB text); erasing the body block frees all but the
    # empty module op. Safe here: extraction is over, ctx is pinned, and the
    # body block is the one binding object still alive (dropped at once).
    if body:
        try:
            body.pop().erase()
        except Exception:  # noqa: BLE001  (bindings drift: keep the bounded leak)
            pass
    return walk, None


_RE_DIAG_POS = re.compile(r'loc\("<ttir>":(\d+):\d+\)')


def _parse_error(diag: str, path: str) -> ModuleParseError:
    diag = diag.replace(path, "<ttir>").strip() or "the MLIR parser rejected the text"
    m = _RE_DIAG_POS.search(diag)
    return ModuleParseError(diag, int(m.group(1)) if m else None)


def _extract(
    mod, body: list[Any]
) -> tuple[list[_BOp], dict[int, _BBlock], dict[int, list[int]]]:
    """Copy the walked module into pure-Python records (post-order walk:
    ops of a block arrive in program order, blocks of a region in order).
    The module's body block (the one binding object kept) goes to ``body``,
    for ``_pinned_extract`` to erase."""
    ops: list[_BOp] = []
    blocks: dict[int, _BBlock] = {}
    region_blocks: dict[int, list[int]] = {}

    def cb(op) -> None:
        blk = op.get_block()
        bid = None
        if blk is not None:
            bid = blk.id()
            rec = blocks.get(bid)
            if rec is None:
                args = [blk.get_argument(i) for i in range(blk.get_num_arguments())]
                parent = blk.get_parent()
                rid = parent.id()
                if parent.get_parent_region() is None:
                    body.append(blk)  # the module's body block
                rec = blocks[bid] = _BBlock(
                    rid,
                    tuple(x.id() for x in args),
                    tuple(str(x.get_type()) for x in args),
                    tuple(str(x.get_loc()) for x in args),
                    [],
                )
                region_blocks.setdefault(rid, []).append(bid)
            rec.ops.append(len(ops))
        name = op.get_name()
        opnds = [op.get_operand(i) for i in range(op.get_num_operands())]
        res = [op.get_result(i) for i in range(op.get_num_results())]
        regs = [op.get_region(i) for i in range(op.get_num_regions())]
        res_types = tuple(str(x.get_type()) for x in res)
        attrs, problems = _read_attrs(op, name, res_types)
        ops.append(
            _BOp(
                name,
                tuple(x.id() for x in opnds),
                tuple(str(x.get_type()) for x in opnds),
                tuple(x.id() for x in res),
                res_types,
                tuple(str(x.get_loc()) for x in res),
                tuple(x.id() for x in regs),
                tuple(x.size() for x in regs),
                bid,
                attrs,
                problems,
            )
        )

    mod.walk(cb)
    return ops, blocks, region_blocks


# ─────────────────────────── alignment ───────────────────────────


@dataclass
class _OpB:  # mutable op builder
    name: str
    operands_raw: tuple[int, ...]
    operand_types: tuple[str, ...]
    results: tuple[int, ...]
    result_types: tuple[str, ...]
    attrs: dict[str, Any]
    path: tuple[int, ...]
    position: int
    line_no: int | None
    end_line: int | None
    loc: SourceLoc | None
    callers: tuple[SourceLoc, ...]
    loc_name: str | None
    implicit: bool
    regions: list[tuple[int, ...]]


@dataclass
class _BlockB:
    op: int
    region: int
    position: int
    label: str | None
    args: tuple[int, ...]
    arg_types: tuple[str, ...]
    arg_names: tuple[str | None, ...]
    ops: list[int]
    path: tuple[int, ...]  # path of the ops inside this block


# what Module.stats counts
_STATS = ("ops", "implicit_ops", "blocks", "values", "funcs", "result_locs", "attrs")
_ELIDES_YIELD = frozenset({"scf.for", "scf.if"})


class _Aligner:
    def __init__(self, tree: _TextTree, bw: _BindWalk) -> None:
        self.tree = tree
        self.bw = bw
        self.locp = _LocParser(tree.aliases, bw.path)
        self.problems: list[tuple[int | None, str]] = []
        self.ops: list[_OpB] = []
        self.blocks: list[_BlockB] = []
        self.values: list[list[Any]] = []  # [type, op, block, position, name]
        self.vmap: dict[int, int] = {}  # raw value id -> value index
        # op lines with / without a printed trailing loc: the printer (debug
        # info on) gives every op one, so a module mixing both is misread
        self.loc_lines: list[int] = []
        self.no_loc_lines: list[int] = []
        self.stats: collections.Counter[str] = collections.Counter(
            dict.fromkeys(_STATS, 0)
        )

    def bad(self, line: int | None, msg: str) -> None:
        self.problems.append((line, f"line {line}: {msg}" if line is not None else msg))

    def new_value(
        self,
        raw: int,
        type_: str,
        op: int | None,
        block: int | None,
        pos: int,
        loc: str,
    ) -> int:
        if raw in self.vmap:
            raise ValueError("a value appears twice in the walk")
        idx = len(self.values)
        tree = self.locp.parse(loc)
        name = _loc_site(tree)[2] if tree is not None else None
        self.values.append([type_, op, block, pos, name])
        self.vmap[raw] = idx
        return idx

    def text_loc(self, t: _TOp) -> tuple | None:
        masked, raw = (
            (t.close_masked, t.close_raw) if t.opens_region else (t.masked, t.raw)
        )
        span = _trailing_loc(masked)
        (self.loc_lines if span is not None else self.no_loc_lines).append(t.line_no)
        if span is None:
            return None
        return self.locp.parse(raw[span[0] : span[1]])

    # ── traversal ──
    def run(self) -> Module:
        root_w = [i for i, b in enumerate(self.bw.ops) if b.block_id is None]
        top = self.tree.root.regions[0][0].ops if self.tree.root.regions[0] else []
        if len(root_w) != 1 or len(top) != 1:
            self.bad(
                None,
                f"expected one top-level op: text has {len(top)}, bindings {len(root_w)}",
            )
            raise self.failure()
        # task: ("op", text op, walk index, parent block index | None, position)
        #    or ("implicit", walk index, parent block index, position)
        stack: list[tuple] = [("op", top[0], root_w[0], None, 0)]
        while stack:
            task = stack.pop()
            if task[0] == "op":
                children = self.visit(*task[1:])
            else:
                children = []
                self.implicit(*task[1:])
            stack.extend(reversed(children))
        if len(self.ops) != len(self.bw.ops):
            self.bad(
                None, f"{len(self.ops)} aligned ops != {len(self.bw.ops)} walked ops"
            )
        if self.loc_lines:
            for line in self.no_loc_lines:
                self.bad(line, "op prints no loc while the module prints locs")
        if self.problems:
            raise self.failure()
        return self.freeze()

    def failure(self) -> MisalignedModule:
        first = next((ln for ln, _ in self.problems if ln is not None), None)
        return MisalignedModule([m for _, m in self.problems], first)

    def implicit(self, wi: int, block: int, pos: int) -> None:
        b = self.bw.ops[wi]
        idx = len(self.ops)
        self.stats["implicit_ops"] += 1
        path = self.blocks[block].path
        self.ops.append(
            _OpB(
                b.name,
                (),
                (),
                (),
                (),
                {},
                path,
                pos,
                None,
                None,
                None,
                (),
                None,
                True,
                [],
            )
        )
        self.blocks[block].ops.append(idx)

    def visit(self, t: _TOp, wi: int, block: int | None, pos: int) -> list:
        b = self.bw.ops[wi]
        line = t.line_no
        idx = len(self.ops)
        path = self.blocks[block].path if block is not None else ()
        if block is not None:
            self.blocks[block].ops.append(idx)
        structural_ok = True
        if t.name != b.name:
            self.bad(line, f"text op {t.name!r} != walked op {b.name!r}")
            structural_ok = False
        if t.n_results != len(b.results):
            self.bad(
                line,
                f"{t.name}: {t.n_results} printed results != {len(b.results)} walked",
            )
            structural_ok = False
        # printed regions: trailing empty regions may be omitted
        if len(t.regions) > len(b.region_ids) or any(
            b.region_sizes[i] for i in range(len(t.regions), len(b.region_ids))
        ):
            self.bad(
                line,
                f"{t.name}: {len(t.regions)} printed regions vs walked sizes {list(b.region_sizes)}",
            )
            structural_ok = False
        # the op's loc: the printed one, which must equal the loc the bindings
        # report for each of its results (the one source for a zero-result op)
        try:
            tloc = self.text_loc(t)
            for k, s in enumerate(b.result_locs):
                tree = self.locp.parse(s)
                if tree is not None:
                    self.stats["result_locs"] += 1
                    if tree != tloc:
                        self.bad(
                            line,
                            f"{t.name}: result {k} loc differs: text {tloc} vs bindings {tree}",
                        )
        except ValueError as e:
            self.bad(line, f"{t.name}: unreadable loc: {e}")
            tloc = None
        site, callers, lname = _loc_site(tloc)
        attrs = dict(b.attrs)
        if structural_ok:
            for p in b.problems:
                self.bad(line, f"{b.name}: {p}")
            try:
                attrs.update(_text_attrs(t, b.name))
            except ValueError as e:
                self.bad(line, f"{b.name}: {e}")
            self.stats["attrs"] += len(attrs)
        rec = _OpB(
            b.name,
            b.operands,
            b.operand_types,
            (),
            b.result_types,
            attrs,
            path,
            pos,
            line,
            t.close_line if t.opens_region else line,
            site,
            callers,
            lname,
            False,
            [],
        )
        self.ops.append(rec)
        rec.results = tuple(
            self.new_value(raw, b.result_types[k], idx, None, k, b.result_locs[k])
            for k, raw in enumerate(b.results)
        )
        if not structural_ok:
            rec.regions = [() for _ in b.region_ids]
            return []
        children: list[tuple] = []
        for ri, rid in enumerate(b.region_ids):
            children += self.region(t, b, idx, ri, rid, rec)
        return children

    def region(self, t: _TOp, b: _BOp, idx: int, ri: int, rid: int, rec: _OpB) -> list:
        line = t.line_no
        wblocks = self.bw.region_blocks.get(rid, [])
        if len(wblocks) != b.region_sizes[ri]:
            self.bad(line, f"{t.name}: region {ri} has a block without ops")
        tblocks = t.regions[ri] if ri < len(t.regions) else []
        if not tblocks and len(wblocks) == 1 and b.name in _ELIDES_YIELD:
            only = self.bw.blocks[wblocks[0]].ops
            tail = self.bw.ops[only[-1]] if len(only) == 1 else None
            if tail is not None and tail.name == "scf.yield" and not tail.operands:
                # `{ }`: the region's one block held only the elided yield
                tblocks = [_TBlock(None, [], line)]
        if len(tblocks) != len(wblocks):
            self.bad(
                line,
                f"{t.name}: region {ri}: {len(tblocks)} printed blocks != {len(wblocks)} walked",
            )
            rec.regions.append(())
            return []
        keys = []
        children: list[tuple] = []
        for bi, (tb, rbid) in enumerate(zip(tblocks, wblocks)):
            bb = self.bw.blocks[rbid]
            bidx = len(self.blocks)
            keys.append(bidx)
            if tb.label is None and bi != 0:
                self.bad(tb.line_no, f"{t.name}: unlabeled block {ri}.{bi}")
            blk = _BlockB(
                idx, ri, bi, tb.label, (), bb.arg_types, (), [], rec.path + (bidx,)
            )
            self.blocks.append(blk)
            blk.args = tuple(
                self.new_value(raw, bb.arg_types[ai], None, bidx, ai, bb.arg_locs[ai])
                for ai, raw in enumerate(bb.args)
            )
            blk.arg_names = tuple(self.values[v][4] for v in blk.args)
            # ops, with the one tolerated gap: an elided trailing scf.yield
            n_text, n_walk = len(tb.ops), len(bb.ops)
            elided = False
            if n_text != n_walk:
                tail = self.bw.ops[bb.ops[-1]] if bb.ops else None
                elided = (
                    n_text == n_walk - 1
                    and tail is not None
                    and tail.name == "scf.yield"
                    and not tail.operands
                    and b.name in _ELIDES_YIELD
                )
                if not elided:
                    self.bad(
                        tb.line_no,
                        f"{t.name}: block {ri}.{bi}: {n_text} printed ops != {n_walk} walked",
                    )
                    continue
            for p, (tchild, wchild) in enumerate(zip(tb.ops, bb.ops)):
                children.append(("op", tchild, wchild, bidx, p))
            if elided:
                children.append(("implicit", bb.ops[-1], bidx, n_text))
        rec.regions.append(tuple(keys))
        return children

    def freeze(self) -> Module:
        values = tuple(Value(i, *v) for i, v in enumerate(self.values))
        blocks = tuple(
            Block(
                i,
                b.op,
                b.region,
                b.position,
                b.label,
                b.args,
                b.arg_types,
                b.arg_names,
                tuple(b.ops),
            )
            for i, b in enumerate(self.blocks)
        )
        ops = []
        funcs = []
        for i, r in enumerate(self.ops):
            operands = tuple(self.vmap[v] for v in r.operands_raw)
            ops.append(
                Op(
                    i,
                    r.name,
                    operands,
                    r.operand_types,
                    r.results,
                    r.result_types,
                    _FrozenMap(r.attrs),
                    tuple(r.regions),
                    r.path,
                    r.position,
                    r.line_no,
                    r.end_line,
                    r.loc,
                    r.callers,
                    r.loc_name,
                    r.implicit,
                )
            )
            if r.name == "tt.func":
                args: tuple[FuncArg, ...] = ()
                if r.regions and r.regions[0]:
                    entry = blocks[r.regions[0][0]]
                    args = tuple(
                        FuncArg(k, v, entry.arg_types[k], entry.arg_names[k])
                        for k, v in enumerate(entry.args)
                    )
                visibility = r.attrs.get("visibility", "public")
                funcs.append(Func(i, r.attrs["sym_name"], visibility, args))
        self.stats.update(
            ops=len(ops), blocks=len(blocks), values=len(values), funcs=len(funcs)
        )
        return Module(
            tuple(ops),
            blocks,
            values,
            tuple(funcs),
            _FrozenMap(self.stats),
        )


# ─────────────────────────── entry point ───────────────────────────


def _walk(text: str, scan_text: str | None = None) -> Module:
    """Uncached walk. ``scan_text`` (tests only) feeds the text layer a
    different string than the bindings parse."""
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ModuleParseError(f"the text is not encodable as UTF-8: {e}") from None
    _screen(text)
    bw = _bind_walk(data)
    try:
        tree = _scan_text(text if scan_text is None else scan_text)
    except _TextError as e:
        raise MisalignedModule(
            [f"line {e.line_no}: {e.msg}" if e.line_no else e.msg], e.line_no
        ) from None
    try:
        return _Aligner(tree, bw).run()
    except MisalignedModule:
        raise
    except (ValueError, KeyError, IndexError, AssertionError, TypeError) as e:
        # a text shape the aligner does not model: fail closed
        raise MisalignedModule([f"aligner: {type(e).__name__}: {e}"]) from None


_CACHE_SIZE = 32
_CACHE: collections.OrderedDict[
    bytes, Module | tuple[tuple[str, ...], int | None]
] = collections.OrderedDict()
_CACHE_LOCK = threading.Lock()


def walk_module(text: str) -> Module:
    """Walk one printed TTIR module (see the module docstring).

    Raises ``MisalignedModule`` when the text layer and the bindings
    disagree and ``ModuleParseError`` when the MLIR parser rejects the text
    (or the text holds a construct the parser cannot be handed, or the
    installed Triton is not the release the walk reads). Results (and
    misalignments) are cached by sha256 of the text, so a text is parsed
    once while it stays among the last ``_CACHE_SIZE`` distinct texts.
    """
    from ..core.config import ir_triton_unsupported

    unsupported = ir_triton_unsupported()
    if unsupported is not None:
        raise ModuleParseError(unsupported)
    key = hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
    if hit is None:
        try:
            hit = _walk(text)
        except MisalignedModule as e:
            hit = (e.problems, e.line_no)
        with _CACHE_LOCK:
            _CACHE[key] = hit
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
    if isinstance(hit, Module):
        return hit
    raise MisalignedModule(*hit)


def _after_fork_in_child() -> None:
    """A fork while another thread holds ``_PARSE_LOCK`` / ``_CACHE_LOCK``
    leaves the child a lock nobody releases (its next walk would hang), and a
    fork inside a parse window leaves the child's fd 2 in the capture buffer.
    The child gets fresh locks and its fd 2 back; the capture's two fds stay
    open (their owner is the parent's thread, which does not run here)."""
    global _PARSE_LOCK, _CACHE_LOCK, _REDIRECT
    _PARSE_LOCK = threading.Lock()
    _CACHE_LOCK = threading.Lock()
    redirect, _REDIRECT = _REDIRECT, None
    if redirect is not None:
        try:
            os.dup2(redirect[0], 2)
        except OSError:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
