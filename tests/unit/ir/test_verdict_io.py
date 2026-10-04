"""Persistence of the IR-mode records: the tilelens.ir.verdict records
and the compiled sanitizer's findings are plain data that tilelens.save() /
tilelens.load() round-trip, with a public source-location type in place of
the reader's private one. No GPU.
"""

from __future__ import annotations

import dataclasses
import enum
import gc
import importlib
import subprocess
import sys
import weakref
import zipfile
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

import tilelens
from tilelens.clients.sanitizer.data import CompiledSanitizerRecord
from tilelens.core.data import Launch, Load, Store
from tilelens.ir import _mlir_walk as W
from tilelens.ir.launch import TensorFacts, tensor_facts
from tilelens.ir.ttir_reader import TTIRKind, UnsupportedTTIR
from tilelens.ir.verdict import ConfigVerdict, IRVerdict, Refusal, SourceLocation
from tilelens.utils.traceback_utils import TracebackInfo

trace_module = importlib.import_module("tilelens.core.trace")
REPO = Path(__file__).resolve().parents[3]


def test_record_modules_import_without_triton():
    code = (
        "import sys\n"
        "import tilelens.clients.sanitizer.data\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'triton')\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


# ======== source locations and refusals =========


def test_refusal_holds_the_reader_loc_as_a_public_source_location():
    exc = UnsupportedTTIR(
        TTIRKind.CONTROL_FLOW, "scf.while", line_no=7, loc=W.SourceLoc("k.py", 3, 5)
    )
    refusal = Refusal.from_exception(exc)

    assert refusal == Refusal(
        "control-flow", "scf.while", 7, SourceLocation("k.py", 3, 5)
    )
    assert type(refusal.loc) is SourceLocation
    # The reader's enum kind is held as its plain string, as a load gives it.
    assert refusal.kind == TTIRKind.CONTROL_FLOW
    assert not isinstance(refusal.kind, enum.Enum)
    # Building one directly converts the reader's loc the same way; an
    # object without a column gets None.
    assert Refusal("call", "m", loc=W.SourceLoc("k.py", 3, 5)).loc == refusal.loc
    no_col = Refusal("call", "m", loc=SimpleNamespace(file="k.py", line=2))
    assert no_col.loc == SourceLocation("k.py", 2, None)
    assert Refusal("call", "m").loc is None


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"loc": ("k.py", 3, 5)}, "Refusal.loc must be a SourceLocation"),
        ({"loc": "k.py:3"}, "Refusal.loc must be a SourceLocation"),
        ({"kind": None}, "Refusal.kind must be a str"),
        ({"message": ValueError("m")}, "Refusal.message must be a str"),
    ],
)
def test_refusal_rejects_fields_a_trace_cannot_hold(fields, match):
    with pytest.raises(TypeError, match=match):
        Refusal(**{"kind": "call", "message": "m", **fields})


def test_record_hashes_agree_with_equality():
    loc = SourceLocation("k.py", 3, 5)
    from_enum = Refusal(TTIRKind.CALL, "m", 4, W.SourceLoc("k.py", 3, 5))
    plain = Refusal("call", "m", 4, loc)
    assert from_enum == plain and hash(from_enum) == hash(plain)
    assert len({from_enum, plain}) == 1
    assert hash(loc) == hash(SourceLocation("k.py", 3, 5))
    # The verdicts hold a config dict: explicitly unhashable, not a hash
    # that fails only for some field values.
    for verdict in (ConfigVerdict("h", {}, "ok"), IRVerdict("toy_ir", "ok")):
        assert type(verdict).__hash__ is None
        with pytest.raises(TypeError, match="unhashable"):
            hash(verdict)


class _Note(str, enum.Enum):
    TIMEOUT = "solver timed out"


def test_verdict_notes_are_a_tuple_of_plain_strings():
    verdict = IRVerdict("toy_ir", "ok", notes=(n for n in ["a", _Note.TIMEOUT]))
    assert verdict.notes == ("a", "solver timed out")
    assert not any(isinstance(note, enum.Enum) for note in verdict.notes)
    assert IRVerdict("toy_ir", "ok").notes == ()
    with pytest.raises(TypeError, match="notes takes a sequence, not a str"):
        IRVerdict("toy_ir", "ok", notes="solver timed out")
    with pytest.raises(TypeError, match="IRVerdict note must be a str, not bytes"):
        IRVerdict("toy_ir", "ok", notes=[b"raw"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="IRVerdict note must be a str, not int"):
        IRVerdict("toy_ir", "ok", notes=b"ab")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="per_config items must be ConfigVerdicts"):
        IRVerdict("toy_ir", "ok", per_config=[{"BLOCK": 16}])  # type: ignore[list-item]


# ======== save / load =========


def _save_and_load(tmp_path, monkeypatch, records):
    monkeypatch.setattr(
        trace_module, "launches", [Launch(grid=(4, 1, 1), records=records)]
    )
    path = tilelens.save(tmp_path / "trace.tvz")
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("manifest.json").decode()
    (launch,) = tilelens.load(path)
    return launch, manifest


def _refused_verdict() -> IRVerdict:
    refusal = Refusal.from_exception(
        UnsupportedTTIR(
            "control-flow", "scf.while", line_no=7, loc=W.SourceLoc("k.py", 3, 5)
        )
    )
    return IRVerdict(
        "sanitizer_ir",
        "unsupported",
        scope="launch",
        refusal=refusal,
        per_config=[
            ConfigVerdict(
                "hash-a", MappingProxyType({"BLOCK": 16, "num_warps": 4}), "proved"
            ),
            ConfigVerdict(None, {"BLOCK": 64}, "refused", refusal, n_reports=2),
            ConfigVerdict(
                "hash-c",
                {"BLOCK": 32},
                "refused",
                Refusal("solver-unknown", "timeout after 10 s"),
            ),
        ],
        notes=["1 of 3 configs proved"],
    )


def test_a_saved_trace_holds_ir_verdicts(tmp_path, monkeypatch):
    verdict = _refused_verdict()

    launch, manifest = _save_and_load(tmp_path, monkeypatch, ["report", verdict])

    assert launch.records == ["report", verdict]
    loaded = launch.records[1]
    assert type(loaded) is IRVerdict and loaded is not verdict
    assert type(loaded.per_config) is tuple and type(loaded.notes) is tuple
    assert [type(c.config) for c in loaded.per_config] == [dict] * 3
    assert type(loaded.refusal.loc) is SourceLocation
    assert not isinstance(loaded.refusal.kind, enum.Enum)
    assert hash(loaded.refusal) == hash(verdict.refusal)
    assert loaded.per_config[1].refusal == loaded.refusal
    # The manifest names only public record types.
    assert "tilelens.ir.verdict:SourceLocation" in manifest
    assert "_mlir_walk" not in manifest


def _findings(facts: TensorFacts) -> list[CompiledSanitizerRecord]:
    traceback = TracebackInfo("k.py", 3, "kernel", "x = tl.load(x_ptr + offs)")
    return [
        CompiledSanitizerRecord(
            kind="out-of-bounds",
            op_type=Load,
            tensor_name="x_ptr",
            tensor_facts=facts,
            witness={"pid_0": np.int64(3), "arange_0_d0": 5},
            config=MappingProxyType({"BLOCK": 64}),
            user_code_tracebacks=(traceback,),
            violation_offset=np.int64(-2),
            violation_address=facts.data_ptr - 2 * facts.elem_size,
            detail="mask is live at the witness",
        ),
        CompiledSanitizerRecord(
            kind="integer-overflow",
            op_type=Store,
            tensor_name="out_ptr",
            tensor_facts=facts,
            witness={"pid_0": 2**20},
            config={},
            user_code_tracebacks=[traceback],
            detail="pid * 4096 overflows i32",
        ),
        CompiledSanitizerRecord(
            kind="division-by-zero",
            op_type=Load,
            tensor_name="x_ptr",
            tensor_facts=facts,
            witness={"pid_0": 0},
            config={"BLOCK": 16},
            user_code_tracebacks=[],
        ),
    ]


def test_compiled_sanitizer_records_hold_no_tensor():
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    tensor_ref = weakref.ref(tensor)
    records = _findings(tensor_facts(tensor))

    first = records[0]
    assert first.tensor_facts.shape == (4, 3)
    assert first.tensor_facts.strides == (6, 2)
    assert first.tensor_facts.dtype == "torch.float32"
    assert first.witness == {"pid_0": 3, "arange_0_d0": 5}
    assert all(isinstance(value, int) for value in first.witness.values())
    assert isinstance(first.violation_offset, int)
    assert isinstance(first.config, dict)
    assert isinstance(first.user_code_tracebacks, list)
    # A record outlives its launch without keeping the tensor alive.
    del tensor
    gc.collect()
    assert tensor_ref() is None
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)


@pytest.mark.parametrize(
    "overrides, error, match",
    [
        ({"kind": "oob"}, ValueError, "unknown compiled sanitizer finding"),
        ({"op_type": object}, TypeError, "op_type must be Load or Store"),
        ({"witness": {"pid_0": 1.5}}, TypeError, "float"),
        ({"violation_offset": 2.0}, TypeError, "float"),
    ],
)
def test_compiled_sanitizer_records_reject_values_a_trace_cannot_hold(
    overrides, error, match
):
    facts = tensor_facts(torch.zeros(4))
    fields = {
        "kind": "out-of-bounds",
        "op_type": Load,
        "tensor_name": "x_ptr",
        "tensor_facts": facts,
        "witness": {},
        "config": {},
        "user_code_tracebacks": [],
        **overrides,
    }
    with pytest.raises(error, match=match):
        CompiledSanitizerRecord(**fields)


def test_a_saved_trace_holds_compiled_sanitizer_records(tmp_path, monkeypatch):
    records = _findings(tensor_facts(torch.arange(24.0).reshape(4, 6)[:, ::2]))
    verdict = IRVerdict(
        "sanitizer_ir",
        "findings",
        per_config=[
            ConfigVerdict("hash-a", {"BLOCK": 64}, "findings", n_reports=len(records))
        ],
    )

    launch, manifest = _save_and_load(tmp_path, monkeypatch, [*records, verdict])

    assert launch.records == [*records, verdict]
    loaded = launch.records[:-1]
    assert [type(r) for r in loaded] == [CompiledSanitizerRecord] * 3
    assert [r.op_type for r in loaded] == [Load, Store, Load]
    for record in loaded:
        assert type(record.tensor_facts) is TensorFacts
        assert all(type(tb) is TracebackInfo for tb in record.user_code_tracebacks)
        assert not any(
            isinstance(getattr(record, f.name), torch.Tensor)
            for f in dataclasses.fields(record)
        )
    # Nothing was stored as a tensor payload.
    assert '"kind": "tensor"' not in manifest


def test_trace_io_registers_tensor_facts_in_its_own_right():
    """TensorFacts is registered from tilelens.ir.launch, not only as a name
    tilelens.clients.sanitizer.data happens to import."""
    code = (
        "import tilelens.clients.sanitizer.data as data\n"
        "del data.TensorFacts\n"
        "from tilelens.core import trace_io\n"
        "from tilelens.ir.launch import TensorFacts\n"
        "assert trace_io._TRACE_CLASSES['tilelens.ir.launch:TensorFacts'] is TensorFacts\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)
