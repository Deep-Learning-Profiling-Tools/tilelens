"""RaceDetector(compile=True) on real launches: each launch is compiled on the
host for the client's target, its TTIR analyzed for global-memory races and
the kernel not run, on CPU tensors with Triton's driver unreachable. The
kernels are tests/unit/race_detector/compiled_kernels.py's, whose analysis
tests/unit/race_detector checks in detail."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.clients import (
    CompiledRaceDetector,
    CompiledRaceRecord,
    RaceDetector,
    RaceVerdict,
)
from tilelens.clients.race_detector.compiled import client as client_module
from tilelens.clients.race_detector.compiled.leaves import LaunchTensor
from tilelens.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from tilelens.core.config import DEFAULT_IR_TARGET
from tilelens.core.config import config as cfg
from tilelens.ir.launch import LaunchBinding, tensor_facts
from tilelens.ir.ttir_reader import parse_ttir

# tests/ is on sys.path (tests/conftest.py's rootdir under pytest's default
# prepend import mode): the kernels and the TTIR injector are the unit
# tests' own.
from unit.ir.inject import inject_ttir
from unit.race_detector import compiled_kernels as K

trace_module = importlib.import_module("tilelens.core.trace")

REPO = Path(__file__).resolve().parents[2]
B = K.B


@pytest.fixture(autouse=True)
def _setup(unreachable_driver, monkeypatch):
    unreachable_driver("IR mode queried Triton's driver")
    for name in ("TILELENS_IR_TARGET", "TRITON_VIZ_IR_TARGET", "TRITON_INTERPRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "ir_target", DEFAULT_IR_TARGET)
    monkeypatch.setattr(cfg, "enable_race_detector", True)
    from triton import knobs

    # A traced launch elsewhere may have left interpret mode pinned on.
    monkeypatch.setitem(knobs.runtime.__dict__, "interpret", False)


def i32(n):
    return torch.zeros(n, dtype=torch.int32)


def f32(n):
    return torch.zeros(n, dtype=torch.float32)


def launch(kernel, grid, detector=None, **args):
    """Launch ``kernel`` (a compiled_kernels name or a JITFunction) traced by
    a compiled race detector with ``args`` (and, for a compiled_kernels
    name, its SPECS constexprs); returns the detector."""
    if isinstance(kernel, str):
        args = {**K.SPECS[kernel][1], **args}
        kernel = getattr(K, kernel)
    det = detector or RaceDetector(compile=True)
    tilelens.trace(det)(kernel)[grid](**args)
    return det


def races(det):
    return {
        (r.race_type, r.first.tensor_name, r.programs[0][0], r.programs[1][0])
        for r in det.last_reports
    }


@pytest.mark.parametrize(
    "name, race",
    [
        ("waw", ("WAW", "out")),
        ("raw", ("RAW", "x")),
        ("war", ("WAR", "x")),
        ("atomic_vs_load", ("RAW", "x")),
    ],
)
def test_each_race_class_is_recorded_and_printed(name, race, capsys):
    tensors = {"x": i32(B), "out": i32(4 * B)}
    det = launch(
        name, (2,), **{k: v for k, v in tensors.items() if k in K.SPECS[name][0]}
    )
    verdict = det.last_verdict
    assert isinstance(verdict, RaceVerdict) and verdict.status == "races"
    assert races(det) == {(*race, 0, 1)}
    records = trace_module.launches[-1].records
    assert records[:-1] == det.last_reports and records[-1] is verdict
    (config,) = verdict.per_config
    assert (config.status, config.n_reports) == ("races", len(det.last_reports))
    record = det.last_reports[0]
    assert isinstance(record, CompiledRaceRecord)
    assert record.first.source_location[0] == K.__file__
    assert record.first.element is not None
    out = capsys.readouterr().out
    assert out.count("Data Race Detected") == len(det.last_reports)
    assert "(compiled race detector)" in out and f"Race Type: {race[0]}" in out


I32 = "assumes no i32 overflow"
TERM = "assumes termination"
CONTENT = "reads tensor contents"


def _perm():
    return torch.randperm(4 * B, generator=torch.Generator().manual_seed(0)).to(
        torch.int32
    )


# (kernel, grid, args, status, rung, qualifiers, refusal kind)
ROWS = [
    # a proof per rung
    ("add", 8, lambda: dict(x=f32(1024), y=f32(1024), out=f32(1024), n=1000),
     "ok", "T0", (I32,), None),
    ("atomic_only", 4, lambda: dict(x=i32(B)), "ok", "T0", (), None),
    ("strided", 4, lambda: dict(x=f32(64 * B), out=f32(64 * B), stride=B),
     "ok", "T1", (I32,), None),
    ("two_slots", 2, lambda: dict(out=i32(2 * B)), "ok", "T1-launch", (), None),
    ("scatter", 4, lambda: dict(idx=_perm(), out=i32(4 * B)),
     "ok", "T1", (I32, CONTENT), None),
    # fences: a global barrier orders; a conditional one does not; in a loop
    # one barrier orders the same iteration only
    ("store_fence_load", 1, lambda: dict(hist=i32(B), out=i32(B)),
     "ok", "T1-launch", (), None),
    ("store_flag_fence_load", 1, lambda: dict(hist=i32(B), out=i32(B), flag=3),
     "races", None, (), None),
    ("loop_rows_fenced", 1, lambda: dict(t=i32(4 * B), u=i32(4 * B), n=4),
     "ok", "T0", (I32,), None),
    ("loop_buffer_fenced", 1, lambda: dict(t=i32(4 * B), u=i32(4 * B), n=4),
     "races", None, (), None),
    # dependency order
    ("inplace", 4, lambda: dict(x=i32(4 * B)), "ok", "T0", (I32,), None),
    ("inplace_shift", 4, lambda: dict(x=i32(4 * B + 1)), "races", None, (), None),
    # awaits
    ("comm_comp", 3, lambda: dict(sem=i32(1), payload=f32(B), out=f32(2 * B)),
     "ok", "T1", (I32, TERM), None),
    ("counted_while", 1, lambda: dict(x=i32(1), n=4),
     "unsupported", None, (), "spin-shape"),
    ("cas_lock", 2, lambda: dict(lock=i32(1), data=i32(B)),
     "ok", "T1", (TERM,), None),
    # the CAS operand gate
    ("cas_bool_operand", 2, lambda: dict(flag=i32(1), out=i32(4)),
     "ok", "T1", (), None),
    ("cas_cast_operand", 2, lambda: dict(flag=i32(1), data=i32(1)),
     "unsupported", None, (), "cas-value"),
    # multipath: an early return, several loops, a loop under a branch whose
    # bounds wrap only where it is not reached
    ("early_return", 4, lambda: dict(x=f32(4 * B), out=f32(4 * B), T=4 * B),
     "ok", "T0", (I32,), None),
    ("sequential_loops", 4, lambda: dict(x=f32(B), out=f32(32 * B), n=2),
     "ok", "T1", (I32,), None),
    ("nested_loops", 4, lambda: dict(x=f32(64 * B), out=f32(64 * B), n=2, m=3),
     "ok", "T1", (I32,), None),
    ("loop_under_guard", 8, lambda: dict(out=i32(8 * B)), "ok", "T0", (), None),
    # width obligations: qualified any-grid, discharged at the launch
    ("wrapped_slots", 2, lambda: dict(out=i32(3)), "ok", "T0", (I32,), None),
    # program 2 wraps to element 0 at this launch
    ("wrapped_slots", 3, lambda: dict(out=i32(3)),
     "unsupported", None, (), "integer-overflow"),
    ("launch_wrap", 2, lambda: dict(x=i32(2), out=i32(2 * B), big=2**30),
     "unsupported", None, (), "integer-overflow"),
    ("where_wrap", 3, lambda: dict(out=i32(4)), "races", None, (), None),
    # a mask on float flags (no snapshot): a possible race, not a witness
    ("table_mask", 4, lambda: dict(flags=f32(4 * B), out=i32(B)),
     "unsupported", None, (), "potential-race"),
]  # fmt: skip


@pytest.mark.parametrize(
    "name, grid, args, status, rung, qualifiers, kind",
    ROWS,
    ids=[f"{row[0]}-{i}" for i, row in enumerate(ROWS)],
)
def test_verdicts(name, grid, args, status, rung, qualifiers, kind):
    det = launch(name, (grid,), **args())
    verdict = det.last_verdict
    (config,) = verdict.per_config
    assert (verdict.status, config.status, config.rung) == (status, status, rung)
    assert config.qualifiers == qualifiers == verdict.qualifiers
    assert (verdict.refusal and verdict.refusal.kind) == kind
    if status == "ok":
        scope = {
            "T0": "any-params-any-grid",
            "T1": "this-params-any-grid",
            "T1-launch": "this-params-this-grid",
        }[rung]
        assert verdict.scope == scope and det.last_reports == []
    if name == "two_slots":
        # the races at three programs and more are evidence only
        assert all(
            max(a[0], b[0]) >= 2 for a, b in (r.programs for r in config.grid_fragile)
        )


def test_the_global_barrier_is_a_fence_only_with_global_bits():
    """The reader records the barrier between the store and the reload; with
    a barrier of shared memory only (injected), the pair races."""
    args = dict(hist=i32(B), out=i32(B), N=B)
    det = RaceDetector(compile=True)
    traced = tilelens.trace(det)(K.store_fence_load)
    texts = []
    with inject_ttir(traced, lambda t: texts.append(t) or t):
        traced[(1,)](**args)
    assert det.last_status == "ok"
    (fence,) = parse_ttir(texts[0], multipath=True, while_regions=True).fences
    assert "ttg.barrier all" in texts[0]
    with inject_ttir(
        traced, lambda t: t.replace("ttg.barrier all", "ttg.barrier local")
    ):
        traced[(1,)](**args)
    assert det.last_status == "races"
    assert races(det) == {("RAW", "hist", 0, 0)}


def test_a_possible_race_beside_a_race_is_withheld():
    det = launch("table_mask_and_waw", (2,), flags=f32(2 * B), out=i32(B), both=i32(B))
    assert det.last_status == "races"
    assert {r.first.tensor_name for r in det.last_reports} == {"both"}
    (note,) = det.last_verdict.notes
    assert "possible race(s) under over-approximation were withheld" in note


def test_the_snapshot_bounds(monkeypatch):
    det = launch("scatter", (4,), idx=_perm(), out=i32(4 * B))
    assert det.last_status == "ok"
    monkeypatch.setattr(CompiledRaceDetector, "SNAPSHOT_MAX_ELEMENTS", 4 * B - 1)
    det = launch("scatter", (4,), idx=_perm(), out=i32(4 * B))
    assert det.last_verdict.refusal.kind == "snapshot-bound"
    monkeypatch.setattr(CompiledRaceDetector, "SNAPSHOT_MAX_ELEMENTS", 4 * B)
    monkeypatch.setattr(CompiledRaceDetector, "SNAPSHOT_CAP_BYTES", 4 * B)
    det = launch("scatter", (4,), idx=_perm(), out=i32(4 * B))
    assert det.last_verdict.refusal.kind == "indirect-address"
    assert "snapshot cap" in det.last_verdict.refusal.message


def test_a_timeout_is_unsupported(monkeypatch):
    monkeypatch.setattr(CompiledRaceDetector, "T0_TIMEOUT_MS", 1)
    monkeypatch.setattr(CompiledRaceDetector, "T1_TIMEOUT_MS", 1)
    idx = torch.arange(4 * B, dtype=torch.int32)
    idx[5] = 40
    det = launch("scatter", (4,), idx=idx, out=i32(4 * B))
    assert det.last_status == "unsupported"
    assert det.last_verdict.refusal.kind == "timeout"


def test_an_undecided_launch_grid_requery_records_no_race(monkeypatch):
    found = TwoCopySymbolicHBSolver.find_races

    def requery_times_out(self, **kw):
        if self.only_pairs is not None:
            raise UnsupportedSymbolicRaceQuery("undecided", reason_unknown="timeout")
        return found(self, **kw)

    monkeypatch.setattr(TwoCopySymbolicHBSolver, "find_races", requery_times_out)
    det = launch("two_slots", (2,), out=i32(2 * B))
    verdict = det.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "timeout")
    # the any-grid races are evidence only
    assert trace_module.launches[-1].records == [verdict] and not det.last_reports
    assert verdict.per_config[0].grid_fragile


def test_a_config_that_needs_two_ctas_is_unsupported_and_the_rest_decided():
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_ctas=1),
            triton.Config({"BLOCK": 16}, num_ctas=2),
        ],
        key=[],
    )
    @triton.jit
    def copy(x, out, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + offs, tl.load(x + offs))

    det = launch(copy, (4,), x=f32(64), out=f32(64))
    verdict = det.last_verdict
    ok, failed = verdict.per_config
    assert (ok.config["num_ctas"], ok.status, ok.rung) == (1, "ok", "T0"), ok.refusal
    assert (failed.config["num_ctas"], failed.status) == (2, "unsupported")
    assert failed.refusal.kind == verdict.refusal.kind == "compile-failed"
    assert "RaceDetector(compile=True, target=...)" in failed.refusal.message
    assert verdict.status == "unsupported"


def test_the_launch_holds_what_every_config_proved():
    @triton.autotune(
        configs=[triton.Config({"SLOTS": False}), triton.Config({"SLOTS": True})],
        key=[],
    )
    @triton.jit
    def rows(out, SLOTS: tl.constexpr):
        # with SLOTS, race-free for two programs only
        pid = tl.program_id(0)
        row = pid % 2 if SLOTS else pid
        tl.store(out + row * 16 + tl.arange(0, 16), 1)

    det = launch(rows, (2,), out=i32(2 * B))
    verdict = det.last_verdict
    assert [c.rung for c in verdict.per_config] == ["T0", "T1-launch"]
    assert (verdict.status, verdict.scope) == ("ok", "this-params-this-grid")


def test_a_race_outranks_a_config_that_failed_to_compile():
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 32}, num_ctas=1),
            triton.Config({"BLOCK": 32}, num_ctas=2),
        ],
        key=[],
    )
    @triton.jit
    def store_rows(out, BLOCK: tl.constexpr):
        tl.store(out + tl.program_id(0) * 16 + tl.arange(0, BLOCK), 1)

    det = launch(store_rows, (4,), out=i32(128))
    assert [c.status for c in det.last_verdict.per_config] == ["races", "unsupported"]
    assert det.last_status == "races"


def test_bindings_that_differ_in_grid_or_contents_are_solved_apart(monkeypatch):
    solves = []
    analyze = client_module.analyze_graph
    monkeypatch.setattr(
        client_module,
        "analyze_graph",
        lambda *a, **kw: solves.append(a[3]) or analyze(*a, **kw),
    )
    det = CompiledRaceDetector()
    out = i32(2 * B)
    facts = tensor_facts(out)

    def binding(grid):
        return LaunchBinding(
            params=MappingProxyType({}),
            tensors=MappingProxyType({"out": facts}),
            constexprs=MappingProxyType({}),
            raw_grid=grid,
            grid=grid,
            config=MappingProxyType({}),
        )

    bindings = [binding((2, 1, 1)), binding((3, 1, 1)), binding((2, 1, 1))]
    bindings.append(binding((2, 1, 1)))
    for b, value in zip(bindings, (0, 0, 0, 1)):
        det._contents[id(b)] = (b, {"out": LaunchTensor(facts, (value,) * 2 * B)})
    _, verdict, _ = det._analyze_config(
        None, K.graph("two_slots"), "text", {}, bindings, {}
    )
    # the third binding is the first's; the fourth differs in its contents
    assert solves == [(2, 1, 1), (3, 1, 1), (2, 1, 1)]
    assert verdict.status == "races"


def test_launches_on_two_threads_take_turns():
    """Z3's main context is not thread-safe: two host threads' traced
    launches (of two kernels: one JITFunction is traced by one launch at a
    time) are analyzed one at a time."""
    verdicts, errors = [], []

    def work(name, **args):
        try:
            for _ in range(5):
                verdicts.append(launch(name, (4,), **args).last_status)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=work, args=("inplace",), kwargs=dict(x=i32(4 * B))),
        threading.Thread(
            target=work,
            args=("add",),
            kwargs=dict(x=f32(64), y=f32(64), out=f32(64), n=64),
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and verdicts == ["ok"] * 10


def test_undecided_parts_always_print_once_and_proofs_only_under_verbose(
    monkeypatch, capsys
):
    monkeypatch.setattr(cfg, "verbose", False)
    det = RaceDetector(compile=True)
    for _ in range(3):
        launch("counted_while", (1,), detector=det, x=i32(1), n=4)
    launch("add", (2,), x=f32(64), y=f32(64), out=f32(64), n=64)
    out = capsys.readouterr().out
    assert out.count("not analyzed: spin-shape") == 1
    assert "race-free" not in out
    monkeypatch.setattr(cfg, "verbose", True)
    launch("add", (2,), x=f32(64), y=f32(64), out=f32(64), n=64)
    assert "race-free" in capsys.readouterr().out


def test_autotune_configs_are_folded_and_identical_ones_solved_once(monkeypatch):
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=2),
            triton.Config({"BLOCK": 16}, num_warps=4),
            triton.Config({"BLOCK": 32}, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def store_rows(out, BLOCK: tl.constexpr):
        # BLOCK=32 programs overlap on 16-element rows
        tl.store(out + tl.program_id(0) * 16 + tl.arange(0, BLOCK), 1)

    solves = []
    analyze = client_module.analyze_graph
    monkeypatch.setattr(
        client_module,
        "analyze_graph",
        lambda *a, **kw: solves.append(a[0].kernel_name) or analyze(*a, **kw),
    )
    det = launch(store_rows, (4,), out=i32(128))
    configs = det.last_verdict.per_config
    assert [(c.config["BLOCK"], c.config["num_warps"], c.status) for c in configs] == [
        (16, 2, "ok"),
        (16, 4, "ok"),
        (32, 4, "races"),
    ]
    assert len(solves) == 2  # the two BLOCK=16 configs share a TTIR
    assert det.last_status == "races"
    assert {r.config["BLOCK"] for r in det.last_reports} == {32}


def test_the_verdict_and_races_survive_save_and_load(tmp_path, monkeypatch):
    det = launch("two_slots", (3,), out=i32(2 * B))
    det = launch("two_slots", (2,), detector=det, out=i32(2 * B))
    monkeypatch.setattr(trace_module, "launches", trace_module.launches[-2:])
    saved = [list(launch.records) for launch in trace_module.launches]
    tilelens.load(tilelens.save(tmp_path / "trace.zip"))
    assert [list(launch.records) for launch in trace_module.launches] == saved
    assert saved[0][-1].status == "races"
    assert saved[1][-1].per_config[0].grid_fragile


# ── fenced CAS publications and the comm/comp pattern ──


@K.real_jit
def guarded_acq_rel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.store(data_ptr, 1, mask=pid == 0)
    tl.debug_barrier()
    old = tl.atomic_cas(flag_ptr, tl.where(pid == 0, 0, 1), 1, sem="acq_rel")
    tl.debug_barrier()
    x = tl.load(data_ptr, mask=(pid == 1) & (old == 1), other=0)
    tl.store(out_ptr + pid, x, mask=(pid == 1) & (old == 1))


@K.real_jit
def acquire_on_failure(flag_ptr, data_ptr, out_ptr):
    # the consumer's CAS never succeeds (cmp 7), yet reading the release
    # acquires it
    pid = tl.program_id(0)
    tl.store(data_ptr, 1, mask=pid == 0)
    tl.debug_barrier()
    old = tl.atomic_cas(flag_ptr, tl.where(pid == 0, 0, 7), 1, sem="acq_rel")
    tl.debug_barrier()
    x = tl.load(data_ptr, mask=(pid == 1) & (old == 1), other=0)
    tl.store(out_ptr + pid, x, mask=(pid == 1) & (old == 1))


@K.real_jit
def role_specific_order(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(data_ptr, 1)
        tl.debug_barrier()
        tl.atomic_cas(flag_ptr, 0, 1, sem="release")
    else:
        old = tl.atomic_cas(flag_ptr, 1, 1, sem="acquire")
        tl.debug_barrier()
        value = tl.load(data_ptr, mask=old == 1, other=0)
        tl.store(out_ptr + pid, value, mask=old == 1)


@K.real_jit
def cas_unlock_mutex(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    while tl.atomic_cas(flag_ptr, 0, 1, sem="acquire") != 0:
        pass
    tl.debug_barrier()
    tl.store(data_ptr, tl.load(data_ptr) + 1)
    tl.debug_barrier()
    tl.atomic_cas(flag_ptr, 1, 0, sem="release")
    tl.store(out_ptr + pid, 1)


@K.real_jit
def fenced_tile_handoff(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tl.store(data_ptr + offs, offs + 1, mask=pid == 0)
    tl.debug_barrier()
    old = tl.atomic_cas(flag_ptr, tl.where(pid == 0, 0, 1), 1, sem="acq_rel")
    tl.debug_barrier()
    value = tl.load(data_ptr + offs, mask=(pid == 1) & (old == 1), other=0)
    tl.store(out_ptr + offs, value, mask=(pid == 1) & (old == 1))


@pytest.mark.parametrize(
    "kernel",
    [
        guarded_acq_rel,
        acquire_on_failure,
        role_specific_order,
        cas_unlock_mutex,
        fenced_tile_handoff,
    ],
)
def test_fenced_cas_publications_are_proved(kernel):
    kw = {"BLOCK": B} if kernel is fenced_tile_handoff else {}
    det = launch(kernel, (2,), flag_ptr=i32(1), data_ptr=i32(B), out_ptr=i32(B), **kw)
    assert det.last_status == "ok", det.last_verdict
    assert det.last_reports == []


def test_comm_comp_is_proved_assuming_termination():
    det = launch("comm_comp", (3,), sem=i32(1), payload=f32(B), out=f32(2 * B))
    assert det.last_status == "ok"
    assert TERM in det.last_verdict.qualifiers


# ── entry points ──


def test_the_cli_compile_flag_runs_the_compiled_race_detector(tmp_path):
    script = tmp_path / "waw.py"
    script.write_text(
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def waw(out, BLOCK: tl.constexpr):\n"
        "    tl.store(out + tl.arange(0, BLOCK), tl.program_id(0))\n"
        "waw[(2,)](torch.zeros(16, dtype=torch.int32), BLOCK=16)\n"
        "print('launch returned')\n"
    )
    cli = (
        f"import sys; sys.argv = ['tile-race', '--compile', {str(script)!r}]; "
        "from tilelens.wrapper import apply_race_detector; apply_race_detector()"
    )
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-c", cli], capture_output=True, text=True, env=env, cwd=REPO
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("(compiled race detector)") == 1
    assert f"File: {script}, Line: 4, in waw" in proc.stdout
    assert "launch returned" in proc.stdout
