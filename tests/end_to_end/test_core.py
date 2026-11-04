import inspect
import warnings

import pytest
import torch

import triton
import triton.language as tl

import tilelens
from tilelens.clients import Profiler, Sanitizer, Tracer
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import Client
from tilelens.core.data import Barrier, InterpreterDivergence
import tilelens.core.trace as trace_module


# ======== Trace Decorator Tests =========
def test_trace_decorator_add_clients():
    """
    Test goal:
    1. Apply @trace("sanitizer") and @trace("profiler") to add the Sanitizer and Profiler clients.
    2. Apply @trace("tracer") to append a Tracer client.
    3. Apply @trace(("sanitizer",)) with a duplicate Sanitizer, which should be
       ignored by the de-duplication logic.

    The final Trace object should contain exactly one instance each of
    Sanitizer, Profiler, and Tracer (total = 3 clients).
    """

    @tilelens.trace("sanitizer")
    @tilelens.trace("profiler")
    @tilelens.trace("tracer")
    @tilelens.trace(
        Sanitizer(abort_on_error=True)
    )  # Duplicate Sanitizer (should be ignored)
    @triton.jit
    def my_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

    # Should be wrapped as a Trace object.
    from tilelens.core.trace import TritonTrace

    assert isinstance(my_kernel, TritonTrace)

    # Verify client de-duplication and addition logic
    clients = my_kernel.client_manager.clients
    assert len(clients) == 3
    assert sum(c == "sanitizer" for c in clients) == 1
    assert sum(c == "profiler" for c in clients) == 1
    assert sum(c == "tracer" for c in clients) == 1


def test_trace_decorator_supports_gluon_frontend():
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl
    from tilelens.core.trace import GluonTrace

    @tilelens.trace("tracer", frontend="gluon")
    @gluon.jit
    def my_gluon_kernel(out):
        pid = ttgl.program_id(0)
        ttgl.store(out + pid, pid)

    assert isinstance(my_gluon_kernel, GluonTrace)
    assert my_gluon_kernel.client_manager.get_client("tracer") is not None


def test_gluon_trace_handles_autotuner_wrapper():
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl
    from triton.runtime import Autotuner
    from tilelens.core.simulation.gluon import GluonInterpretedFunction
    from tilelens.core.trace import GluonTrace

    @tilelens.trace("tracer", frontend="gluon")
    @triton.autotune(configs=[triton.Config({"BLOCK": 1})], key=[])
    @gluon.jit
    def my_gluon_kernel(out, BLOCK: ttgl.constexpr):
        pid = ttgl.program_id(0)
        ttgl.store(out + pid, pid)

    assert isinstance(my_gluon_kernel, GluonTrace)
    assert isinstance(my_gluon_kernel.runner, Autotuner)
    assert isinstance(my_gluon_kernel.runner.fn, GluonInterpretedFunction)
    assert my_gluon_kernel.runner.fn.fn is my_gluon_kernel.base_fn


def test_gluon_trace_handles_heuristics_wrapper():
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl
    from triton.runtime.autotuner import Heuristics
    from tilelens.core.simulation.gluon import GluonInterpretedFunction
    from tilelens.core.trace import GluonTrace

    @tilelens.trace("tracer", frontend="gluon")
    @triton.heuristics({"BLOCK": lambda args: 1})
    @gluon.jit
    def my_gluon_kernel(out, BLOCK: ttgl.constexpr):
        pid = ttgl.program_id(0)
        ttgl.store(out + pid, pid)

    assert isinstance(my_gluon_kernel, GluonTrace)
    assert isinstance(my_gluon_kernel.runner, Heuristics)
    assert isinstance(my_gluon_kernel.runner.fn, GluonInterpretedFunction)
    assert my_gluon_kernel.runner.fn.fn is my_gluon_kernel.base_fn


def test_gluon_sanitizer_run_preserves_instrumentation_mode(monkeypatch):
    from triton import knobs
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl

    @tilelens.trace(client=Sanitizer(abort_on_error=False), frontend="gluon")
    @gluon.jit
    def my_gluon_kernel(out):
        pid = ttgl.program_id(0)
        ttgl.store(out + pid, pid)

    seen_modes = []
    saved_mode = knobs.compilation.instrumentation_mode

    def fake_run(*args, **kwargs):
        seen_modes.append(knobs.compilation.instrumentation_mode)
        return "compiled"

    monkeypatch.setattr(my_gluon_kernel.runner, "run", fake_run)

    try:
        ret = my_gluon_kernel[(1,)](object())
    finally:
        knobs.compilation.instrumentation_mode = saved_mode

    assert ret == "compiled"
    assert seen_modes == [saved_mode]
    assert knobs.compilation.instrumentation_mode == saved_mode


class _PreRunSkippingClient(Client):
    NAME = "pre_run_skipping"

    def __init__(self):
        super().__init__()
        self.pre_run_calls = 0
        self.post_run_calls = 0

    def pre_run_callback(self, fn):
        self.pre_run_calls += 1
        return False

    def post_run_callback(self, fn):
        self.post_run_calls += 1
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid):
        pass

    def grid_idx_callback(self, grid_idx):
        pass

    def register_op_callback(self, op_type, *args, **kwargs):
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        return []


def test_gluon_run_does_not_use_pre_run_as_launch_gate(monkeypatch):
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl

    client = _PreRunSkippingClient()

    @tilelens.trace(client=client, frontend="gluon")
    @gluon.jit
    def my_gluon_kernel(out):
        pid = ttgl.program_id(0)
        ttgl.store(out + pid, pid)

    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        return "compiled"

    monkeypatch.setattr(my_gluon_kernel.runner, "run", fake_run)

    ret = my_gluon_kernel[(1,)](object())

    assert ret == "compiled"
    assert len(run_calls) == 1
    assert client.pre_run_calls == 0
    assert client.post_run_calls == 1


# ======== Unpatch Tests =========
def test_unpatch_lang_restores_builtins():
    @triton.jit
    def dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs))

    # e2e run: make sure jit'd triton kernel can run after tracing
    if not torch.cuda.is_available():
        pytest.skip("cuda required for triton kernel execution")
    size = 16
    block_size = 8
    x = torch.arange(size, device="cuda")
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    for client in ["tracer", "sanitizer", "profiler"]:
        traced = tilelens.trace(client)(dummy_kernel)
        traced[grid](x, BLOCK_SIZE=block_size)
        dummy_kernel[grid](x, BLOCK_SIZE=block_size)


def test_trace_patches_extra_cuda_builtins():
    @tilelens.trace("tracer")
    @triton.jit
    def extra_cuda_builtin_kernel(out):
        offs = tl.arange(0, tl.extra.cuda.num_threads())
        tl.store(out + offs, offs)

    out = torch.empty((128,), dtype=torch.int32)
    extra_cuda_builtin_kernel[(1,)](out, num_warps=4)
    torch.testing.assert_close(out, torch.arange(128, dtype=torch.int32))


def test_trace_supports_symbolic_sort():
    @tilelens.trace("sanitizer")
    @triton.jit
    def sort_kernel(inp, out, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        values = tl.load(inp + offs)
        values = tl.sort(values, 0)
        tl.store(out + offs, values)

    inp = torch.tensor([3, 1, 7, 0, 2, 5, 4, 6], dtype=torch.int32)
    out = torch.empty_like(inp)
    sort_kernel[(1,)](inp, out, BLOCK=inp.numel())


# ======== Nested JIT Call Tests =========
@tilelens.trace(client=Sanitizer(abort_on_error=True))
@triton.jit
def trace_nested_inner_kernel(x):
    return x * 2


@tilelens.trace(client=Profiler())
@triton.jit
def trace_nested_inner_profiler_kernel(x):
    return x * 2


def test_trace_nested_jit_calls():
    """
    Test that Trace class properly handles nested JIT function calls via __call__ method.

    When a traced JIT function is called from within another JIT function,
    the Trace wrapper needs to properly delegate to the underlying function.
    This test ensures compatibility with the command line tile-sanitizer wrapper.
    """

    @tilelens.trace(client=Sanitizer(abort_on_error=True))
    @triton.jit
    def trace_nested_call_kernel(ptr, n: tl.constexpr):
        x = tl.load(ptr + tl.arange(0, n))
        y = trace_nested_inner_kernel(
            x
        )  # This nested call requires __call__ method when wrapped by trace
        tl.store(ptr + tl.arange(0, n), y)

    # Test execution
    data = torch.ones(8)
    trace_nested_call_kernel[(1,)](data, 8)


def test_trace_nested_jit_calls_mismatched_clients():
    @tilelens.trace(client=Sanitizer(abort_on_error=True))
    @triton.jit
    def trace_nested_call_kernel(ptr, n: tl.constexpr):
        x = tl.load(ptr + tl.arange(0, n))
        y = trace_nested_inner_profiler_kernel(x)
        tl.store(ptr + tl.arange(0, n), y)

    data = torch.ones(8)
    with pytest.raises(
        RuntimeError, match="nested traced calls require matching clients"
    ):
        trace_nested_call_kernel[(1,)](data, 8)


# ======== Autotuner Compatibility =========
if torch.cuda.is_available():  # Only test if CUDA is available

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        ],
        key=["n_elements"],
    )
    @tilelens.trace(client=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel_no_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """
        A Triton kernel that loads and stores values without boundary checks (mask).
        This can lead to out-of-bound access if n_elements exceeds the buffer size.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # No mask is applied here, so loading/storing beyond the valid range can occur.
        x_val = tl.load(x_ptr + offsets)
        y_val = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x_val + y_val)

    def test_autotune_add_inrange():
        """
        This test uses n_elements = 128, matching the size of the input tensors.
        It should NOT cause any out-of-bound access.
        """
        x = torch.randn(128)
        y = torch.randn(128)
        out = torch.empty_like(x)

        # The kernel launch uses n_elements=128, aligned with the tensor size.
        grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
        add_kernel_no_mask[grid](x_ptr=x, y_ptr=y, out_ptr=out, n_elements=128)

    def test_autotune_add_out_of_bound():
        """
        This test deliberately sets n_elements = 256, exceeding the actual buffer size (128).
        It will likely cause out-of-bound reads/writes, which may trigger errors or warnings.
        """
        x = torch.randn(128)
        y = torch.randn(128)
        out = torch.empty_like(x)

        # The kernel launch uses n_elements=256, exceeding the valid tensor size.
        grid = lambda META: (triton.cdiv(256, META["BLOCK_SIZE"]),)
        with pytest.raises(SystemExit):
            add_kernel_no_mask[grid](x_ptr=x, y_ptr=y, out_ptr=out, n_elements=256)


# ======== Autotuner + Interpreter Mode =========
def test_autotune_interpreter_mode():
    """
    Test that tracing an autotuned kernel does not crash in interpreter mode
    (TRITON_INTERPRET=1), where jit_fn is None and warmup should be skipped.
    """

    @triton.autotune(configs=[triton.Config({"BLOCK": 32})], key=["n"])
    @triton.jit
    def noop_kernel(n, BLOCK: tl.constexpr):
        pass

    traced = tilelens.trace(client=Sanitizer())(noop_kernel)
    traced[(1,)](n=32)


# ======== Barrier and Interpreter-Divergence Ops =========
class _OpRecorder(Tracer):
    def __init__(self):
        super().__init__()
        self.ops: list = []

    def register_op_callback(self, op_type):
        if op_type in (Barrier, InterpreterDivergence):
            return OpCallbacks(before_callback=lambda *a: self.ops.append((op_type, a)))
        return super().register_op_callback(op_type)


@triton.jit
def _block_truthiness_kernel(out):
    offs = tl.arange(0, 8)
    m = (offs < 2) and (offs < 4)  # diverges
    m = m or (offs < 1)  # diverges
    v = 1.0 if offs < 4 else 2.0  # diverges
    if not (offs < 4):  # diverges
        v = 3.0
    if tl.arange(0, 1) < 1:  # a one-element block does not diverge
        tl.store(out + offs, v, mask=m)


@triton.jit
def _scalar_if_and_mask_kernel(x, out, n):
    offs = tl.arange(0, 8)
    if n > 0:
        tl.store(out + offs, tl.load(x + offs, mask=offs < 4, other=0.0), mask=offs < 4)
    tl.debug_barrier()


def test_block_truthiness_and_barrier_reach_clients_as_ops():
    recorder = _OpRecorder()
    tilelens.trace(client=recorder)(_block_truthiness_kernel)[(1,)](torch.zeros(8))
    src, first = inspect.getsourcelines(_block_truthiness_kernel.fn)
    lines = [first + i for i, ln in enumerate(src) if "# diverges" in ln]
    assert [(op, a[0], a[1][1]) for op, a in recorder.ops] == [
        (InterpreterDivergence, "block-truthiness", line) for line in lines
    ]
    # scalar `if n > 0` and Triton's own `if mask and ...` are no divergence
    recorder = _OpRecorder()
    args = (torch.zeros(8), torch.zeros(8), 3)
    tilelens.trace(client=recorder)(_scalar_if_and_mask_kernel)[(2,)](*args)
    assert recorder.ops == [(Barrier, ()), (Barrier, ())]


def _profiler():
    profiler = Profiler()
    profiler.disable_buffer_load_check = True  # its warmup needs a GPU
    return profiler


@pytest.mark.parametrize("make_client", [Sanitizer, Tracer, _profiler])
def test_block_truthiness_noticed_once_per_launch(make_client, tmp_path, monkeypatch):
    client = make_client()  # the sanitizer with abort_on_error=True
    traced = tilelens.trace(client=client)(_block_truthiness_kernel)
    grids = [(2,), (1,), (1,)]  # the sanitizer must not skip the repeat
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("default")  # Python's own, not pytest.warns' always
        for grid in grids:
            traced[grid](torch.zeros(8))
    notices = [w for w in warned if "triton-lang/triton/pull/12150" in str(w.message)]
    assert len(notices) == len(grids)
    if make_client is Sanitizer:
        verdict = client.unsupported
        assert verdict.status == "unsupported"
        assert verdict.refusal.kind == "block-truthiness"
        ours = trace_module.launches[-len(grids) :]
        assert [launch.records for launch in ours] == [[verdict]] * len(grids)
        monkeypatch.setattr(trace_module, "launches", ours[-1:])  # save only ours
        tilelens.load(tilelens.save(tmp_path / "trace.zip"))
        assert trace_module.launches[-1].records == [verdict]
