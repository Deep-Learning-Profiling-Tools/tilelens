from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import ast
import inspect
from queue import Empty, SimpleQueue
import sys
import threading
import time
from typing import Any, ClassVar, cast
import warnings

import numpy as np
import triton.language as tl
from triton import knobs
from triton.runtime import JITFunction
from triton.runtime.interpreter import (
    ASTTransformer as _OrigASTTransformer,
    FunctionRewriter,
    GridExecutor,
    TensorHandle,
    _implicit_cvt,
    _patch_builtin,
    _patch_lang as triton_patch_lang,
    _get_signed_np_dtype,
    _rewrap_tensor,
    _tuple_create,
    _unwrap_tensor,
    interpreter_builder,
    interpreter_semantic,
)
from triton.tools.tensor_descriptor import TensorDescriptor

from ..config import config as cfg
from ..data import (
    AddPtr,
    Advance,
    Ashr,
    AtomicCas,
    AtomicRMW,
    BinaryOp,
    Bitcast,
    Broadcast,
    CastImpl,
    CumSum,
    Dot,
    ExpandDims,
    Fabs,
    Fma,
    FpToFp,
    Idiv,
    IntToPtr,
    Join,
    Load,
    MakeBlockPointer,
    MakeRange,
    Op,
    ProgramId,
    PtrToInt,
    RawLoad,
    RawStore,
    ReduceMax,
    ReduceMin,
    ReduceOr,
    ReduceSum,
    ReduceXor,
    Reshape,
    Rsqrt,
    Sort,
    Splat,
    Split,
    Store,
    TensorPointerLoad,
    TensorPointerStore,
    TernaryOp,
    Trans,
    Umulhi,
    Unsplat,
    UnaryOp,
)
from ..symbolic_metadata import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    FLOAT8_E4B8,
    FLOAT8_E4M3,
    FLOAT8_E4M3FN,
    FLOAT8_E5M2,
    FLOAT8_E5B16,
    INT1,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    SymbolicDType,
    SymbolicPointerDType,
    SymbolicScalarDType,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    block_type as symbolic_block_type,
    dtype_to_numpy,
    pointer_type as symbolic_pointer_type,
)
from ..patch import LoopSite, loop_file_token
from .base import (
    AdapterResult,
    Frontend,
    _LangPatchScope,
    _MISSING,
    register_frontend,
)


TRITON_NAMESPACES: dict[Any, dict[str, type[Op]]] = {
    interpreter_builder: {
        "create_get_program_id": ProgramId,
        "create_store": RawStore,
        "create_masked_store": Store,
        "create_load": RawLoad,
        "create_masked_load": Load,
        "create_dot": Dot,
        "unary_op": UnaryOp,
        "binary_op": BinaryOp,
        "ternary_op": TernaryOp,
        "create_fma": Fma,
        "create_make_range": MakeRange,
        "create_addptr": AddPtr,
        "create_expand_dims": ExpandDims,
        "create_broadcast": Broadcast,
        "create_splat": Splat,
        "create_unsplat": Unsplat,
        "create_make_block_ptr": MakeBlockPointer,
        "create_tensor_pointer_load": TensorPointerLoad,
        "create_tensor_pointer_store": TensorPointerStore,
        "create_idiv": Idiv,
        "create_rsqrt": Rsqrt,
        "cast_impl": CastImpl,
        "create_reshape": Reshape,
        "create_trans": Trans,
        "create_join": Join,
        "create_split": Split,
        "create_fabs": Fabs,
        "create_ashr": Ashr,
        "create_advance": Advance,
        "create_fp_to_fp": FpToFp,
        "create_umulhi": Umulhi,
        "create_bitcast": Bitcast,
        "create_ptr_to_int": PtrToInt,
        "create_int_to_ptr": IntToPtr,
        "create_atomic_cas": AtomicCas,
        "create_atomic_rmw": AtomicRMW,
    },
    tl: {
        "max": ReduceMax,
        "min": ReduceMin,
        "sum": ReduceSum,
        "xor_sum": ReduceXor,
        "reduce_or": ReduceOr,
        "sort": Sort,
        "cumsum": CumSum,
    },
    tl.standard: {
        "max": ReduceMax,
        "min": ReduceMin,
        "sum": ReduceSum,
        "xor_sum": ReduceXor,
        "reduce_or": ReduceOr,
    },
    tl.math: {
        "umulhi": Umulhi,
    },
}


TRITON_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {
    ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
    RawStore: lambda ptr, value, *_args, **_kwargs: AdapterResult(ptr, value),
    Store: lambda ptr, _value, mask, *_args, **kwargs: AdapterResult(
        ptr,
        mask,
        kwargs.get("keys"),
    ),
    RawLoad: lambda ptr, *_args, **_kwargs: AdapterResult(ptr),
    Load: lambda ptr, mask, _other, *_args, **kwargs: AdapterResult(
        ptr,
        mask,
        kwargs.get("keys"),
    ),
    AtomicRMW: lambda _rmw_op, ptr, _val, mask, *_args, **_kwargs: AdapterResult(
        ptr, mask
    ),
    AtomicCas: lambda ptr, *_args, **_kwargs: AdapterResult(ptr),
    Dot: lambda a, b, *_args, **_kwargs: AdapterResult(a, b),
    ReduceSum: lambda input_tensor,
    axis=None,
    keep_dims=False,
    *_args,
    **_kwargs: AdapterResult(input_tensor, axis, keep_dims),
    AddPtr: lambda ptr, offset, *_args, **_kwargs: AdapterResult(ptr, offset),
}


@dataclass(frozen=True)
class FakeTensor:
    _data_ptr: int
    dtype: str
    shape: tuple[int, ...] = ()
    _stride: tuple[int, ...] = ()
    _is_contiguous: bool = True
    _element_size: int = 1
    device: str = "fake_tensor"

    def data_ptr(self) -> int:
        return self._data_ptr

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def is_contiguous(self) -> bool:
        return self._is_contiguous

    def numel(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def element_size(self) -> int:
        return self._element_size


class TritonFrontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="triton",
            builder=interpreter_builder,
            namespaces=TRITON_NAMESPACES,
            adapters=TRITON_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )
        self._thread_local_interpreter_state = threading.local()
        self._thread_local_interpreter_state.grid_idx = None
        self._current_client_manager = None
        self._loop_wrapper_arg = "_tilelens_loop_iter_wrapper"
        self._loop_ast_methods: dict[str, Callable | object] = {}
        self._loop_ast_patched = False
        self._patch_calls_scope = 0
        # These targets are stable for the process; compute them once so each
        # launch only snapshots current values, not the list of attributes.
        self._triton_language_targets = tuple(self._triton_language_attr_targets())
        self._triton_extra_modules = self._triton_extra_builtin_modules()
        self._triton_builtin_attr_cache: dict[int, tuple[str, ...]] = {}
        # Most patched ops are interpreter-builder ops and can call overriders
        # directly. Cache the smaller frontend-wrapped sets for quick routing.
        self._math_op_types = frozenset(self.namespaces[tl.math].values())
        self._tl_op_types = frozenset(self.namespaces[tl].values())
        self._frontend_wrapped_op_types = self._math_op_types | self._tl_op_types
        self._bind_interpreter_builder_thread_state()

    def _bind_interpreter_builder_thread_state(self) -> None:
        def set_grid_idx(_builder, x: int, y: int, z: int) -> None:
            self._thread_local_interpreter_state.grid_idx = (x, y, z)

        # Triton's interpreter builder is a process-global singleton. Store grid
        # coordinates in TileLens state so concurrent block workers do not race
        # through one shared builder attribute.
        interpreter_cls = interpreter_builder.__class__
        interpreter_cls.set_grid_idx = set_grid_idx  # type: ignore[attr-defined]
        interpreter_cls.grid_idx = property(  # type: ignore[attr-defined]
            lambda _builder: self._thread_local_interpreter_state.grid_idx
        )

    def _visit_triton_for(self, transformer: ast.NodeTransformer, node: ast.For):
        transformer.generic_visit(node)

        # Preserve which range spelling the user wrote so loop clients can
        # distinguish Python loops, `tl.range`, and `tl.static_range`.
        range_type = "unknown"
        if isinstance(node.iter, ast.Call):
            func = node.iter.func
            if isinstance(func, ast.Name) and func.id == "range":
                range_type = "python_range"
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "tl"
            ):
                if func.attr == "range":
                    range_type = "tl_range"
                elif func.attr == "static_range":
                    range_type = "tl_static_range"

        # Rebuild the loop iterator as a callable plus explicit args. This lets
        # range-wrapper clients replace the iterable before the original call is
        # evaluated, which is needed for symbolic loop bounds.
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

        return ast.fix_missing_locations(
            ast.For(
                target=node.target,
                iter=ast.Call(
                    # `_loop_wrapper_arg` is injected as a hidden keyword-only
                    # default by `_visit_triton_function_def`, avoiding any
                    # helper name in user kernel globals.
                    func=ast.Name(id=self._loop_wrapper_arg, ctx=ast.Load()),
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
        )

    def _visit_triton_function_def(
        self, transformer: ast.NodeTransformer, node: ast.FunctionDef
    ):
        node = cast(ast.FunctionDef, transformer.generic_visit(node))
        # Triton compiles rewritten kernels with a local namespace, but loop
        # bodies resolve names from the function defaults/globals. Add the
        # wrapper as a defaulted kw-only arg so the rewritten loop can call it
        # without mutating the user's global dictionary.
        if self._loop_wrapper_arg in {
            arg.arg
            for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
        }:
            return node

        node.args.kwonlyargs.append(ast.arg(arg=self._loop_wrapper_arg))
        node.args.kw_defaults.append(
            ast.Name(id=self._loop_wrapper_arg, ctx=ast.Load())
        )
        return ast.fix_missing_locations(node)

    def loop_iter_wrapper(
        self,
        iterable_callable: Callable,
        iter_args: Any,
        iter_kwargs: Any,
        lineno: int,
        range_type: str,
    ):
        client_manager = self.current_client_manager()
        if client_manager is None:
            # Device-function rewrites can run outside a top-level traced launch.
            # In that case, preserve Triton's normal behavior and just evaluate
            # the original iterable.
            args = tuple(iter_args) if iter_args is not None else ()
            kwargs = dict(iter_kwargs) if iter_kwargs is not None else {}
            return iterable_callable(*args, **kwargs)
        # The rewritten lineno is function-relative (Triton parses the kernel
        # source with `def` at line 1), so loops in different files can share
        # it. The caller frame is the rewritten kernel, compiled under the
        # kernel's real filename; combine both pieces into the loop identity.
        caller_file = sys._getframe(1).f_code.co_filename
        loop_site = LoopSite(lineno, loop_file_token(caller_file))
        return client_manager.loop_iter_wrapper(
            iterable_callable,
            iter_args,
            iter_kwargs,
            loop_site,
            range_type,
        )

    @staticmethod
    def _triton_language_attr_targets() -> list[tuple[Any, tuple[str, ...]]]:
        tensor_attrs = ("__index__", "__bool__", "__repr__", "__str__", "T")
        lang_attrs = (
            "range",
            "static_range",
            "static_assert",
            "static_print",
            "multiple_of",
            "max_contiguous",
            "max_constancy",
            "reduce",
            "associative_scan",
        )

        targets: list[tuple[Any, tuple[str, ...]]] = []
        for lang in (tl, tl.core):
            targets.append((lang.tensor, tensor_attrs))
            targets.append((lang, lang_attrs))
            targets.append((lang.dtype, ("to_ir",)))
            if lang == tl:
                targets.append((lang.math, ()))
        targets.append((tl.core.tensor_descriptor_base, ()))
        return targets

    def _triton_snapshot_scope(self, fn: Callable[..., Any]) -> _LangPatchScope:
        """
        Stores Triton attributes into a LangPatchScope for later unpatching.
        This is to be run before patching with the interpreter.
        """

        scope = _LangPatchScope()
        # Triton's interpreter patch mutates global tl/tl.core objects. Snapshot
        # both unconditionally so nested or generated JIT functions without a
        # direct `tl` global still restore the language state after tracing.
        for obj, attrs in self._triton_language_targets:
            obj_id = id(obj)
            builtin_names = self._triton_builtin_attr_cache.get(obj_id)
            if builtin_names is None:
                builtin_names = tuple(
                    name
                    for name, member in inspect.getmembers(obj)
                    if tl.core.is_builtin(member)
                )
                self._triton_builtin_attr_cache[obj_id] = builtin_names
            for name in builtin_names:
                scope.set_attr(obj, name, getattr(obj, name))
            for attr in attrs:
                scope.save_attr(obj, attr)

        return scope

    @staticmethod
    def _triton_extra_builtin_modules() -> tuple[Any, ...]:
        modules = []
        extra = getattr(tl, "extra", None)
        if extra is None:
            return ()
        for name in ("cuda", "hip"):
            module = getattr(extra, name, None)
            if module is not None:
                modules.append(module)
        return tuple(modules)

    @staticmethod
    def _patch_triton_inline_asm(scope: _LangPatchScope) -> None:
        warned = False

        def _warn_inline_asm_approximation_once() -> None:
            nonlocal warned
            if warned:
                return
            warned = True
            warnings.warn(
                "Triton inline assembly is approximated in trace mode by returning "
                "input tensor values; traced values may differ from the numeric result "
                "of the real inline assembly.",
                RuntimeWarning,
                stacklevel=3,
            )

        def _inline_asm_elementwise_fallback(
            asm,
            constraints,
            args,
            dtype,
            is_pure,
            pack,
            **kwargs,
        ):
            _warn_inline_asm_approximation_once()
            if isinstance(dtype, (list, tuple)):
                return tuple(args[0].to(_dtype) for _dtype in dtype)
            return args[0].to(dtype)

        scope.set_attr(tl, "inline_asm_elementwise", _inline_asm_elementwise_fallback)

    @staticmethod
    def _patch_triton_semantic_to_tensor(scope: _LangPatchScope) -> None:
        original_to_tensor = interpreter_semantic.to_tensor

        def _to_tensor_symbolic_aware(x, check_type=True):
            unwrapped = x.value if isinstance(x, tl.constexpr) else x
            if isinstance(unwrapped, tl.core.tensor):
                return unwrapped
            return original_to_tensor(x, check_type)

        scope.set_attr(interpreter_semantic, "to_tensor", _to_tensor_symbolic_aware)

    def _init_args_hst(self, args_dev, kwargs):
        def _to_cpu(arg):
            if isinstance(arg, tuple):
                return _tuple_create(arg, map(_to_cpu, arg))
            elif isinstance(arg, TensorDescriptor):
                return TensorDescriptor(
                    _to_cpu(arg.base),
                    arg.shape,
                    arg.strides,
                    arg.block_shape,
                )
            elif not hasattr(arg, "data_ptr"):
                return arg

            unwrapped_arg = _unwrap_tensor(arg)
            cpu_arg = FakeTensor(
                _data_ptr=unwrapped_arg.data_ptr(),
                dtype=unwrapped_arg.dtype,
                shape=unwrapped_arg.shape,
                _stride=unwrapped_arg.stride(),
                _is_contiguous=unwrapped_arg.is_contiguous(),
                _element_size=unwrapped_arg.element_size(),
            )
            cpu_arg = _rewrap_tensor(cpu_arg, original_tensor=arg)
            return cpu_arg

        args_hst = [_to_cpu(arg) for arg in args_dev]

        kwargs_hst = {}
        for key, value in kwargs.items():
            kwargs_hst[key] = _to_cpu(value)
        return args_hst, kwargs_hst

    def current_client_manager(self):
        return self._current_client_manager

    def maybe_yield_for_multism(self) -> None:
        if cfg.num_sms <= 1:
            return

        # Periodically sleep briefly so other worker threads can run.
        yield_interval_sec = 0.005  # Request GIL handoff roughly every 5ms.
        yield_sleep_sec = 0.0005  # Small positive sleep to encourage OS-level yield.
        now = time.perf_counter()
        last = getattr(self._thread_local_interpreter_state, "last_yield_ts", 0.0)
        if now - last >= yield_interval_sec:
            self._thread_local_interpreter_state.last_yield_ts = now
            time.sleep(yield_sleep_sec)

    @staticmethod
    def concrete_fn_for_op_type(
        namespace_ops: dict[str, Callable], op_type: type[Op], original_op: Callable
    ) -> Callable:
        if op_type == RawLoad:
            return namespace_ops["create_masked_load"]
        if op_type == RawStore:
            return namespace_ops["create_masked_store"]
        if op_type == Ashr:
            return partial(TritonFrontend._concrete_ashr, namespace_ops["binary_op"])
        return original_op

    @staticmethod
    def _concrete_ashr(binary_op: Callable, lhs: Any, rhs: Any) -> Any:
        lhs_signed = TensorHandle(
            lhs.data.astype(_get_signed_np_dtype(lhs.data.dtype)),
            lhs.dtype,
            dict(lhs.attr),
        )
        rhs_signed = TensorHandle(
            rhs.data.astype(_get_signed_np_dtype(rhs.data.dtype)),
            rhs.dtype,
            dict(rhs.attr),
        )
        return binary_op(lhs_signed, rhs_signed, np.right_shift)

    @staticmethod
    def symbolic_ops_for_op_type(op_type: type[Op]) -> tuple[str, ...]:
        if op_type == UnaryOp:
            return (
                "cos",
                "exp",
                "exp2",
                "abs",
                "floor",
                "ceil",
                "log",
                "log2",
                "sqrt",
                "sin",
                "negative",
            )
        if op_type == BinaryOp:
            return (
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "less",
                "less_equal",
                "greater",
                "greater_equal",
                "not_equal",
                "equal",
                "maximum",
                "minimum",
                "bitwise_and",
                "bitwise_or",
                "bitwise_xor",
                "right_shift",
                "left_shift",
            )
        symbolic_ops_by_type = {
            ProgramId: ("pid",),
            RawLoad: ("load",),
            Load: ("load",),
            RawStore: ("store",),
            Store: ("store",),
            TernaryOp: ("where",),
            Fma: ("fma",),
            Dot: ("dot",),
            MakeRange: ("arange",),
            AddPtr: ("addptr",),
            ExpandDims: ("expand_dims",),
            Broadcast: ("broadcast",),
            ReduceSum: ("sum",),
            ReduceXor: ("xor_sum",),
            ReduceOr: ("reduce_or",),
            ReduceMax: ("max", "argmax"),
            ReduceMin: ("min", "argmin"),
            Sort: ("sort",),
            Splat: ("splat",),
            Unsplat: ("unsplat",),
            MakeBlockPointer: ("make_block_ptr",),
            TensorPointerLoad: ("tensor_pointer_load",),
            TensorPointerStore: ("tensor_pointer_store",),
            Idiv: ("idiv",),
            Rsqrt: ("rsqrt",),
            CastImpl: ("cast_impl",),
            Reshape: ("reshape",),
            Trans: ("trans",),
            Join: ("join",),
            Split: ("split",),
            Fabs: ("fabs",),
            Ashr: ("ashr",),
            Advance: ("advance",),
            FpToFp: ("fp_to_fp",),
            Umulhi: ("umulhi",),
            CumSum: ("cumsum",),
            Bitcast: ("bitcast",),
            PtrToInt: ("bitcast",),
            IntToPtr: ("bitcast",),
            AtomicCas: ("atomic_cas",),
            AtomicRMW: ("atomic_rmw",),
        }
        return symbolic_ops_by_type.get(op_type, ())

    _TRITON_TO_SYMBOLIC_DTYPES: ClassVar[dict[str, SymbolicScalarDType]] = {
        "int1": INT1,
        "int8": INT8,
        "int16": INT16,
        "int32": INT32,
        "int64": INT64,
        "uint8": UINT8,
        "uint16": UINT16,
        "uint32": UINT32,
        "uint64": UINT64,
        "fp16": FLOAT16,
        "fp32": FLOAT32,
        "fp64": FLOAT64,
        "bf16": BFLOAT16,
        "fp8e4nv": FLOAT8_E4M3,
        "fp8e4b8": FLOAT8_E4B8,
        "fp8e5": FLOAT8_E5M2,
        "fp8e5b16": FLOAT8_E5B16,
        "fp8e4b15": FLOAT8_E4M3FN,
    }

    _SYMBOLIC_TO_TRITON_DTYPE_NAMES: ClassVar[dict[SymbolicScalarDType, str]] = {
        INT1: "int1",
        INT8: "int8",
        INT16: "int16",
        INT32: "int32",
        INT64: "int64",
        UINT8: "uint8",
        UINT16: "uint16",
        UINT32: "uint32",
        UINT64: "uint64",
        FLOAT16: "float16",
        FLOAT32: "float32",
        FLOAT64: "float64",
        BFLOAT16: "bfloat16",
        FLOAT8_E4M3: "float8e4nv",
        FLOAT8_E4B8: "float8e4b8",
        FLOAT8_E5M2: "float8e5",
        FLOAT8_E5B16: "float8e5b16",
        FLOAT8_E4M3FN: "float8e4b15",
    }

    @classmethod
    def _normalize_triton_type(cls, dtype: Any) -> SymbolicDType | SymbolicTypeSpec:
        if isinstance(dtype, tl.block_type):
            scalar = cls._normalize_triton_type(dtype.scalar)
            if isinstance(scalar, SymbolicTypeSpec):
                raise TypeError(f"Unexpected nested block dtype: {dtype}")
            return symbolic_block_type(scalar, [int(dim) for dim in dtype.shape])
        if isinstance(dtype, tl.pointer_type):
            element = cls._normalize_triton_type(dtype.element_ty)
            if isinstance(element, SymbolicTypeSpec):
                raise TypeError(f"Unexpected block pointer element dtype: {dtype}")
            return symbolic_pointer_type(element)
        if isinstance(dtype, tl.core.dtype):
            try:
                return cls._TRITON_TO_SYMBOLIC_DTYPES[str(dtype)]
            except KeyError as exc:
                raise TypeError(f"Unsupported Triton dtype: {dtype}") from exc
        if isinstance(dtype, (SymbolicScalarDType, SymbolicPointerDType)):
            return dtype
        if isinstance(dtype, SymbolicTypeSpec):
            return dtype
        raise TypeError(f"Unsupported Triton type: {dtype}")

    @classmethod
    def _to_triton_dtype(cls, dtype: SymbolicDType) -> Any:
        if isinstance(dtype, SymbolicPointerDType):
            return tl.pointer_type(cls._to_triton_dtype(dtype.element_ty))
        try:
            return getattr(tl, cls._SYMBOLIC_TO_TRITON_DTYPE_NAMES[dtype])
        except KeyError as exc:
            raise TypeError(f"Unsupported symbolic dtype for Triton: {dtype}") from exc

    @classmethod
    def _to_triton_type(cls, value: SymbolicDType | SymbolicTypeSpec) -> Any:
        if isinstance(value, SymbolicTypeSpec):
            dtype = cls._to_triton_dtype(value.dtype)
            return tl.block_type(dtype, list(value.shape)) if value.shape else dtype
        return cls._to_triton_dtype(value)

    def normalize_symbolic_value(self, value: Any) -> Any:
        if isinstance(
            value,
            (
                SymbolicTensorValue,
                SymbolicScalarDType,
                SymbolicPointerDType,
                SymbolicTypeSpec,
                int,
                np.integer,
                bool,
                float,
                np.floating,
                type(None),
            ),
        ):
            return value
        if isinstance(value, tl.core.tensor):
            value = value.handle
        if isinstance(value, TensorHandle):
            dtype_or_spec = self._normalize_triton_type(value.dtype)
            dtype = (
                dtype_or_spec.dtype
                if isinstance(dtype_or_spec, SymbolicTypeSpec)
                else dtype_or_spec
            )
            return SymbolicTensorValue(
                np.asarray(value.data),
                dtype,
                dict(getattr(value, "attr", {})),
            )
        if isinstance(value, (tl.block_type, tl.pointer_type, tl.core.dtype)):
            return self._normalize_triton_type(value)
        if isinstance(value, tl.core.tuple):
            return tuple(self.normalize_symbolic_value(item) for item in value)
        if isinstance(value, tuple):
            return tuple(self.normalize_symbolic_value(item) for item in value)
        if isinstance(value, list):
            return [self.normalize_symbolic_value(item) for item in value]
        return super().normalize_symbolic_value(value)

    def to_frontend_symbolic_value(self, value: Any) -> Any:
        if isinstance(value, SymbolicTensorValue):
            return TensorHandle(
                np.asarray(value.data, dtype=dtype_to_numpy(value.dtype)),
                self._to_triton_dtype(value.dtype),
                dict(value.attr),
            )
        if isinstance(value, (SymbolicScalarDType, SymbolicPointerDType)):
            return self._to_triton_dtype(value)
        if isinstance(value, SymbolicTypeSpec):
            return self._to_triton_type(value)
        if isinstance(value, tuple):
            return tuple(self.to_frontend_symbolic_value(item) for item in value)
        if isinstance(value, list):
            return [self.to_frontend_symbolic_value(item) for item in value]
        return value

    def from_frontend_symbolic_value(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(self.from_frontend_symbolic_value(item) for item in value)
        if isinstance(value, list):
            return [self.from_frontend_symbolic_value(item) for item in value]
        return self.normalize_symbolic_value(value)

    def wrap_symbolic_loop_index(self, expr: Any, dtype: Any) -> Any:
        return tl.tensor(expr, self._to_triton_dtype(dtype))

    def wrap_symbolic_concrete_fn(self, concrete_fn: Callable) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            frontend_args = tuple(self.to_frontend_symbolic_value(arg) for arg in args)
            frontend_kwargs = {
                key: self.to_frontend_symbolic_value(value)
                for key, value in kwargs.items()
            }
            ret = concrete_fn(*frontend_args, **frontend_kwargs)
            return self.from_frontend_symbolic_value(ret)

        return wrapped

    def _wrap_tl_ret(self, ret: Any, fallback_dtype: Any) -> Any:
        if isinstance(ret, tl.tensor):
            return ret
        if isinstance(ret, tuple):
            return tuple(
                self._wrap_tl_ret(ret_elem, fallback_dtype) for ret_elem in ret
            )

        shape = getattr(ret, "shape", ())
        dtype = getattr(ret, "dtype", None)
        if isinstance(dtype, (SymbolicScalarDType, SymbolicPointerDType)):
            dtype = self._to_triton_dtype(dtype)
        ret_dtype = tl.block_type(dtype, list(shape)) if shape and dtype else dtype
        return tl.core.tensor(ret, ret_dtype or fallback_dtype)

    def run_op_overrider(
        self,
        op: Callable,
        op_type: type[Op],
        op_overrider: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        # Interpreter-builder ops already use TensorHandle-style arguments; tl
        # namespace ops need Triton tensor <-> handle conversion around clients.
        if op_type not in self._frontend_wrapped_op_types:
            return op_overrider(*args, **kwargs)
        if op_type in self._math_op_types:
            raise NotImplementedError("Patching math ops not yet supported")

        # see triton.runtime.interpreter:ReduceOps.sum
        # First, convert input from tl.tensor to TensorHandle. Here, input tensor is args[0]
        # Then, convert return value from TensorHandle to tl.tensor
        ret = op_overrider(args[0].handle, *args[1:], **kwargs)
        return self._wrap_tl_ret(ret, args[0].dtype)

    def patch_for_loop(self) -> None:
        if self._loop_ast_patched:
            return

        # Patch Triton's process-global AST transformer, but keep exact originals
        # so nested tracing scopes and later unpatching restore Triton behavior.
        self._loop_ast_methods = {
            "visit_For": getattr(_OrigASTTransformer, "visit_For", _MISSING),
            "visit_FunctionDef": getattr(
                _OrigASTTransformer, "visit_FunctionDef", _MISSING
            ),
        }

        frontend = self

        def visit_for(transformer, node):
            return frontend._visit_triton_for(transformer, node)

        def visit_function_def(transformer, node):
            return frontend._visit_triton_function_def(transformer, node)

        _OrigASTTransformer.visit_For = visit_for
        _OrigASTTransformer.visit_FunctionDef = visit_function_def
        self._loop_ast_patched = True

    def unpatch_for_loop(self) -> None:
        if not self._loop_ast_patched:
            return

        for method_name, original in self._loop_ast_methods.items():
            if original is _MISSING:
                delattr(_OrigASTTransformer, method_name)
            else:
                setattr(_OrigASTTransformer, method_name, original)
        self._loop_ast_methods = {}
        self._loop_ast_patched = False

    def patch_lang(self, fn, client_manager=None) -> _LangPatchScope:
        # Snapshot before calling Triton's patcher because Triton mutates many
        # attributes in-place but only records the language modules visible
        # from `fn`; nested/generated kernels without a `tl` global need both.
        scope = self._triton_snapshot_scope(fn)
        triton_patch_lang(fn)
        for module in self._triton_extra_modules:
            _patch_builtin(module, interpreter_builder, scope)
        self._patch_triton_inline_asm(scope)
        self._patch_triton_semantic_to_tensor(scope)
        scope.set_attr(knobs.runtime, "interpret", True)
        return scope

    def _prepare_grid_launch(self, grid_executor, args_dev, kwargs):
        # Removes not used reserved keywords from kwargs
        # Triton doesn't support keyword-only, variable positional or variable keyword arguments
        # It's safe to inspect only positional or keyword arguments (i.e., argspec.args)
        argspec = inspect.getfullargspec(grid_executor.fn)
        tilelens_args = ["client_manager", "jit_fn"]
        kwargs = {
            k: v for k, v in kwargs.items() if k in argspec.args or k in tilelens_args
        }
        client_manager = kwargs.pop("client_manager")
        self._current_client_manager = client_manager

        # Triton's CPU interpreter materializes host-side arguments. In virtual
        # memory mode we keep only fake tensor metadata so sanitizer/profiler
        # runs do not need to read real device memory.
        kwargs.pop("jit_fn")
        if cfg.virtual_memory:
            args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
        else:
            args_hst, kwargs_hst = grid_executor._init_args_hst(args_dev, kwargs)

        args = inspect.getcallargs(
            grid_executor.fn,
            *args_hst,
            **kwargs_hst,
        )
        # The hidden loop-wrapper default is an implementation detail of the
        # rewritten function; clients should not see it as a kernel argument.
        args.pop(self._loop_wrapper_arg, None)
        call_args = {}
        for name, arg in args.items():
            if name in grid_executor.constexprs:
                call_args[name] = arg
                ret = arg
            else:
                ret = _implicit_cvt(arg)
            client_manager.arg_callback(name, arg, ret)
            call_args[name] = ret
        call_args.pop("self", None)

        grid = (
            grid_executor.grid(call_args)
            if callable(grid_executor.grid)
            else grid_executor.grid
        )
        assert len(grid) <= 3
        grid = grid + (1,) * (3 - len(grid))

        self.builder.set_grid_dim(*grid)
        client_manager.grid_callback(grid)
        total_blocks = grid[0] * grid[1] * grid[2]
        max_workers = min(cfg.num_sms, total_blocks)
        return (
            client_manager,
            kwargs,
            args_hst,
            kwargs_hst,
            call_args,
            grid,
            max_workers,
        )

    def _run_grid(self, grid_executor, call_args, client_manager, grid, max_workers):
        def run_block(x, y, z):
            # Each worker sets its own thread-local grid index before callbacks
            # and kernel execution observe `tl.program_id`.
            interpreter_builder.set_grid_idx(x, y, z)
            client_manager.grid_idx_callback((x, y, z))
            if not client_manager.pre_run_callback(grid_executor.fn):
                return True
            grid_executor.fn(**call_args)
            return client_manager.post_run_callback(grid_executor.fn)

        if max_workers == 1:
            for x in range(grid[0]):
                for y in range(grid[1]):
                    for z in range(grid[2]):
                        if not run_block(x, y, z):
                            return
            return

        tasks: SimpleQueue = SimpleQueue()
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    tasks.put((x, y, z))

        stop_event = threading.Event()

        def worker():
            while not stop_event.is_set():
                try:
                    x, y, z = tasks.get_nowait()
                except Empty:
                    return
                if not run_block(x, y, z):
                    stop_event.set()
                    return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker) for _ in range(max_workers)]
            for fut in futures:
                fut.result()

    @contextmanager
    def _builder_options_scope(self, **updates: Any):
        # Triton's builder is process-global. Restore missing attributes as
        # missing too, not merely to None, so launches do not leak option state.
        originals = {
            name: getattr(self.builder.options, name, _MISSING) for name in updates
        }
        for name, value in updates.items():
            object.__setattr__(self.builder.options, name, value)
        try:
            yield
        finally:
            for name, value in originals.items():
                if value is _MISSING:
                    object.__delattr__(self.builder.options, name)
                else:
                    object.__setattr__(self.builder.options, name, value)

    def _grid_executor_call(self, grid_executor, *args_dev, **kwargs):
        if kwargs.pop("warmup", False):
            return

        launch_num_warps = kwargs.get("num_warps", 4)

        previous_client_manager = self._current_client_manager
        try:
            (
                client_manager,
                kwargs,
                args_hst,
                kwargs_hst,
                call_args,
                grid,
                max_workers,
            ) = self._prepare_grid_launch(grid_executor, args_dev, kwargs)

            if cfg.enable_timing:
                start_time = time.time()

            # Triton's interpreter overflow asserts add runtime work that
            # TileLens clients do not consume, so keep them off for launches.
            with self._builder_options_scope(
                num_warps=launch_num_warps,
                sanitize_overflow=False,
            ):
                self._run_grid(
                    grid_executor,
                    call_args,
                    client_manager,
                    grid,
                    max_workers,
                )

            if cfg.enable_timing:
                end_time = time.time()
                elapsed_time = end_time - start_time
                name = grid_executor.fn.__name__
                print(
                    f"TileLens: execution time for {name}: "
                    f"{elapsed_time * 1000:.3f} ms"
                )

            if not cfg.virtual_memory:
                grid_executor._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
        finally:
            self._current_client_manager = previous_client_manager

    def _jit_function_call(self, jit_function, *args, **kwargs):
        if (
            self._current_client_manager is not None
            and not self._jit_function_uses_core_module(jit_function)
        ):
            return jit_function.fn(*args, **kwargs)

        scope = self.patch_lang(
            jit_function.fn,
            client_manager=self._current_client_manager,
        )
        try:
            return jit_function.fn(*args, **kwargs)
        finally:
            scope.restore()

    @staticmethod
    def _jit_function_uses_core_module(jit_function) -> bool:
        fn = jit_function.fn
        # JIT helpers that call tl.core directly need a fresh language patch for
        # semantic injection. Plain user/device JIT helpers can reuse the active
        # top-level launch patch and skip another scope.
        return any(fn.__globals__.get(name) is tl.core for name in fn.__code__.co_names)

    @contextmanager
    def patch_calls(self):
        if self._patch_calls_scope == 0:
            # Only patch at top-level scope.
            old_grid_executor_call = GridExecutor.__call__
            old_jit_function_call = JITFunction.__call__
            old_compile_and_exec = FunctionRewriter._compile_and_exec

            # GridExecutor is the top-level kernel launch path; JITFunction is
            # the device-function path that can be called from interpreted code.
            def grid_executor_call(grid_executor, *args_dev, **kwargs):
                return self._grid_executor_call(grid_executor, *args_dev, **kwargs)

            def jit_function_call(jit_function, *args, **kwargs):
                return self._jit_function_call(jit_function, *args, **kwargs)

            def compile_and_exec(rewriter, transformed_ast):
                # Triton's FunctionRewriter executes code with `rewriter.kwargs`
                # as the local namespace. Temporarily add the hidden loop wrapper
                # there so rewritten function defaults can bind it.
                old_kwargs = rewriter.kwargs
                rewriter.kwargs = {
                    **old_kwargs,
                    self._loop_wrapper_arg: self.loop_iter_wrapper,
                }
                try:
                    return old_compile_and_exec(rewriter, transformed_ast)
                finally:
                    rewriter.kwargs = old_kwargs

            GridExecutor.__call__ = grid_executor_call
            JITFunction.__call__ = jit_function_call
            FunctionRewriter._compile_and_exec = compile_and_exec

        self._patch_calls_scope += 1
        try:
            yield
        finally:
            self._patch_calls_scope -= 1
            if self._patch_calls_scope == 0:
                # Only unpatch at top-level scope.
                GridExecutor.__call__ = old_grid_executor_call
                JITFunction.__call__ = old_jit_function_call
                FunctionRewriter._compile_and_exec = old_compile_and_exec


frontend = register_frontend(TritonFrontend())
