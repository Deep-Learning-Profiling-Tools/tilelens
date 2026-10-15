"""What one traced launch bound its kernel parameters to.

A ``LaunchBinding`` is built from a core ``LaunchEvent``: its ``bound_args``
split into integer scalars, tensor facts and constexprs, plus the grid and the
config kwargs an Autotuner/Heuristics layer added. Mechanism only: which of
these facts an analysis trusts (the view footprint or the allocation, whether
non-contiguous tensors are refused, what a missing fact means) is the client's
call. Building a binding never raises; a fact that cannot be read leaves the
argument out and names it in ``LaunchBinding.error``.

A binding is not a complete account of the kernel's arguments: see
``LaunchBinding`` for what it leaves out.

Importing this module does not import Triton.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.client import LaunchCall, LaunchEvent


@dataclass(frozen=True)
class TensorFacts:
    """Launch-time facts about one tensor argument, read without touching
    its values."""

    # The view's first element; already includes the storage offset, so a
    # lowering must never add that offset again.
    data_ptr: int
    elem_size: int  # bytes
    numel: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]  # in elements
    dtype: str  # str(tensor.dtype), e.g. "torch.float32"
    contiguous: bool
    # The underlying allocation, independent of the view's data_ptr, shape
    # and strides; None when the tensor exposes no storage.
    storage_data_ptr: int | None = None
    storage_nbytes: int | None = None

    def allocation_interval(self) -> tuple[int, int] | None:
        """Verified byte bounds [start, end) of the allocation, or None when
        the address extent is unknown.

        Without storage metadata only a contiguous view's own extent is
        known. Partial or inconsistent storage metadata never falls back to
        numel, which could silently deactivate valid accesses.
        """
        if self.elem_size <= 0 or self.numel < 0 or self.data_ptr < 0:
            return None
        if self.storage_data_ptr is None and self.storage_nbytes is None:
            if not self.contiguous:
                return None
            return self.data_ptr, self.data_ptr + self.numel * self.elem_size
        if self.storage_data_ptr is None or self.storage_nbytes is None:
            return None
        start, size = self.storage_data_ptr, self.storage_nbytes
        end = start + size
        if start < 0 or size < 0 or not start <= self.data_ptr <= end:
            return None
        if self.numel and self.data_ptr + self.elem_size > end:
            return None
        if self.contiguous and self.data_ptr + self.numel * self.elem_size > end:
            return None
        return start, end


@dataclass(frozen=True)
class LaunchBinding:
    """One call's kernel parameters, by name, as a launch bound them.

    Only int/bool scalars, tensors and constexprs are bound. Arguments of
    other kinds (floats, None, tuples, ...) are left out without an error,
    although a tuple argument is several TTIR function arguments (e.g. two
    pointers). A descriptor-style argument is bound as its ``.base`` tensor
    alone: the shape, stride and flag fields it adds to the TTIR function
    are not bound. So a consumer must treat a TTIR function argument with no
    entry in ``params`` or ``tensors`` as unknown (e.g. refuse an access that
    depends on it), never as unconstrained.
    """

    # Non-constexpr int and bool arguments (bools as 0/1).
    params: Mapping[str, int]
    tensors: Mapping[str, TensorFacts]
    # Arguments to tl.constexpr parameters, as passed.
    constexprs: Mapping[str, Any]
    # The grid as passed: a tuple, a callable, or None.
    raw_grid: Any
    # The grid canonicalized to three int dims; None if it cannot be resolved,
    # or if a dim is no integer (named in ``error``).
    grid: tuple[int, int, int] | None
    # The keyword arguments Autotuner/Heuristics layers added to the
    # caller's call (see config_kwargs).
    config: Mapping[str, Any]
    # The facts that could not be read, "; "-joined, e.g. "argument 'x':
    # AttributeError: ...".
    error: str | None = None


def tensor_facts(value: Any) -> TensorFacts:
    """Read the TensorFacts of a torch-like tensor. Raises if a fact is
    unreadable; bind_launch contains that."""
    storage_data_ptr = storage_nbytes = None
    untyped_storage = getattr(value, "untyped_storage", None)
    if callable(untyped_storage):
        try:
            storage = untyped_storage()
            storage_data_ptr = int(storage.data_ptr())
            storage_nbytes = int(storage.nbytes())
        except Exception:  # duck-typed tensors without a storage
            storage_data_ptr = storage_nbytes = None
    return TensorFacts(
        data_ptr=int(value.data_ptr()),
        elem_size=int(value.element_size()),
        numel=int(value.numel()),
        shape=tuple(int(size) for size in value.shape),
        strides=tuple(int(stride) for stride in value.stride()),
        dtype=str(value.dtype),
        contiguous=bool(value.is_contiguous()),
        storage_data_ptr=storage_data_ptr,
        storage_nbytes=storage_nbytes,
    )


_SCALARS = (bool, int, float, str)


def _is_passed(passed: Any, value: Any) -> bool:
    # A layer that recomputes a caller's scalar to an equal value may hand on
    # another object (e.g. an int above the small-int cache); anything else
    # (tensors, callables) is the caller's only as the same object.
    if passed is value:
        return True
    return type(passed) is type(value) and type(value) in _SCALARS and passed == value


def config_kwargs(event: LaunchEvent, call: LaunchCall | None) -> dict[str, Any]:
    """The kwargs of ``event`` its launch's caller did not pass: what the
    Autotuner/Heuristics layers added (config kwargs, num_warps, ... and
    heuristic values that differ from the caller's). Without ``call`` every
    kwarg counts."""
    if call is None:
        return dict(event.kwargs)
    passed = call.kwargs
    return {
        name: value
        for name, value in event.kwargs.items()
        if name not in passed or not _is_passed(passed[name], value)
    }


def _constexpr_names(jit_fn: Any) -> frozenset[str]:
    return frozenset(
        param.name
        for param in getattr(jit_fn, "params", None) or ()
        if getattr(param, "is_constexpr", False)
    )


def _described_tensor(value: Any) -> Any:
    # A descriptor-style argument (e.g. triton.tools.tensor_descriptor.
    # TensorDescriptor) addresses its .base tensor.
    base = getattr(value, "base", None)
    if base is not None and hasattr(base, "data_ptr"):
        return base
    return value


def _int_grid(resolved: Any) -> tuple[int, int, int] | None:
    if resolved is None:
        return None
    # operator.index, as the launcher converts: a float dim is an error, not
    # truncated into a grid the untraced launch would reject.
    x, y, z = (operator.index(dim) for dim in resolved)
    return x, y, z


def bind_launch(event: LaunchEvent, call: LaunchCall | None = None) -> LaunchBinding:
    """Bind ``event`` (its ``bound_args`` and ``resolved_grid``); ``call`` is
    the launch's LaunchCall, which tells config kwargs from the caller's.
    Never raises."""
    params: dict[str, int] = {}
    tensors: dict[str, TensorFacts] = {}
    constexprs: dict[str, Any] = {}
    errors: list[str] = []
    grid = None
    config: dict[str, Any] = {}
    try:
        constexpr_names = _constexpr_names(event.jit_fn)
        for name, value in event.bound_args.items():
            try:
                if name in constexpr_names:
                    constexprs[name] = value
                    continue
                value = _described_tensor(value)
                if hasattr(value, "data_ptr"):
                    tensors[name] = tensor_facts(value)
                elif isinstance(value, (bool, int)):
                    params[name] = int(value)
            except Exception as exc:
                errors.append(f"argument {name!r}: {type(exc).__name__}: {exc}")
        try:
            grid = _int_grid(event.resolved_grid)
        except Exception as exc:
            errors.append(f"grid {event.resolved_grid!r}: {type(exc).__name__}: {exc}")
        config = config_kwargs(event, call)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return LaunchBinding(
        params=MappingProxyType(params),
        tensors=MappingProxyType(tensors),
        constexprs=MappingProxyType(constexprs),
        raw_grid=getattr(event, "grid", None),
        grid=grid,
        config=MappingProxyType(config),
        error="; ".join(errors) if errors else None,
    )
