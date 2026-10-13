import json
import zipfile
from dataclasses import fields, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..clients.profiler import data as profiler_data
from ..clients.race_detector import data as race_detector_data
from ..clients.sanitizer import data as sanitizer_data
from ..ir import launch as ir_launch
from ..ir import verdict as ir_verdict
from ..utils import traceback_utils
from . import data as trace_data
from .data import Launch, TensorSnapshot


_TRACE_FORMAT_VERSION = 1
_TRACE_FORMAT = "tilelens_trace"
_LEGACY_TRACE_FORMAT = "triton_viz_trace"
_MANIFEST_NAME, _TENSORS_NAME = "manifest.json", "tensors.npz"
ArrayMap = dict[str, np.ndarray]
_TRACE_CLASSES = {
    f"{cls.__module__}:{cls.__qualname__}": cls
    for module in (
        trace_data,
        profiler_data,
        race_detector_data,
        sanitizer_data,
        traceback_utils,
        # Pure data: neither imports Triton.
        ir_launch,
        ir_verdict,
    )
    for cls in vars(module).values()
    if isinstance(cls, type) and is_dataclass(cls)
}
# _TRACE_CLASSES rebuilds serialized dataclass instances like Load(...), Dot(...),
# Launch(...), etc. _TRACE_OP_TYPES is narrower: it restores class-valued fields
# such as OutOfBoundsRecord.op_type = Load, where the payload stores the class
# object itself rather than an instance of that class.
_TRACE_OP_TYPES = {cls.__name__: cls for cls in (trace_data.Load, trace_data.Store)}


def _store_array(arrays: ArrayMap, value) -> str:
    """Store an array payload in the NPZ sidecar and return its key."""
    key = f"arr_{len(arrays)}"
    arrays[key] = np.asarray(value)
    return key


def _store_tensor_payload(arrays: ArrayMap, tensor: torch.Tensor) -> dict[str, object]:
    """Store a tensor payload, preserving dtypes that NumPy cannot represent."""
    data = tensor.detach().cpu()
    payload = data.numpy()  # TODO: use ml_dtypes for bf16/others
    return {
        "kind": "tensor",
        "dtype": str(data.dtype),
        "key": _store_array(arrays, payload),
    }


def _encode_trace_value(value, arrays: ArrayMap):
    """Convert a trace object graph into JSON-safe manifest data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, type):
        name = value.__name__
        if name not in _TRACE_OP_TYPES:
            raise TypeError(f"Unsupported trace class reference: {value}")
        return {"kind": "type", "name": name}
    if hasattr(value, "data_ptr") and hasattr(value, "detach"):
        return _encode_trace_value(TensorSnapshot.from_tensor(value), arrays)
    if isinstance(value, np.ndarray):
        return {"kind": "array", "key": _store_array(arrays, value)}
    if isinstance(value, (list, tuple, set)):
        items = [_encode_trace_value(item, arrays) for item in value]
        return (
            items
            if isinstance(value, list)
            else {"kind": type(value).__name__, "items": items}
        )
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                {
                    "key": _encode_trace_value(key, arrays),
                    "value": _encode_trace_value(item, arrays),
                }
                for key, item in value.items()
            ],
        }
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        cls_key = f"{cls.__module__}:{cls.__qualname__}"
        if cls_key not in _TRACE_CLASSES:
            raise TypeError(f"Unsupported trace dataclass: {cls_key}")
        payload_fields: dict[str, Any] = {}
        for field in fields(value):
            field_value = getattr(value, field.name)
            if isinstance(value, sanitizer_data.OutOfBoundsRecordZ3) and field.name in {
                "constraints",
                "symbolic_expr",
            }:
                field_value = None if field_value is None else str(field_value)
            elif isinstance(value, TensorSnapshot) and field.name == "data":
                payload_fields[field.name] = _store_tensor_payload(arrays, field_value)
                continue
            payload_fields[field.name] = _encode_trace_value(field_value, arrays)
        return {
            "kind": "dataclass",
            "type": cls_key,
            "fields": payload_fields,
        }
    raise TypeError(f"Unsupported trace value: {type(value)}")


def _load_tensor_payload(payload: dict[str, object], arrays: ArrayMap) -> torch.Tensor:
    """Rebuild a tensor payload, including dtypes like bfloat16."""
    key = payload["key"]
    if not isinstance(key, str):
        raise TypeError("Invalid tensor payload")
    array = np.asarray(arrays[key])
    return torch.from_numpy(array.copy())


def _build_dataclass(cls: type, kwargs: dict):
    """Instantiate a dataclass without calling generated __init__/__post_init__."""
    obj: Any = object.__new__(cls)
    for name, value in kwargs.items():
        object.__setattr__(obj, name, value)
    return obj


def _decode_trace_value(value, arrays: ArrayMap):
    """Reconstruct trace objects from manifest data and NPZ payloads."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_decode_trace_value(item, arrays) for item in value]
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported trace payload: {type(value)}")

    kind = value.get("kind")
    if kind == "array":
        return np.asarray(arrays[value["key"]]).copy()
    if kind == "tensor":
        return _load_tensor_payload(value, arrays)
    if kind == "path":
        return Path(value["value"])
    if kind in {"tuple", "set"}:
        container = tuple if kind == "tuple" else set
        return container(_decode_trace_value(item, arrays) for item in value["items"])
    if kind == "dict":
        return {
            _decode_trace_value(item["key"], arrays): _decode_trace_value(
                item["value"], arrays
            )
            for item in value["items"]
        }
    if kind == "type":
        return _TRACE_OP_TYPES[value["name"]]
    if kind == "dataclass":
        cls_key = value["type"]
        if cls_key.startswith("triton_viz."):
            cls_key = "tilelens." + cls_key.removeprefix("triton_viz.")
        cls = _TRACE_CLASSES.get(cls_key)
        if cls is None:
            raise TypeError(f"Unsupported trace dataclass: {value['type']}")
        kwargs = {
            name: _decode_trace_value(field_value, arrays)
            for name, field_value in value["fields"].items()
        }
        return _build_dataclass(cls, kwargs)
    raise TypeError(f"Unsupported trace payload kind: {kind}")


def save(path: str | Path) -> Path:
    """Save the current trace launches to a zip archive."""
    from .trace import launches

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "format": _TRACE_FORMAT,
        "version": _TRACE_FORMAT_VERSION,
        "launches": _encode_trace_value(launches, arrays),
    }
    tensor_bytes = BytesIO()
    np.savez_compressed(tensor_bytes, allow_pickle=False, **arrays)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_MANIFEST_NAME, json.dumps(manifest).encode("utf-8"))
        archive.writestr(_TENSORS_NAME, tensor_bytes.getvalue())
    return path


def load(path: str | Path, append: bool = False) -> list[Launch]:
    """Load trace launches from a zip archive into the global trace state."""
    from .trace import launches

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(_MANIFEST_NAME))
        if manifest.get("format") not in {_TRACE_FORMAT, _LEGACY_TRACE_FORMAT}:
            raise TypeError(f"{path} is not a tilelens trace archive")
        if manifest.get("version") != _TRACE_FORMAT_VERSION:
            raise TypeError(
                f"Unsupported trace format version: {manifest.get('version')}"
            )
        with np.load(BytesIO(archive.read(_TENSORS_NAME)), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    loaded = _decode_trace_value(manifest["launches"], arrays)
    if not isinstance(loaded, list) or any(
        not isinstance(launch, Launch) for launch in loaded
    ):
        raise TypeError(f"{path} does not contain tilelens trace launches")
    if not append:
        launches.clear()
    launches.extend(loaded)
    return launches
