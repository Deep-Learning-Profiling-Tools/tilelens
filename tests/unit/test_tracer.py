from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from tilelens.clients.tracer.tracer import Tracer, _convert_grid_idx
from tilelens.core.data import AtomicRMW, IntToPtr, Load, Transfer


# ======== _convert_grid_idx Tests ===========


def test_convert_grid_idx_none():
    assert _convert_grid_idx(None) is None


def test_convert_grid_idx_int():
    assert _convert_grid_idx(1) == (1, 0, 0)


def test_convert_grid_idx_tuple_1d():
    assert _convert_grid_idx((2,)) == (2, 0, 0)


def test_convert_grid_idx_tuple_2d():
    assert _convert_grid_idx((2, 3)) == (2, 3, 0)


def test_convert_grid_idx_tuple_3d():
    assert _convert_grid_idx((2, 3, 4)) == (2, 3, 4)


# ======== Tracer Init Tests ===========


def test_tracer_init_defaults():
    tracer = Tracer()
    assert tracer.callpath is True
    assert tracer.grid_idx is None
    assert tracer.records == []
    assert tracer.tensors == []
    assert tracer.sample is True


def test_tracer_init_with_grid_idx_int():
    tracer = Tracer(grid_idx=1)
    assert tracer.grid_idx == (1, 0, 0)


def test_tracer_init_with_grid_idx_tuple():
    tracer = Tracer(grid_idx=(2, 3))
    assert tracer.grid_idx == (2, 3, 0)


# ======== Tracer Callback Tests ===========


def test_tracer_pre_run_callback():
    tracer = Tracer()
    assert tracer.pre_run_callback(lambda: None) is True


def test_tracer_post_run_callback():
    tracer = Tracer()
    assert tracer.post_run_callback(lambda: None) is True


def test_tracer_pre_warmup_callback():
    tracer = Tracer()
    assert tracer.pre_warmup_callback(None) is False


def test_tracer_arg_callback_with_tensor():
    tracer = Tracer()
    mock_tensor = MagicMock()
    mock_tensor.data_ptr = MagicMock(return_value=12345)
    tracer.arg_callback("x", mock_tensor, None)
    assert mock_tensor in tracer.tensors


def test_tracer_arg_callback_without_tensor():
    tracer = Tracer()
    non_tensor = 42
    tracer.arg_callback("x", non_tensor, None)
    assert tracer.tensors == []


def test_tracer_grid_idx_callback_sampling_match():
    tracer = Tracer(grid_idx=(1, 0, 0))
    tracer.grid_idx_callback((1, 0, 0))
    assert tracer.sample is True


def test_tracer_grid_idx_callback_sampling_no_match():
    tracer = Tracer(grid_idx=(1, 0, 0))
    tracer.grid_idx_callback((2, 0, 0))
    assert tracer.sample is False


def test_tracer_grid_idx_callback_no_filter():
    tracer = Tracer()
    tracer.grid_idx_callback((0, 0, 0))
    assert tracer.sample is True
    tracer.grid_idx_callback((5, 3, 2))
    assert tracer.sample is True


# ======== Tracer _get_tensor Tests ===========


def test_tracer_get_tensor():
    tracer = Tracer()

    mock_tensor1 = MagicMock()
    mock_tensor1.data_ptr = MagicMock(return_value=1000)
    mock_tensor2 = MagicMock()
    mock_tensor2.data_ptr = MagicMock(return_value=2000)
    mock_tensor3 = MagicMock()
    mock_tensor3.data_ptr = MagicMock(return_value=3000)

    tracer.tensors = [mock_tensor1, mock_tensor2, mock_tensor3]

    assert tracer._get_tensor(1500) == mock_tensor1
    assert tracer._get_tensor(2000) == mock_tensor2
    assert tracer._get_tensor(2500) == mock_tensor2
    assert tracer._get_tensor(3500) == mock_tensor3


def test_tracer_transfer_records_element_strides_as_bytes():
    """Test that transfer tracing converts element strides into byte offsets."""
    tracer = Tracer()
    callback = tracer.register_op_callback(Transfer).before_callback
    src_data = np.zeros((2, 3), dtype=np.float32)
    dst_data = np.zeros((2, 3), dtype=np.float16)
    src = SimpleNamespace(
        shape=src_data.shape,
        data=src_data,
        buffer="hbm",
        _parent=None,
        data_ptr=lambda: 1000,
        stride=lambda: (3, 1),
        element_size=lambda: 4,
    )
    dst = SimpleNamespace(
        shape=dst_data.shape,
        data=dst_data,
        buffer="sbuf",
        _parent=None,
        data_ptr=lambda: 2000,
        stride=lambda: (3, 1),
        element_size=lambda: 2,
    )

    callback(src, dst, src.buffer, dst.buffer)

    record = tracer.records[-1]
    linear = np.arange(6, dtype=np.int64).reshape(2, 3)
    assert np.array_equal(record.src_offsets, linear * 4)
    assert np.array_equal(record.dst_offsets, linear * 2)
    assert record.bytes == dst_data.nbytes


# ======== Out-of-bounds Guard Tests ===========


def test_check_in_bounds_accepts_python_bool_mask():
    tracer = Tracer()
    tensor = torch.zeros(4)
    tracer.arg_callback("x_ptr", tensor, None)
    base = tensor.data_ptr()

    def ptr(*offsets):
        lanes = np.array([base + offset for offset in offsets], dtype=np.uint64)
        return TensorHandle(lanes, tl.pointer_type(tl.float32))

    tracer._check_in_bounds("load", ptr(0, 4, 8, 12), True)
    with pytest.raises(IndexError, match="out-of-bounds load"):
        tracer._check_in_bounds("load", ptr(8, 12, 16), True)


def test_tracer_traces_op_subclasses_as_their_core_op():
    # frontends subclass a core op to pick another argument adapter (Gluon's
    # builder atomics, int->pointer casts and async copies)
    class FrontendLoad(Load):
        pass

    class FrontendAtomicRMW(AtomicRMW):
        pass

    class FrontendIntToPtr(IntToPtr):
        pass

    tracer = Tracer()
    tensor = torch.zeros(4)
    tracer.arg_callback("x_ptr", tensor, None)
    lanes = np.array([tensor.data_ptr() + offset for offset in (8, 12, 16)])
    ptr = TensorHandle(lanes.astype(np.uint64), tl.pointer_type(tl.float32))

    with pytest.raises(IndexError, match="out-of-bounds load"):
        tracer.register_op_callback(FrontendLoad).before_callback(ptr, None, None)
    with pytest.raises(IndexError, match="out-of-bounds atomic"):
        tracer.register_op_callback(FrontendAtomicRMW).before_callback(ptr, None)

    tracer.grid_idx_callback((0, 0, 0))
    tracer.register_op_callback(FrontendIntToPtr).after_callback(None)
    assert tracer._get_thread_local("int_to_ptr") is True
