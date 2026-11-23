import numpy as np
import pytest

from tilelens.core.data import Load, Tensor, Transfer
from tilelens.visualizer.draw import extract_load_coords, prepare_visualization_data


def _tensor(ptr, data):
    return Tensor(
        ptr=ptr,
        dtype=str(data.dtype),
        stride=tuple(data.strides),
        shape=tuple(data.shape),
        element_size=data.dtype.itemsize,
        data=data,
    )


@pytest.mark.parametrize(
    "mem_src,mem_dst,expected_kind,expected_ptr,expected_offsets",
    [
        ("HBM", "SBUF", "load", 101, np.array([[0, 4], [8, 12]])),
        ("PSUM", "SBUF", "transfer", 101, np.array([[0, 4], [8, 12]])),
        ("SBUF", "HBM", "store", 202, np.array([[4, 8], [12, 16]])),
    ],
)
def test_prepare_visualization_data_records_transfer_focus(
    mem_src, mem_dst, expected_kind, expected_ptr, expected_offsets
):
    """Test that Transfer ops resolve focus tensors and overall maps correctly."""
    src_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    dst_data = np.arange(6, dtype=np.float32).reshape(2, 3) + 10
    record = Transfer(
        src_ptr=101,
        dst_ptr=202,
        src_offsets=np.array([[0, 4], [8, 12]], dtype=np.int64),
        dst_offsets=np.array([[4, 8], [12, 16]], dtype=np.int64),
        mem_src=mem_src,
        mem_dst=mem_dst,
        bytes=16,
        time_idx=3,
    )
    tensor_table = {
        101: (_tensor(101, src_data), 0),
        202: (_tensor(202, dst_data), 1),
    }

    viz_data, raw_tensor_data, _, _, _, transfer_overall = prepare_visualization_data(
        [record], tensor_table
    )

    op = viz_data[0]
    record_uuid = op["uuid"]
    ptr_key = f"TRANSFER:{expected_ptr}"
    coords = transfer_overall[ptr_key]["tiles"][0]["global_coords"]
    linear = np.asarray(expected_offsets, dtype=np.int64).reshape(-1)
    linear_indices = linear // src_data.dtype.itemsize
    expected_coords = np.column_stack(np.unravel_index(linear_indices, (2, 3)))
    expected_coords = [
        tuple(float(value) for value in coord) for coord in expected_coords
    ]

    assert op["type"] == "Transfer"
    assert op["transfer_kind"] == expected_kind
    assert op["overall_key"] == ptr_key
    assert raw_tensor_data[record_uuid]["offsets"] == expected_offsets.tolist()
    assert raw_tensor_data[record_uuid]["transfer_kind"] == expected_kind
    assert raw_tensor_data["__sbuf_events__"] == []
    assert coords == expected_coords


def test_extract_load_coords_uses_recorded_element_size():
    """Unknown dtype names fall back to 4 bytes; the recorded size must win."""
    data = np.zeros((2, 4), dtype=np.uint8)
    tensor = Tensor(
        ptr=0,
        dtype="cutile_only_dtype",
        stride=tuple(data.strides),
        shape=tuple(data.shape),
        element_size=1,
        data=data,
    )
    record = Load(
        ptr=0, offsets=np.arange(8).reshape(2, 4), masks=np.ones((2, 4), bool)
    )
    assert len(extract_load_coords(record, tensor)) == 8
