"""Opt-in enqueue floors; legacy compute lookups and match tags are preserved."""
import pytest

from triton_viz.tools.nki_cost_model import ComputeCalibration


def test_floor_shared_by_streams_without_changing_affine_bandwidth():
    calibration = ComputeCalibration({('vector', 'float32', 1): (16., .9),
                                      ('vector', 'float32', 2): (88., 1.8)},
                                     {('vector', 'float32'): 86.})
    assert calibration.instruction_lookup('vector', 'float32', 1, 1) == (86., 'exact')
    assert calibration.instruction_ns('vector', 'float32', 2, 1) == pytest.approx(89.8)
    assert calibration.instruction_ns('vector', 'float32', 1, 1024) == pytest.approx(16 + .9*1024)


def test_legacy_csv_remains_affine_and_dtype_fallback_retains_match_tag(tmp_path):
    path = tmp_path / 'compute.csv'
    path.write_text('engine,dtype,input_stream_count,startup_ns,ns_per_free_elem\nvector,float32,1,16,0.9\n')
    calibration = ComputeCalibration.from_csv(path)
    assert calibration.issue_floors_ns == {}
    assert calibration.instruction_lookup('vector', 'bfloat16', 1, 1) == (16.9, 'dtype_fallback')
    path.write_text(path.read_text().replace('ns_per_free_elem\n', 'ns_per_free_elem,issue_floor_ns\n').replace('16,0.9\n', '16,0.9,86\n'))
    floored = ComputeCalibration.from_csv(path)
    assert floored.instruction_lookup('vector', 'bfloat16', 1, 1) == (86., 'dtype_fallback')
    assert floored.instruction_lookup('vector', 'bfloat16', 1, 1, strict_dtype=True) == (None, 'missing')


def test_conflicting_per_stream_floors_are_not_a_pattern_table(tmp_path):
    path = tmp_path / 'compute.csv'
    path.write_text('engine,dtype,input_stream_count,startup_ns,ns_per_free_elem,issue_floor_ns\n'
                    'vector,float32,1,16,0.9,86\nvector,float32,2,88,1.8,90\n')
    with pytest.raises(ValueError, match='Conflicting'):
        ComputeCalibration.from_csv(path)


@pytest.mark.parametrize('floor', ['nan', 'inf', '-1'])
def test_invalid_csv_floors_are_rejected(tmp_path, floor):
    path = tmp_path / 'compute.csv'
    path.write_text('engine,dtype,input_stream_count,startup_ns,ns_per_free_elem,issue_floor_ns\n'
                    f'vector,float32,1,16,0.9,{floor}\n')
    with pytest.raises(ValueError):
        ComputeCalibration.from_csv(path)


def test_derived_streams_keep_their_engine_floor_and_fallback_label():
    calibration = ComputeCalibration({('vector', 'float32', 1): (16., .9),
                                      ('vector', 'float32', 2): (88., 1.8),
                                      ('scalar', 'float32', 1): (15., .7)},
                                     {('scalar', 'float32'): 73.})
    assert calibration.instruction_lookup('scalar', 'float32', 2, 1) == (73., 'derived_streams')
    assert calibration.instruction_ns('scalar', 'float32', 2, 1024) == pytest.approx(15+1.4*1024)
