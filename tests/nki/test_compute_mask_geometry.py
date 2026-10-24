import numpy as np
from triton_viz.core.compute_mask import compute_mask_geometry


def test_sparse_mask_is_not_promoted_to_its_bounding_box():
    mask=np.array([[True,False,True]])
    result=compute_mask_geometry(mask,(2,3))
    assert result['bounds']==[[0,2],[0,3]]
    assert result['active_elements']==4 and not result['rectangular']


def test_empty_and_incompatible_masks_are_distinct():
    assert compute_mask_geometry(False,(2,3))['status']=='empty'
    assert compute_mask_geometry([True,False],(2,3))['status']=='unsupported_broadcast'


def test_rectangular_predicates_keep_the_full_allocation_separate():
    mask=(np.arange(128)[:,None]<1) & (np.arange(16384)[None,:]<192)
    result=compute_mask_geometry(mask,(128,16384))
    assert result['shape']==[128,16384]
    assert result['bounds']==[[0,1],[0,192]]
    assert result['active_elements']==192 and result['rectangular']
