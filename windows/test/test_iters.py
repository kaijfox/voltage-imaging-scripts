import numpy as np
import pytest
import awkward as ak
from imaging_scripts.windows.ragged_ops import ak_apply_1d, ak_reduce_1d, ak_unique

def demean(x):
    if len(x) == 0:
        return x.astype(float)
    return x - np.mean(x)

def test_ak_apply_1d_basic():
    # Test demean function
    array = ak.Array([[1, 2, 3], [4, 5]])
    result = ak_apply_1d(array, demean)
    expected = ak.Array([[-1, 0, 1], [-0.5, 0.5]])
    assert ak.almost_equal(result, expected)

def test_ak_apply_1d_empty():
    # Test with empty subarrays
    array = ak.Array([[], [1, 2, 3], []])
    result = ak_apply_1d(array, demean)
    expected = ak.Array([[], [-1., 0, 1], []])
    assert ak.almost_equal(result, expected)

def test_ak_apply_1d_single_element():
    # Test with single-element subarrays
    array = ak.Array([[1], [2], [3]])
    result = ak_apply_1d(array, demean)
    expected = ak.Array([[0.], [0], [0]])
    assert ak.almost_equal(result, expected)

def test_ak_apply_1d_char():
    # Test with non-numeric data
    array = ak.Array([['a', 'b'], ['c']])
    result = ak_apply_1d(array, lambda x: x)
    assert ak.almost_equal(result, array)

def test_ak_apply_1d_string():
    # Test with non-numeric data
    array = ak.Array([['aa', 'bb'], ['cc']])
    result = ak_apply_1d(array, np.char.upper)
    expected = ak.Array([['AA', 'BB'], ['CC']])
    assert ak.almost_equal(result, expected)

def test_ak_apply_1d_input_1d():
    # Test with 1D input
    array = ak.Array([1, 2, 3])
    result = ak_apply_1d(array, demean)
    expected = ak.Array([-1., 0, 1])
    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_basic():
    # Test reduction with min function
    array = ak.Array([[1, 2, 3], [4, 5]])
    result = ak_reduce_1d(array, np.min)
    expected = ak.Array([1, 4])
    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_empty():
    # Test with an empty array
    array = ak.Array([])
    result = ak_reduce_1d(array, np.min)
    expected = ak.Array([])
    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_single():
    # Test with single-element segments
    array = ak.Array([[1], [2], [3]])
    result = ak_reduce_1d(array, np.min)
    expected = ak.Array([1, 2, 3])
    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_char():
    # Test with non-numeric data
    array = ak.Array([['a', 'b'], ['c']])
    expected = ak.Array(['ab', 'c'])
    result = ak_reduce_1d(array, lambda x: ''.join(x))
    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_string():
    # Test with non-numeric data
    array = ak.Array([['aa', 'bb'], ['cc']])
    expected = ak.Array(['aa', 'cc'])
    result = ak_reduce_1d(array, lambda x: x[0])
    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_string_to_int():
    # Test with non-numeric data
    array = ak.Array([['aa', 'bbb'], ['cc']])
    expected = ak.Array([2, 2])
    result = ak_reduce_1d(array, lambda x: np.strings.str_len(x).min())

    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_nested_empty():
    # Test with nested empty segments
    array = ak.Array([[], []])
    result = ak_reduce_1d(array, np.min)
    expected = ak.Array([[], []])
    assert ak.almost_equal(result, expected)

def test_ak_reduce_1d_input_1d():
    # Test with 1D input
    array = ak.Array([1, 2, 3])
    result = ak_reduce_1d(array, np.min)
    expected = ak.Array([1])
    assert ak.almost_equal(result, expected)

def test_ak_unique():
    # Test unique operation
    array = ak.Array([[1, 2, 2, 3], [4, 5, 4]])
    result = ak_unique(array)
    expected = ak.Array([1, 2, 3, 4, 5])
    assert ak.almost_equal(result, expected)
