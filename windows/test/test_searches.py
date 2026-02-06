import numpy as np
import pytest

from imaging_scripts.windows.ragged_ops import boundsorted, digitize

def naive_lefts_rights_per_segment(data, offsets, max_val):
    """
    Compute expected lefts and rights per segment using numpy.searchsorted.
    Lefts: insertion indices for values 0..max_val (inclusive)
    Rights: insertion indices for values 1..max_val+1 (i.e., v+1)
    Returns (lefts, rights) each shaped (num_segments, max_val+1)
    """
    num_segments = len(offsets) - 1
    targets_left = np.arange(0, max_val + 1)
    targets_right = np.arange(1, max_val + 2)
    lefts = np.zeros((num_segments, max_val + 1), dtype=int)
    rights = np.zeros((num_segments, max_val + 1), dtype=int)
    for i in range(num_segments):
        start, end = offsets[i], offsets[i + 1]
        seg = data[start:end]
        if seg.size == 0:
            lefts[i, :] = 0
            rights[i, :] = 0
        else:
            lefts[i, :] = np.searchsorted(seg, targets_left, side="left")
            rights[i, :] = np.searchsorted(seg, targets_right, side="left")
    return lefts, rights


def test_boundsorted_given_example():
    # Example:
    # [[0, 0, 1, 2, 4], [2, 3, 3]] with max_val=4
    data = np.array([0, 0, 1, 2, 4, 2, 3, 3], dtype=int)
    offsets = np.array([0, 5, 8], dtype=int)
    max_val = 4
    bins = np.arange(0, max_val + 2)

    lefts, rights = boundsorted(data, offsets, bins)
    assert lefts.shape == (2, max_val+1)
    assert rights.shape == (2, max_val+1)

    expected_lefts = np.array([[0, 2, 3, 4, 4], [0, 0, 0, 1, 3]], dtype=int)
    assert np.array_equal(lefts, expected_lefts)

    expected_rights = np.array([[2, 3, 4, 4, 5], [0, 0, 1, 3, 3]], dtype=int)
    assert np.array_equal(rights, expected_rights)

    counts = rights - lefts
    expected = np.array([[2, 1, 1, 0, 1], [0, 0, 1, 2, 0]], dtype=int)
    assert np.array_equal(counts, expected)


def test_boundsorted_empty_segment_in_middle():
    # offsets include an empty first segment
    # [[0, 1], [], [2]]
    data = np.array([0, 1, 2], dtype=int)
    offsets = np.array([0, 2, 2, 3], dtype=int)
    max_val = 2
    bins = np.arange(0, max_val + 2)

    lefts, rights = boundsorted(data, offsets, bins)

    # Expected left and right indices for each bin in each segment
    expected_lefts = np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]], dtype=int)
    expected_rights = np.array([[1, 2, 2], [0, 0, 0], [0, 0, 1]], dtype=int)
    assert np.array_equal(lefts, expected_lefts)
    assert np.array_equal(rights, expected_rights)

    counts = rights - lefts
    expected_counts = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=int)
    assert np.array_equal(counts, expected_counts)


def test_boundsorted_all_same_value_segments():
    # [[1,1,1], [0,0]]
    data = np.array([1, 1, 1, 0, 0], dtype=int)
    offsets = np.array([0, 3, 5], dtype=int)
    max_val = 2
    bins = np.arange(0, max_val + 2)

    lefts, rights = boundsorted(data, offsets, bins)

    # Expected left and right indices for each bin in each segment
    expected_lefts = np.array([[0, 0, 3], [0, 2, 2]], dtype=int)
    expected_rights = np.array([[0, 3, 3], [2, 2, 2]], dtype=int)
    assert np.array_equal(lefts, expected_lefts)
    assert np.array_equal(rights, expected_rights)

    counts = rights - lefts
    expected_counts = np.array([[0, 3, 0], [2, 0, 0]], dtype=int)
    assert np.array_equal(counts, expected_counts)


def test_boundsorted_values_out_of_range_are_ignored():
    # data values are > max_val; counts for 0..max_val should be zero
    data = np.array([5, 6, 7], dtype=int)
    offsets = np.array([0, 3], dtype=int)
    max_val = 4
    bins = np.arange(0, max_val + 2)

    lefts, rights = boundsorted(data, offsets, bins)

    # Expected left and right indices for each bin in each segment
    expected_lefts = np.array([[0, 0, 0, 0, 0]], dtype=int)
    expected_rights = np.array([[0, 0, 0, 0, 0]], dtype=int)
    assert np.array_equal(lefts, expected_lefts)
    assert np.array_equal(rights, expected_rights)

    counts = rights - lefts
    expected_counts = np.zeros((1, max_val + 1), dtype=int)
    assert np.array_equal(counts, expected_counts)


@pytest.mark.parametrize("num_segments, max_val, seed", [(1, 5, 1), (5, 7, 2), (10, 4, 3)])
def test_boundsorted_random_segments_match_naive(num_segments, max_val, seed):
    rng = np.random.RandomState(seed)
    
    # Build random sorted segments
    lens = rng.randint(0, 20, size=num_segments)
    segments = []
    offsets = [0]
    for L in lens:
        if L == 0:
            seg = np.empty(0, dtype=int)
        else:
            # random values in [0, max_val+2] (some values beyond max_val)
            seg = rng.randint(0, max_val + 3, size=L)
            seg.sort()
        segments.append(seg)
        offsets.append(offsets[-1] + L)

    data = np.concatenate(segments) if len(segments) > 0 else np.empty(0, dtype=int)
    offsets = np.array(offsets, dtype=int)
    bins = np.arange(0, max_val + 2)

    lefts, rights = boundsorted(data, offsets, bins)
    expected_lefts, expected_rights = naive_lefts_rights_per_segment(data, offsets, max_val)

    assert lefts.shape == expected_lefts.shape
    assert rights.shape == expected_rights.shape
    assert np.array_equal(lefts, expected_lefts)
    assert np.array_equal(rights, expected_rights)

    # also verify counts (rights - lefts) are non-negative and integer
    counts = rights - lefts
    assert np.issubdtype(counts.dtype, np.integer)
    assert np.all(counts >= 0)


def test_boundsorted_counts_return():
    # Test case for count=True
    data = np.array([0, 0, 1, 2, 4, 2, 3, 3], dtype=int)
    offsets = np.array([0, 5, 8], dtype=int)
    bins = np.array([0, 1, 2, 3, 4, 5], dtype=int)

    counts = boundsorted(data, offsets, bins, count=True)

    # Expected counts for each bin in each segment
    expected = np.array([[2, 1, 1, 0, 1], [0, 0, 1, 2, 0]], dtype=int)
    assert np.array_equal(counts, expected)


def test_digitize_basic():
    data = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    bins = np.array([0, 1, 2, 3, 4])
    result = digitize(data, bins)
    expected = np.array([0, 1, 2, 3, 4, 5])
    assert np.array_equal(result, expected)

def test_digitize_left():
    data = np.array([0, 1, 2, 3, 4])
    bins = np.array([0, 1, 2, 3, 4])
    result = digitize(data, bins, right=False)
    expected = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(result, expected)

def test_digitize_right():
    data = np.array([0, 1, 2, 3, 4])
    bins = np.array([0, 1, 2, 3, 4])
    result = digitize(data, bins, right=True)
    expected = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(result, expected)

def test_digitize_empty_data():
    data = np.array([])
    bins = np.array([0, 1, 2, 3, 4])
    result = digitize(data, bins)

    expected = np.array([], dtype=int)
    assert np.array_equal(result, expected)

def test_digitize_empty_bins():
    data = np.array([0.5, 1.5, 2.5, 3.5])
    bins = np.array([])

    with pytest.raises(ValueError, match="bins must be a 1-D array"):
        digitize(data, bins)

def test_digitize_large_data():
    data = np.linspace(-1, 101, 500)
    bins = np.linspace(0, 100, 10)

    result = digitize(data, bins, right=False)    
    expected = np.digitize(data, bins, right=False)
    assert np.array_equal(result, expected)

def test_digitize_non_uniform_bins():
    data = np.array([0.5, 1.5, 2.5, 3.5])
    bins = np.array([0, 1, 3, 4])

    result = digitize(data, bins)
    expected = np.array([1, 2, 2, 3])
    assert np.array_equal(result, expected)

def test_digitize_out_of_bounds():
    data = np.array([-1, 0, 5])
    bins = np.array([0, 1, 2, 3, 4])

    result = digitize(data, bins, right=True)
    expected = np.array([0, 1, 5])
    assert np.array_equal(result, expected)