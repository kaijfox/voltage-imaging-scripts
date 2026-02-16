import numpy as np
import awkward as ak #type: ignore

from imaging_scripts.timeseries import spike_analysis as sa


# ======================================================== neighbor_events ====
def test_ne_example():
    """Given two per-batch event lists, return neighbor lists within +/-2 samples."""
    a = ak.Array([[10, 20, 30], [100]])
    b = ak.Array([[9, 11, 25], [200]])
    out = sa.neighbor_events(a, b, n_pre=2, n_post=2)
    expected = ak.Array([[[9, 11], [], []], [[]]])
    assert ak.to_list(out) == ak.to_list(expected)


def test_ne_empty():
    """Empty B lists produce empty neighbor lists for every event in A."""
    a = ak.Array([[1, 2], []])
    b = ak.Array([[], []])
    out = sa.neighbor_events(a, b, n_pre=1, n_post=1)
    assert ak.to_list(out) == [[[], []], []]


def test_ne_broadcast():
    """events_b with singleton batch dimension should broadcast to events_a batch dims."""
    a = ak.Array([[10], [20]])
    b = ak.Array([[9, 11]])  # shape (1, 2) broadcasts to (2, 2)
    out = sa.neighbor_events(a, b, n_pre=2, n_post=2)
    assert ak.to_list(out) == [[[9, 11]], [[]]]


def test_ne_maxn():
    """neighbor_events should return nearest neighbors when max_n is provided."""
    a = ak.Array([[10]])
    b = ak.Array([[8, 9, 11, 12]])

    out1 = sa.neighbor_events(a, b, n_pre=2, n_post=2, max_n=1)
    v1 = ak.to_list(out1)[0][0]
    # Accept either a scalar int or a length-1 list containing the int
    if isinstance(v1, list):
        assert len(v1) == 1
        v1 = v1[0]
    # single nearest neighbor (tie between 9 and 11 allowed)
    assert isinstance(v1, int)
    assert v1 in (9, 11)

    out2 = sa.neighbor_events(a, b, n_pre=2, n_post=2, max_n=2)
    v2 = ak.to_list(out2)[0][0]
    assert sorted(v2) == [9, 11]

    out3 = sa.neighbor_events(a, b, n_pre=2, n_post=2, max_n=10)
    v3 = sorted(ak.to_list(out3)[0][0])
    assert v3 == sorted([8, 9, 11, 12])


# =========================================================== onset_offset ====

def test_onoff_mean_diff():
    """For a 1D numpy array, mean_diff equals mean(last2) - mean(first2)."""
    arr = np.array([0, 1, 2, 3, 4, 5])
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert mean_diff == 4.0


def test_onoff_dprime():
    """For a 1D numpy array, dprime computed from pooled std matches expected value."""
    arr = np.array([0, 1, 2, 3, 4, 5])
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert np.isclose(dprime, 8.0)

def test_onoff_axis_reduction():
    """onset_offset_stat reduces along the provided axis for numpy arrays."""
    arr = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=1, n=2)
    assert np.allclose(mean_diff, np.array([2.0, 2.0]))

def test_onoff_no_variance_eq(recwarn):
    """If pre and post windows are same with zero variance, dprime = 0.0
    - not inf or NaN
    - no divide-by-zero warnings
    """
    # -- numpy impl path --
    arr = np.array([1, 1, 1, 1])  # mean_diff = 0, pooled std = 0
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert dprime == 0.0
    assert len(recwarn) == 0  # no warnings

    # -- ak impl path --
    arr = ak.Array([1, 1, 1, 1])  # mean_diff = 0, pooled std = 0
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert dprime == 0.0
    assert len(recwarn) == 0  # no warnings

def test_onoff_no_variance_noneq(recwarn):
    """If pre and post windows are different but both have zero variance, dprime = +-inf
    """
    # -- numpy impl path --
    arr = np.array([1, 1, 2, 2])  # mean_diff = 1, pooled std = 0
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert dprime == np.inf

    arr = np.array([2, 2, 1, 1])  # mean_diff = 1, pooled std = 0
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert dprime == -np.inf
    assert len(recwarn) == 0  # no warnings

    # -- ak impl path --
    arr = ak.Array([1, 1, 2, 2])  # mean_diff = 1, pooled std = 0
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert dprime == np.inf

    arr = ak.Array([2, 2, 1, 1])  # mean_diff = 1, pooled std = 0
    mean_pre, mean_post, mean_diff, dprime = sa.onset_offset_stat(arr, axis=-1, n=2)
    assert dprime == -np.inf
    assert len(recwarn) == 0  # no warnings



# ========================================================= event_template ====

def test_event_template_values():
    """Extract mean waveform around events from a short trace (ms->samples 1:1).

    Note: slice_by_events returns n_pre + n_post + 1 frames per window (inclusive of event).
    """
    trace = np.arange(5)[None, :]
    events = ak.Array([[1, 3]])
    fs = 1000
    template = sa.event_template(trace, events, fs, ms_pre=1, ms_post=1)[0]
    assert np.allclose(template, np.array([1.0, 2.0, 3.0]))


# ===================================================== template_magnitude ====

def test_template_magnitude_basic():
    """Template magnitude is dot product with L2-normalized template per window."""
    tpl = np.array([1.0, 0.0, -1.0])
    windows = ak.Array([[[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]]])
    mags = sa.template_magnitude(tpl, windows)
    assert np.isclose(ak.to_list(mags)[0][0], np.sqrt(2.0))
    assert ak.to_list(mags)[0][1] == 0.0


def test_template_magnitude_multi_template():
    """Template magnitude is dot product with L2-normalized PER-template per window."""
    tpl = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    windows = ak.Array([[[1.0, 0.0, -1.0], [0.0, 0.0, -1.0]]])
    mags = sa.template_magnitude(tpl, windows)
    assert np.isclose(ak.to_list(mags)[0][0], np.sqrt(2.0))
    assert np.isclose(ak.to_list(mags)[0][1], 1.0)

# ============================================================ peak_trough ====

def test_peak_to_trough_value():
    """Peak-to-trough per-event equals max(window)-min(window)."""
    trace2 = np.array([[0, 1, 2, 3, 4]])
    events2 = ak.Array([[1]])
    pt = sa.peak_to_trough(trace2, events2, fs=1000, ms_pre=1, ms_post=1)
    assert ak.to_list(pt)[0][0] == 2

# ======================================================== classify_events ====

def test_ce_bs_label():
    """Two nearby events within threshold should be labeled 'BS'."""
    traces = np.zeros((1, 20))
    events = ak.Array([[5, 7]])
    fs = 1000
    threshold_ms = 3  # n_pre = n_post = 3
    mean_window_ms = 1

    res = sa.classify_events(traces, events, fs, threshold_ms, mean_window_ms)
    assert ak.to_list(res['labels']) == [['BS', 'BS']]


def test_ce_ss_adp_label():
    """Isolated event with larger post-window mean should be 'SS-ADP'."""
    traces = np.array([[0, 0, 0, 0, 0, 5, 5, 5, 5, 5]])
    events = ak.Array([[4]])
    fs = 1000
    threshold_ms = 2
    mean_window_ms = 2

    res = sa.classify_events(traces, events, fs, threshold_ms, mean_window_ms)
    assert ak.to_list(res['labels']) == [['SS-ADP']]


def test_ce_ss_noadp_label():
    """Isolated event with post-window mean <= pre-window mean is 'SS-noADP'."""
    traces = np.array([[5, 5, 0, 0, 0]])
    events = ak.Array([[2]])
    fs = 1000
    threshold_ms = 1
    mean_window_ms = 1

    res = sa.classify_events(traces, events, fs, threshold_ms, mean_window_ms)
    assert ak.to_list(res['labels']) == [['SS-noADP']]


def test_ce_meta_returns():
    """classify_events returns dict containing labels, counts, mean_diff, dprime with correct structure."""
    traces = np.zeros((1, 10))
    events = ak.Array([[3, 6]])
    res = sa.classify_events(traces, events, fs=1000, threshold_ms=2, mean_window_ms=1)
    assert set(res.keys()) >= {'labels', 'counts', 'mean_diff', 'dprime'}
    assert ak.num(res['labels'], axis=-1)[0] == 2
    assert ak.num(res['counts'], axis=-1)[0] == 2
    assert ak.num(res['mean_diff'], axis=-1)[0] == 2
    assert ak.num(res['dprime'], axis=-1)[0] == 2
