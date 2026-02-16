import numpy as np
import awkward as ak
import matplotlib
matplotlib.use('Agg')

from imaging_scripts.viz import spike_types as st
from imaging_scripts.viz import spike_images as si
from imaging_scripts.timeseries import spike_analysis as sa


def test_type_proportions_runs():
    labels = ak.Array([["BS", "SS-ADP", "SS-noADP"], ["SS-ADP", "SS-ADP"]])
    ids = ["r1", "r2"]
    fig, ax = st.type_proportions(labels, ids)
    # Expect 3 bars stacked per ROI -> legend entries equal 3
    assert len(ax.patches) >= 3


def test_pre_post_scatter_runs():
    mean_pre = ak.Array([[0, 1], [2]])
    mean_post = ak.Array([[1, 0], [3]])
    labels = ak.Array([["SS-ADP", "SS-noADP"], ["BS"]])
    fig, ax = st.pre_post_scatter(mean_pre, mean_post, labels)
    # one scatter collection
    assert len(ax.collections) == 1


def test_neighbors_v_diff_runs():
    counts = ak.Array([[0, 1], [2]])
    mean_diff = ak.Array([[0.5, -0.2], [0.1]])
    labels = ak.Array([["BS", "SS-ADP"], ["SS-noADP"]])
    fig, ax = st.neighbors_v_diff(counts, mean_diff, labels)
    assert len(ax.collections) == 1


def test_characteristic_traces_runs():
    traces = np.zeros((2, 10))
    events = ak.Array([[2, 5], [3]])
    labels = ak.Array([["BS", "SS-ADP"], ["SS-noADP"]])
    fig, axes = st.characteristic_traces(traces, events, labels, fs=1000, ms_pre=1, ms_post=1)
    # Expect at least one line plotted
    ax_list = axes if isinstance(axes, list) else [axes]
    assert any(len(a.lines) > 0 for a in ax_list)


def test_characteristic_traces_averaged_runs():
    traces = np.zeros((2, 10))
    events = ak.Array([[2, 5], [3]])
    labels = ak.Array([["BS", "SS-ADP"], ["SS-noADP"]])
    ids = ["soma1", "branch1"]
    # minimal fake tree object with root_of
    class FakeTree:
        def root_of(self, rid):
            return rid
    tree = FakeTree()
    fig, axes = st.characteristic_traces_averaged(traces, events, labels, fs=1000, ms_pre=1, ms_post=1, ids=ids, tree=tree)
    assert isinstance(axes, dict)


def test_nearest_neighbor_events():
    a = ak.Array([[10, 20, 30]])
    b = ak.Array([[11, 25]])
    out = sa.nearest_neighbor_events(a, b, n_pre=2, n_post=2)
    assert ak.to_list(out) == [[11, None, None]]


def test_spatial_magnitudes_runs():
    traces = np.zeros((2, 10))
    events = ak.Array([[2, 5], [3, 4]])
    footprints = [np.array([[0,0],[0,1]]), np.array([[1,0]])]
    img = si.spatial_magnitudes(traces, events, footprints, ref_idx=0, fs=1000, ms_pre=1, ms_post=1, magnitude_fn=sa.peak_to_trough)
    assert hasattr(img, 'shape')
