"""Spike-type plotting helpers.

Implements functions described in design/handoff/26-02-11_spike-plots.md.
"""

from __future__ import annotations

import collections
from typing import Sequence

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from ..windows.ragged_ops import ak_reduce_1d, ak_boundsorted
from mplutil import util as vu

SPIKE_TYPE_COLORS = {"BS": "C0", "SS-ADP": "C1", "SS-noADP": "C2"}


def _ensure_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    if isinstance(ax, np.ndarray):
        return ax.flatten()[0].figure, ax
    return ax.figure, ax


def _label_counts(labels: ak.Array, label_list: Sequence[str]):
    """Count occurrences of each label in label_list for each ROI.

    Returns dict: {label: ak.Array of counts per ROI}
    """
    # Convert labels to integer for easier counting
    lab_ints = {l: i for i, l in enumerate(label_list)}
    int_labels = ak_reduce_1d(labels[..., None], lambda x: lab_ints.get(x[0], -1))

    # Sort and count numbers of labels
    labels_sorted = int_labels[ak.argsort(int_labels, axis=-1)]
    label_int_counts = ak_boundsorted(
        labels_sorted, max_val=len(lab_ints) + 1, return_count=True
    )
    label_counts = {l: label_int_counts[..., i] for i, l in enumerate(lab_ints.keys())}
    return label_counts


def type_proportions(labels: ak.Array, ids: Sequence[str], ax=None, colors=None):

    fig, ax = _ensure_ax(ax)
    colors = colors or SPIKE_TYPE_COLORS

    # Count and convert to proportions
    counts = _label_counts(labels, list(colors.keys()))
    totals = sum(counts[t] for t in colors.keys())
    zero_total = totals == 0
    totals = ak.where(zero_total, 1, totals)  # avoid zero division
    proportions = {
        t: ak.where(zero_total, 0, counts[t] / totals) for t in colors.keys()
    }

    types = list(colors.keys())
    n_type = len(types)
    n_x = len(labels)
    x = np.arange(n_x)

    bottom = np.zeros(n_x)
    for t in types:
        vals = np.array(proportions[t])
        ax.bar(x, vals, bottom=bottom, color=[colors.get(t)] * n_x, label=t)
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90, ha="center")
    ax.set_ylabel("Fraction of spikes")
    vu.legend(ax)

    return fig, ax


def pre_post_scatter(
    mean_pre: ak.Array, mean_post: ak.Array, labels: ak.Array, ax=None, colors=None
):
    fig, ax = _ensure_ax(ax)
    colors = colors or {"SS-ADP": "C1", "SS-noADP": "C2"}

    mp = ak.to_list(mean_post)
    mpr = ak.to_list(mean_pre)
    labs = ak.to_list(labels)

    xs = []
    ys = []
    cs = []
    for pre_row, post_row, lab_row in zip(mpr, mp, labs):
        for pre, post, lab in zip(pre_row, post_row, lab_row):
            if lab == "BS":
                continue
            xs.append(pre)
            ys.append(post)
            cs.append(colors.get(lab, "k"))

    ax.scatter(xs, ys, c=cs)
    if len(xs) > 0:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3)

    ax.set_xlabel("mean_pre")
    ax.set_ylabel("mean_post")
    return fig, ax


def neighbors_v_diff(
    counts: ak.Array, mean_diff: ak.Array, labels: ak.Array, ax=None, colors=None
):
    fig, ax = _ensure_ax(ax)
    colors = colors or SPIKE_TYPE_COLORS

    cts = (
        np.array(ak.flatten(counts, axis=None).to_list())
        if hasattr(counts, "to_list")
        else np.array(ak.to_list(counts))
    )
    md = (
        np.array(ak.flatten(mean_diff, axis=None).to_list())
        if hasattr(mean_diff, "to_list")
        else np.array(ak.to_list(mean_diff))
    )
    labs = ak.to_list(labels)
    xs = []
    ys = []
    cs = []
    for c_row, m_row, l_row in zip(ak.to_list(counts), ak.to_list(mean_diff), labs):
        for c, m, l in zip(c_row, m_row, l_row):
            xs.append(c)
            ys.append(m)
            cs.append(colors.get(l, "k"))

    ax.scatter(xs, ys, c=cs)
    ax.set_xlabel("neighbor_count")
    ax.set_ylabel("mean_diff")
    return fig, ax


def _grouped_type_stats(
    trace_data,
    event_frames: ak.Array,
    labels: ak.Array,
    fs: float,
    near_ms_pre: float,
    near_ms_post: float,
    window_ms_pre: float,
    window_ms_post: float,
    groups: Sequence[Sequence[int]],
    types: Sequence[str],
    mode: str = "pool",
    ref_idx: int = 0,
):
    """Compute mean and std templates for each spike type and ROI-groups.

    Parameters
    - trace_data: (n_rois, n_frames) ndarray-like
    - event_frames: ak.Array (n_rois, <n_events>)
    - labels: ak.Array (n_rois, <n_events>)
    - groups: list of lists of ROI indices (group_ixs)
    - mode: 'pool' or 'mean'
    - ref_idx: index of reference ROI whose events drive the per-type lists
    - types: sequence of types to compute for


    Returns
    - mean, std: np.ndarray (n_types, n_groups, window_len)
    """
    from ..timeseries.spike_analysis import ms_to_samples, nearest_neighbor_events
    from ..windows.ragged_ops import slice_by_events

    evs = ak.Array(event_frames)
    labs = ak.Array(labels)
    groups = ak.Array(groups)

    n_pre = ms_to_samples(near_ms_pre, fs)
    n_post = ms_to_samples(near_ms_post, fs)
    window_n_pre = ms_to_samples(window_ms_pre, fs)
    window_n_post = ms_to_samples(window_ms_post, fs)
    window_len = window_n_pre + window_n_post + 1

    n_types = len(types)
    n_groups = len(groups)

    mean_out = np.full((n_types, n_groups, window_len), np.nan, dtype=float)
    std_out = np.full((n_types, n_groups, window_len), np.nan, dtype=float)

    for i_t, t in enumerate(types):
        # mask & select ref events of this type
        ref_labels = labs[ref_idx]
        ref_events = evs[ref_idx]

        ref_events_of_type = ref_events[ref_labels == t]
        if len(ref_events_of_type) == 0:
            # nothing to match for this type
            continue
        
        n_group = ak.num(groups, axis=1)
        groups_flat = ak.flatten(groups, axis=1)
        
        # (group * <roi_in_group>, <events_of_ref>)
        nearest_flat = nearest_neighbor_events(
            # (1, <events_of_ref>)
            ref_events_of_type[None],
            # (group * roi_in_group>, <events_of_roi>)
            evs[groups_flat],
            n_pre,
            n_post,
        )
        # (group * <roi_in_group>, <events_of_both>)
        nearest_flat = ak.drop_none(nearest_flat)
        
        # (group * <roi_in_group>, <events>, window_size)
        windows = slice_by_events(
            # (group * <roi_in_group>, n_frames)
            trace_data[groups_flat.to_numpy()],
            # (group * <roi_in_group>, <events>)
            nearest_flat,
            window_n_pre,
            window_n_post,
        )
        # (group, <roi_in_group>, <events>, window_size)
        windows = ak.unflatten(windows, n_group, axis=0)

        if mode == 'pool':
            # (group, <roi_in_group * events>, window_size)
            windows_flat = ak.flatten(windows, axis=1)
            # If <roi_in_group * events> = 0, mean returns zero-length
            windows_n = ak.flatten(ak.num(windows_flat, axis=1), axis=None)
            valid = (windows_n > 0).to_numpy()
            # Set return for this type
            mean_out[i_t, valid] = ak.mean(windows_flat, axis=1)[valid]
            std_out[i_t, valid] = ak.std(windows_flat, axis=1)[valid]
        elif mode == 'mean':
            window_mean = ak.mean(windows, axis=1)
            # If no spikes for group & type, do not set return values
            windows_n = ak.flatten(ak.num(window_mean, axis=1), axis=None)
            valid = (windows_n > 0).to_numpy()
            # Set return for this type
            mean_out[i_t, valid] = ak.mean(window_mean, axis=1)[valid]
            std_out[i_t, valid] = ak.std(window_mean, axis=1)[valid]
        else:
            raise ValueError(f"Invalid mode: {mode}")

    return mean_out, std_out


def characteristic_traces(
    trace_data,
    event_frames: ak.Array,
    labels: ak.Array,
    fs: float,
    near_ms_pre: float,
    near_ms_post: float,
    window_ms_pre: float,
    window_ms_post: float,
    ax=None,
    type_colors: dict | None = None,
    roi_names: Sequence[str] | None = None,
    ref_idx: int = 0,
    types: Sequence[str] | None = None,
    **psth_kw,
):
    """Per-ROI characteristic traces (rows=types, cols=ROIs).

    Accepts separate `near_ms_*` (neighbor search window) and `window_ms_*` (PSTH window),
    plus `ref_idx` and optional `types`. Returns (fig, ax) from mean_psth_grid.
    """
    from .psth_grid import mean_psth_grid
    from ..timeseries.spike_analysis import ms_to_samples

    fig, ax = _ensure_ax(ax)
    type_colors = type_colors or SPIKE_TYPE_COLORS

    # groups: one ROI per group
    n_rois = len(ak.to_list(event_frames))
    groups = [[i] for i in range(n_rois)]

    # Determine types order used in helper
    if types is None:
        if type_colors is None:
            labs = ak.to_list(ak.flatten(labels[ref_idx], axis=None))
            types = sorted(list(dict.fromkeys(labs)))
        else:
            types = list(type_colors.keys())

    mean, std = _grouped_type_stats(
        trace_data,
        event_frames,
        labels,
        fs,
        near_ms_pre,
        near_ms_post,
        window_ms_pre,
        window_ms_post,
        groups,
        types if types is not None else None,
        mode="pool",
        ref_idx=ref_idx,
    )

    trace_colors = [type_colors.get(t, "k") for t in types]
    n_pre = ms_to_samples(window_ms_pre, fs)
    fig, ax = mean_psth_grid(
        mean=mean,
        std=std,
        fs=fs,
        zero=n_pre,
        trace_colors=trace_colors,
        trace_names=types,
        event_names=roi_names,
        ax=ax,
        **psth_kw,
    )
    return fig, ax


def characteristic_traces_averaged(
    trace_data,
    event_frames: ak.Array,
    labels: ak.Array,
    groups,
    fs: float,
    near_ms_pre: float,
    near_ms_post: float,
    window_ms_pre: float,
    window_ms_post: float,
    mode: str = "pool",
    ax=None,
    type_colors: dict | None = None,
    ref_idx: int = 0,
    types: Sequence[str] | None = None,
    **psth_kw,
):
    """Characteristic traces averaged across ROI groups.

    `groups` is a sequence with length == n_rois containing group labels
    (string/int) or None to omit an ROI.

    Accepts separate `near_ms_*` and `window_ms_*`, plus `ref_idx` and optional `types`.
    Returns (fig, ax) from mean_psth_grid.
    """
    from .psth_grid import mean_psth_grid
    from ..timeseries.spike_analysis import ms_to_samples

    type_colors = type_colors or SPIKE_TYPE_COLORS

    # Build mapping from group label -> list of ROI indices
    grp_arr = list(groups)
    unique_labels = []
    for g in grp_arr:
        if g is None:
            continue
        try:
            if np.isnan(g):
                continue
        except Exception:
            pass
        if g not in unique_labels:
            unique_labels.append(g)

    group_ixs = [
        [i for i, gg in enumerate(grp_arr) if gg == label] for label in unique_labels
    ]

    if len(group_ixs) == 0:
        # nothing to plot — return empty figure
        fig, ax = _ensure_ax(ax)
        return fig, ax

    mean, std = _grouped_type_stats(
        trace_data,
        event_frames,
        labels,
        fs,
        near_ms_pre,
        near_ms_post,
        window_ms_pre,
        window_ms_post,
        group_ixs,
        types if types is not None else None,
        mode=mode,
        ref_idx=ref_idx,
    )

    # Determine types order used in helper
    if types is None:
        try:
            labs_list = ak.to_list(labels[ref_idx])
            types_used = sorted(list(dict.fromkeys(labs_list)))
        except Exception:
            all_labs = []
            for row in ak.to_list(labels):
                all_labs.extend(row)
            types_used = sorted(list(dict.fromkeys(all_labs)))
    else:
        types_used = list(types)

    trace_colors = [type_colors.get(t, "k") for t in types_used]

    n_pre = ms_to_samples(window_ms_pre, fs)
    fig, ax = mean_psth_grid(
        mean=mean,
        std=std,
        fs=fs,
        zero=n_pre,
        trace_colors=trace_colors,
        trace_names=types_used,
        event_names=unique_labels,
        ax=ax,
        **psth_kw,
    )
    return fig, ax
