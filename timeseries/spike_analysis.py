
import numpy as np
import awkward as ak


from ..windows.ragged_ops import slice_by_events
from ..timeseries.events import Traces

def ms_to_samples(ms, fs):
    return int(np.ceil(ms * fs / 1000))


def neighbor_events(events_a, events_b, n_pre, n_post, max_n=None):
    # Fully vectorized Awkward implementation (no Python per-event loops).
    # Steps:
    # 1. Broadcast batch dims (stop before ragged axis)
    # 2. Compute cartesian product per-list: shape (*batch, <n_a>, <n_b>, 2)
    # 3. Unzip to obtain a_vals and b_vals with shape (*batch, <n_a>, <n_b>)
    # 4. Mask pairs within window and collect b-values; unflatten back to (*batch, <n_a>, <n_neighbors>)

    events_a = ak.Array(events_a)
    events_b = ak.Array(events_b)

    # Broadcast batch dims (stop before ragged axis)
    events_a, events_b = ak.broadcast_arrays(
        events_a, events_b, depth_limit=events_a.ndim - 1
    )

    # Cartesian product per-list: for each batch element, every a paired with every b
    pairs = ak.cartesian([events_a, events_b], nested=True)
    a_vals, b_vals = ak.unzip(pairs)

    # Differences and boolean mask of neighbors
    diff = b_vals - a_vals
    mask = (diff >= -n_pre) & (diff <= n_post)

    # Optionally sort neighbors by absolute difference and trim to max_n
    if max_n is not None:
        # push non-neighbors to end by assigning large distance
        abs_diff = ak.where(mask, np.abs(diff), 999999)
        order = ak.argsort(abs_diff, axis=-1)
        # reorder b_vals/mask along the innermost axis using the computed order
        b_vals = b_vals[order]
        mask = mask[order]
        # trim to max_n entries
        b_vals = b_vals[..., :max_n]
        mask = mask[..., :max_n]

    # Counts per (batch, event_a)
    counts = ak.sum(mask, axis=-1)

    # Flatten b_vals and mask across all nested dims, select masked values
    b_flat = ak.flatten(b_vals, axis=None)
    mask_flat = ak.flatten(mask, axis=None)
    selected = b_flat[mask_flat]

    # Unflatten selected values by counts per event (flattened)
    counts_flat = ak.flatten(counts, axis=None)
    neigh_per_event_flat = ak.unflatten(selected, counts_flat)

    # Now regroup per-batch using number of events per batch
    n_a = ak.num(events_a, axis=-1)
    if n_a.ndim == 0:  # scalar case
        n_a_flat = n_a
        result = ak.unflatten(neigh_per_event_flat, n_a_flat)
    else:
        n_a_flat = ak.flatten(n_a, axis=None)
        result = ak.unflatten(neigh_per_event_flat, n_a_flat)

    return result


def nearest_neighbor_events(events_a, events_b, n_pre, n_post):
    """For each event in A, return the single closest event in B (or None).

    Implemented by calling neighbor_events with max_n=1 and collapsing the
    inner length-1 lists to scalars, padding with None where needed.
    """
    neigh = neighbor_events(events_a, events_b, n_pre, n_post, max_n=1)
    # pad empty inner lists to length 1 with None, then take the sole element
    neigh_padded = ak.pad_none(neigh, 1, axis=-1)
    return neigh_padded[..., 0]


def count_neighbors(event_frames, fs, threshold_ms):
    n = ms_to_samples(threshold_ms, fs)
    counts = ak.num(neighbor_events(event_frames, event_frames, n, n), axis=-1) - 1
    return counts


def onset_offset_stat(arr, axis, n):
    """Compare first-n vs last-n samples along `axis`.

    Returns (mean_pre, mean_post, mean_diff, dprime) with `axis` reduced away.
    Supports numpy ndarrays and awkward Arrays (ragged along last axis). For awkward
    arrays this routine requires the tested axis to be the last axis (common
    use-case in this repo).
    """
    import math

    # Normalize axis for numpy arrays; for awkward we'll assume last-axis usage
    if isinstance(arr, ak.Array):
        # require axis refer to last dimension for ragged arrays
        if axis != -1 and axis != (arr.ndim - 1):
            raise NotImplementedError(
                "onset_offset_stat for awkward arrays only supports the last axis"
            )
        
        # Issues with scalar promotion if attempted on 1d array
        if arr.ndim == 1:
            mean_pre, mean_post, mean_diff, dprime = onset_offset_stat(arr[None], axis=-1, n=n)
            return mean_pre[0], mean_post[0], mean_diff[0], dprime[0]

        lengths = ak.num(arr, axis=-1)
        if ak.any(lengths < [n]):
            raise ValueError("n is larger than some segments along the reduction axis")

        pre = arr[..., :n]
        post = arr[..., -n:]

        mean_pre = ak.mean(pre, axis=-1)
        mean_post = ak.mean(post, axis=-1)

        std_pre = ak.std(pre, axis=-1)
        std_post = ak.std(post, axis=-1)

        mean_diff = mean_post - mean_pre

        pooled = np.sqrt((std_pre**2 + std_post**2) / 2)

        # Compute dprime robustly without triggering divide-by-zero warnings.
        invalid = pooled == 0
        diff_sign = np.sign(mean_diff)
        pooled_safe = ak.where(invalid, [1.0], pooled)
        raw_dprime = mean_diff / pooled_safe
        dprime = ak.where(
            invalid,
            np.where(diff_sign == 0, [0.0], np.where(diff_sign > 0, [np.inf], [-np.inf])),
            raw_dprime,
        )

        return mean_pre, mean_post, mean_diff, dprime

    else:
        # numpy path (or array-like convertible to ndarray)
        a = np.asarray(arr)
        axis_pos = axis if axis >= 0 else a.ndim + axis
        if n > a.shape[axis_pos]:
            raise ValueError(
                "n is larger than array dimension along the specified axis"
            )

        # Move axis to last for easy slicing
        a_moved = np.moveaxis(a, axis_pos, -1)
        pre = a_moved[..., :n]
        post = a_moved[..., -n:]

        mean_pre = np.mean(pre, axis=-1)
        mean_post = np.mean(post, axis=-1)
        std_pre = np.std(pre, axis=-1, ddof=0)
        std_post = np.std(post, axis=-1, ddof=0)

        mean_diff = mean_post - mean_pre
        pooled = np.sqrt((std_pre**2 + std_post**2) / 2)

        # Robust dprime
        invalid = pooled == 0
        diff_sign = np.sign(mean_diff)
        pooled_safe = np.where(invalid, 1.0, pooled)
        raw_dprime = mean_diff / pooled_safe
        dprime = np.where(
            invalid,
            np.where(diff_sign == 0, 0.0, np.where(diff_sign > 0, np.inf, -np.inf)),
            raw_dprime,
        )

        return mean_pre, mean_post, mean_diff, dprime


def event_template(trace_data, event_frames, fs, ms_pre, ms_post):
    n_pre, n_post = ms_to_samples(ms_pre, fs), ms_to_samples(ms_post, fs)
    windows = slice_by_events(trace_data, event_frames, n_pre, n_post)
    return ak.mean(windows, axis=-2)


def template_magnitude(template, windows):
    from ..windows.ragged_ops import ak_reduce_1d, ak_infer_shape

    # L2-normalize template, avoiding zero division
    tpl = ak.Array(template)
    norm = ak_reduce_1d(tpl, np.linalg.norm)[..., None]
    norm = ak.where(norm == 0, 1.0, norm)
    template_norm = tpl / norm

    # Broadcast to match window shape, then compute dot product and return
    template_norm = template_norm[(None,) * (windows.ndim - template_norm.ndim)]
    return ak.sum(windows * template_norm, axis=-1)


def peak_to_trough(trace_data, event_frames, fs, ms_pre, ms_post):
    n_pre, n_post = ms_to_samples(ms_pre, fs), ms_to_samples(ms_post, fs)
    w = slice_by_events(trace_data, event_frames, n_pre, n_post)
    return ak.max(w, axis=-1) - ak.min(w, axis=-1)


LABELS = {True: "BS", (False, True): "SS-ADP", (False, False): "SS-noADP"}


def classify_events(trace_data, event_frames, fs, threshold_ms, mean_window_ms):
    n_pre = n_post = ms_to_samples(threshold_ms, fs)
    counts = count_neighbors(event_frames, fs, threshold_ms)
    windows = slice_by_events(trace_data, event_frames, n_pre, n_post)
    n_stat = ms_to_samples(mean_window_ms, fs)
    mean_pre, mean_post, mean_diff, dprime = onset_offset_stat(windows, axis=-1, n=n_stat)

    # Label: counts > 0 → 'BS', else mean_diff > 0 → 'SS-ADP', else 'SS-noADP'
    labels = ak.where(
        counts > 0, ["BS"], ak.where(mean_diff > 0, ["SS-noADP"], ["SS-ADP"])
    )

    return {
        "labels": labels,
        "counts": counts,
        "mean_pre": mean_pre,
        "mean_post": mean_post,
        "mean_diff": mean_diff,
        "dprime": dprime,
    }


def events_to_trace(events, hop_ms, window_ms, fs, max_frame=None):
    """Convert per-batch event frames into binned spike-count traces.

    Purpose
    -------
    Produce a per-batch 2D numpy array of spike counts binned at anchors spaced by
    `hop_ms` milliseconds. This routine builds anchors with
    `anchors = np.arange(0, max_frame, hop)` where `hop = ms_to_samples(hop_ms, fs)`
    and counts spikes that fall within the half-window around each anchor.

    Behavior contract
    -----------------
    - Output shape is (*batch, n_bins) where n_bins = len(np.arange(0, max_frame, hop)).
    - Counting: a spike at frame `t` contributes +1 to bin at anchor `a` iff
      `t in [a - n_pre, a + n_post]` where `n_pre = n_post = ms_to_samples(window_ms/2, fs)`.
    - Non-overlapping: when `hop_ms == window_ms` (so hop == window), a spike is
      counted at most once across bins.
    - Empty events: a batch item with no spikes produces an all-zero row.
    - max_frame inference: if `max_frame` is None, it should be inferred from the
      events (e.g., `int(ak.max(events.spike_frames))`) so that the output covers
      at least the latest spike.

    Parameters
    ----------
    events : awkward.Array-like
        Per-batch ragged array of spike frame indices (e.g., ak.Array([[10,20], [5]])).
    hop_ms : float
        Spacing between bin centers in milliseconds.
    window_ms : float
        Full window width in milliseconds; half-window is used to compute n_pre/n_post.
    fs : float
        Sampling rate in samples per second (e.g., 1000 for 1 sample per ms).
    max_frame : int or None, optional
        Number of frames to cover; if None, infer from the events so the output
        includes the latest spike.

    Returns
    -------
    numpy.ndarray
        Integer array of shape (*batch, n_bins) containing counts per bin.
    """
    hop = ms_to_samples(hop_ms, fs)
    n_pre = n_post = ms_to_samples(window_ms / 2, fs)

    # events.spike_frames is expected per-spec; accept either an object with that
    # attribute or an awkward array directly.
    spike_frames = getattr(events, "spike_frames", events)
    spike_frames = ak.Array(spike_frames)

    # Infer max_frame to include the latest spike (add 1 so range is inclusive)
    if max_frame is None:
        max_val = ak.max(spike_frames)
        if max_val is None:
            max_frame = 0

    anchors_1d = np.arange(0, max_frame, hop)
    anchors_wrapped = ak.Array([anchors_1d])

    result = neighbor_events(anchors_wrapped, spike_frames, n_pre, n_post)
    return Traces(
        ak.to_numpy(ak.num(result, axis=-1)),
        ids=events.ids if hasattr(events, "ids") else None,
        fs=fs / hop,
    )
