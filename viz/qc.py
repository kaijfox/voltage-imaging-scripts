"""QC helpers for timeseries analysis and plotting."""

import numpy as np
import awkward as ak


def rolling_window(data, window_size_ms, fs, window_hop_ms=None, pad=False):
    """Create rolling windows from time-series data.

    Parameters
    ----------
    data : ndarray, shape (..., T)
        Input time-series array with arbitrary leading dimensions and a final
        time dimension of length T.
    window_size_ms : float or int
        Window length in milliseconds.
    fs : float
        Sampling frequency in Hz (frames per second).
    window_hop_ms : float or int, optional
        Step between successive windows in milliseconds. If ``None``, defaults
        to ``window_size_ms`` (non-overlapping windows).
    pad : bool, optional
        If True, pad the windows output along the n_steps axis with NaN slices
        (half_win on each side) so its length matches the original number of
        timepoints T, and return times = np.arange(T) / fs. If False (default),
        existing behavior unchanged.

    Returns
    -------
    windows : ndarray, shape (..., n_steps, W) or (..., T, W) if pad=True
        Windowed views of the input data along the last axis. ``W`` is the
        window length in samples (frames), and ``n_steps`` is the number of
        windows extracted (or T when pad=True).
    times : ndarray, shape (n_steps,)
        Center times for each extracted window, in seconds.
    """
    # convert ms → frames; ensure odd window size
    window_size_frames = int(round(float(window_size_ms) * float(fs) / 1000.0))
    if window_size_frames <= 0:
        raise ValueError("window_size_ms too small; results in 0 frames")
    # make window size odd
    if window_size_frames % 2 == 0:
        window_size_frames += 1
    half_win = window_size_frames // 2

    if window_hop_ms is None:
        window_hop_ms = window_size_ms
    hop_frames = int(round(float(window_hop_ms) * float(fs) / 1000.0))
    if hop_frames < 1:
        hop_frames = 1

    # sliding window view along last axis
    windows_all = np.lib.stride_tricks.sliding_window_view(data, window_size_frames, axis=-1)
    # subsample along the window-step axis (second-to-last)
    if hop_frames > 1:
        windows = windows_all[..., ::hop_frames, :]
    else:
        windows = windows_all

    T = data.shape[-1]
    center_idxs = np.arange(half_win, T - half_win, hop_frames)
    if pad:
        # create padded windows of length T along the step axis, fill with NaN
        padded_shape = windows.shape[:-2] + (T, window_size_frames)
        padded_windows = np.full(padded_shape, np.nan, dtype=float)
        # assign windows at their center indices (cast to float to allow NaN)
        padded_windows[..., center_idxs, :] = windows.astype(float)
        times = np.arange(T) / float(fs)
        return padded_windows, times

    times = np.arange(T)[half_win : T - half_win : hop_frames] / float(fs)

    return windows, times


def embed_events(values, spike_frames, n_frames):
    """Embed per-event values into dense per-ROI time series.

    Parameters
    ----------
    values : ak.Array, shape (n_rois, <n_events>)
        Awkward array of scalar event values for each ROI; ``<n_events>``
        denotes a variable-length inner dimension (events per ROI).
    spike_frames : ak.Array, shape (n_rois, <n_events>)
        Awkward array of integer frame indices corresponding to events in
        ``values``. Indices are in the range ``[0, n_frames-1]``.
    n_frames : int
        Length of the output time axis (number of frames).

    Returns
    -------
    out : ndarray, shape (n_rois, n_frames)
        Dense NumPy array with embedded event values at their corresponding
        frame indices and ``np.nan`` elsewhere.
    """
    # values: ak array (n_rois, <n_events>) of scalars
    # spike_frames: ak array (n_rois, <n_events>) of int frame indices
    n_rois = len(values)
    out = np.full((n_rois, n_frames), np.nan, dtype=float)

    for i in range(n_rois):
        # get indices and values for ROI i
        idxs = spike_frames[i].to_list()
        if len(idxs) == 0:
            continue
        vals = values[i].to_numpy()
        out[i, idxs] = vals

    return out


def plot_distribution(ax, t, data, label=None, color=".7", mean_color="k"):
    """Plot a distribution summary (IQR, min/max, mean) across observations over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib Axes instance on which to draw.
    t : array-like, shape (T,)
        Time vector corresponding to the second axis of ``data``.
    data : ndarray, shape (n, T)
        Observations array where rows are independent samples and columns are
        timepoints. May contain ``np.nan`` values which are ignored in
        statistical reductions (``nanmean``, ``nanquantile``).
    label : str or None, optional
        Base label used for legend entries; if ``None``, legend labels are
        omitted or simplified.
    color : color-like, optional
        Color used for the IQR fill and min/max lines.
    mean_color : color-like, optional
        Color used for the mean line.

    Returns
    -------
    None
        Draws on ``ax`` and does not return a value.
    """
    # data: (n, time) array, may contain nan; reduce across axis=0
    mean = np.nanmean(data, axis=0)
    q0, q25, q75, q100 = np.nanquantile(data, [0, .25, .75, 1], axis=0)
    # labels: prefix with provided label if given
    iqr_label = f"{label} IQR" if label else "IQR"
    avg_label = f"{label} Avg." if label else "Avg."
    # draw IQR as filled area
    ax.fill_between(t, q25, q75, color=color, lw=0, label=iqr_label)
    # dashed min/max lines; only label the first (min) one
    ax.plot(t, q0, color=color, ls="--", lw=0.5, label="Min/Max")
    ax.plot(t, q100, color=color, ls="--", lw=0.5)
    # plot mean line
    ax.plot(t, mean, color=mean_color, label=avg_label)
