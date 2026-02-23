import numpy as np
import matplotlib.pyplot as plt
from ..timeseries.types import Traces, Events


def nancat(arr, n=5):
    """Pad with nans and concatenate along the specified axis.
    Shapes: [k, m,] => [k * (m+n),]"""
    pad = np.full(arr.shape[:-1] + (n,), np.nan)
    padded = np.concatenate([arr, pad], axis=-1)
    return padded.reshape(arr.shape[:-2] + (-1,))

def segment_to_traces(array_1d, start_s, end_s, segment_s, fs) -> Traces:
    """Convert a 1D frame array into fixed-length segment traces.

    Parameters
    ----------
    array_1d : array-like
        1D array of frames indexed by global frame number.
    start_s : float
        Start time in seconds (inclusive).
    end_s : float
        End time in seconds (exclusive).
    segment_s : float
        Segment length in seconds.
    fs : float
        Sampling frequency (frames per second).

    Returns
    -------
    Traces
        Traces object containing segmented data, segment ids, and fs.
    """
    arr = np.asarray(array_1d)
    if arr.ndim != 1:
        raise ValueError("array_1d must be 1D")
    if not (start_s < end_s):
        raise ValueError("start_s must be less than end_s")
    if segment_s <= 0:
        raise ValueError("segment_s must be > 0")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    frames_per_segment = int(round(segment_s * fs))
    if frames_per_segment <= 0:
        raise ValueError("segment duration too short for given fs")

    start_frame = int(start_s * fs)
    end_frame = int(end_s * fs)
    trimmed = arr[start_frame:end_frame]

    n_segments = trimmed.size // frames_per_segment
    if n_segments == 0:
        seg_data = np.empty((0, frames_per_segment))
        seg_ids = []
        return Traces(seg_data, seg_ids, fs=fs)

    trimmed = trimmed[: n_segments * frames_per_segment]
    seg_data = trimmed.reshape((n_segments, frames_per_segment))

    seg_start_times = start_s + np.arange(n_segments) * segment_s
    seg_ids = [f"{t:.2f}-{(t + segment_s):.2f}s" for t in seg_start_times]

    return Traces(seg_data, seg_ids, fs=fs)


def segment_events(
    event_frames,
    start_s,
    fs=None,
    n_segments=None,
    frames_per_segment=None,
    segment_traces=None,
) -> Events:
    """Map global event frame indices into per-segment local frames.

    Parameters
    ----------
    event_frames : array-like
        1D array of global frame indices (integers).
    segment_traces : Traces
        Traces object returned by segment_to_traces used as reference.

    Returns
    -------
    Events
        Events object containing per-segment local frames and segment ids.
    """
    ef = np.asarray(event_frames, dtype=int)

    if segment_traces is not None:
        n_segments, frames_per_segment = segment_traces.data.shape
        fs = segment_traces.fs
    elif fs is None or n_segments is None or frames_per_segment is None:
        raise ValueError(
            "Must provide either segment_traces or fs, n_segments, and frames_per_segment"
        )

    if n_segments == 0:
        return Events([[] for _ in range(0)], [])

    start_frame0 = int(round(start_s * fs))
    start_frames = start_frame0 + np.arange(n_segments) * frames_per_segment

    events_per_segment = []
    for sf in start_frames:
        local = ef - sf
        mask = (local >= 0) & (local < frames_per_segment)
        events_per_segment.append((local[mask]).tolist())

    ids = [
        f"{(sf / fs):.2f}-{((sf + frames_per_segment) / fs):.2f}s"
        for sf in start_frames
    ]

    return Events(events_per_segment, ids)


def display_event_windows(
    windows,
    n_pre,
    fs,
    ids=None,
    n_gap=5,
    fig=None,
    axes=None,
    sharex=True,
    line_kws=None,
    event_line_kws=None,
):
    """Plot event-triggered windows, one row per ROI.

    windows : (n_roi, n_events, frames) or (n_events, frames) numpy array
    n_pre   : frames before event onset (used to mark the onset time)
    event_line_kws : axvline kwargs; None = default faint gray; False = skip markers
    Returns (fig, axes, meta) where meta = dict(t_ms=..., event_onset_ms=...)
    """
    arr = np.asarray(windows)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError("windows must be a 2D or 3D array")

    n_roi, n_events, n_frames = arr.shape

    # defaults
    if line_kws is None:
        line_kws = dict(lw=1)
    if event_line_kws is False:
        draw_event_lines = False
    else:
        draw_event_lines = True
        if event_line_kws is None:
            event_line_kws = dict(color=".8", lw=0.5, zorder=-5)

    if ids is None:
        ids = [f"ROI {i}" for i in range(n_roi)]

    step = n_frames + n_gap
    t_ms = np.arange(n_events * step) / fs * 1000.0
    onset_idx = np.arange(n_events) * step + n_pre
    event_onset_ms = t_ms[onset_idx]

    # axes handling
    if axes is None:
        fig, axes = plt.subplots(n_roi, 1, sharex=sharex)
        axes = np.atleast_1d(axes).reshape(-1)
    else:
        axes = np.asarray(axes)
        if axes.ndim == 0:
            axes = np.array([axes.item()])
        axes = axes.reshape(-1)
        if fig is None:
            fig = axes[0].figure

    for i in range(n_roi):
        flat = nancat(arr[i], n=n_gap)
        ax = axes[i]
        ax.plot(t_ms, flat, **line_kws)
        ax.set_ylabel(ids[i])

    if draw_event_lines:
        for ax in axes:
            for x in event_onset_ms:
                ax.axvline(x=x, **event_line_kws)

    return fig, np.asarray(axes), dict(t_ms=t_ms, event_onset_ms=event_onset_ms)
