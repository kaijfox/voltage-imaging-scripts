from typing import Tuple
from mplutil import util as vu
import matplotlib.pyplot as plt

from ..windows.ragged_ops import ak_infer_shape
import numpy as np
from matplotlib.collections import LineCollection


def _time_grid(ndim, nframes, fs, zero):
    """
    ndim: int
        Number of batch dimensions to include
    zero: int
        Index of time zero in the resultng `nframes` dimension

    Returns
    time_grid : (1, ..., nframes)
        time_grid.ndim = `ndim + 1`
    """
    # Normalize inputs
    if zero is None:
        zero = 0

    # Create a 1D time vector first
    if fs is None:
        t1 = np.arange(nframes) - zero
    else:
        t1 = (np.arange(nframes, dtype=float) - float(zero)) / float(fs)

    # Reshape to have `ndim` leading singleton dimensions, then a final time axis
    shape = (1,) * ndim + (nframes,)
    return t1.reshape(shape)


# Helper to pick color
def _pick_color(
    i_trace, i_event, default_color=None, trace_colors=None, event_colors=None
):
    color = None
    if trace_colors is not None:
        try:
            color = trace_colors[i_trace]
        except Exception:
            color = trace_colors[0] if len(trace_colors) > 0 else None
    elif event_colors is not None:
        try:
            color = event_colors[i_event]
        except Exception:
            color = event_colors[0] if len(event_colors) > 0 else None

    if color is None:
        color = default_color
    return color


def override_kws(*overrides, **default):
    merged = default.copy()
    for d in overrides:
        if d is not None:
            merged.update(d)
    return merged


def mean_psth_grid(
    traces=None,
    mean=None,
    std=None,
    sem=None,
    fs=None,
    zero=None,
    subplot_kw={},
    trace_kw={},
    mean_kw={},
    std_kw={},
    sem_kw={},
    trace_colors=None,
    event_colors=None,
    trace_names=None,
    event_names=None,
    ax=None,
):
    """
    mean, std, sem: (trace_roi, event_roi, frames_per_window)
        Numpy or awkward array
    traces:  (trace_roi, event_roi, <n_event>, frames_per_window)
        Ragged in axis 2
    subplot_kw: dict
        Arguments to vu.subplots
    fs : float
        Sampling frequency used to construct time grid
    trace_colors, event_colors: Sequence[color]
        Colors convertible with matplotlib to_rgba. One of the two may be
        provided, indicating colors to be used along the `trace_rois` or
        `event_rois` dimension.
    """

    # infer shapes (ak_infer_shape or .shape if ndarray)
    infer_from = mean
    if infer_from is None:
        # only traces was passed
        # (sem & std require mean to be passed as well)
        raise NotImplementedError("Shape inference over ragged <n_events> dimension")
    n_trace, n_event, n_frames = ak_infer_shape(infer_from)

    # Build axis grid: rows = trace, cols = event
    # ax: (n_trace, n_event)
    if ax is None:
        fig, ax = vu.subplots(
            **override_kws(
                subplot_kw,
                {"grid_size": (n_trace, n_event)},
                ax_size=(2, 2),
                sharex=True,
                sharey=True,
            )
        )
    else:
        fig = ax[0, 0].get_figure()

    # Build time grid:
    # - default to time 0 = index 0
    t = _time_grid(1, n_frames, fs, zero or 0).squeeze(0)  # shape: (1, n_frames,)

    # For each axis (i_trace, i_event)
    # Select color
    # used for all, once passed through vu.lighten

    # Plot in stacking order to ax[i_trace, i_event]:
    # Other than popped lighten kw, use {**{defaults}, **kw} pattern for
    # passed kwargs
    # If relevant data is None, simply do not plot

    # Plot stds
    # fill_between, vu.lighten(color, pct)
    # default edge width 0
    # std_kw.pop('lighten') w/ default 0.8 lighten pct

    # Plot sems
    # fill_between, vu.lighten(color, pct)
    # default edge width 0
    # sem_kw.pop('lighten') w/ default 0.4 lighten pct

    # Plot traces
    # LineCollection x = (1, nframe), y = (<n_event>, nframe)
    # default linewidth=0.5
    # trace_kw.pop('lighten') w/ default 0.4 lighten pct

    # Plot means
    # default lw 1

    # Ensure ax is indexable in 2D
    ax_arr = np.asarray(ax)
    try:
        ax_arr = ax_arr.reshape(n_trace, n_event)
    except Exception:
        raise ValueError(f"Could not reshape ax to {(n_trace, n_event)}")
    if ax_arr.shape != (n_trace, n_event):
        raise ValueError(f"Expected ax shape {(n_trace, n_event)}, got {ax_arr.shape}")

    std_lighten = std_kw.pop("lighten", 0.8)
    sem_lighten = sem_kw.pop("lighten", 0.4)
    trace_lighten = trace_kw.pop("lighten", 0.4)

    for i_trace in range(n_trace):
        for i_event in range(n_event):
            ax_i = ax_arr[i_trace, i_event]

            default = ax_i._get_lines.get_next_color()
            color = _pick_color(
                i_trace,
                i_event,
                default_color=default,
                trace_colors=trace_colors,
                event_colors=event_colors,
            )

            # Set column titles (top row) and row ylabels (first column) if names provided
            if event_names is not None and i_trace == 0:
                try:
                    ax_i.set_title(str(event_names[i_event]))
                except Exception:
                    # fallback to first element or empty string
                    try:
                        ax_i.set_title(str(event_names[0]))
                    except Exception:
                        ax_i.set_title("")
            if trace_names is not None and i_event == 0:
                try:
                    ax_i.set_ylabel(str(trace_names[i_trace]))
                except Exception:
                    try:
                        ax_i.set_ylabel(str(trace_names[0]))
                    except Exception:
                        ax_i.set_ylabel("")

            # Plot stds (also requires mean to define center)
            if (std is not None) and (mean is not None):
                lower = np.asarray(mean[i_trace, i_event]) - np.asarray(
                    std[i_trace, i_event]
                )
                upper = np.asarray(mean[i_trace, i_event]) + np.asarray(
                    std[i_trace, i_event]
                )
                ax_i.fill_between(
                    t,
                    lower,
                    upper,
                    **override_kws(
                        std_kw,
                        color=vu.lighten(color, std_lighten),
                        linewidth=0,
                    ),
                )

            # Plot sems (also requires mean to define center)
            if (sem is not None) and (mean is not None):
                lower = np.asarray(mean[i_trace, i_event]) - np.asarray(
                    sem[i_trace, i_event]
                )
                upper = np.asarray(mean[i_trace, i_event]) + np.asarray(
                    sem[i_trace, i_event]
                )
                ax_i.fill_between(
                    t,
                    lower,
                    upper,
                    **override_kws(
                        sem_kw,
                        color=vu.lighten(color, sem_lighten),
                        linewidth=0,
                    ),
                )

            # Plot traces (individual trials) if provided
            if traces is not None:
                try:
                    ys = np.asarray(traces[i_trace, i_event])
                except Exception:
                    ys = None
                if ys is not None:
                    # If ys is 2D (n_trials, n_frames) produce LineCollection
                    if ys.ndim == 2:
                        segments = [np.column_stack((t, y)) for y in ys]
                        lc = LineCollection(
                            segments,
                            **override_kws(
                                trace_kw,
                                colors=vu.lighten(color, trace_lighten),
                                linewidths=0.5,
                            ),
                        )
                        ax_i.add_collection(lc)
                        ax_i.autoscale_view()

            # Plot mean if provided
            if mean is not None:
                y_mean = np.asarray(mean[i_trace, i_event])
                ax_i.plot(t, y_mean, **override_kws(mean_kw, color=color, linewidth=1))

            # Tidy axis: set x-limits to time range
            if t.size > 0:
                ax_i.set_xlim(t[0], t[-1])

    return fig, ax
