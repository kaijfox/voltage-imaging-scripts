import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Union

from imaging_scripts.windows.ragged_ops import slice_by_events
from imaging_scripts.viz.rois import apply_cmap, setup_video
from imaging_scripts.viz.psth_grid import mean_psth_grid
from imaging_scripts.viz.grid_dispatchers import _finalize_axes_with_scalebars


def extract_mean_videos(
    svd_video,
    event_frames,
    fs: float,
    pre_ms: float,
    post_ms: float,
    max_rank: Optional[int] = None,
    row_slice: slice = slice(None),
    col_slice: slice = slice(None),
) -> np.ndarray:
    """
    Compute event-triggered mean videos from SVD components.

    Parameters
    ----------
    svd_video : object
        Object with attributes:
            U : np.ndarray, shape (n_t, rank)
                Temporal components.
            S : np.ndarray, shape (rank,)
                Singular values / component scalings.
            Vt : np.ndarray, shape (rank, row, col)
                Spatial components.
    event_frames : Sequence[Sequence[int]]
        Per-batch sequence of event frame indices (ragged). For a 1-D batch
        this is e.g. [np.array([10, 20]), np.array([5])].
    fs : float
        Sampling rate in Hz.
    pre_ms : float
        Time before event to include, in milliseconds.
    post_ms : float
        Time after event to include, in milliseconds.
    max_rank : positive int or None, default None
        Maximum rank to use for reconstruction. If None, use all available.
    row_slice, col_slice : slice, default slice(None)
        Spatial slicing applied to Vt before reconstruction.

    Returns
    -------
    np.ndarray
        Array of shape (batch..., window_t, row, col) where
        window_t = round(pre_ms*fs/1000) + round(post_ms*fs/1000) + 1.
    """
    # number of frames before/after event (inclusive)
    n_pre = int(round(pre_ms * fs / 1000.0))
    n_post = int(round(post_ms * fs / 1000.0))

    # materialize arrays
    U = np.asarray(svd_video.U)
    S = np.asarray(svd_video.S)
    Vt = np.asarray(svd_video.Vt)

    # determine rank to use
    rank_avail = U.shape[1]
    r = rank_avail if (max_rank is None) else min(int(max_rank), rank_avail)

    # prepare temporal components and events for slicing
    events = ak.Array(event_frames)
    n_batch_dim = events.ndim - 1
    U_T = U[:, :r].T  # (r, n_t)
    # insert singleton axes so U_T and events have matching ndim
    idx = (slice(None),) + (None,) * (n_batch_dim) + (slice(None),)
    U_T = U_T[idx]
    events_exp = events[None]  # (1, *batch, <n_events>)

    # extract windows (r, *batch, <n_events>, window_t)
    sampled = slice_by_events(U_T, events_exp, n_pre, n_post)

    # mean over ragged events axis -> (r, *batch, window_t)
    mean_U = ak.mean(sampled, axis=-2)
    mean_U_np = ak.to_numpy(mean_U)  # now regular

    # reconstruct frames: contract over rank dimension
    Vt_sel = Vt[:r, row_slice, col_slice]
    scaled_U = mean_U_np * S[:r, *((None,) * (n_batch_dim + 1))]

    # dot over rank axis -> (*batch, window_t, rows, cols)
    frames = np.tensordot(scaled_U, Vt_sel, axes=([0], [0]))

    return frames


def slice_videos_by_events(
    svd_video,
    event_frames,
    fs: float,
    pre_ms: float,
    post_ms: float,
    max_rank: Optional[int] = None,
    row_slice: slice = slice(None),
    col_slice: slice = slice(None),
    spatial_mask = None,
) -> ak.Array:
    """
    Compute per-event triggered videos from SVD components (no averaging).

    Parameters
    ----------
    svd_video : object
        Object with attributes U (n_t, rank), S (rank,), Vt (rank, row, col).
    event_frames : Sequence[Sequence[int]]
        Per-batch ragged event frame indices.
    fs : float
        Sampling rate in Hz.
    pre_ms, post_ms : float
        Window bounds in milliseconds.
    max_rank : int or None
        Maximum SVD rank to use.
    row_slice, col_slice : slice
        Spatial slicing applied to Vt.

    Returns
    -------
    ak.Array
        Ragged array of shape (*batch, <n_events>, window_t, row, col).
    """
    from ..windows.ragged_ops import ak_flatten_n_times, ak_infer_shape, ak_unflatten_by_batch_shape

    n_pre = int(round(pre_ms * fs / 1000.0))
    n_post = int(round(post_ms * fs / 1000.0))

    U = np.asarray(svd_video.U)
    S = np.asarray(svd_video.S)
    Vt = np.asarray(svd_video.Vt)

    rank_avail = U.shape[1]
    r = rank_avail if (max_rank is None) else min(int(max_rank), rank_avail)

    events = ak.Array(event_frames)
    n_batch_dim = events.ndim - 1
    U_T = U[:, :r].T  # (r, n_t)
    idx = (slice(None),) + (None,) * n_batch_dim + (slice(None),)
    U_T = U_T[idx]
    events_exp = events[None]  # (1, *batch, <n_events>)

    # extract windows: (r, *batch, <n_events>, window_t)
    U_sampled = slice_by_events(U_T, events_exp, n_pre, n_post)

    # reshape to remove batch dimensions
    # -> (r, prod(batch) * event, window_t)
    n_events = ak.num(events, axis=-1)
    batch_shape = ak_infer_shape(events)[:-1]
    U_sampled_np = ak_flatten_n_times(U_sampled, n_batch_dim, axis=-2)
    U_sampled_np = ak.to_numpy(U_sampled_np)

    # reconstruct frames: contract over rank dimension
    # -> (prod(batch) * event, window_t, rows, cols)
    if spatial_mask is not None:
        Vt_sel = Vt[:r][:, spatial_mask]
    else:
        Vt_sel = Vt[:r, row_slice, col_slice]
    scaled_U = U_sampled_np * S[:r, None, None]
    frames_flat = np.tensordot(scaled_U, Vt_sel, axes=([0], [0]))

    # reshape back to ragged
    # -> (*batch, <event>, window_t, rows, cols)
    frames = ak_unflatten_by_batch_shape(frames_flat, n_events, batch_shape, axis=0)
    # print("ragged frames:", ak_infer_shape(frames))

    return frames






def video_and_trace(
    output_path,
    trace_mean: np.ndarray,
    frames: np.ndarray,
    fs: float,
    pre_ms: float,
    roi_collection,
    ms_per_s: float = 20.0,
    height: float = 2.0,
    roi_name: str = "",
    n_events: int = 0,
    cmap: str = "RdBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    x_scalebar: Optional[Union[float, dict]] = None,
    y_scalebar: Optional[Union[float, dict]] = None,
) -> None:
    """
    Render an mp4 file showing the mean trace and the mean video side-by-side.

    Parameters
    ----------
    output_path : str or Path
        Path to the output mp4 file. Parent directories will be created.
    trace_mean : np.ndarray, shape (window_t,)
        Pre-windowed mean trace.
    frames : np.ndarray, shape (window_t, row, col)
        Video frames for the same window.
    fs : float
        Sampling rate (Hz).
    pre_ms : float
        Pre-window offset in milliseconds (used for playback line zero).
    ms_per_s : float, default 20.0
        Real milliseconds per second of the output video (controls playback speed).
    roi_collection : optional
        ROICollection-like object; if provided, used for footprint and colors.
    roi_name : str
        Optional ROI name added to the title.

    Returns
    -------
    None
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trace_mean = np.asarray(trace_mean)
    frames = np.asarray(frames)

    if trace_mean.shape[0] != frames.shape[0]:
        raise ValueError("trace_mean and frames must have the same temporal length")

    zero = int(round(pre_ms * fs / 1000.0))

    # colormap for video
    if frames.ndim == 4 and frames.shape[-1] == 3:
        # Already RGB frames (T, H, W, 3); setup_video handles RGB input.
        color_frames = frames
        _use_colorbar = False
    else:
        _use_colorbar = True
        if vmin is None and vmax is None:
            vmax = np.abs(frames).max()
            vmin = -vmax
            color_frames = apply_cmap(
                frames, cmap=cmap, vmin=vmin, vmax=vmax, mode="centered"
            )
        else:
            vmin = frames.min() if vmin is None else vmin
            vmax = frames.max() if vmax is None else vmax
            color_frames = apply_cmap(
                frames, cmap=cmap, vmin=vmin, vmax=vmax, mode="absolute"
            )

    # figure + axes
    vid_width = height * frames.shape[2] / frames.shape[1]
    trace_width = height
    fig, (vid_ax, trace_ax) = plt.subplots(
        1,
        2,
        figsize=(vid_width + trace_width + 1, height),
        gridspec_kw={"width_ratios": [vid_width, height]},
    )

    # setup video display
    im_artist, update_fn = setup_video(
        color_frames,
        vid_ax,
        roi_collection=roi_collection,
        roi_kws={"text_kws": {"do_text": False}},
    )
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if _use_colorbar:
        fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
            ax=vid_ax,
            fraction=0.046,
            pad=0.04,
        )

    # trace color
    trace_color = roi_collection.colors[0]

    # convert scalebar floats to dict
    if isinstance(x_scalebar, (int, float)):
        x_scalebar = dict(size=float(x_scalebar))
    if isinstance(y_scalebar, (int, float)):
        y_scalebar = dict(size=float(y_scalebar))

    # render trace using mean_psth_grid
    ax_grid = np.array([[trace_ax]])
    trace_colors = [trace_color] if trace_color is not None else None
    mean_psth_grid(
        mean=trace_mean[None, None, :],
        fs=fs / 1000.0,
        zero=zero,
        trace_colors=trace_colors,
        ax=ax_grid,
    )

    # finalize scalebars and aesthetics
    _finalize_axes_with_scalebars(ax_grid, x_kws=x_scalebar, y_kws=y_scalebar)
    trace_ax.set_title(f"{roi_name}, n={n_events}")
    fig.tight_layout()

    # playback line (x axis is ms)
    fs_khz = fs / 1000.0
    x0 = -zero / float(fs_khz)
    line = trace_ax.axvline(x=x0, color="k", lw=0.5)

    n_frames = frames.shape[0]
    output_fps = float(ms_per_s) * float(fs) / 1000.0
    if output_fps <= 0:
        output_fps = float(fs)

    def update(frame_i):
        artists = update_fn(frame_i)
        pos = (frame_i - zero) / float(fs_khz)
        line.set_xdata([pos, pos])
        try:
            artists_t = tuple(artists)
        except Exception:
            artists_t = (artists,)
        return artists_t + (line,)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    try:
        anim.save(str(out_path), dpi=150, fps=max(1.0, float(output_fps)))
    finally:
        plt.close(fig)

    return None
