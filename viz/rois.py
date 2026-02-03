from ..timeseries.rois import ROICollection
from ..io.svd_video import SVDVideo
from ..timeseries.types import Traces, Events
from ..roigui.roi import compute_boundary_edges

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Dict, Any
import scipy.stats
from matplotlib.collections import LineCollection
from matplotlib.patheffects import Stroke, Normal


def _ensure_axis(
    fig: Optional[plt.Figure],
    ax: Optional[plt.Axes],
    shape: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
):
    """Return (fig, ax). Create a new figure/axis if not provided.

    Keeps caller pseudocode comments about aspect ratio handling.
    """
    if ax is None:
        if fig is None:
            if shape is None:
                fig, ax = plt.subplots()
            else:
                h, w = shape
                fig, ax = plt.subplots(figsize=(w / scale, h / scale))
        else:
            ax = fig.add_subplot(111)
    else:
        fig = fig or ax.figure
    return fig, ax


def _infer_names_and_colors(
    roi_collection: ROICollection, select_idx: Sequence[int] | None = None
):
    """Infer display names and per-ROI colors.

    Returns (names, colors) where names is list[str] and colors is list of matplotlib color specs.
    """
    import matplotlib as mpl

    n = len(roi_collection.rois)
    if select_idx is not None:
        n = len(select_idx)

    # names
    if getattr(roi_collection, "ids", None):
        names_all = list(roi_collection.ids)
    else:
        names_all = [f"ROI {i}" for i in range(len(roi_collection.rois))]

    if select_idx is None:
        names = names_all
    else:
        names = [names_all[i] for i in select_idx]

    # colors
    if getattr(roi_collection, "colors", None):
        colors_all = list(roi_collection.colors)
    else:
        cmap = plt.get_cmap("tab10")
        colors_all = [cmap(i % cmap.N) for i in range(len(roi_collection.rois))]

    if select_idx is None:
        colors = colors_all
    else:
        colors = [colors_all[i] for i in select_idx]

    return names, colors


def mean_image(svd_video: SVDVideo) -> np.ndarray:
    """Compute mean image in SVD video space.

    The mean image across time is computed without reconstructing full video by
    taking the temporal mean of U, scaling by singular values, and multiplying
    with Vt: mean(U, axis=0) * S @ Vt.

    Returns
    -------
    image : ndarray, shape spatial...
    """
    # Compute mean(U) per component
    xp = np
    U = np.asarray(svd_video.U)
    S = np.asarray(svd_video.S)
    Vt = np.asarray(svd_video.Vt)

    mean_u = U.mean(axis=0)  # (rank,)
    coeff = mean_u * S  # (rank,)

    rank = coeff.shape[0]
    spatial_shape = Vt.shape[1:]

    # Flatten spatial dims, multiply and reshape
    Vt_flat = Vt.reshape(rank, -1)  # (rank, n_pixels)
    img_flat = coeff @ Vt_flat  # (n_pixels,)
    image = img_flat.reshape(spatial_shape)
    return image


def gamma_correct(images: np.ndarray, target: float = 0.5) -> np.ndarray:
    """Gamma-correct image(s) to push their geometric mean intensity toward target.

    images : array, shape (..., H, W)
    target : desired geometric mean in normalized [0, 1] space (default 0.5)

    Steps kept as pseudocode comments in the function body.
    """
    # 1. Normalize each image [0, 1]
    img = np.asarray(images)
    orig_shape = img.shape
    if img.ndim == 2:
        imgs = img[None, ...]
    else:
        imgs = img.reshape((-1,) + orig_shape[-2:])

    out = np.empty_like(imgs, dtype=float)
    eps = 1e-12

    for i, im in enumerate(imgs):
        minv = float(np.nanmin(im))
        maxv = float(np.nanmax(im))
        if maxv - minv < eps:
            norm = np.zeros_like(im, dtype=float)
        else:
            norm = (im - minv) / (maxv - minv)

        # 2. Compute geometric mean intensity for each image
        gmean = float(scipy.stats.gmean(np.clip(norm.ravel(), eps, None)))

        # 3. compute gamma = ln(target) / ln(mean) and apply
        if gmean <= 0:
            gamma = 1.0
        else:
            gamma = np.log(target) / np.log(gmean)
        corrected = np.clip(norm, 0.0, 1.0) ** gamma
        out[i] = corrected

    if img.ndim == 2:
        return out[0]
    else:
        return out.reshape(orig_shape)


def overlay_rois(
    roi_collection: ROICollection,
    names: Sequence[str],
    ax: plt.Axes,
    colors: Sequence[Any],
    boundary_kws: Optional[Dict[str, Any]] = None,
    text_kws: Optional[Dict[str, Any]] = None,
    stroke_kws: Optional[Dict[str, Any]] = None,
):
    """Display polygons mapping ROI edges.

    See also `roigui.roi.compute_boundary_edges`

    Parameters
    ----------
    names : list
    colors : dict or list
        Values convertible to RGB by matplotlib. Treated as cyclic if list.
    boundary_kws : dict, optional
        Additional kwargs forwarded to matplotlib.collections.LineCollection. Keys
        provided here override the defaults used for color and linewidth.
    text_kws : dict, optional
        Additional kwargs forwarded to ``ax.text``. Keys provided here override
        the default color/fontsize.
    stroke_kws : dict, optional
        Additional kwargs forwarded to matplotlib.patheffects.Stroke used for
        the text outline. Keys provided here override the default linewidth/
        foreground.

    Pseudocode comments for steps are retained in the body.
    """
    # Ensure kw dicts are not None
    boundary_kws = {} if boundary_kws is None else dict(boundary_kws)
    text_kws = {} if text_kws is None else dict(text_kws)
    stroke_kws = {} if stroke_kws is None else dict(stroke_kws)

    # For each ROI
    for i, roi in enumerate(roi_collection.rois):
        name = names[i] if i < len(names) else f"ROI {i}"
        color = colors[i % len(colors)] if len(colors) > 0 else "r"

        #   1. Compute boundaries
        boundary_edges = compute_boundary_edges(roi.footprint)  # [(r0, c0), (r1, c1)]
        coords = np.array(list(boundary_edges))  # (N, start/end, row/col)

        #   3. Create linecollection
        #   4. Create text artist
        if len(boundary_edges):
            # default kwargs for LineCollection; allow boundary_kws to override
            lc_defaults = {"colors": color, "linewidths": 1}
            lc_kwargs = {**lc_defaults, **boundary_kws}
            lines = LineCollection(coords[:, :, ::-1], **lc_kwargs)
            ax.add_collection(lines)
            # place label at centroid
            centroid = (
                roi.footprint.mean(axis=0)
                if len(roi.footprint) > 0
                else np.array([0, 0])
            )
            # default text kwargs; allow text_kws to override
            txt_defaults = {"color": "k", "fontsize": 10}
            txt_kwargs = {**txt_defaults, **text_kws}
            text = ax.text(centroid[1], centroid[0], name, **txt_kwargs)
            # default stroke kwargs; allow stroke_kws to override
            stroke_defaults = {"linewidth": 1, "foreground": "white"}
            stroke_kwargs = {**stroke_defaults, **stroke_kws}
            text.set_path_effects([Stroke(**stroke_kwargs), Normal()])


def display_rois(
    roi_collection: ROICollection,
    svd_video: SVDVideo = None,
    target_gamma: Optional[float] = None,
    fig: Optional[plt.Figure] = None,
    axis: Optional[plt.Axes] = None,
    dpi: Optional[float] = None,
    boundary_kws: Optional[Dict[str, Any]] = None,
    text_kws: Optional[Dict[str, Any]] = None,
    stroke_kws: Optional[Dict[str, Any]] = None,
):
    """Display mean image from SVD video with ROI overlays.

    Parameters
    ----------
    svd_video : SVDVideo
    roi_collection : ROICollection
    target_gamma : float, optional
        If provided, gamma-correct the mean image toward this target geometric
        mean (in normalized [0,1] space).
    fig, axis : matplotlib objects, optional
        If axis is provided it will be used, otherwise a new figure/axis is
        created.
    dpi : float, optional
        If provided, sets figure size as multiple of image size in inches.

    Returns
    -------
    fig, axis
    """

    if svd_video is not None:
        # 1. Load SVD video and compute mean image
        display_img = mean_image(svd_video)

        # 2. gamma-correct mean image
        if target_gamma is not None:
            display_img = gamma_correct(display_img, target=target_gamma)

        shape = display_img.shape
    else:
        print("no svd video: shape: ", roi_collection.image_shape)
        shape = roi_collection.image_shape

        if shape is None:
            # default size if no video or inferreable shape
            shape = (6 * (dpi or 100.0), (8 * (dpi or 100.0)))

    # 3. create figure/axis and display
    fig, ax = _ensure_axis(fig, axis, shape=shape, scale=dpi or 100.0)

    if svd_video is not None:
        ax.imshow(display_img, cmap="gray", origin="upper")
        ax.set_axis_off()

    # 4. overlay ROIs
    names, colors = _infer_names_and_colors(roi_collection)
    overlay_rois(
        roi_collection,
        names,
        ax,
        colors,
        boundary_kws=boundary_kws,
        text_kws=text_kws,
        stroke_kws=stroke_kws,
    )

    return fig, ax


def display_traces(
    traces: Traces,
    roi_collection: ROICollection,
    select: Optional[Sequence[int] | Sequence[str]] = None,
    fig: Optional[plt.Figure] = None,
    axis: Optional[plt.Axes] = None,
    ticks: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Plot traces for a set of ROIs.

    select : list of indices or ids

    Returns metadata dict with 'positions' (y positions), 'colors', and 'names'.
    """
    data = np.asarray(traces.data)
    n_cells, n_frames = data.shape

    # Convert select to indices
    if select is None:
        sel_idx = list(range(n_cells))
    else:
        # If select contains strings and roi_collection.ids present, map them
        if (
            len(select) > 0
            and isinstance(select[0], str)
            and getattr(roi_collection, "ids", None)
        ):
            ids = list(roi_collection.ids)
            sel_idx = [ids.index(s) for s in select]
        else:
            sel_idx = list(select)

    # Infer / default metadata
    names, colors = _infer_names_and_colors(roi_collection, sel_idx)

    # fs/time axis
    if traces.fs is not None:
        x = np.arange(n_frames) / traces.fs
        xlabel = "time (s)"
    else:
        x = np.arange(n_frames)
        xlabel = "frame"

    # Calculate a good vertical offset between the traces
    spans = [np.nanmax(data[i]) - np.nanmin(data[i]) for i in sel_idx]
    span = max(spans) if len(spans) > 0 else 1.0
    offset = span * 1.2

    # Plot traces
    fig, ax = _ensure_axis(fig, axis)
    positions = []
    for k, (i, name, color) in enumerate(zip(sel_idx, names, colors)):
        y = data[i] + k * offset
        ax.plot(x, y, color=color)
        positions.append(k * offset)

    # Labeled ticks around the traces & axis labels
    if ticks is not None:
        # Interleave trace label names with data coordinate ticks
        ticks = np.array(ticks, dtype=float)
        if 0. not in ticks:
            ticks = np.concatenate(([0.0], ticks))
        zero_idx = np.where(ticks == 0.0)[0][0]
        tick_locations = np.array(positions)[None, :] + np.array(ticks)[:, None]
        tick_labels = np.full(tick_locations.shape, "", dtype=object)
        tick_labels[zero_idx, :] = names
        # Flatten and sort
        tick_sort = np.argsort(tick_locations.ravel())
        tick_locations = tick_locations.ravel()[tick_sort]
        tick_labels = tick_labels.ravel()[tick_sort]
    else:
        # Only label trace ids
        tick_locations = positions
        tick_labels = names

    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel(xlabel)

    return fig, ax, {"positions": positions, "colors": colors, "names": names}


def display_events(
    events: Events,
    traces: Optional[Traces] = None,
    trace_locations: Optional[Sequence[Tuple[float, float]]] = None,
    select: Optional[Sequence[int] | Sequence[str]] = None,
    fig: Optional[plt.Figure] = None,
    axis: Optional[plt.Axes] = None,
    **plot_kws,
) -> Dict[str, Any]:
    """Display events aligned to traces or as separate event rows."""
    # Convert select similar to display_traces
    n = len(events.spike_frames)
    if select is None:
        sel_idx = list(range(n))
    else:
        if (
            len(select) > 0
            and isinstance(select[0], str)
            and getattr(events, "ids", None)
        ):
            ids = list(events.ids)
            sel_idx = [ids.index(s) for s in select]
        else:
            sel_idx = list(select)

    # x axis conversion
    if traces is not None and traces.fs is not None:
        to_x = lambda frames: np.asarray(frames) / traces.fs
        xlabel = "time (s)"
    else:
        to_x = lambda frames: np.asarray(frames)
        xlabel = "frame"

    fig, ax = _ensure_axis(fig, axis)

    plotted = []
    for k, i in enumerate(sel_idx):
        spikes = np.atleast_1d(events.spike_frames[i])
        xvals = to_x(spikes)

        if traces is not None:
            # plot events as points along the trace
            y_trace = traces.data[i]
            # sample y positions at integer frame indices (clip indices)
            idx = np.clip(spikes.astype(int), 0, y_trace.size - 1)
            yvals = y_trace[idx]
            plot_kws = {**dict(
                linestyle="None",
                marker=".",
                label=getattr(events, "ids", None) and events.ids[i],
            ), **plot_kws}
            ax.plot(
                xvals,
                yvals,
                **plot_kws
            )
        elif trace_locations is not None:
            # events as vertical bars matching extent
            ymin, ymax = trace_locations[i]
            for xv in xvals:
                ax.vlines(xv, ymin, ymax, **plot_kws)
        else:
            plot_kws = {**dict(
                linestyle="None", marker="|",
                label=getattr(events, "ids", None) and events.ids[i],
            ), **plot_kws}
            # events as points on separate row
            y = -k  # separate row per ROI
            ax.plot(xvals, np.full_like(xvals, y), **plot_kws)
        plotted.append(i)

    ax.set_xlabel(xlabel)
    return {"plotted": plotted}
