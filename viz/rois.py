from ..timeseries.rois import ROICollection
from ..timeseries.types import Traces, Events
from .traces import _resolve_selection, _compute_positions, _infer_names_and_colors

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Dict, Any
import scipy.stats
from matplotlib.collections import LineCollection
from matplotlib.patheffects import Stroke, Normal

# SVDVideo contains some python 3.11-specific features
try:
    from ..io.svd_video import SVDVideo
except SyntaxError:
    pass

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


def get_pixel_edges(r: int, c: int) -> tuple:
    """Return the 4 edges of pixel (r, c) as corner coordinate pairs.

    Corners of pixel (r, c): (r, c), (r, c+1), (r+1, c), (r+1, c+1)
    Returns edges as (start_corner, end_corner) tuples, normalized so start < end.
    """
    return (
        ((r, c), (r, c + 1)),      # top
        ((r + 1, c), (r + 1, c + 1)),  # bottom
        ((r, c), (r + 1, c)),      # left
        ((r, c + 1), (r + 1, c + 1)),  # right
    )


def compute_boundary_edges(pixel_set: set) -> set:
    """Compute all boundary edges from a set of pixel coordinates.

    An edge is on the boundary iff exactly one of its adjacent pixels is in the set.
    """
    edge_counts = {}
    for r, c in pixel_set:
        for edge in get_pixel_edges(r, c):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    # Boundary edges appear exactly once (interior edges appear twice)
    return {edge for edge, count in edge_counts.items() if count == 1}


def footprint_mask(image_shape, footprint):
    mask = np.zeros(image_shape, dtype=bool)
    mask[footprint[:, 0], footprint[:, 1]] = True
    return mask


def mean_image(svd_video: "SVDVideo") -> np.ndarray:
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
    img = np.asarray(images)
    orig_shape = img.shape
    if img.ndim == 2:
        imgs = img[None, ...]
    else:
        imgs = img.reshape((-1,) + orig_shape[-2:])

    out = np.empty_like(imgs, dtype=float)
    eps = 1e-12

    for i, im in enumerate(imgs):
        # 1. Normalize each image [0, 1]
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

# def quantile_normalize(arr, n_quantiles=256, vmin=None, vmax=None, cmap=None)

from matplotlib.colors import LinearSegmentedColormap

def quantile_normalize(
    arr: np.ndarray,
    n_quantiles: int = 256,
    vmin: float | None = None,
    vmax: float | None = None,
    gamma: float | None = None,
    cmap=None,
):
    """Map values to a uniform [0,1] using an interpolated inverse CDF (quantile normalization).

    Map an array to a uniform distribution by approximating the inverse CDF using
    N quantiles. Data are first linearly scaled to [0,1] using ``vmin``/``vmax``
    (inferred from finite data if not provided) and clipped; quantiles are
    computed on that scaled range.

    Parameters
    ----------
    arr: ndarray
        Input array of any shape. NaNs are preserved and excluded from fitting.
    n_quantiles: int, default 256
        Number of quantile levels used to approximate the CDF.
    vmin, vmax: float or None
        Minimum and maximum used to linearly scale the data before quantile
        estimation. If ``None`` they are inferred from the finite values of
        ``arr``. If ``vmin == vmax`` a ValueError is raised.
    cmap: optional
        Anything accepted by ``plt.get_cmap``. If provided the function will
        return a ``LinearSegmentedColormap`` whose color sampling follows the
        inverse CDF; if ``None`` the function returns the normalized array.

    Returns
    -------
    ndarray or matplotlib.colors.Colormap
        If ``cmap`` is ``None`` returns a float ndarray with the same shape as
        ``arr`` with values in [0,1] (NaNs preserved). If ``cmap`` is provided
        returns a ``LinearSegmentedColormap`` constructed by sampling the base
        colormap through the inverse CDF.
    """
    arr = np.asarray(arr)
    
    # Infer vmin/vmax if not provided
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))

    if vmin == vmax:
        raise ValueError("vmin and vmax must differ (constant data cannot be scaled).")

    # Scale valid data to [0,1] and clip
    scaled = (arr - vmin) / (vmax - vmin)

    # Only fit quantiles on non-nan values within [vmin, vmax]
    flat = scaled.ravel()
    finite_mask = np.isfinite(flat) & (flat >= 0) & (flat <= 1)
    valid = flat[finite_mask]

    # Compute quantiles on the scaled data
    q = np.linspace(0.0, 1.0, n_quantiles)
    v = np.quantile(valid, q)
    
    # Deduplicate equal quantile values for safe interpolation
    uvals, idx = np.unique(v, return_index=True)
    uq = q[idx]

    if uq.size < 2:
        raise ValueError(
            "Input array must contain at least two unique finite values after scaling."
        )

    # Map scaled values -> uniform [0,1] via interpolated inverse CDF (value -> prob)
    if cmap is None:
        arr_scaled = (arr - vmin) / (vmax - vmin)
        mapped = np.interp(arr_scaled, uvals, uq, left=0.0, right=1.0)
        return mapped

    # Build a new colormap sampled through the inverse CDF: cmap_new(q) = base_cmap(icdf(q))
    base_cmap = plt.get_cmap(cmap)
    normed_val = np.linspace(0.0, 1.0, 256)
    umin, umax = uvals[0], uvals[-1]
    gamma = gamma if gamma is not None else 1.0
    icdf_values = np.interp(
        normed_val,
        ((uvals - umin) / (umax - umin)) ** gamma,
        uq,
        left=0.,
        right=1.0
    )
    cmap_name = f"quantile_{getattr(base_cmap, 'name', str(base_cmap))}"
    new_cmap = LinearSegmentedColormap.from_list(
        cmap_name, base_cmap(icdf_values), N=len(normed_val)
    )

    return new_cmap


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
        the default color/fontsize. Optionally may contain do_text: bool
        (default True) to disable text labels.
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

    text_artists = {}
    line_artists = {}

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
            line_artists[i] = lines
            if text_kws.get("do_text", True) is True:
                # place label at centroid
                centroid = (
                    roi.footprint.mean(axis=0)
                    if len(roi.footprint) > 0
                    else np.array([0, 0])
                )
                # default text kwargs; allow text_kws to override
                txt_defaults = {"color": color, "fontsize": 10}
                txt_kwargs = {**txt_defaults, **text_kws}
                text = ax.text(centroid[1], centroid[0], name, **txt_kwargs)
                # default stroke kwargs; allow stroke_kws to override
                stroke_defaults = {"linewidth": 0.5, "foreground": "white"}
                stroke_kwargs = {**stroke_defaults, **stroke_kws}
                text.set_path_effects([Stroke(**stroke_kwargs), Normal()])
                text_artists[i] = text

    return text_artists, line_artists


def apply_cmap(
    video: np.ndarray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mode: str = "absolute",
) -> np.ndarray:
    """Apply colormap to monochrome (T,H,W) video. Returns (T,H,W,4) RGBA float.

    mode: 'absolute' (min/max) or 'centered' (+/- max(abs)).
    """
    # 1. compute vmin, vmax from mode
    # 2. normalize to [0,1] with clip
    # 3. apply plt.get_cmap(cmap) over flattened array, reshape back
    vid = np.asarray(video)

    # Accept single-frame (H,W) as convenience
    added_axis = False
    if vid.ndim == 2:
        vid = vid[None, ...]
        added_axis = True

    if vid.ndim != 3:
        raise ValueError("apply_cmap expects a monochrome video of shape (T,H,W)")

    # compute vmin/vmax
    if mode == "absolute":
        vmin = float(np.nanmin(vid)) if vmin is None else vmin
        vmax = float(np.nanmax(vid)) if vmax is None else vmax
    elif mode == "centered":
        m = max(abs(float(np.nanmin(vid))), abs(float(np.nanmax(vid))))
        vmin = -m if vmin is None else vmin
        vmax = m if vmax is None else vmax
    else:
        raise ValueError("mode must be 'absolute' or 'centered'")

    T, H, W = vid.shape

    # Handle constant video: map to midpoint 0.5
    eps = 1e-12
    if vmax - vmin < eps:
        norm = np.full((T, H, W), 0.5, dtype=float)
    else:
        norm = (vid.astype(float) - vmin) / (vmax - vmin)
        np.clip(norm, 0.0, 1.0, out=norm)

    # get colormap
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, (str,)) else cmap

    # apply colormap over flattened data and reshape to (T,H,W,4)
    flat = norm.ravel()
    rgba_flat = cmap_obj(flat)
    rgba = rgba_flat.reshape((T, H, W, 4))

    # Ensure float dtype
    return rgba.astype(float)


def setup_video(
    video: np.ndarray,
    ax: plt.Axes,
    roi_collection: Optional[ROICollection] = None,
    roi_kws: Optional[Dict[str, Any]] = None,
    extent: Tuple[int, int, int, int] = None,
) -> Tuple:
    """Create imshow artist + update closure for FuncAnimation.

    video: (T,H,W) monochrome or (T,H,W,3|4) color.
    roi_kws: passed through to overlay_rois (keys: names, colors,
             boundary_kws, text_kws, stroke_kws).
    extent: tuple (left, right, bottom, top)
        Rectangle where video should be displayed in data coords.
    Returns (im_artist, update_fn) where update_fn(frame_idx) sets frame data.
    """
    # 1. imshow first frame; use cmap='gray' + clim for monochrome
    # 2. ax.set_axis_off()
    # 3. if roi_collection: infer names/colors, call overlay_rois
    # 4. build update_fn closure: im.set_data(video[frame_idx])
    vid = np.asarray(video)

    if vid.ndim == 2:
        # single frame -> treat as single-frame monochrome video
        vid = vid[None, ...]

    if vid.ndim == 3:
        # monochrome (T,H,W)
        mono = True
    elif vid.ndim == 4 and vid.shape[-1] in (3, 4):
        # color video (T,H,W,3|4)
        mono = False
    else:
        raise ValueError("video must be shape (T,H,W) or (T,H,W,3|4)")

    # Parse (left, right, bottom, top) from extent to imshow extent argument
    imshow_extent = None
    if extent is not None:
        left, right, bottom, top = extent
        imshow_extent = (left, right, bottom, top)

    # display first frame
    if mono:
        # compute stable clim across video
        vmin = float(np.nanmin(vid))
        vmax = float(np.nanmax(vid))
        im = ax.imshow(
            vid[0],
            cmap="gray",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            extent=imshow_extent,
        )
    else:
        im = ax.imshow(vid[0], origin="upper", extent=imshow_extent)

    ax.set_axis_off()

    # overlay ROIs if provided
    if roi_collection is not None:
        inferred_names, inferred_colors = _infer_names_and_colors(roi_collection)
        roi_kws = {} if roi_kws is None else dict(roi_kws)
        names = roi_kws.pop("names", None) or inferred_names
        colors = roi_kws.pop("colors", None) or inferred_colors
        boundary_kws = roi_kws.pop("boundary_kws", None)
        text_kws = roi_kws.pop("text_kws", None)
        stroke_kws = roi_kws.pop("stroke_kws", None)

        _, line_artists = overlay_rois(
            roi_collection,
            names,
            ax,
            colors,
            boundary_kws=boundary_kws,
            text_kws=text_kws,
            stroke_kws=stroke_kws,
        )
    else:
        line_artists = {}

    # update function
    def update_fn(frame_idx: int):
        im.set_data(vid[frame_idx])
        return (im, *line_artists.values())

    return im, update_fn


def display_rois(
    roi_collection: ROICollection,
    svd_video: "SVDVideo" = None,
    target_gamma: Optional[float] = None,
    fig: Optional[plt.Figure] = None,
    axis: Optional[plt.Axes] = None,
    dpi: Optional[float] = None,
    names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Any]] = None,
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
            shape = (6 / (dpi or 100.0), (8 / (dpi or 100.0)))
    print(shape, dpi or 100.)

    # 3. create figure/axis and display
    fig, ax = _ensure_axis(fig, axis, shape=shape, scale=dpi or 100.0)

    if svd_video is not None:
        ax.imshow(display_img, cmap="gray", origin="upper")
        ax.set_axis_off()
    else:
        ax.set_xlim(0, shape[1])
        ax.set_ylim(shape[0], 0)

    # 4. overlay ROIs
    inferred_names, inferred_colors = _infer_names_and_colors(roi_collection)
    names = names or inferred_names
    colors = colors or inferred_colors
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
    roi_collection: Optional[ROICollection] = None,
    select: Optional[Sequence[int] | Sequence[str]] = None,
    fig: Optional[plt.Figure] = None,
    axis: Optional[plt.Axes] = None,
    ticks: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Any]] = None,
    positions: Optional[Sequence[float]] = None,
    tick_mode: str = "overwrite",
    scale_to: Optional[float] = None,
    line_kws: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Plot traces for a set of ROIs.

    select : list of indices or ids

    Optional metadata arguments allow re-using positions/colors/names from a
    previous call. If provided, the length of `names`, `colors`, or
    `positions` must match the number of selected traces and a ValueError will
    be raised otherwise.

    If `scale_to` is provided each selected trace is scaled so its maximum
    absolute value fits within +/- (scale_to / 2). Scaling is applied before
    computing vertical spacing so the span/offset computation is consistent.

    tick_mode : str, one of 'overwrite', 'add', 'none'
        Controls how yticks/yticklabels are applied.

    Returns metadata dict with 'positions' (y positions), 'colors', 'names', and
    'scale_factors' (list or None).
    """
    from .psth_grid import override_kws
    
    data = np.asarray(traces.data)
    if scale_to is not None:
        # make working float copy for potential scaling
        data = data.astype(float, copy=True)
    n_cells, n_frames = data.shape

    # Build full id list corresponding to rows in `data`
    if roi_collection is not None and getattr(roi_collection, "ids", None) is not None:
        full_ids = list(roi_collection.ids)
    elif getattr(traces, "ids", None) is not None:
        full_ids = list(traces.ids)
    else:
        full_ids = [str(i) for i in range(n_cells)]

    # Resolve selection of rows
    sel_idx, sel_roi_ids = _resolve_selection(traces, select, roi_collection)

    # Compute positions and optionally scale data in-place; positions_map maps roi_id -> offset
    # Note: pass None for scale_to to preserve matplotlib path behavior
    positions_map, scale_factors_used = _compute_positions(
        full_ids, [full_ids[i] for i in sel_idx], data=data, scale_to=None, provided=positions
    )
    positions_used = [positions_map[full_ids[i]] for i in sel_idx]

    # Infer / default metadata as needed
    if roi_collection is None:
        cmap = plt.get_cmap("tab10")
        if getattr(traces, "ids", None) is not None:
            names_all = list(traces.ids)
        else:
            names_all = [f"Trace {i}" for i in range(n_cells)]
        colors_all = [cmap(i % cmap.N) for i in range(n_cells)]

        inferred_names = [names_all[i] for i in sel_idx]
        inferred_colors = [colors_all[i] for i in sel_idx]
    else:
        inferred_names, inferred_colors = _infer_names_and_colors(roi_collection, sel_idx)

    # Resolve names
    if names is None:
        names_used = inferred_names
    else:
        if len(names) != len(sel_idx):
            raise ValueError("length of 'names' must match number of selected traces")
        names_used = list(names)

    # Resolve colors
    if colors is None:
        colors_used = inferred_colors
    else:
        if len(colors) != len(sel_idx):
            raise ValueError("length of 'colors' must match number of selected traces")
        colors_used = list(colors)

    # fs/time axis
    if traces.fs is not None:
        x = np.arange(n_frames) / traces.fs
        xlabel = "time (s)"
    else:
        x = np.arange(n_frames)
        xlabel = "frame"

    # Calculate a good vertical offset between the traces unless positions provided
    # (Position computation delegated to _compute_positions above.)
    # positions_used already assigned from positions_map returned by _compute_positions.

    # Plot traces
    fig, ax = _ensure_axis(fig, axis)
    for k, i in enumerate(sel_idx):
        name = names_used[k]
        color = colors_used[k]
        y = data[i] + positions_used[k]
        ax.plot(x, y, **override_kws(line_kws, color=color))

    # Labeled ticks around the traces & axis labels
    if ticks is not None:
        # Interleave trace label names with data coordinate ticks
        ticks = np.array(ticks, dtype=float)
        if 0.0 not in ticks:
            ticks = np.concatenate(([0.0], ticks))
        zero_idx = np.where(ticks == 0.0)[0][0]
        tick_locations = np.array(positions_used)[None, :] + np.array(ticks)[:, None]
        tick_labels = np.full(tick_locations.shape, "", dtype=object)
        tick_labels[zero_idx, :] = names_used
        # Flatten and sort
        tick_sort = np.argsort(tick_locations.ravel())
        tick_locations = tick_locations.ravel()[tick_sort]
        tick_labels = tick_labels.ravel()[tick_sort]
    else:
        # Only label trace ids
        tick_locations = positions_used
        tick_labels = names_used

    # Apply tick_mode
    if tick_mode not in ("overwrite", "add", "none"):
        raise ValueError("tick_mode must be one of 'overwrite', 'add', or 'none'")

    if tick_mode == "overwrite":
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
    elif tick_mode == "add":
        existing_ticks = ax.get_yticks()
        existing_labels = [t.get_text() for t in ax.get_yticklabels()]
        # Append new ticks/labels
        combined_ticks = np.concatenate([np.asarray(existing_ticks), np.asarray(tick_locations)])
        combined_labels = list(existing_labels) + list(tick_labels)
        ax.set_yticks(combined_ticks)
        ax.set_yticklabels(combined_labels)
    else:  # 'none'
        pass

    ax.set_xlabel(xlabel)

    return (
        fig,
        ax,
        {"positions": positions_used, "colors": colors_used, "names": names_used, "scale_factors": scale_factors_used},
    )


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
    sel_idx, _ = _resolve_selection(events, select, None)

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
            yvals = y_trace[idx] + trace_locations[k]
            merged_kws = {
                **dict(
                    linestyle="None",
                    marker=".",
                    label=getattr(events, "ids", None) is not None and events.ids[i],
                ),
                **plot_kws,
            }
            ax.plot(xvals, yvals, **merged_kws)
        elif trace_locations is not None:
            # events as vertical bars matching extent
            ymin, ymax = trace_locations[k]
            for xv in xvals:
                ax.vlines(xv, ymin, ymax, **plot_kws)
        else:
            merged_kws = {
                **dict(
                    linestyle="None",
                    marker="|",
                    label=getattr(events, "ids", None) is not None and events.ids[i],
                ),
                **plot_kws,
            }
            # events as points on separate row
            y = -k  # separate row per ROI
            ax.plot(xvals, np.full_like(xvals, y), **merged_kws)
        plotted.append(i)

    ax.set_xlabel(xlabel)
    return {"plotted": plotted}
