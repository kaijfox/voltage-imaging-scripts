from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from ..timeseries.types import Traces
from ..timeseries.rois import ROIHierarchy, ROICollection
from .scalebars import scale_bar
from .psth_grid import override_kws, mean_psth_grid
from ..windows.ragged_ops import slice_by_events


def _make_id_maps(roi_collection, tree) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    """Create id->index and index->id maps using the tree keying used elsewhere.

    Returns (id_to_idx, idx_to_id).
    """
    id_to_idx = {
        tree._key(roi_id): idx for idx, roi_id in enumerate(roi_collection.ids)
    }
    idx_to_id = {
        idx: tree._key(roi_id) for idx, roi_id in enumerate(roi_collection.ids)
    }
    return id_to_idx, idx_to_id


def _standardize_ids_and_idxs(
    tree,
    subtree,
    id_to_idx: Optional[Dict] = None,
    idx_to_id: Optional[Dict] = None,
    roi_collection=None,
) -> Tuple[List, List, Dict, Dict]:
    """Return canonical subtree ids and indices. If id/index maps are not provided,
    build them from roi_collection and tree.

    Returns (subtree_ids, subtree_ixs, id_to_idx, idx_to_id).
    """
    if id_to_idx is None or idx_to_id is None:
        if roi_collection is None:
            raise ValueError("roi_collection required when id maps are not provided")
        id_to_idx, idx_to_id = _make_id_maps(roi_collection, tree)

    subtree_ids = [tree._to_hid(node).long(all_ids=tree._hids) for node in subtree]
    # assume elements of `subtree` are keys acceptable to id_to_idx
    try:
        subtree_ixs = [id_to_idx[n] for n in subtree]
    except Exception:
        # try using the canonical ids we just created
        subtree_ixs = [id_to_idx[tree._key(n)] for n in subtree]

    return subtree_ids, subtree_ixs, id_to_idx, idx_to_id


def filter_spiking_rois(
    subtree_ixs: List[int], spikes, tree, idx_to_id: Dict[int, Any]
) -> Tuple[List[int], List]:
    """Return indices in subtree that have spikes and their corresponding canonical ids."""
    spiking_ixs = [i for i in subtree_ixs if len(spikes.spike_frames[i]) > 0]
    spiking_ids = [
        tree._to_hid(idx_to_id[i]).long(all_ids=tree._hids) for i in spiking_ixs
    ]
    return spiking_ixs, spiking_ids


def _prepare_and_extract_windows(
    traces,
    spikes,
    subtree_ixs: List[int],
    spiking_ixs: List[int],
    n_pre: int,
    n_post: int,
) -> ak.Array:
    """Prepare trace / spike arrays and extract event-centered windows using provided slice_by_events.

    This combines the notebook steps that broadcast and call slice_by_events.
    """
    spike_frames = ak.Array(spikes.spike_frames)
    trace_data = traces.data[subtree_ixs, None]  # (trace_roi, event_roi, frames)
    spike_frames = spike_frames[None, spiking_ixs]  # (trace_roi, event_roi, <event>)

    sampled = slice_by_events(
        trace_data,
        spike_frames,
        n_pre=n_pre,
        n_post=n_post,
    )
    return sampled


def _broadcast_compared_indices(
    trace_ids: List, event_ids: List, ref_id
) -> Tuple[np.ndarray, np.ndarray]:
    """Return broadcasted index arrays for comparing every (trace,event) to a reference id.
    """
    try:
        ref_trace_idx = trace_ids.index(ref_id)
    except ValueError:
        raise ValueError("ref_id not found in trace_ids")
    # try:
    #     ref_event_idx = event_ids.index(ref_id)
    # except ValueError:
    #     raise ValueError("ref_id not found in event_ids")

    compare_ix = np.broadcast_arrays(
        np.full(len(trace_ids), ref_trace_idx)[:, None],
        np.arange(len(event_ids))[None, :],
    )
    return compare_ix


def _get_names(tree, ids):
    return [tree._to_hid(r).short(all_ids=tree._hids) for r in ids]


def precompute_psth_grid(
    traces: Traces,
    spikes: Any,
    window_ms: Tuple[int, int],
    fs: float,
    roi_collection: ROICollection,
    tree: ROIHierarchy,
    subtree: List[tuple[Any]],
    event_subtree: List[tuple[Any]] = None,
    compare_id: Optional[Any] = None,
    color_by: str = "trace",
):
    # Extract names and indices
    id_to_idx, idx_to_id = _make_id_maps(roi_collection, tree)
    trace_ids, trace_ixs, id_to_idx, idx_to_id = _standardize_ids_and_idxs(
        tree, subtree, id_to_idx, idx_to_id, roi_collection
    )
    if event_subtree is not None:
        _, event_ixs, _, _ = _standardize_ids_and_idxs(
            tree, event_subtree, id_to_idx, idx_to_id, roi_collection
        )
    else:
        event_ixs = trace_ixs
    event_ixs, event_ids = filter_spiking_rois(event_ixs, spikes, tree, idx_to_id)

    # Names and colors
    trace_names = _get_names(tree, trace_ids)
    event_names = _get_names(tree, event_ids)
    if color_by == "trace":
        trace_colors = roi_collection[trace_ixs].colors
        event_colors = None
    elif color_by == "event":
        trace_colors = None
        event_colors = roi_collection[event_ixs].colors
    else:
        raise ValueError("color_by must be 'trace' or 'event'")

    # Compute windows
    if isinstance(window_ms, (int, float)):
        window_ms = (window_ms, window_ms)
    n_pre = int(np.ceil(window_ms[0] * fs / 1000))
    n_post = int(np.ceil(window_ms[1] * fs / 1000))
    samples = _prepare_and_extract_windows(
        traces, spikes, trace_ixs, event_ixs, n_pre, n_post
    )

    # Set up comparison
    compare_ix = (
        _broadcast_compared_indices(trace_ids, event_ids, compare_id)
        if compare_id is not None
        else None
    )

    return dict(
        windows=samples,
        trace_names=trace_names,
        event_names=event_names,
        trace_colors=trace_colors,
        event_colors=event_colors,
        fs=fs,
        zero=n_pre,
        compare_ix=compare_ix,
    )


def psth_grid_dispatch(
    mode: str,
    windows: ak.Array,
    trace_colors=None,
    event_colors=None,
    trace_names: Optional[List[str]] = None,
    event_names: Optional[List[str]] = None,
    fs: float = None,
    zero: int = None,
    ax=None,    
    compare_ix: Optional[Any] = None,
    compare_kws: Optional[Dict] = None,
    compare_mode: str = None,
    y_scalebar: Optional[Dict] = None,
    x_scalebar: Optional[Dict] = None,
    finalize: bool = True,
    **kws,
):
    """Dispatch to mean_psth_grid with common argument handling.


    Returns (fig, ax).
    """
    # Normalize argument defaults
    compare_kws = {} if compare_kws is None else compare_kws
    compare_mode = compare_mode or mode
    if isinstance(y_scalebar, (int, float)):
        y_scalebar = dict(size=y_scalebar)
    if isinstance(x_scalebar, (int, float)):
        x_scalebar = dict(size=x_scalebar)
    y_scalebar = {} if y_scalebar is None else y_scalebar
    x_scalebar = {} if x_scalebar is None else x_scalebar

    # Check silly inputs
    if trace_colors is None and event_colors is None:
        raise ValueError(
            "At least one of trace_colors or event_colors must be provided."
        )

    # Set up mean_psth_grid kwargs to add data to
    base_kwargs = dict(
        trace_colors=trace_colors,
        trace_names=trace_names,
        event_names=event_names,
        fs=fs,
        zero=zero,
    )

    # Compute relevant stats & insert to kws dictionary
    mean = sem = std = None
    if "mean" in mode or "sem" in mode or "std" in mode:
        mean = ak.mean(windows, axis=-2)
        base_kwargs["mean"] = mean

    if "sem" in mode or "std" in mode:
        std = ak.std(windows, axis=-2)
        if "std" in mode:
            base_kwargs["std"] = std
    if "sem" in mode:
        count = ak.count(windows, axis=-2)
        sem = std / np.sqrt(count)
        base_kwargs["sem"] = sem
    if "trace" in mode:
        base_kwargs["traces"] = windows

    # Compute arrays and colors for compare mode
    if compare_ix is not None:
        print(np.asarray(mean).shape)
        print(compare_ix)
        compare_mean = mean[compare_ix] if mean is not None else None
        compare_sem = sem[compare_ix] if sem is not None else None
        compare_std = std[compare_ix] if std is not None else None
        compare_traces = windows[compare_ix]
        compare_color = plt.matplotlib.colors.to_rgb(".8")

        # Set color, fs, and alignment args for the comparison plot
        compare_default = dict(fs=fs, zero=zero)
        if trace_colors is not None:
            compare_default["trace_colors"] = [compare_color] * len(trace_colors)
        if event_colors is not None:
            compare_default["event_colors"] = [compare_color] * len(event_colors)

        # Add data to the comparison plot
        if "mean" in compare_mode or "sem" in compare_mode or "std" in compare_mode:
            compare_default["mean"] = compare_mean
        if "sem" in compare_mode:
            compare_default["sem"] = compare_sem
        if "std" in compare_mode:
            compare_default["std"] = compare_std
        if "trace" in compare_mode:
            compare_default["traces"] = compare_traces

        fig, ax = mean_psth_grid(
            **override_kws(compare_default, compare_kws, {"ax": ax})
        )

    # Plot main data over comparison
    main_kw = override_kws(base_kwargs, kws, {"ax": ax})
    fig, ax = mean_psth_grid(**main_kw)

    if finalize:
        _finalize_axes_with_scalebars(ax, x_kws=x_scalebar, y_kws=y_scalebar)

    return fig, ax


def _finalize_axes_with_scalebars(
    ax,
    y_kws: Optional[Dict] = None,
    x_kws: Optional[Dict] = None,
) -> None:
    """Add y/x scale bars to the grid of axes and optionally call an axes-off function.

    y_kws/x_kws are merged with sensible defaults but can be overridden.
    """

    y_kws = override_kws(
        y_kws,
        max_nbins=4,
        loc="lower right",
    )
    x_kws = override_kws(x_kws, max_nbins=4, loc="lower right")

    scale_bar(ax[0, -1], "y", **y_kws)
    scale_bar(ax[-1, 0], "x", **x_kws)

    import mplutil.util as vu

    vu.axes_off(ax)
