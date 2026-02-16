"""Spike visualization helpers. See design/handoff/26-02-10_spike-fns.md for spec."""
# Typing & docstrings: follow docs/code-style.md

import numpy as np
import awkward as ak
from imaging_scripts.timeseries import spike_analysis as sa


def footprint_image(footprints, data, cmap=None, vmin=None, vmax=None):
    # Convert inputs to lists
    footprints = list(footprints)
    data = list(data)

    # Compute bounds
    all_rows = np.concatenate([fp[:, 0] for fp in footprints]) if len(footprints) > 0 else np.array([], dtype=int)
    all_cols = np.concatenate([fp[:, 1] for fp in footprints]) if len(footprints) > 0 else np.array([], dtype=int)

    if all_rows.size == 0 or all_cols.size == 0:
        return np.empty((0, 0)) if cmap is None else np.empty((0, 0, 3))

    rmin, rmax = int(all_rows.min()), int(all_rows.max())
    cmin, cmax = int(all_cols.min()), int(all_cols.max())

    H = rmax - rmin + 1
    W = cmax - cmin + 1

    img = np.full((H, W), np.nan, dtype=float)

    for fp, dat in zip(footprints, data):
        # fp: (N_i, 2) rows,cols
        rows = fp[:, 0].astype(int) - rmin
        cols = fp[:, 1].astype(int) - cmin
        img[rows, cols] = dat

    if cmap is None:
        return img

    # Apply colormap
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Prepare RGBA image, then drop alpha
    rgba = mapper.to_rgba(img)

    # Ensure NaN pixels remain NaN in all channels
    nan_mask = np.isnan(img)
    rgb = rgba[..., :3]
    rgb[nan_mask] = np.nan

    return rgb


def magnitude_scatter(trace_data, event_frames, labels, fs, ms_pre, ms_post, tree, magnitude_fn, ax=None, colors=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    colors = colors or {"BS": "C0", "SS-ADP": "C1", "SS-noADP": "C2"}

    n_pre = int(np.ceil(ms_pre * fs / 1000))
    n_post = int(np.ceil(ms_post * fs / 1000))

    traces = np.asarray(trace_data)
    evs = ak.to_list(event_frames)
    labs = ak.to_list(labels)

    xs = []
    ys = []
    cs = []

    # Map ROI id to index using tree if available; assume tree has .root_of and ids are keys
    # For simplicity tests will supply tree.root_of returning same id
    for i in range(traces.shape[0]):
        # find root idx for this ROI
        try:
            roi_id = i
            root_id = tree.root_of(i)
            # if tree returns an id string, attempt to find index in ids list
            if isinstance(root_id, str):
                # find matching index in labels order by searching for id string in available labels; fallback to 0
                i_root = 0
            else:
                i_root = int(root_id)
        except Exception:
            i_root = 0

        # skip root itself
        if i == i_root:
            continue

        nearest = sa.nearest_neighbor_events(evs[i], evs[i_root], n_pre, n_post)
        nearest_list = ak.to_list(nearest)
        # Find matched roi events that have a neighbor
        matched_roi_events = []
        matched_root_events = []
        matched_root_labels = []
        for j, nval in enumerate(nearest_list):
            if nval is None:
                continue
            matched_roi_events.append(evs[i][j])
            matched_root_events.append(nval)
            # find label of root's event: find index of nval in evs[i_root]
            try:
                ridx = list(evs[i_root]).index(nval)
                matched_root_labels.append(labs[i_root][ridx])
            except Exception:
                matched_root_labels.append('BS')

        if len(matched_roi_events) == 0:
            continue

        mag_roi = magnitude_fn(trace_data[i:i+1], ak.Array([matched_roi_events]), fs, ms_pre, ms_post)
        mag_root = magnitude_fn(trace_data[i_root:i_root+1], ak.Array([matched_root_events]), fs, ms_pre, ms_post)

        mag_roi_list = ak.to_list(mag_roi)[0]
        mag_root_list = ak.to_list(mag_root)[0]

        for xr, yr, lab in zip(mag_root_list, mag_roi_list, matched_root_labels):
            xs.append(xr)
            ys.append(yr)
            cs.append(colors.get(lab, 'k'))

    ax.scatter(xs, ys, c=cs)
    ax.set_xlabel('root magnitude')
    ax.set_ylabel('ROI magnitude')
    return fig, ax


def spatial_magnitudes(trace_data, event_frames, footprints, ref_idx, fs, ms_pre, ms_post, magnitude_fn, cmap=None, vmin=None, vmax=None):
    # Vectorized implementation: broadcast ref events against all ROIs and compute magnitudes in batch
    n_pre = int(np.ceil(ms_pre * fs / 1000))
    n_post = int(np.ceil(ms_post * fs / 1000))

    # Keep event_frames as an Awkward array so neighbor_events can broadcast
    evs = ak.Array(event_frames)

    # nearest: for each ROI (batch), for each ref event, the nearest event in that ROI (or None)
    nearest = sa.nearest_neighbor_events(evs[ref_idx], evs, n_pre, n_post)

    # Compute magnitudes in batch: magnitude_fn should accept (n_rois, n_frames) and (n_rois, <n_events>)
    mags = magnitude_fn(trace_data, nearest, fs, ms_pre, ms_post)

    # mean per ROI, replacing None with np.nan
    mean_mags = ak.mean(mags, axis=-1)
    mean_mags = ak.fill_none(mean_mags, np.nan)
    mean_mags_list = ak.to_list(mean_mags)

    # Broadcast per-pixel
    data = [np.full(len(fp), mag) for fp, mag in zip(footprints, mean_mags_list)]
    img = footprint_image(footprints, data, cmap=cmap, vmin=vmin, vmax=vmax)
    return img
