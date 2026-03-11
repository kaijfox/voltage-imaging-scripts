from .rois import ROICollection, ROI
from .types import Traces, Events
from ..io.svd_video import SVDVideo
from .spike_analysis import ms_to_samples
from ..viz.rois import mean_image
from .events import detect_spikes, despike_impute
from .filtering import filter_dff
from .ols_streaming import extract_traces

from scipy.ndimage import binary_closing
import numpy as np
import itertools as iit
from typing import Tuple, Optional, Dict


def spatial_traces(
    raw_video: SVDVideo,
    grid_n: int = 8,
    percentile: float = 90.0,
    closing_diameter: int = 5,  # not used here; mask is provided
    min_footprint: int = 50,
    fs: Optional[float] = None,
    # df/f kwargs
    hpf_ms: Optional[float] = 400.,
    dff_mode: str = "savgol_add",
    # spike detection kwargs
    spike_hpf_ms: Optional[float] = 150.,
    sd_threshold: float = 4.0,
    # despike kwargs (match despike_impute signature)
    despike_window_ms: int = 5.0,
    despike_savgol_ms: int = 100.0,
) -> Tuple[ROICollection, Events, Dict[str, Traces]]:
    """Build grid ROIs from a boolean mask, extract traces, detect spikes, and despike."""
    # Identify bright pixels and extract ROIs within grid
    mean = mean_image(raw_video)  # (H, W)
    threshold = np.percentile(mean.ravel(), percentile)
    struct = np.ones((closing_diameter, closing_diameter))
    mask = binary_closing(mean >= threshold, structure=struct)
    footprint = np.column_stack(np.nonzero(mask))

    # Build footprint coordinates within grid cells
    cell_H = mask.shape[0] // grid_n
    cell_W = mask.shape[1] // grid_n
    rois, ids = [], []
    for i, j in iit.product(range(grid_n), range(grid_n)):
        cell_mask = (
            (footprint[:, 0] >= i * cell_H)
            & (footprint[:, 0] < (i + 1) * cell_H)
            & (footprint[:, 1] >= j * cell_W)
            & (footprint[:, 1] < (j + 1) * cell_W)
        )
        if np.count_nonzero(cell_mask) < min_footprint:
            continue

        fp = footprint[cell_mask]
        rois.append(ROI(footprint=fp, weights=np.ones(len(fp)), code=np.ones(1)))
        ids.append(f"r={i+1} c={j+1}")

    roi_collection = ROICollection(rois=rois, image_shape=mask.shape, ids=ids)

    # Extract traces, dF/F transform standarize (no neuropil subtraction)
    traces, _ = extract_traces(
        raw_video,
        roi_collection,
        neuropil_range=(-1, -1),
        fs=fs,
        ols=False,
        weighted=False,
    )
    traces_hp, baseline = filter_dff(
        traces,
        mode=dff_mode,
        window_length=ms_to_samples(hpf_ms, fs),
        polyorder=2,
    )
    traces_std = Traces(
        data=traces_hp.data / traces_hp.data.std(axis=1, keepdims=True),
        ids=traces.ids,
        fs=traces.fs,
    )

    # Detect spikes
    _, events, _ = detect_spikes(
        traces_std,
        sd_threshold=sd_threshold,
        sg_window_frames=ms_to_samples(spike_hpf_ms, fs),
    )

    # Despike/impute using the high-pass filtered traces and detected events
    if despike_window_ms is not None:
        despiked_traces = despike_impute(
            traces_std,
            events,
            window_size=ms_to_samples(despike_window_ms, fs),
            savgol_window_frames=ms_to_samples(despike_savgol_ms, fs),
        )
    else:
        despiked_traces = None

    return roi_collection, events, {
        "raw": traces,
        "highpass": traces_hp,
        "baseline": baseline,
        "std": traces_std,
        "despiked": despiked_traces,
    }
