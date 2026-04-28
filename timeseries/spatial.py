from .rois import ROICollection, ROI
from ..viz.rois import footprint_mask
from .types import Traces, Events
from ..io.svd_video import SVDVideo
from .spike_analysis import ms_to_samples
from ..viz.rois import mean_image
from .events import detect_spikes, despike_impute
from .filtering import filter_dff
from .ols_streaming import extract_traces

from scipy import ndimage, stats
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
    hpf_ms: Optional[float] = 400.0,
    dff_mode: str = "savgol_add",
    # spike detection kwargs
    spike_hpf_ms: Optional[float] = 150.0,
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
    mask = ndimage.binary_closing(mean >= threshold, structure=struct)
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

    return (
        roi_collection,
        events,
        {
            "raw": traces,
            "highpass": traces_hp,
            "baseline": baseline,
            "std": traces_std,
            "despiked": despiked_traces,
        },
    )


def dark_mask(
    image: np.ndarray,
    roi_collection: ROICollection,
    thresh_global: float = 0.3,
    thresh_local: float = 0.5,
    sigma=20,
    bite=10,
    radius=30,
    min_size=3,
    smooth_size=None
):
    """
    Identify locally-dark regions near ROIs.

    Includes:
    i) globally dark regions, as quantile threshold
    ii) locally dark regions, based on quantile threshold rank-ordered and highpass filtered image
    iii) pixels within a "radius"-sized donut region around the ROI
    Excludes:
    iv) regions near any ROI - removing a "bite"-sized dilation of all ROI footprints
    v) regions smaller than min_size in diameter

    Arguments
    ----------
    image: np.ndarray
        The input image to process.
    roi_collection: ROICollection
        The collection of ROIs to consider.
    thresh_global: float
        The global threshold for dark region detection.
    thresh_local: float
        The local threshold for dark region detection.
    sigma: float
        The standard deviation for Gaussian filtering.
    bite: int
        The size of the dilation to remove regions near ROIs.
    radius: int
        The radius for the donut-shaped region around each ROI.
    min_size: int
        The minimum size of regions to keep.

    Returns
    -------
    rois: ROICollection
        With same ids as input ROIs, but with footprints on the dark annuli
    mask: np.ndarray
        Boolean mask of all pixels satisfying the shared criteria for all ROIs,
        (i, ii, and iv)
    masks: Tuple[np.ndarray]
        Boolean masks for each of the criteria (i, ii, and iv)
    """
    if smooth_size is not None:
        image = ndimage.median_filter(image, size=smooth_size)

    # Globally dark regions
    dark = image < np.quantile(image.ravel(), thresh_global)
    # Locally dark regions
    rank = stats.rankdata(image.ravel()).reshape(image.shape)
    local = rank - ndimage.gaussian_filter(rank, sigma=sigma)
    dark_local = local < np.quantile(local.ravel(), thresh_local)
    # Exclude regions near any ROI
    all_fp = np.concatenate([r.footprint for r in roi_collection.rois])
    fp = footprint_mask(image.shape, all_fp)
    fp = ~ndimage.binary_dilation(fp, iterations=bite)

    # Find region near each ROI satisfying the above
    mask = (dark | dark_local) & fp
    annuli = []
    for r in roi_collection.rois:
        # Radius-sized area around each ROI footprint consistent with the mask
        halo = footprint_mask(image.shape, r.footprint)
        halo = ndimage.binary_dilation(halo, iterations=radius)
        halo &= mask
        # Remove small regions
        halo = ndimage.binary_opening(halo, iterations=min_size)
        annuli.append(np.column_stack(np.where(halo)))

    # Merge into ROICollection with same ids as input ROIs and return
    annuli_coll = roi_collection.__replace__(
        rois=[ROI(ann, weights=np.ones(len(ann)), code=np.ones(1)) for ann in annuli],
    )
    return annuli_coll, mask, (dark, dark_local, fp)
