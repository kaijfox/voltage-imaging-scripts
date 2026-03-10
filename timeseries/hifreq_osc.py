from imaging_scripts.viz.grid_dispatchers import mean_psth_grid
from imaging_scripts.timeseries.rois import ROICollection, ROIHierarchy, ProcessROIId
from imaging_scripts.timeseries.types import Traces, Events
import awkward as ak
from imaging_scripts.io.svd_video import SVDVideo
from imaging_scripts.windows.ragged_ops import ak_infer_shape, slice_by_events
from imaging_scripts.viz.psth_videos import extract_mean_videos, video_and_trace
from imaging_scripts.timeseries import extract_traces
from imaging_scripts.timeseries.analysis import pairwise_coherence, power_spectrum

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import cm
from pathlib import Path
import gc
from scipy.signal import hilbert
from typing import Dict, Tuple, Optional, Sequence
from imaging_scripts.viz.rois import _ensure_axis, quantile_normalize, apply_cmap
import mplutil.util as vu


from scipy.signal import butter, filtfilt


def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def _determine_cmap_params(
    plot_data, vmin=None, vmax=None, cmap=None, center=None, robust=False
):
    """
    Determine color mapping and range for plotting an array.

    Parameters
    ----------
    plot_data : array-like
        Data array to inspect (can be a masked array).
    vmin, vmax : float, optional
        Manual min/max to use. If None, determined from data.
    cmap : {None, str, list, Colormap}, optional
        Colormap specification.
    center : float, optional
        If provided, recenter a divergent colormap around this value.
    robust : bool, optional
        Use percentile-based robust range (2/98) instead of min/max.

    Returns
    -------
    vmin, vmax, colormap
        Tuple of numeric vmin, vmax and a Matplotlib Colormap instance to use.
    """
    import matplotlib as mpl
    from seaborn._compat import get_colormap

    # plot_data may be a masked array
    if hasattr(plot_data, "filled"):
        plot_data = plot_data.astype(float).filled(np.nan)

    if vmin is None:
        if robust:
            vmin = np.nanpercentile(plot_data, 2)
        else:
            vmin = np.nanmin(plot_data)
    if vmax is None:
        if robust:
            vmax = np.nanpercentile(plot_data, 98)
        else:
            vmax = np.nanmax(plot_data)
    _vmin, _vmax = vmin, vmax

    # Choose default colormaps if not provided
    if cmap is None:
        if center is None:
            _cmap = cm.rocket
        else:
            _cmap = cm.icefire
    elif isinstance(cmap, str):
        _cmap = get_colormap(cmap)
    elif isinstance(cmap, list):
        _cmap = mpl.colors.ListedColormap(cmap)
    else:
        _cmap = cmap

    # Recenter a divergent colormap
    if center is not None:
        bad = _cmap(np.ma.masked_invalid([np.nan]))[0]
        under = _cmap(-np.inf)
        over = _cmap(np.inf)
        under_set = under != _cmap(0)
        over_set = over != _cmap(_cmap.N - 1)

        vrange = max(vmax - center, center - vmin)
        normlize = mpl.colors.Normalize(center - vrange, center + vrange)
        cmin, cmax = normlize([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        _cmap = mpl.colors.ListedColormap(_cmap(cc))
        _cmap.set_bad(bad)
        if under_set:
            _cmap.set_under(under)
        if over_set:
            _cmap.set_over(over)

    return _vmin, _vmax, _cmap





def spatial_traces(
    raw_video: SVDVideo,
    mask: np.ndarray,           # bool (H, W)
    grid_n: int = 8,
    percentile: float = 90.,
    closing_diameter: int = 5,
    # extract_traces kwargs passed through
    neuropil_range: tuple = None,
    bg_smooth_size: int = 0,
    fs: float = None,
    # despike kwargs
    sg_window_frames: int = None,
    sd_threshold: float = 4.0,
):
    # Identify bright pixels and extract ROIs within grid
    # mean_image = io.pipeline.svd_video_mean
    # threshold = np.percentile(mean_image, percentile)
    # struct = np.ones((closing_diameter, closing_diameter))
    # mask = ndimage.binary_closing(mean_image >= threshold, structure=struct)
    
    # split spatial into (components, grid_n,  grid_n, cell_size, cell_size)
    # H, W = mask.shape
    # cell_size = floor(H / grid_n), floor(W / grid_n))
    # Vt_grid = raw_video.Vt.reshape(...)
    # 




    pass




def _footprint_mask(image_shape, footprint):
    mask = np.zeros(image_shape, dtype=bool)
    mask[footprint[:, 0], footprint[:, 1]] = True
    return mask


def hilo_corr_maps(
    raw_lorank,
    soma_collection: ROICollection,
    max_rank: int = 200,
    scale: int = 50,
    anatomy=True,
    corr=True,
    overlay=True,
):
    """
    Generate correlation and anatomical overlay figures for a reference soma code.

    Parameters
    ----------
    raw_lorank : object
        Object exposing .S and .Vt (low-rank decomposition components).
        Typically the loaded S/Vt from SVDVideo or similar.
    soma_collection : ROICollection
        Collection of ROIs (Regions of Interest), from which the first ROI will
        be selected as the reference to correlate with.
    max_rank : int, optional
        Maximum rank to use from the decomposition (not currently used but kept for API compatibility).
    scale : int, optional
        Scale factor passed to plotting helper for sizing.

    Returns
    -------
    corr_fig, anatomy_fig, overlay_fig : dict
        Dicts containing figure and axis handles for correlation map, anatomical image and overlay image.
    """
    import cv2
    from skimage import filters, morphology

    S = raw_lorank.S[:max_rank]
    Vt = raw_lorank.Vt[:max_rank]

    soma1_mask = _footprint_mask(
        soma_collection.image_shape, soma_collection.rois[0].footprint
    )

    # Build spatial components: S[:, None, None] * Vt -> (rank, H, W)
    loaded_spatial = S[:, None, None] * Vt
    # Mean code across soma mask
    soma1_code = loaded_spatial[:, soma1_mask].mean(axis=1)
    dots = np.tensordot(loaded_spatial, soma1_code, axes=(0, 0))
    code_norm = np.linalg.norm(soma1_code)
    loaded_spatial_norm = np.linalg.norm(loaded_spatial, axis=0)

    soma1_corr = dots / (loaded_spatial_norm * code_norm)
    vmin_corr, vmax_corr, corr_cmap = _determine_cmap_params(
        soma1_corr, vmin=None, vmax=None, cmap="RdBu", center=0
    )

    # Display anatomical / std image
    dff_std = np.linalg.norm(loaded_spatial, axis=0)
    disp1 = np.log(dff_std)
    disp1 = 255 * (disp1 - disp1.min()) / (disp1.max() - disp1.min())
    vmin_std, vmax_std = disp1.min(), disp1.max()

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # disp2 = clahe.apply(disp1.astype(np.uint8).copy())
    # disp3 = filters.median(disp2.copy(), morphology.disk(2))
    # edge = filters.scharr(disp3)
    # edge = 255 * edge.astype(float) / edge.max()
    # disp4 = 0.8 * disp3 + 0.2 * edge
    # disp5 = 0.6 * disp1 + 0.4 * disp4

    img_corr = apply_cmap(soma1_corr, cmap=corr_cmap, vmin=vmin_corr, vmax=vmax_corr)[0]
    img_std = apply_cmap(disp1, cmap="Greys_r", vmin=vmin_std, vmax=vmax_std * 0.8)[0]
    mappable = cm.ScalarMappable(
        cmap=corr_cmap, norm=plt.Normalize(vmin=vmin_corr, vmax=vmax_corr)
    )

    if corr:
        fig, ax = _ensure_axis(None, None, soma1_corr.shape, scale=scale)
        ax.imshow(img_corr)
        fig.colorbar(mappable, ax=ax, pad=0.01)
        vu.axes_off(ax)
        corr_fig = {"fig": fig, "ax": ax}
    else:
        corr_fig = None

    if anatomy:
        fig, ax = _ensure_axis(None, None, soma1_corr.shape, scale=scale)
        ax.imshow(img_std)
        vu.axes_off(ax)
        anatomy_fig = {"fig": fig, "ax": ax}
    else:
        anatomy_fig = None

    if overlay:
        # Overlay correlation (alpha) on anatomy
        fig, ax = _ensure_axis(None, None, soma1_corr.shape, scale=scale)
        ax.imshow(img_std)
        img_corr[..., -1] = np.abs(soma1_corr) / np.abs(soma1_corr).max()
        ax.imshow(img_corr)
        fig.colorbar(mappable, ax=ax, pad=0.01)
        vu.axes_off(ax)
        overlay_fig = {"fig": fig, "ax": ax}
    else:
        overlay_fig = None

    return corr_fig, anatomy_fig, overlay_fig


def corr_map_from_stddev_seed(
    raw_lorank,
    seed_rank: int = 10,
    max_rank: int = 200,
    percentile: float = 90,
    dilation: int = None,
):
    """
    Compute a correlation map seeded by high-variance pixels, without requiring an ROI.

    Uses a truncated low-rank reconstruction (seed_rank components) to estimate
    per-pixel standard deviation, then selects pixels above `percentile` as the
    seed set. The mean spatial code of those seed pixels (computed from up to
    max_rank components) is then correlated with every pixel via cosine similarity.

    Parameters
    ----------
    raw_lorank : object
        Object with .S (rank,) and .Vt (rank, H, W) attributes.
    seed_rank : int, optional
        Number of components to use for stddev-based seed selection.
        Should be much smaller than max_rank to focus on dominant structure.
    max_rank : int, optional
        Maximum rank to use for correlation map computation.
    percentile : float, optional
        Percentile threshold for selecting high-stddev seed pixels (0-100).

    Returns
    -------
    corr_map : ndarray, shape (H, W)
        Cosine-similarity correlation of each pixel with the seed code.
    seed_mask : ndarray of bool, shape (H, W)
        Boolean mask indicating which pixels were selected as the seed.
    """
    S_seed = raw_lorank.S[:seed_rank]
    Vt_seed = raw_lorank.Vt[:seed_rank]
    loaded_seed = S_seed[:, None, None] * Vt_seed  # (seed_rank, H, W)
    dff_std_seed = np.linalg.norm(loaded_seed, axis=0)  # (H, W)

    threshold = np.percentile(dff_std_seed, percentile)
    seed_mask = dff_std_seed >= threshold
    if dilation is not None:
        struct = np.ones((dilation, dilation), dtype=bool)
        seed_mask = ndimage.binary_dilation(seed_mask, structure=struct)

    S = raw_lorank.S[:max_rank]
    Vt = raw_lorank.Vt[:max_rank]
    loaded_spatial = S[:, None, None] * Vt  # (max_rank, H, W)

    seed_code = loaded_spatial[:, seed_mask].mean(axis=1)  # (max_rank,)
    dots = np.tensordot(loaded_spatial, seed_code, axes=(0, 0))  # (H, W)
    code_norm = np.linalg.norm(seed_code)
    loaded_spatial_norm = np.linalg.norm(loaded_spatial, axis=0)  # (H, W)
    corr_map = dots / (loaded_spatial_norm * code_norm)

    return corr_map, seed_mask


def trace_from_spatial_code(svd_video, spatial_code, max_rank: int = 200):
    """
    Extract a temporal trace by projecting a spatial weight map onto the SVD
    reconstruction.

    Computes: trace[t] = sum_{i,j} spatial_code[i,j] * X[t,i,j]
                       = U @ (S * (Vt_flat @ w))

    Parameters
    ----------
    svd_video : SVDVideo
        Video with .U (T, rank), .S (rank,), .Vt (rank, H, W) attributes.
    spatial_code : ndarray, shape (H, W)
        Spatial weight map (e.g. a correlation map). Need not be normalized.
    max_rank : int, optional
        Maximum rank components to use.

    Returns
    -------
    trace : ndarray, shape (T,)
        Temporal trace weighted by spatial_code.
    """
    U = svd_video.U[:, :max_rank]  # (T, rank)
    S = svd_video.S[:max_rank]     # (rank,)
    Vt = svd_video.Vt[:max_rank]   # (rank, H, W)
    print(U.shape, S.shape, Vt.shape, spatial_code.shape)

    w = spatial_code.ravel()                          # (H*W,)
    vt_flat = Vt.reshape(len(S), -1)                  # (rank, H*W)
    code_proj = vt_flat @ w                           # (rank,)
    trace = U @ (S * code_proj)                       # (T,)
    return trace


def filtered_video(
    svd_video: SVDVideo,
    fs,
    max_rank=None,
    cuts=None,
    f=None,
    order=4,
    orthogonalize=True,
):
    if f is None:
        raise ValueError("Must provide filter function f")
    U = svd_video.U[:, :max_rank]
    U = f(U, fs, freqs=cuts, order=order, axis=0)
    if orthogonalize:

        return SVDVideo(
            *SVDVideo.orthogonal(svd_video.S[:max_rank], U, svd_video.Vt[:max_rank])
        )
    else:
        return SVDVideo(U, svd_video.S[:max_rank], svd_video.Vt[:max_rank])


def _filter_traces(
    traces_dff: Dict[str, Traces],
    fs: float,
    freq_range: Tuple[float, float] = (80, 120),
    order=4,
) -> Dict[str, Traces]:
    """
    Bandpass filter a dictionary of Traces objects.

    Parameters
    ----------
    traces_dff : dict
        Mapping of keys (e.g. 'hi','lo','soma') -> Traces instances containing .data and .ids
    fs : float
        Sampling frequency in Hz.
    freq_range : (low, high)
        Bandpass range in Hz.

    Returns
    -------
    dff_band : dict
        Same keys as input mapping to Traces with filtered data and updated sampling rate metadata.
    """
    low, high = freq_range
    dff_band = {
        k: Traces(
            bandpass_filter(traces_dff[k].data, fs, low, high, order=order),
            traces_dff[k].ids,
            fs,
        )
        for k in traces_dff.keys()
    }
    return dff_band


def pair_roi_traces(
    dff_video,
    soma_roi,
    hi_rois,
    lo_rois,
    fs: float = 800,
    neuropil_kw: Optional[Dict] = None,
    freq_range: Tuple[float, float] = (80, 120),
):
    """
    Extract dF/F traces for ROI collections and return both raw dff traces and bandpassed traces.

    Parameters
    ----------
    dff_video : SVDVideo or similar
        Source video object used by extract_traces.
    soma_roi, hi_rois, lo_rois : ROI collections
        Collections passed to extract_traces.
    fs : float, optional
        Sampling frequency in Hz.
    neuropil_kw : dict, optional
        Keyword args passed to extract_traces (e.g. neuropil_range, ols, weighted).
    freq_range : tuple, optional
        Frequency range used for bandpass filtering.

    Returns
    -------
    dict
        {'dff': traces_dff, 'bandpass': dff_band}
    """
    if neuropil_kw is None:
        dff_nrpl_kw = dict(neuropil_range=(15, 30), ols=False, weighted=False)
    else:
        dff_nrpl_kw = neuropil_kw

    collections = dict(soma=soma_roi, hi=hi_rois, lo=lo_rois)

    # Run trace extraction (extract_traces returns (traces, metadata) in this codebase)
    traces_dff = {
        k: extract_traces(dff_video, coll, **dff_nrpl_kw)[0]
        for k, coll in collections.items()
    }

    dff_band = _filter_traces(traces_dff, fs, freq_range)

    return {"dff": traces_dff, "bandpass": dff_band}


def pair_roi_spectra(traces_dff: Dict[str, Traces], fs: float = 800):
    """
    Compute and plot power spectra and pairwise coherence for hi/lo traces.

    Parameters
    ----------
    traces_dff : dict
        Dictionary from pair_roi_traces()['dff'] mapping keys to Traces instances.
    fs : float, optional
        Sampling frequency.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    Shi, freqs_hi = power_spectrum(traces_dff["hi"].data, fs=fs, nperseg=2 * fs)
    Slo, freqs_lo = power_spectrum(traces_dff["lo"].data, fs=fs, nperseg=2 * fs)
    Sx, freqs = pairwise_coherence(
        traces_dff["hi"].data, traces_dff["lo"].data, fs=fs, nperseg=2 * fs
    )

    n_hi = traces_dff["hi"].data.shape[0]
    fig, ax = vu.subplots((1.5, 2), (3, n_hi), sharex=True)

    for i in range(n_hi):
        ax[0, i].plot(freqs_hi, Shi[i], color="g", lw=1)
        ax[1, i].plot(freqs_lo, Slo[i], color="r", lw=1)
        ax[2, i].plot(freqs, Sx[i, i], color="k", lw=1)
        ax[0, i].set_title(f"Pair {i+1}")
        ax[0, i].set_yscale("log")
        ax[1, i].set_yscale("log")
    ax[-1, 0].set_xlabel("Frequency (Hz)")
    ax[-1, 0].set_xscale("log")

    return fig, ax


def phase_aligned_windows(
    traces_dff,
    dff_video,
    fs: float = 800,
    freq_range: Tuple[float, float] = (80, 120),
    target_phase: float = -np.pi / 4,
    pair: int = 0,
    N: int = 40,
    window_ms: int = 50,
    max_rank=100,
    order=4,
):
    """
    Find phase-aligned centers and extract trace windows and mean videos centered on those times.

    Parameters
    ----------
    traces_dff : dict or mapping
        Dict with keys 'hi' and 'lo' mapping to Traces objects containing .data and .ids.
    dff_video : SVDVideo-like
        Video used to extract mean videos.
    fs : float, optional
        Sampling frequency in Hz.
    freq_range : (low, high), optional
        Bandpass range for extracting phase signal.
    target_phase : float, optional
        Phase value to find rising crossings through (radians).
    pair : int, optional
        Index of hi trace pair to use to find phase events.
    N : int, optional
        Maximum number of centers to keep.
    window_ms : int, optional
        Window length on each side in milliseconds.

    Returns
    -------
    center_frames, trace_windows, mean_video
        center_frames : 1D array of frame indices
        trace_windows : dict of awkward arrays for 'hi' and 'lo' windows
        mean_video : array-like mean video around each center
    """
    hi_sig = traces_dff["hi"].data[pair].astype(float)
    hi_bp = bandpass_filter(hi_sig, fs, freq_range[0], freq_range[1], order=order)

    analytic = hilbert(hi_bp)
    ang = np.angle(analytic)

    # Find rising crossings through target_phase
    crossings = np.where((ang[:-1] < target_phase) & (ang[1:] > target_phase))[0] + 1

    # Avoid edges so windows extracted later have room
    margin_ms = 100
    margin_frames = int(round(margin_ms / 1000 * fs))
    zc = crossings[
        (crossings > margin_frames) & (crossings < len(hi_sig) - margin_frames)
    ]

    step = max(1, len(zc) // N) if len(zc) > N else 1
    centers = zc[::step]
    center_frames = centers[:N].astype(int)

    kw = dict(
        n_pre=int(round(window_ms / 1000 * fs)),
        n_post=int(round(window_ms / 1000 * fs)),
    )
    trace_windows = {
        # (n_hi/n_lo, n_events, window_size)
        k: slice_by_events(traces_dff[k].data, center_frames[None], **kw)
        for k in ["hi", "lo"]
    }

    # Calculate the mean videos around these frames
    mean_video_kw = dict(pre_ms=window_ms, post_ms=window_ms, fs=fs, max_rank=max_rank)
    mean_video = extract_mean_videos(dff_video, center_frames, **mean_video_kw)

    return center_frames, trace_windows, mean_video


def phase_aligned_traces(trace_windows, fs: float = 800, window_ms: int = 50):
    """
    Plot mean PSTH-style traces aligned to centers.

    Parameters
    ----------
    trace_windows : dict
        Dictionary with keys 'hi' and 'lo' containing awkward arrays of shape
        (n_lo/_hi, <event>, window_size).
    fs : float, optional
        Sampling frequency in Hz.
    window_ms : int, optional
        Window half-width in milliseconds used to compute zero index.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes used for the plot.
    """
    # Determine number of pre frames
    n_pre = int(round(window_ms / 1000 * fs))

    # Use ak_infer_shape to determine number of hi ROIs
    # (n_lo/n_hi, n_events, window_size)
    shape_hi = ak_infer_shape(trace_windows["hi"])
    n_hi = shape_hi[0]

    fig, ax = vu.subplots((1.5, 1.5), (1, n_hi), sharey=True)
    ax = np.atleast_1d(ax)

    means = {
        k: ak.mean(trace_windows[k], axis=-1, keepdims=True)
        for k in trace_windows.keys()
    }
    centered = {
        k: trace_windows[k] - means[k] for k in trace_windows.keys()
    }

    grid_kw = lambda k, c: dict(
        mean=ak.mean(centered[k], axis=-2)[None, :],
        std=ak.std(centered[k], axis=-2)[None, :],
        event_colors=[c] * n_hi,
        event_names=[f"Pair {i+1}" for i in range(n_hi)],
        fs=fs / 1000.0,
        zero=n_pre,
        std_kw=dict(lighten=0.8),
        ax=ax[None],
    )

    if 'hi' in trace_windows:
        mean_psth_grid(**grid_kw("hi", "g"))
    if 'lo' in trace_windows:
        mean_psth_grid(**grid_kw("lo", "r"))

    return fig, ax


def phase_aligned_video(
    path,
    trace_windows,
    mean_video,
    hi_rois,
    fs: float = 800,
    window_ms: int = 50,
    vmin=-3,
    vmax=3,
    cmap="inferno_r",
):
    """
    Produce two variant videos (unbounded and bounded) showing the mean phase-aligned video and a trace.

    Parameters
    ----------
    path : os.PathLike
        Video file path ending in .mp4 or similar
    trace_windows : dict
        As returned by `phase_aligned_windows` (used to compute the trace overlay).
    mean_video : array-like
        Mean video frames returned by `extract_mean_videos`.
    hi_rois : sequence
        ROI collection used for producing trace overlays (first ROI used for name/coloring).
    fs : float, optional
        Sampling frequency.
    window_ms : int, optional
        Window half-width in ms used to compute scalebar lengths.

    Returns
    -------
    out_paths : tuple
        Paths of the generated unbounded and bounded videos.
    """
    roi_collection = hi_rois[0] if hi_rois is not None else None
    # derive mean roi trace from hi trace windows
    roi_trace = np.mean(trace_windows["hi"], axis=-2)[0]

    kws = dict(
        frames=mean_video * 100,
        trace_mean=roi_trace * 100,
        fs=fs,
        pre_ms=window_ms,
        ms_per_s=15,
        height=6,
        roi_collection=roi_collection,
        roi_name="Hi 1",
        cmap=cmap,
        x_scalebar=dict(
            size=10, fmt="{:.0f} ms", loc="lower center", xy=(6, 0), text_loc="inside"
        ),
        y_scalebar=dict(max_nbins=5, fmt="{:.1f}% dF/F", loc="center right"),
    )

    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    video_and_trace(path, vmin=vmin, vmax=vmax, **kws)
    plt.close("all")
    gc.collect()


def phase_aligned_keyframes(
    trace_windows,
    mean_video,
    hi_rois=None,
    fs: float = 800.0,
    N: int = 4,
    cmap: str = "inferno_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    height: float = 3.0,
    ax_size: Tuple[float, float] = None,
    upsample: int = 1,
):
    """
    Create a static figure showing mean keyframes sampled across phase bins of the
    ROI trace and the corresponding mean ROI trace with markers for the sampled
    frames.

    New behavior:
    - Compute the analytic signal (hilbert) of the mean ROI trace, take its
      phase (angle) and bin the phase into `N` bins based on quantiles so each
      bin tends to contain an even number of samples.
    - `indices_list` contains the frame indices (on the temporally-upsampled
      timeline) that fall into each phase bin.
    - `upsample` (int) linearly interpolates `roi_trace` and `mean_video` along
      their temporal dimension before binning & hilbert transform.
    """
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from imaging_scripts.viz.rois import setup_video

    mv = np.asarray(mean_video)
    if mv.ndim != 3:
        raise ValueError("mean_video must have shape (T, H, W)")
    T, H, W = mv.shape

    # mirror roi_collection handling from phase_aligned_video
    roi_collection = hi_rois[0] if hi_rois is not None else None

    # derive roi_trace (same as phase_aligned_video)
    roi_trace = np.asarray(ak.mean(trace_windows["hi"], axis=-2)[0])

    # optionally upsample temporally (linear interpolation)
    if upsample is None or int(upsample) < 1:
        upsample = 1
    upsample = int(upsample)

    if upsample == 1:
        mv_u = mv
        roi_trace_u = roi_trace
        fs_effective = fs
    else:
        orig_t = np.arange(T)
        new_T = int(T * upsample)
        new_t = np.linspace(0, T - 1, new_T)

        # upsample roi trace
        roi_trace_u = np.interp(new_t, orig_t, roi_trace)

        # upsample mean_video by interpolating each spatial pixel over time
        mv_flat = mv.reshape(T, -1)
        cols = mv_flat.shape[1]
        mv_up_cols = np.stack(
            [np.interp(new_t, orig_t, mv_flat[:, j]) for j in range(cols)], axis=1
        )
        mv_u = mv_up_cols.reshape(new_T, H, W)

        fs_effective = fs * upsample

    T_u = mv_u.shape[0]

    # compute analytic phase of the roi trace and bin into N quantile-based bins
    analytic = hilbert(roi_trace_u)
    phase = np.angle(analytic)

    # compute quantile edges so each bin tends to have an even number of samples
    edges = np.quantile(phase, np.linspace(0.0, 1.0, N + 1))
    # digitize into 0..N-1 bins; exclude first and last edges when passing to digitize
    binidx = np.digitize(phase, edges[1:-1])

    # produce averaged keyframes for each phase bin and record indices
    keyframes = np.zeros((N, H, W), dtype=float)
    indices_list = []
    for i in range(N):
        idx = np.where(binidx == i)[0].astype(int)
        indices_list.append(idx)
        if idx.size > 0:
            keyframes[i] = np.nanmean(mv_u[idx], axis=0)

    # colorize keyframes using same logic as video_and_trace
    if vmin is None and vmax is None:
        _vmax = np.abs(keyframes).max() if keyframes.size else 0.0
        _vmin = -_vmax
        color_keyframes = apply_cmap(
            keyframes, cmap=cmap, vmin=_vmin, vmax=_vmax, mode="centered"
        )
        _use_colorbar = True
    else:
        _vmin = keyframes.min() if vmin is None else vmin
        _vmax = keyframes.max() if vmax is None else vmax
        color_keyframes = apply_cmap(
            keyframes, cmap=cmap, vmin=_vmin, vmax=_vmax, mode="absolute"
        )
        _use_colorbar = True

    # determine ax_size for vu.subplots
    if ax_size is None:
        ax_size = (height, height)

    # create 2 x N grid via vu.subplots so axes are regular and sized consistently
    fig, ax = vu.subplots(ax_size, (2, N))
    ax = np.asarray(ax).reshape(2, N)

    # display images using setup_video so ROI overlays are consistent
    im_artists = []
    for i in range(N):
        ax_img = ax[0, i]
        single = color_keyframes[i : i + 1]
        im, update_fn = setup_video(
            single,
            ax_img,
            roi_collection=roi_collection,
            roi_kws={"text_kws": {"do_text": False}},
        )
        try:
            update_fn(0)
        except Exception:
            pass
        ax_img.set_title(f"[{edges[i]:.2f}, {edges[i+1]:.2f}] rad")
        im_artists.append(im)

    # plot the same roi trace on each bottom axis and overlay markers for that column
    fs_khz = fs_effective / 1000.0
    trace_color = (
        hi_rois.colors[0]
        if (hi_rois is not None and hasattr(hi_rois, "colors"))
        else "k"
    )

    marker_kwargs = dict(s=40, c="k", edgecolors="white", zorder=5)

    # mean_psth_grid expects mean shaped (trace, event, frames) and ax shaped (1,1)
    for i in range(N):
        ax_trace = ax[1, i]
        ax_grid = np.array([[ax_trace]])
        mean_psth_grid(
            mean=roi_trace_u[None, None, :], fs=fs_khz, zero=None, ax=ax_grid
        )

        idx = indices_list[i]
        if idx.size > 0:
            x = idx / fs_khz
            # guard indexing length of roi_trace_u
            idx_safe = idx[idx < roi_trace_u.shape[-1]]
            y = roi_trace_u[idx_safe]
            ax_trace.scatter(x[: len(y)], y, **marker_kwargs)

        # ensure title for each column's trace
        ax_trace.set_title(f"Pair 1")

    fig.tight_layout()

    # add a single colorbar axis positioned to the right of the image row
    if _use_colorbar:
        # approximate placement to the right of the images
        cb_ax = fig.add_axes([1.0, 0.55, 0.015, 0.3])
        mappable = ScalarMappable(norm=Normalize(vmin=_vmin, vmax=_vmax), cmap=cmap)
        fig.colorbar(mappable, cax=cb_ax)

    return fig, ax
