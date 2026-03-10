import numpy as np
import awkward as ak
import pandas as pd
from typing import Tuple, Optional, Sequence, Any, Union


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


from ..windows.ragged_ops import slice_by_events
from ..io.svd_video import SVDVideo
from .rois import ROICollection
from .types import Traces, Events
from .ols_streaming import extract_traces
from .spike_analysis import ms_to_samples, peak_to_trough
from .filtering import filter_dff


def qc_trace_extraction_1(
    video: SVDVideo,
    soma_rois: ROICollection,
    background_rois: ROICollection,
    fs: float,
    hpf_ms: float,
) -> Tuple[Traces, np.ndarray]:
    """Background regression trace extraction.

    Extract raw soma and background traces from a video using the provided ROI
    collections, regress out background activity per-ROI, clip extreme
    outliers, subtract the median and apply dF/F filtering.

    Parameters
    ----------
    video : SVDVideo
        Video or video-like object providing frames for trace extraction.
    soma_rois : ROICollection
        ROICollection describing soma ROIs to extract.
    background_rois : ROICollection
        ROICollection containing one or more background ROIs; if multiple the
        mean is used as the background vector.
    fs : float
        Sampling frequency in Hz.
    hpf_ms : float
        High-pass filter length in milliseconds (used to compute window length).

    Returns
    -------
    traces_dff : Traces
        dF/F filtered traces after background regression and clipping.
    baselines : ndarray
        Baseline values produced by ``filter_dff`` (one per ROI/time as
        returned by that function).
    """

    # Extract raw soma and background traces
    soma_traces_raw, _ = extract_traces(
        video, soma_rois, neuropil_range=(-1, -1), ols=False, weighted=False, fs=fs
    )
    bg_traces_raw, _ = extract_traces(
        video,
        background_rois,
        neuropil_range=(-1, -1),
        ols=False,
        weighted=False,
        fs=fs,
    )

    soma_data = soma_traces_raw.data.copy()
    # Use single background ROI (collapse if necessary)
    bg_data = bg_traces_raw.data
    if bg_data.shape[0] == 1:
        bg_vec = bg_data[0]
    else:
        bg_vec = bg_data.mean(axis=0)

    # Regress out background from each soma trace
    for i in range(soma_data.shape[0]):
        lr = LinearRegression()
        lr.fit(bg_vec.reshape(-1, 1), soma_data[i])
        m = lr.coef_[0]
        b = lr.intercept_
        corrected = soma_data[i] - (m * bg_vec + b)

        # Clip extreme positive outliers (> mean + 10*std)
        mean = np.mean(corrected)
        std = np.std(corrected)
        upper = mean + 10 * std
        corrected = np.where(corrected > upper, upper, corrected)

        # Subtract median to produce Vm-like trace
        corrected = corrected - np.median(corrected)

        soma_data[i] = corrected

    traces_vm = Traces(soma_data, soma_traces_raw.ids, fs)

    # Apply dF/F filtering (same as qc_trace_extraction_2)
    traces_vmhigh, baselines = filter_dff(
        traces_vm,
        mode="savgol_add",
        window_length=ms_to_samples(hpf_ms, fs),
        polyorder=2,
    )

    return traces_vmhigh, baselines, soma_traces_raw, bg_traces_raw, traces_vm


def qc_trace_extraction_2(
    video: Traces,
    soma_rois: ROICollection,
    fs: float,
    hpf_ms: float,
    neuropil_range: Tuple[int, int] = (-1, -1),
) -> Tuple[Traces, Traces, Traces, Traces]:
    """Extract traces from provided video and ROI collection, then apply dF/F filtering.

    This function expects already-loaded ``video``/frame data (e.g., a Traces
    or SVDVideo-like object) and an ROICollection describing soma ROIs.

    Parameters
    ----------
    video : Traces
        Video-like object or preloaded frames passed to ``extract_traces``.
    soma_rois : ROICollection
        ROICollection describing soma ROIs to extract.
    fs : float
        Sampling frequency in Hz.
    hpf_ms : float
        High-pass filter length in milliseconds (used to compute window length).

    Returns
    -------
    traces_dff : Traces
        dF/F filtered traces (shifted so baseline corresponds to 0).
    baselines : ndarray
        Baselines returned by `filter_dff`.
    """

    traces_raw, traces_raw_neuropil = extract_traces(
        video,
        soma_rois,
        neuropil_range=neuropil_range,
        ols=False,
        weighted=False,
        fs=fs,
    )

    traces_dff, baselines = filter_dff(
        traces_raw,
        mode="savgol_mult",
        window_length=ms_to_samples(hpf_ms, fs),
        polyorder=2,
    )
    traces_dff = Traces(traces_dff.data - 1, traces_dff.ids, traces_dff.fs)

    return traces_raw, traces_dff, baselines, traces_raw_neuropil


def qc_trace_extraction_3(
    video: Traces,
    soma_rois: ROICollection,
    fs: float,
    hpf_ms: float,
    neuropil_range: Tuple[int, int] = (-1, -1),
):
    """
    Parameters
    ----------
    See qc_trace_extraction_2.

    Returns
    -------
    traces : dict[str, Traces]
        With keys
        - "raw", "neuropil_raw",
        - "hp", "neuropil_hp",
        - "baseline", "neuropil_baseline"
        - "neuropil_component",
        - "raw_corrected", "hp_corrected"
    """
    # Compute roi and neuropil raw using extract_traces with remove_neuropil=False
    # -> raw, neuropil_raw
    # Compute roi and neuropil dff with savgol_add
    # -> hp, neuropil_hp, baseline, neuropil_baseline
    # Regress neuropil dff out of roi dff (linear regression per roi)
    # -> neuropil_component, hp_corrected
    # Add baseline back
    # -> raw_corrected
    


def embed_events(
    values: ak.Array,
    spike_frames: ak.Array,
    n_frames: int,
    fill_value=np.nan,
) -> np.ndarray:
    """Embed per-event values into dense per-ROI time series.

    Parameters
    ----------
    values : ak.Array
        Awkward array of scalar event values for each ROI; shape is
        (n_rois, <n_events>) where ``<n_events>`` denotes a variable-length
        inner dimension (events per ROI).
    spike_frames : ak.Array
        Awkward array of integer frame indices corresponding to events in
        ``values``. Indices are in the range ``[0, n_frames-1]``.
    n_frames : int
        Length of the output time axis (number of frames).
    fill_value : scalar, default np.nan
        Value to fill in for non-event frames

    Returns
    -------
    out : np.ndarray
        Dense NumPy array with embedded event values at their corresponding
        frame indices and ``np.nan`` elsewhere. Shape is (n_rois, n_frames).
    """
    # values: ak array (n_rois, <n_events>) of scalars
    # spike_frames: ak array (n_rois, <n_events>) of int frame indices
    n_rois = len(values)
    out = np.full((n_rois, n_frames), fill_value, dtype=float)

    for i in range(n_rois):
        # get indices and values for ROI i
        idxs = spike_frames[i].to_list()
        if len(idxs) == 0:
            continue
        vals = values[i].to_numpy()
        out[i, idxs] = vals

    return out


def rolling_window(
    data: np.ndarray,
    window_size_ms: Union[float, int],
    fs: float,
    window_hop_ms: Optional[Union[float, int]] = None,
    pad: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling windows from time-series data.

    Parameters
    ----------
    data : np.ndarray
        Input time-series array with arbitrary leading dimensions and a final
        time dimension of length T.
    window_size_ms : float or int
        Window length in milliseconds.
    fs : float
        Sampling frequency in Hz (frames per second).
    window_hop_ms : float or int, optional
        Step between successive windows in milliseconds. If ``None``, defaults
        to ``window_size_ms`` (non-overlapping windows).
    pad : bool, optional
        If True, pad the windows output along the n_steps axis with NaN slices
        (half_win on each side) so its length matches the original number of
        timepoints T, and return times = np.arange(T) / fs. If False (default),
        existing behavior unchanged.

    Returns
    -------
    windows : np.ndarray
        Windowed views of the input data along the last axis. Shape is
        (..., n_steps, W) or (..., T, W) if pad=True where ``W`` is the window
        length in samples and ``n_steps`` is the number of windows extracted.
    times : np.ndarray
        Center times for each extracted window, in seconds.
    """
    # convert ms → frames; ensure odd window size
    window_size_frames = int(round(float(window_size_ms) * float(fs) / 1000.0))
    if window_size_frames <= 0:
        raise ValueError("window_size_ms too small; results in 0 frames")
    # make window size odd
    if window_size_frames % 2 == 0:
        window_size_frames += 1
    half_win = window_size_frames // 2

    if window_hop_ms is None:
        window_hop_ms = window_size_ms
    hop_frames = int(round(float(window_hop_ms) * float(fs) / 1000.0))
    if hop_frames < 1:
        hop_frames = 1

    # sliding window view along last axis
    windows_all = np.lib.stride_tricks.sliding_window_view(
        data, window_size_frames, axis=-1
    )
    # subsample along the window-step axis (second-to-last)
    if hop_frames > 1:
        windows = windows_all[..., ::hop_frames, :]
    else:
        windows = windows_all

    T = data.shape[-1]
    center_idxs = np.arange(half_win, T - half_win, hop_frames)
    if pad:
        # create padded windows of length T along the step axis, fill with NaN
        padded_shape = windows.shape[:-2] + (T, window_size_frames)
        padded_windows = np.full(padded_shape, np.nan, dtype=float)
        # assign windows at their center indices (cast to float to allow NaN)
        padded_windows[..., center_idxs, :] = windows.astype(float)
        times = np.arange(T) / float(fs)
        return padded_windows, times

    times = np.arange(T)[half_win : T - half_win : hop_frames] / float(fs)

    return windows, times


def compute_spike_metrics(
    traces_dff: Traces, events: Events, fs: float
) -> pd.DataFrame:
    """Compute per-spike metrics and return a pandas DataFrame.

    Background is estimated as 2 * rolling std of positive-going dF/F using a
    1s window (1000 ms) and hop of 1 frame, padded so background value aligns
    with frame indices.

    Parameters
    ----------
    traces_dff : Traces
        dF/F traces used for spike metric computations (n_rois, n_frames).
    events : Events
        Detected spike events corresponding to the traces.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with one row per spike and columns:
        ['roi_id', 'spike_time_s', 'raw_dff', 'peak_to_trough_dff', 'sbr', 'background']
    """
    # Prepare arrays
    data = np.asarray(traces_dff.data, dtype=float)
    n_rois, n_frames = data.shape

    # per-event peak-to-trough (ak.Array of shape (n_rois, <n_events>))
    print("Computing peak-to-trough.")
    peak_trough = peak_to_trough(data, events.spike_frames, fs=fs, ms_pre=0, ms_post=10)

    # positive-going dff with NaN elsewhere and rolling background
    print("Computing background.")
    pos_going = np.where(data > 0, data, np.nan)
    windows, times = rolling_window(
        pos_going, window_size_ms=1000.0, fs=fs, window_hop_ms=1.0 / fs, pad=True
    )
    background_series = 2 * np.nanstd(windows, axis=-1)  # shape (n_rois, n_frames)

    # Build awkward arrays for vectorized indexing
    pf = ak.Array(events.spike_frames)
    pt = ak.Array(peak_trough)
    bg_ak = ak.Array(background_series)
    raw_ak = ak.Array(data)

    # Index background and raw traces at spike frames (ragged arrays)
    bg_at_spikes = bg_ak[pf]
    raw_at_spikes = raw_ak[pf]

    # Compute sbr elementwise, guarding against zero/NaN/None backgrounds
    print("Computing SBR.")
    sbr = pt / bg_at_spikes

    # Accumulate per-ROI ragged arrays into lists, then concatenate once
    cols = {
        "roi_id": [],
        "spike_time_s": [],
        "raw_dff": [],
        "peak_to_trough_dff": [],
        "sbr": [],
        "background": [],
    }

    print("Assembling DataFrame.")

    for i in range(n_rois):
        roi_id = traces_dff.ids[i] if traces_dff.ids is not None else i
        frames_i = ak.to_list(pf[i])
        if len(frames_i) == 0:
            continue
        frames_np = np.asarray(frames_i, dtype=int)

        # per-spike arrays for this ROI
        raw_vals = np.asarray(ak.to_list(raw_at_spikes[i]), dtype=float)
        pt_vals = np.asarray(ak.to_list(pt[i]), dtype=float)
        bg_vals = np.asarray(ak.to_list(bg_at_spikes[i]), dtype=float)
        sbr_vals = np.asarray(ak.to_list(sbr[i]), dtype=float)

        cols["roi_id"].append(np.repeat(roi_id, len(frames_np)))
        cols["spike_time_s"].append(frames_np / float(fs))
        cols["raw_dff"].append(raw_vals)
        cols["peak_to_trough_dff"].append(pt_vals)
        cols["sbr"].append(sbr_vals)
        cols["background"].append(bg_vals)

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    return df


def compute_binned_stats(
    raw_traces: Traces,
    traces_dff: Traces,
    events: Events,
    fs: float,
    bin_size_s: float = 10.0,
) -> pd.DataFrame:
    """Compute binned statistics per ROI and return a pandas DataFrame.

    mean_sbr and mean_dff are per-spike means inside each bin (NaN if no spikes).
    background_noise is the mean of the rolling background within the bin.

    Parameters
    ----------
    raw_traces : Traces
        Raw fluorescence traces (n_rois, n_frames).
    traces_dff : Traces
        dF/F traces corresponding to raw_traces.
    events : Events
        Detected spike events for each ROI.
    fs : float
        Sampling frequency in Hz.
    bin_size_s : float, optional
        Bin size in seconds used to aggregate metrics (default 10.0).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with rows for each (roi_id, bin_time_s) containing metrics:
        ['mean_F','spike_rate','mean_sbr','mean_dff','background_noise']
    """
    data_raw = np.asarray(raw_traces.data, dtype=float)
    data_dff = np.asarray(traces_dff.data, dtype=float)
    n_rois, n_frames = data_raw.shape

    bin_frames = int(np.round(bin_size_s * fs))
    if bin_frames < 1:
        bin_frames = 1
    n_bins = int(np.ceil(n_frames / bin_frames))

    # rolling background same as compute_spike_metrics
    print("Computing background.")
    pos_going = np.where(data_dff > 0, data_dff, np.nan)
    windows, times = rolling_window(
        pos_going, window_size_ms=1000.0, fs=fs, window_hop_ms=1.0 / fs, pad=True
    )
    background_series = 2 * np.nanstd(windows, axis=-1)

    # per-spike peak-to-trough and per-spike raw dff (ragged)
    print("Computing peak-to-trough and SBR")
    peak_trough = peak_to_trough(
        data_dff, events.spike_frames, fs=fs, ms_pre=0, ms_post=10
    )
    pf = ak.Array(events.spike_frames)
    pt = ak.Array(peak_trough)
    bg_ak = ak.Array(background_series)
    raw_spike_ak = ak.Array(data_dff)[pf]

    # compute per-spike sbr
    sbr_ak = pt / bg_ak[pf]

    # embed per-spike metrics into time series (roi, frames)
    print("Computing time-averaged event metrics.")
    event_sbr_ts = embed_events(sbr_ak, pf, n_frames)
    event_dff_ts = embed_events(raw_spike_ak, pf, n_frames)

    # pad to full bins length and reshape to (n_rois, n_bins, bin_frames)
    total_len = n_bins * bin_frames
    pad_len = total_len - n_frames

    def _pad_and_reshape(arr):
        if pad_len > 0:
            pad = np.full((n_rois, pad_len), np.nan)
            arr_p = np.concatenate([arr, pad], axis=1)
        else:
            arr_p = arr[:, :total_len]
        return arr_p.reshape(n_rois, n_bins, bin_frames)

    sbr_bins = _pad_and_reshape(event_sbr_ts)
    dff_bins = _pad_and_reshape(event_dff_ts)
    raw_bins = _pad_and_reshape(np.asarray(data_raw, dtype=float))
    bg_bins = _pad_and_reshape(np.asarray(background_series, dtype=float))

    # compute statistics per (roi, bin)
    mean_sbr = np.nanmean(sbr_bins, axis=-1)
    mean_dff = np.nanmean(dff_bins, axis=-1)
    mean_F = np.nanmean(raw_bins, axis=-1)
    background_noise = np.nanmean(bg_bins, axis=-1)

    # counts (spikes per bin) and frames per bin (handle last partial bin)
    print("Computing spike counts.")
    spike_counts = np.sum(~np.isnan(sbr_bins), axis=-1)
    frames_per_bin = np.full(n_bins, bin_frames)
    last_frames = n_frames - bin_frames * (n_bins - 1)
    if last_frames > 0:
        frames_per_bin[-1] = last_frames

    # spike rate: spikes / frames_in_bin * fs
    spike_rate = spike_counts / frames_per_bin[None, :] * float(fs)

    # assemble DataFrame via per-ROI lists then concatenate
    print("Assembling DataFrame.")
    bin_times = (np.arange(n_bins) * bin_frames) / float(fs)
    cols = {
        "roi_id": [],
        "bin_time_s": [],
        "mean_F": [],
        "spike_rate": [],
        "mean_sbr": [],
        "mean_dff": [],
        "background_noise": [],
    }

    for i in range(n_rois):
        roi_id = raw_traces.ids[i] if raw_traces.ids is not None else i
        cols["roi_id"].append(np.repeat(roi_id, n_bins))
        cols["bin_time_s"].append(bin_times)
        cols["mean_F"].append(mean_F[i])
        cols["spike_rate"].append(spike_rate[i])
        cols["mean_sbr"].append(mean_sbr[i])
        cols["mean_dff"].append(mean_dff[i])
        cols["background_noise"].append(background_noise[i])

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    return df


def _linear_fit(df, col_x, col_y, log=False):
    X = df[col_x].to_numpy().reshape(-1, 1)
    y = df[col_y].to_numpy()

    # Remove nans and non-positive values if log transform is requested
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan
    if log:
        if np.any(y[valid] <= 0):
            return np.nan, np.nan, np.nan
        y = np.log(y[valid])
        X = X[valid]
    else:
        y = y[valid]
        X = X[valid]

    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)
    r2 = float(r2_score(y, y_pred))
    return slope, intercept, r2


def compute_trend_fits(binned_df: pd.DataFrame, spike_df: pd.DataFrame) -> pd.DataFrame:
    """Fit linear and log trends for specified metrics per roi.

    Fits both linear and log-linear trends for selected metrics across time
    bins. Rows are returned per (roi_id, metric, fit_type).

    Parameters
    ----------
    binned_df : pd.DataFrame
        DataFrame produced by ``compute_binned_stats`` with binned metrics.
    spike_df : pd.DataFrame
        Per-spike DataFrame (unused in current implementation but accepted
        for API compatibility).

    Returns
    -------
    out : pd.DataFrame
        DataFrame with columns ['roi_id','metric','fit_type','slope','intercept','r2']
    """
    metrics_bin = ["mean_F", "spike_rate", "background_noise"]
    metrics_spike = ["raw_dff", "peak_to_trough_dff", "sbr"]
    rows_lin = []
    rows_log = []

    # group by session and roi_id
    grouped = binned_df.groupby("roi_id")
    for roi, group in grouped:
        # sort by time
        g = group.sort_values("bin_time_s")
        X = g["bin_time_s"].to_numpy().reshape(-1, 1)
        for metric in metrics_bin:
            slope, intercept, r2 = _linear_fit(g, "bin_time_s", metric)
            slope_log, intercept_log, r2_log = _linear_fit(
                g, "bin_time_s", metric, log=True
            )
            slope_per_min = slope * 60
            pct_per_min = 100 * (np.exp(slope_log) - 1) * 60

            rows_lin.append((roi, metric, "linear", slope_per_min, intercept, r2))
            rows_log.append(
                (roi, metric, "log", pct_per_min, np.exp(intercept_log), r2_log)
            )

    grouped = spike_df.groupby("roi_id")
    for roi, group in grouped:
        for metric in metrics_spike:
            slope, intercept, r2 = _linear_fit(group, "spike_time_s", metric)
            slope_log, intercept_log, r2_log = _linear_fit(
                group, "spike_time_s", metric, log=True
            )
            slope_per_min = slope * 60
            pct_per_min = 100 * (np.exp(slope_log) - 1) * 60
            rows_lin.append((roi, metric, "linear", slope_per_min, intercept, r2))
            rows_log.append(
                (roi, metric, "log", pct_per_min, np.exp(intercept_log), r2_log)
            )

    out_lin = pd.DataFrame(
        rows_lin,
        columns=["roi_id", "metric", "fit_type", "change_per_min", "intercept", "r2"],
    )
    out_log = pd.DataFrame(
        rows_log,
        columns=["roi_id", "metric", "fit_type", "pct_per_min", "initial", "r2"],
    )
    return out_lin, out_log


def update_session(
    existing_df: Optional[pd.DataFrame], new_df: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Replace rows for new_df.session.iloc[0] in existing_df with new_df and return combined DataFrame.

    Parameters
    ----------
    existing_df : pandas.DataFrame or None
        Existing session-level DataFrame to update.
    new_df : pandas.DataFrame or None
        New DataFrame containing rows for a single session; rows for that
        session will replace any matching rows in ``existing_df``.

    Returns
    -------
    out : pandas.DataFrame or None
        Combined DataFrame with the updated session rows, or None if both
        inputs are None.
    """
    if new_df is None or new_df.empty:
        return existing_df
    sess = new_df["session"].iloc[0]
    if existing_df is None or existing_df.empty:
        return new_df.copy()
    # drop rows matching session
    out = existing_df[existing_df["session"] != sess].copy()
    out = pd.concat([out, new_df], ignore_index=True)
    return out
