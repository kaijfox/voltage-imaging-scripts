"""Spike detection via high-pass filtering."""

from typing import Any

import numpy as np
from scipy.signal import savgol_filter, convolve
from scipy import ndimage
import awkward as ak

from .types import Events, Traces
from ..cli.common import configure_logging
from .qc_analysis import embed_events


def detect_spikes(
    traces: Traces,
    sd_threshold: float,
    sg_window_frames: int | None = None,
    positive_going: bool = False,
) -> tuple[Traces, Events, dict[str, Any]]:
    """Detect spikes using Savitzky-Golay high-pass filtering.

    Applies a second-order Savitzky-Golay filter to estimate baseline,
    subtracts to isolate high-frequency components, then detects spikes
    as peaks exceeding a threshold in standard deviations.

    Args:
        traces: Input traces object
        sd_threshold: Number of standard deviations for spike detection
        sg_window_frames: Savitzky-Golay filter window in frames (must be odd). If None, no HPF is applied.
        positive_going: If True, detect positive-going spikes; if False, negative-going

    Returns:
        hpf_traces: Traces object containing high-pass filtered data (Vmhigh)
        events: Events object with detected spike frames
        info: dict with per-cell statistics:
            - 'baseline_noise': array of baseline noise (SD) per cell
            - 'spike_amplitudes': list of arrays of spike amplitudes per cell
            - 'spike_sbr': list of arrays of signal-to-baseline ratios per cell
    """

    logger, (error, warning, info, debug) = configure_logging("events")

    # Only adjust window length if one was provided
    if sg_window_frames is not None and sg_window_frames % 2 == 0:
        sg_window_frames += 1

    data = traces.data
    n_cells, n_frames = data.shape

    # Compute baseline using Savitzky-Golay filter (2nd order)
    if sg_window_frames is not None:
        baseline = savgol_filter(
            data, window_length=sg_window_frames, polyorder=2, axis=1
        )
        vmhigh = data - baseline
    else:
        info("Not applying high-pass filter! No window length specified.")
        vmhigh = data.copy()

    spike_frames_list = []
    spike_amplitudes_list = []
    spike_sbr_list = []
    baseline_noise = np.zeros(n_cells)

    for i in range(n_cells):
        cell_vmhigh = vmhigh[i]
        cell_data = data[i]

        # Baseline noise: SD of the half opposite to spike direction
        if positive_going:
            noise_values = cell_vmhigh[cell_vmhigh < 0]
        else:
            noise_values = cell_vmhigh[cell_vmhigh > 0]

        if len(noise_values) > 0:
            noise_sd = np.std(noise_values)
        else:
            noise_sd = np.std(cell_vmhigh)
        baseline_noise[i] = noise_sd

        # Detect spikes
        threshold = sd_threshold * noise_sd
        if positive_going:
            above_thresh = cell_vmhigh > threshold
        else:
            above_thresh = cell_vmhigh < -threshold

        # Find peaks (local extrema among threshold crossings)
        spike_frames = []
        in_spike = False
        spike_start = 0

        for j in range(n_frames):
            if above_thresh[j] and not in_spike:
                in_spike = True
                spike_start = j
            elif not above_thresh[j] and in_spike:
                in_spike = False
                spike_region = cell_vmhigh[spike_start:j]
                if positive_going:
                    peak_idx = spike_start + np.argmax(spike_region)
                else:
                    peak_idx = spike_start + np.argmin(spike_region)
                spike_frames.append(peak_idx)

        if in_spike:
            spike_region = cell_vmhigh[spike_start:n_frames]
            if positive_going:
                peak_idx = spike_start + np.argmax(spike_region)
            else:
                peak_idx = spike_start + np.argmin(spike_region)
            spike_frames.append(peak_idx)

        spike_frames = np.array(spike_frames, dtype=np.int64)

        # Compute amplitudes and SBR
        amplitudes = np.zeros(len(spike_frames))
        for k, frame in enumerate(spike_frames):
            peak_val = cell_data[frame]
            pre_start = max(0, frame - 3)
            if positive_going:
                pre_val = np.min(cell_data[pre_start:frame]) if frame > 0 else peak_val
                amplitudes[k] = peak_val - pre_val
            else:
                pre_val = np.max(cell_data[pre_start:frame]) if frame > 0 else peak_val
                amplitudes[k] = pre_val - peak_val

        sbr = amplitudes / noise_sd if noise_sd > 0 else amplitudes

        spike_frames_list.append(spike_frames)
        spike_amplitudes_list.append(amplitudes)
        spike_sbr_list.append(sbr)

    hpf_traces = Traces(
        data=vmhigh,
        ids=traces.ids,
        fs=traces.fs,
    )

    events = Events(
        spike_frames=spike_frames_list,
        ids=traces.ids,
        detection_params={
            "sg_window_frames": sg_window_frames,
            "sd_threshold": sd_threshold,
            "positive_going": positive_going,
        },
    )

    info = {
        "baseline_noise": baseline_noise,
        "spike_amplitudes": spike_amplitudes_list,
        "spike_sbr": spike_sbr_list,
    }

    return hpf_traces, events, info


def despike(
    traces: Traces,
    events: Events,
    n_pre: int,
    n_post: int,
) -> tuple[Traces, np.ndarray]:
    """Subtract estimated spike waveforms from traces.

    Estimates a mean spike waveform per cell by averaging peri-event windows,
    then reconstructs the spike contribution via convolution of an impulse
    train with the waveform and subtracts it.

    Parameters
    ----------
    traces: Traces
        Input traces, data shape (n_cells, n_frames).
    events: Events
        Spike frames, spike_frames length n_cells, each an array of frame indices.
    n_pre, n_post: int
        Samples before/after each spike used to estimate the waveform.

    Returns
    -------
    despiked: Traces
        Traces with spike waveforms subtracted.
    waveforms: ndarray, shape (n_cells, n_pre + n_post + 1)
        Estimated mean spike waveform per cell.
    """
    import awkward as ak
    from ..windows.ragged_ops import slice_by_events, ak_infer_shape

    data = np.asarray(traces.data, dtype=float)
    n_cells, n_frames = data.shape
    window_len = n_pre + n_post + 1

    # (n_cells, <n_events>, window_len)
    windows = slice_by_events(data, events.spike_frames, n_pre, n_post)
    waveforms = ak.mean(windows, axis=1)  # (n_cells, window_len)

    event_component = np.zeros_like(data)
    for i in range(n_cells):
        frames = np.asarray(events.spike_frames[i], dtype=int)
        valid = frames[(frames >= 0) & (frames < n_frames)]
        if not len(frames) or len(valid) == 0:
            continue
        event_component[i, valid] = 1.0
        event_component[i, n_post:-n_pre] = convolve(
            event_component[i], np.asarray(waveforms[i]), mode="valid"
        )

    despiked = Traces(
        data=data - event_component,
        ids=traces.ids,
        fs=traces.fs,
    )
    return despiked, waveforms


def despike_impute(
    traces: Traces,
    events: Events,
    window_size: int,
    savgol_window_frames: int,
) -> Traces:
    # build boolean valid mask (n_cells, n_frames), start all True
    data = np.asarray(traces.data)
    n_cells, n_frames = data.shape

    # mark spike frames as invalid (False), clipping to valid frame range
    nonspike_mask = ~embed_events(
        ak.ones_like(events.spike_frames),
        ak.Array(events.spike_frames),
        n_frames,
        fill_value=0,
    ).astype(bool)

    window_size += 1 - (window_size % 2)  # ensure odd window size
    structure = np.ones([1, window_size], dtype=bool)
    nonspike_mask = ndimage.binary_erosion(nonspike_mask, structure)

    # smooth baseline with savgol_filter (polyorder=3) along time axis
    savgol_window_frames += 1 - (savgol_window_frames % 2)
    sav = savgol_filter(
        data,
        window_length=savgol_window_frames,
        polyorder=2,
        axis=-1,
    )
    imputed = np.where(nonspike_mask, data, sav)

    return Traces(data=imputed, ids=traces.ids, fs=traces.fs)
