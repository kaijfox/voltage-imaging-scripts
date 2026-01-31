"""Spike detection via high-pass filtering."""

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .types import Events, Traces


def detect_spikes(
    traces: Traces,
    sg_window_frames: int,
    sd_threshold: float,
    positive_going: bool = False,
) -> tuple[Traces, Events, dict[str, Any]]:
    """Detect spikes using Savitzky-Golay high-pass filtering.

    Applies a second-order Savitzky-Golay filter to estimate baseline,
    subtracts to isolate high-frequency components, then detects spikes
    as peaks exceeding a threshold in standard deviations.

    Args:
        traces: Input traces object
        sg_window_frames: Savitzky-Golay filter window in frames (must be odd)
        sd_threshold: Number of standard deviations for spike detection
        positive_going: If True, detect positive-going spikes; if False, negative-going

    Returns:
        hpf_traces: Traces object containing high-pass filtered data (Vmhigh)
        events: Events object with detected spike frames
        info: dict with per-cell statistics:
            - 'baseline_noise': array of baseline noise (SD) per cell
            - 'spike_amplitudes': list of arrays of spike amplitudes per cell
            - 'spike_sbr': list of arrays of signal-to-baseline ratios per cell
    """
    if sg_window_frames % 2 == 0:
        sg_window_frames += 1

    data = traces.data
    n_cells, n_frames = data.shape

    # Compute baseline using Savitzky-Golay filter (2nd order)
    baseline = savgol_filter(data, window_length=sg_window_frames, polyorder=2, axis=1)
    vmhigh = data - baseline

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
            'sg_window_frames': sg_window_frames,
            'sd_threshold': sd_threshold,
            'positive_going': positive_going,
        },
    )

    info = {
        'baseline_noise': baseline_noise,
        'spike_amplitudes': spike_amplitudes_list,
        'spike_sbr': spike_sbr_list,
    }

    return hpf_traces, events, info
