import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from typing import Tuple, Sequence

from .types import Traces
from ..cli.common import configure_logging


def double_exponential_curve(t, a, b, c, d, e):
    b = min(b, 0)
    d = min(d, 0)
    a = max(a, 0)
    c = max(c, 0)
    return a * np.exp(b * t) + c * np.exp(d * t) + e


def filter_dff(
    traces: Traces, mode: str, window_length: int = 301, polyorder: int = 2
) -> Tuple[Traces]:
    """
    Compute dF/F0 from raw traces.

    mode : str, (savgol_add, savgol_mult, 2exp, both)
        The mode to use for filtering

    Returns
    -------
    dff_traces : Traces
        The dF/F0 traces.
    baselines : Traces
        The computed baselines.
    """

    logger, (error, warning, info, debug) = configure_logging("filter")

    if mode not in ["savgol_add", "savgol_mult", "2exp"]:
        raise ValueError(f"Unknown df/f filtering mode: {mode}")

    data = traces.data

    if mode == "2exp":
        dff = np.zeros_like(data)
        baseline = np.zeros_like(data)

        T = data.shape[1]
        time = np.arange(T)
        for i in range(data.shape[0]):

            # No-op on constant traces
            if np.all(data[i] == data[i, 0]):
                info(f"Trace {i} is constant, not filtering")
                dff[i] = data[i]
                continue

            # Heuristic initial parameters
            data_min, data_max = data[i].min(), data[i].max()
            data_range = data_max - data_min
            log_data = np.log(data[i] - data_min + 1e-3 * data_range)
            decay_start = min(np.median(np.diff(log_data[: T // 4])), 0)
            decay_end = min(np.median(np.diff(log_data[T // 4 :])), 0)
            p0 = (data_range, decay_start, data_range, decay_end, data_min)
            bounds = (
                (0, -np.inf, 0, -np.inf, -np.inf),
                (np.inf, 0, np.inf, 0, np.inf),
            )

            # Curve fit and remove baseline
            try:
                popt, _ = curve_fit(
                    double_exponential_curve,
                    time,
                    data[i],
                    p0=p0,
                    bounds=bounds,
                    maxfev=10000,
                )
            except RuntimeError:
                warning(f"Curve fit failed for trace {i}, not filtering")
                dff[i] = data[i]
                continue
            baseline[i] = double_exponential_curve(time, *popt)
            dff[i] = data[i] / baseline[i]

    if mode == "savgol_add":
        baseline = signal.savgol_filter(data, window_length, polyorder, axis=1)
        dff = data - baseline

    if mode == "savgol_mult":
        baseline = signal.savgol_filter(data, window_length, polyorder, axis=1)
        dff = data / baseline

    return (
        Traces(dff, traces.ids, traces.fs),
        Traces(baseline, traces.ids, traces.fs),
    )


def frequency_filter(
    data: np.ndarray,
    fs: float,
    freqs: Sequence[str],
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Apply frequency-domain filtering described by `freqs`.

    Parameters
    ----------
    data: array-like
        Input array. Filtering is applied along `axis`.
    fs: float
        Sampling frequency in Hz.
    freqs: sequence of str
        Frequency specifications. Examples:
        - '<50'  -> high-pass at 50 Hz (remove <50)
        - '100-200' -> band-cut 100-200 Hz implemented as data - bandpass(data)
        - '>2000' -> low-pass at 2000 Hz (remove >2000)
    order: int
        Butterworth filter order (shared across all filters).
    axis: int
        Axis along which to filter.

    Returns
    -------
    filtered: np.ndarray
        Filtered array of same shape as input.
    """
    x = np.asarray(data, dtype=float)
    nyq = 0.5 * fs

    for spec in freqs:
        s = str(spec).strip()
        if s.startswith("<"):
            cutoff = float(s[1:])
            b, a = signal.butter(order, cutoff / nyq, btype="high")
            x = signal.filtfilt(b, a, x, axis=axis)
        elif s.startswith(">"):
            cutoff = float(s[1:])
            b, a = signal.butter(order, cutoff / nyq, btype="low")
            x = signal.filtfilt(b, a, x, axis=axis)
        elif "-" in s:
            lo, hi = s.split("-", 1)
            low = float(lo)
            high = float(hi)
            b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
            band = signal.filtfilt(b, a, x, axis=axis)
            x = x - band
        else:
            raise ValueError(f"Unrecognized frequency spec: {spec}")

    return x
