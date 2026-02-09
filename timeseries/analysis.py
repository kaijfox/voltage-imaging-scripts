from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len, fft, ifft, fftshift


def _standardize(x: NDArray, axis: int = -1) -> NDArray:
    """Subtract mean and divide by std along `axis`."""
    x = x - x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    std[std == 0] = 1.0
    return x / std


def cross_correlate(
    x: NDArray,
    y: NDArray,
    fs: float,
    max_lag: float,
    mode: Literal["covariance", "correlation"] = "covariance",
) -> tuple[NDArray, NDArray]:
    """Broadcasted cross-correlation via FFT.

    Sign convention: corr(τ) ~ Σ x(t+τ)·y(t). Positive lag = y leads x.

    Args:
        x, y: (..., samples). Must broadcast against each other.
        fs: Sampling rate (Hz).
        max_lag: Maximum lag (seconds).
        mode: 'covariance' (mean-subtracted) or 'correlation' (standardized).

    Returns:
        corr: (..., n_lags) cross-correlation values.
        lags: (n_lags,) lag times in seconds.
    """
    # Standardize or demean
    # Determine FFT length for linear (non-circular) correlation
    # n = next_fast_len(2 * samples - 1)
    # FFT both, cross-spectrum, ifft
    # fftshift to reorder lags to [-max, ..., 0, ..., +max]
    # Truncate to ±max_lag samples, divide by samples
    # Build lags array from truncation indices and fs

    # Implemented: prepare signals
    S = x.shape[-1]
    if mode == "correlation":
        x_proc = _standardize(x, axis=-1)
        y_proc = _standardize(y, axis=-1)
    else:
        x_proc = x - x.mean(axis=-1, keepdims=True)
        y_proc = y - y.mean(axis=-1, keepdims=True)

    # Broadcast leading dimensions so FFT operations align
    prefix = np.broadcast_shapes(x_proc.shape[:-1], y_proc.shape[:-1])
    x_b = np.broadcast_to(x_proc, prefix + (S,))
    y_b = np.broadcast_to(y_proc, prefix + (S,))

    # FFT-based cross-correlation
    n = next_fast_len(2 * S - 1)
    X = fft(x_b, n=n, axis=-1)
    Y = fft(y_b, n=n, axis=-1)
    corr_full = ifft(X * np.conj(Y), n=n, axis=-1).real / S

    # Shift so zero-lag is centered and slice to requested max_lag
    corr_shifted = fftshift(corr_full, axes=-1)

    max_lag_samples = int(max(0, int(max_lag * fs)))
    if max_lag_samples >= S:
        # clamp to valid range
        max_lag_samples = S - 1

    center = n // 2
    start = center - max_lag_samples
    end = center + max_lag_samples + 1
    corr = corr_shifted[..., start:end]

    lags = np.arange(-max_lag_samples, max_lag_samples + 1) / fs

    return corr, lags


def pairwise_cross_correlations(
    x: NDArray,
    y: NDArray | None,
    fs: float,
    max_lag: float,
    mode: Literal["covariance", "correlation"] = "covariance",
) -> tuple[NDArray, NDArray]:
    """All-pairs cross-correlations between two sets of timeseries.

    Sign convention: corr(τ) ~ Σ x(t+τ)·y(t). Positive lag = y leads x.

    Args:
        x: (n_x, samples).
        y: (n_y, samples) or None (auto-correlation: y = x).
        fs: Sampling rate (Hz).
        max_lag: Maximum lag (seconds).
        mode: 'covariance' or 'correlation'.

    Returns:
        corr: (n_x, n_y, n_lags).
        lags: (n_lags,) lag times in seconds.
    """
    if y is None:
        y = x
    # x[:, None, :] vs y[None, :, :] → broadcast to (n_x, n_y, samples)
    return cross_correlate(x[:, None, :], y[None, :, :], fs, max_lag, mode)
