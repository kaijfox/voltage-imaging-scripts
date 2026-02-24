from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len, fft, ifft, fftshift, rfft, rfftfreq
from scipy.signal import get_window


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


def power_spectrum(
    x: NDArray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    mode: Literal["density", "amplitude"] = "density",
) -> tuple[NDArray, NDArray]:
    """Compute one-sided power or amplitude spectrum for real-valued signals.

    Parameters
    ----------
    x : NDArray
        Input time series with shape (..., samples). Leading dimensions are
        broadcast across when averaging across segments.
    fs : float
        Sampling frequency in Hz.
    nperseg : int or None, optional
        Length of each segment in samples. If None, the full signal length is
        used as a single, non-overlapped segment.
    noverlap : int or None, optional
        Number of samples to overlap between segments. If None, defaults to
        int(nperseg // 2) when nperseg is provided.
    window : str, optional
        Window name (default 'hann') applied to each segment prior to FFT.
    mode : {'density', 'amplitude'}, optional
        'density' (default) returns the one-sided power spectral density (PSD)
        with units signal_units**2/Hz. 'amplitude' returns a one-sided
        amplitude spectrum in the same units as the input signal.

    Returns
    -------
    psd : NDArray
        Spectral estimate with shape (..., n_freqs). For 'density' this is the
        power spectral density; for 'amplitude' this is the amplitude spectrum.
    freqs : NDArray
        Frequencies (Hz) corresponding to the spectrum, length n_freqs as
        produced by rfftfreq(seg_len, d=1/fs).

    Notes
    -----
    - When nperseg is provided the implementation performs Welch-style
      averaging: the signal is split into overlapping segments, each windowed,
      rfft'd, and the per-segment spectra are averaged.
    - Scaling for 'density' divides by the equivalent noise bandwidth and by
      the sampling frequency so that integrating the PSD over frequency yields
      the signal variance (signal_units**2). For 'amplitude' a window-dependent
      normalization (e.g., division by the RMS of the window and a factor of
      sqrt(2) for one-sided spectra) is applied so that the returned spectrum
      represents linear amplitudes.
    - For real-valued inputs a one-sided spectrum is returned: non-DC and
      non-Nyquist frequency bins are doubled to conserve total power.
    - The frequency axis contains n_freqs = seg_len//2 + 1 points for even seg_len.
    """
    # if nperseg set, slice x into overlapping segments → (..., n_segs, nperseg)
    # apply window to last axis; rfft along last axis → (..., [n_segs,] n_freqs)
    # average |X|^2 across segments (if any)
    # scale by window power and fs for 'density', or sqrt(2)/rms(window) for 'amplitude'
    # double non-DC/Nyquist bins for one-sided; return (psd, rfftfreq(seg_len, d=1/fs))
    S = x.shape[-1]
    # determine segment length and overlap
    seg_len = S if nperseg is None else int(nperseg)
    if seg_len <= 0:
        raise ValueError("nperseg must be > 0")
    if seg_len > S:
        seg_len = S
    if nperseg is None:
        noverlap = 0
    else:
        if noverlap is None:
            noverlap = seg_len // 2
        noverlap = int(noverlap)
        if noverlap >= seg_len:
            noverlap = max(0, seg_len - 1)
    step = seg_len - (noverlap or 0)
    # build segment start indices
    if step <= 0:
        starts = np.array([0], dtype=int)
    else:
        starts = np.arange(0, S - seg_len + 1, step, dtype=int)
        if starts.size == 0:
            starts = np.array([0], dtype=int)
    n_segs = starts.size
    # form segments (..., n_segs, seg_len)
    if n_segs == 1:
        segs = x[..., starts[0] : starts[0] + seg_len][..., None, :]
    else:
        segs = np.stack([x[..., s : s + seg_len] for s in starts], axis=-2)
    # windowing and rfft
    win = get_window(window, seg_len)
    # protect against zero window power/sum
    win_pow = float(np.sum(win * win))
    if win_pow == 0:
        win_pow = 1.0
    win_sum = float(np.sum(win))
    if win_sum == 0:
        win_sum = 1.0
    X = rfft(segs * win, axis=-1)
    # average power across segments
    mean_power = np.mean(np.abs(X) ** 2, axis=-2)
    freqs = rfftfreq(seg_len, d=1.0 / fs)
    if mode == "density":
        # scale so integrating PSD over freq yields variance
        psd = mean_power / (fs * win_pow)
        # double non-DC/Nyquist bins for one-sided output
        if seg_len % 2 == 0:
            if psd.shape[-1] > 2:
                psd[..., 1:-1] *= 2.0
        else:
            if psd.shape[-1] > 1:
                psd[..., 1:] *= 2.0
        return psd, freqs
    elif mode == "amplitude":
        # amplitude: convert mean power to RMS amplitude per frequency bin,
        # correct for window coherent gain (sum of window); 2x for one-sided
        amp = 2.0 * np.sqrt(mean_power) / win_sum
        return amp, freqs
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'density' or 'amplitude'")


def cross_spectrum(
    x: NDArray,
    y: NDArray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray, NDArray]:
    """Compute the complex-valued cross-spectrum between two real-valued signals.

    Parameters
    ----------
    x : NDArray
        Input time series with shape (..., samples). Leading dimensions must
        broadcast against y; the last axis is interpreted as time/samples.
    y : NDArray
        Second input time series with shape (..., samples), broadcastable to x.
    fs : float
        Sampling frequency in Hz.
    nperseg : int or None, optional
        Segment length in samples for Welch-style averaging. If None, the
        entire signal length is used as a single segment.
    noverlap : int or None, optional
        Number of samples to overlap between segments. If None, defaults to
        int(nperseg // 2) when nperseg is provided.
    window : str, optional
        Window name (default 'hann') applied to each segment prior to FFT.

    Returns
    -------
    csd : NDArray
        Complex-valued cross-spectrum with shape (..., n_freqs). By convention
        S_xy(f) = <X(f) * conj(Y(f))>, where X and Y are the rfft of the
        windowed segments for x and y respectively.
    freqs : NDArray
        Frequencies (Hz) corresponding to the spectrum, length n_freqs.

    Notes
    -----
    - The cross-spectrum is computed per-segment as X * conj(Y) and then
      averaged across segments (Welch method) when segmentation is used.
    - Scaling by window power and the sampling frequency is applied so that
      S_xy has units signal_x * signal_y / Hz; for real inputs the returned
      spectrum is one-sided and non-DC/non-Nyquist bins are doubled to
      conserve total cross-power.
    - The complex phase of S_xy encodes relative timing/phase between x and y.
    - Coherence can be derived from the cross-spectrum via |S_xy|^2 / (S_xx * S_yy).
    """
    # broadcast x, y to common leading shape
    # segment + window both signals; rfft each → X, Y
    # cross-spectrum per segment: X * conj(Y); average across segments
    # scale by window power and fs; double non-DC/Nyquist bins
    # return (csd, freqs)
    S = x.shape[-1]
    # determine segment length and overlap
    seg_len = S if nperseg is None else int(nperseg)
    if seg_len <= 0:
        raise ValueError("nperseg must be > 0")
    if seg_len > S:
        seg_len = S
    if nperseg is None:
        noverlap = 0
    else:
        if noverlap is None:
            noverlap = seg_len // 2
        noverlap = int(noverlap)
        if noverlap >= seg_len:
            noverlap = max(0, seg_len - 1)
    step = seg_len - (noverlap or 0)
    # segment start indices
    if step <= 0:
        starts = np.array([0], dtype=int)
    else:
        starts = np.arange(0, S - seg_len + 1, step, dtype=int)
        if starts.size == 0:
            starts = np.array([0], dtype=int)
    n_segs = starts.size
    # broadcast leading dims
    prefix = np.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    x_b = np.broadcast_to(x, prefix + (S,))
    y_b = np.broadcast_to(y, prefix + (S,))
    # build segments
    if n_segs == 1:
        x_segs = x_b[..., starts[0] : starts[0] + seg_len][..., None, :]
        y_segs = y_b[..., starts[0] : starts[0] + seg_len][..., None, :]
    else:
        x_segs = np.stack([x_b[..., s : s + seg_len] for s in starts], axis=-2)
        y_segs = np.stack([y_b[..., s : s + seg_len] for s in starts], axis=-2)
    # window and rfft
    win = get_window(window, seg_len)
    win_pow = float(np.sum(win * win))
    if win_pow == 0:
        win_pow = 1.0
    X = rfft(x_segs * win, axis=-1)
    Y = rfft(y_segs * win, axis=-1)
    # cross-spectrum averaged across segments
    csd = np.mean(X * np.conj(Y), axis=-2) / (fs * win_pow)
    freqs = rfftfreq(seg_len, d=1.0 / fs)
    # double non-DC/Nyquist bins
    if seg_len % 2 == 0:
        if csd.shape[-1] > 2:
            csd[..., 1:-1] *= 2.0
    else:
        if csd.shape[-1] > 1:
            csd[..., 1:] *= 2.0
    return csd, freqs


def coherence(
    x: NDArray,
    y: NDArray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray, NDArray]:
    """Estimate magnitude-squared coherence between two signals.

    Parameters
    ----------
    x : NDArray
        Input time series array with shape (..., samples). Leading dimensions
        must broadcast with y; computation is performed along the last axis.
    y : NDArray
        Second input time series with shape (..., samples), broadcastable to x.
    fs : float
        Sampling frequency in Hz.
    nperseg : int or None, optional
        Segment length in samples for Welch-style averaging. If None the full
        signal is used as a single segment.
    noverlap : int or None, optional
        Number of samples to overlap between segments. If None, defaults to
        int(nperseg // 2) when nperseg is provided.
    window : str, optional
        Window name (default 'hann') applied to each segment prior to FFT.

    Returns
    -------
    coh : NDArray
        Magnitude-squared coherence with shape (..., n_freqs), whose values lie
        in the interval [0, 1].
    freqs : NDArray
        Frequencies (Hz) corresponding to the coherence estimates, length n_freqs.

    Notes
    -----
    - Coherence is defined as |S_xy|^2 / (S_xx * S_yy) where S_xy is the
      cross-spectrum and S_xx, S_yy are the auto-spectral densities (PSD).
    - The implementation reuses the same segmentation, windowing and scaling
      conventions as power_spectrum and cross_spectrum so that coherence is
      unitless and independent of the input signal units.
    - Numerical error can produce values slightly outside [0, 1]; the output
      is clipped to [0, 1] for numerical stability.
    """
    # Sxx, freqs = power_spectrum(x, ...)
    # Syy, _     = power_spectrum(y, ...)
    # Sxy, _     = cross_spectrum(x, y, ...)
    # coh = |Sxy|^2 / (Sxx * Syy); clip to [0, 1] for numerical safety
    # return (coh, freqs)
    Sxx, freqs = power_spectrum(x, fs, nperseg=nperseg, noverlap=noverlap, window=window, mode="density")
    Syy, _ = power_spectrum(y, fs, nperseg=nperseg, noverlap=noverlap, window=window, mode="density")
    Sxy, _ = cross_spectrum(x, y, fs, nperseg=nperseg, noverlap=noverlap, window=window)
    denom = Sxx * Syy
    # Avoid division by zero; where denom == 0 yield 0 coherence
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(Sxy) ** 2 / np.where(denom == 0, np.inf, denom)
    coh = np.clip(coh, 0.0, 1.0)
    return coh, freqs


def pairwise_coherence(
    x: NDArray,
    y: NDArray | None,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray, NDArray]:
    """Compute pairwise coherence between two collections of signals.

    Parameters
    ----------
    x : NDArray
        Array of shape (n_x, samples) containing n_x time series.
    y : NDArray or None
        Array of shape (n_y, samples) containing n_y time series, or None to
        compute pairwise coherences within x (i.e., y = x).
    fs : float
        Sampling frequency in Hz.
    nperseg : int or None, optional
        Segment length in samples for Welch-style averaging; see power_spectrum.
    noverlap : int or None, optional
        Number of samples to overlap between segments.
    window : str, optional
        Window name applied to each segment prior to FFT.

    Returns
    -------
    coh : NDArray
        Pairwise coherence with shape (n_x, n_y, n_freqs), values in [0, 1].
    freqs : NDArray
        Frequency axis (Hz), length n_freqs.

    Notes
    -----
    - If y is None, pairwise coherences among the rows of x are computed.
    - The function computes coherence by treating x as shape (n_x, 1, samples)
      and y as (1, n_y, samples) and then applying the coherence computation
      along the last axis; this produces a (n_x, n_y, n_freqs) result.
    - Uses the same scaling, segmentation, and clipping conventions as
      coherence() and power_spectrum().
    """
    # if y is None, y = x
    # coherence(x[:, None, :], y[None, :, :], ...) → (n_x, n_y, n_freqs)
    if y is None:
        y = x
    return coherence(x[:, None, :], y[None, :, :], fs, nperseg=nperseg, noverlap=noverlap, window=window)
