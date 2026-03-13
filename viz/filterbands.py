from ..timeseries.types import Traces
from ..timeseries.analysis import pairwise_coherence, power_spectrum
from ..viz.qc import plot_distribution
from mplutil import util as vu
import numpy as np


def plot_coherence_overlay(
    # (n_traces, n_freq)
    coherence: np.ndarray,
    # (n_freq,)
    freqs: np.ndarray,
):
    """Plot distribution of coherence values across unique signal pairs.

    Parameters
    ----------
    coherence : np.ndarray (n_signals, n_signals, n_freq)
        Full pairwise coherence matrix where ``coherence[i, j, :]`` is the
        coherence between signals i and j across frequencies.
    freqs : np.ndarray
        1-D array of frequency values (Hz) corresponding to the coherence axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes object with the plotted coherence distribution.
    """

    # No duplicate- or self-comparisons
    # (n_comparisons, n_freq)
    uniq = coherence[*np.tril_indices(len(coherence), k=-1)]

    fig, ax = vu.subplots((3, 7), (1, 1))
    plot_distribution(ax, freqs, uniq, mean_color="k")
    ax.set_xscale("log")
    vu.label(ax, "Frequency (Hz)", "Coherence")
    vu.legend(ax)
    return fig, ax


def plot_filtered_spectra(despiked: Traces, filtered: Traces, fs: float):
    """Plot mean power spectra for despiked and filtered traces.

    Parameters
    ----------
    despiked : Traces
        Traces object containing the original (despiked) traces. Must provide
        a ``.data`` attribute of shape (n_traces, n_samples).
    filtered : Traces
        Traces object containing the filtered traces with the same shape as
        ``despiked``.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the power spectral density comparison.
    ax : matplotlib.axes.Axes
        Axes with plotted mean PSDs for despiked (black) and filtered (red).

    Notes
    -----
    Power spectral densities are computed via ``power_spectrum`` with
    ``nperseg=2*fs``. The function plots the mean PSD across traces and sets
    the frequency axis to log scale.
    """
    psd_full, freqs = power_spectrum(despiked.data, fs, nperseg=2 * fs)
    psd_filt, _ = power_spectrum(filtered.data, fs, nperseg=2 * fs)

    fig, ax = vu.subplots((2, 3), (1, 1))

    ax.plot(freqs, psd_full.mean(axis=0), "k-", lw=0.5)
    ax.plot(freqs, psd_filt.mean(axis=0), "r-", lw=0.5)
    ax.set_xscale("log")
    vu.label(ax, "Frequency (Hz)", "Power")
    return fig, ax


def robust_coherence(
    data: np.ndarray, fs: float, nperseg: int, noise_scale=0.1, rng=None
):
    """Compute pairwise coherence after adding small additive noise for stability.

    Adds Gaussian noise scaled by each signal's standard deviation and
    ``noise_scale`` to reduce numerical degeneracies when estimating coherence.
    The noised data are passed to :func:`pairwise_coherence`.

    Parameters
    ----------
    data : np.ndarray
        Time-series array where the last axis is time. Typical shape is
        (n_signals, n_samples) but other leading dimensions are accepted as
        long as the last axis represents time.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Number of samples per segment passed to the coherence estimator.
    noise_scale : float, optional
        Relative scale of additive Gaussian noise (default 0.1). Noise for each
        point is sampled as data[i, j] = eps[i, j] * data_std[i] * noise_scale
        where eps are independent standard normal.

    Returns
    -------
    coherence : np.ndarray
        The pairwise coherence array returned by :func:`pairwise_coherence`.
        For square inputs this is typically shaped (n_signals, n_signals, n_freq).

    Notes
    -----
    Because a small random perturbation is added, results will vary across
    runs unless a random seed is set externally (e.g., via ``np.random.seed``).
    """
    rng = np.random.default_rng(rng)
    data_std = data.std(axis=-1, keepdims=True)
    noise = rng.normal(size=data.shape) * data_std * noise_scale
    return pairwise_coherence(data + noise, None, fs=fs, nperseg=nperseg)
