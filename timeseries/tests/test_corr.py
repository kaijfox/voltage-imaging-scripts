import numpy as np
import pytest

from imaging_scripts.timeseries.analysis import (
    cross_correlate,
    pairwise_cross_correlations,
)


def test_auto_peak_at_zero():
    rng = np.random.default_rng(0)
    S = 1000
    fs = 1000.0
    max_lag = 0.05
    x = rng.standard_normal(S)

    corr, lags = cross_correlate(x, x, fs, max_lag)
    # zero lag index should be at center (max_lag_samples)
    max_idx = int(np.argmax(corr))
    center_idx = len(lags) // 2
    assert max_idx == center_idx


def test_shifted_peak():
    rng = np.random.default_rng(1)
    S = 1000
    fs = 1000.0
    max_lag = 0.05
    k = 5  # samples shift
    x = rng.standard_normal(S)
    y = np.roll(x, k)

    corr, lags = cross_correlate(x, y, fs, max_lag)
    max_idx = int(np.argmax(corr))
    # expected lag is +k / fs
    assert pytest.approx(lags[max_idx], rel=1e-6, abs=1e-8) == -k / fs


def test_symmetry_pairwise():
    rng = np.random.default_rng(2)
    n = 4
    S = 500
    fs = 1000.0
    max_lag = 0.02

    x = rng.standard_normal((n, S))
    corr, lags = pairwise_cross_correlations(x, None, fs, max_lag)
    # corr shape (n, n, n_lags)
    assert corr.shape[0] == corr.shape[1] == n
    # symmetry: corr[i,j,t] == corr[j,i,-t] → equivalent to flipping last axis
    for i in range(n):
        for j in range(n):
            assert np.allclose(corr[i, j, :], np.flip(corr[j, i, :]))


def test_correlation_bounded_and_auto_peak_one():
    rng = np.random.default_rng(3)
    S = 1000
    fs = 1000.0
    max_lag = 0.05
    x = rng.standard_normal(S)

    corr, lags = cross_correlate(x, x, fs, max_lag, mode="correlation")
    # values must be within [-1, 1]
    assert np.all(corr <= 1.0 + 1e-12)
    assert np.all(corr >= -1.0 - 1e-12)
    # auto-correlation peak should be 1 at zero lag
    center_idx = len(lags) // 2
    assert pytest.approx(corr[center_idx], rel=1e-6) == 1.0


def test_constant_signal_zero():
    S = 400
    fs = 1000.0
    max_lag = 0.01
    x = np.ones(S)

    corr_cov, lags = cross_correlate(x, x, fs, max_lag, mode="covariance")
    corr_corr, _ = cross_correlate(x, x, fs, max_lag, mode="correlation")

    assert np.allclose(corr_cov, 0.0)
    assert np.allclose(corr_corr, 0.0)


def test_broadcasting_shape():
    rng = np.random.default_rng(4)
    n_x = 3
    n_y = 5
    S = 800
    fs = 1000.0
    max_lag = 0.02

    x = rng.standard_normal((n_x, S))
    y = rng.standard_normal((n_y, S))

    corr, lags = pairwise_cross_correlations(x, y, fs, max_lag)
    max_lag_samples = int(max_lag * fs)
    expected_nlags = 2 * max_lag_samples + 1
    assert corr.shape == (n_x, n_y, expected_nlags)
    assert lags.shape == (expected_nlags,)
