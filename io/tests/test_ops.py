import numpy as np
from scipy.signal import savgol_filter

from imaging_scripts.io.svd_video import SVDVideo
from imaging_scripts.io.ops import subtractive_lowpass


def make_video(U, S, Vt):
    return SVDVideo(
        U.astype(np.float64), np.asarray(S, dtype=np.float64), Vt.astype(np.float64)
    )


def interior_slice(T, wl):
    if T <= 2 * wl:
        return slice(0, 0)  # empty
    return slice(wl, T - wl)


def test_sub_dff_constant_input():
    # T1: Constant temporal -> interior nearly zero after subtraction
    rng = np.random.RandomState(0)
    T = 200
    r = 2
    nx, ny = 10, 10

    # U: identical rows (DC)
    base = rng.randn(r)
    U = np.tile(base[None, :], (T, 1))

    S = np.ones(r)
    Vt = rng.randn(r, nx, ny)

    video = make_video(U, S, Vt)

    wl = 11
    po = 3

    out = subtractive_lowpass(video, wl, po, reorthogonalize=True)

    # Reconstruct full pixel videos and check interior frames ~ 0
    frames_in = video[:]
    frames_out = out[:]

    Tslice = interior_slice(T, wl)
    if Tslice.start < Tslice.stop:
        # Check that interior frames are near zero
        interior_norm = np.max(np.abs(frames_out[Tslice, :]))
        assert interior_norm < 1e-6


def test_sub_dff_hf_preserved():
    # T2: Alternating ±1 rows preserved with reorthogonalize=False
    rng = np.random.RandomState(1)
    T = 300
    r = 1
    nx, ny = 8, 8

    U = (np.arange(T) % 2 * 2 - 1).astype(np.float64).reshape(T, 1)
    S = np.array([2.0])
    Vt = rng.randn(r, nx, ny)

    video = make_video(U, S, Vt)

    wl = 101  # long window
    po = 0

    out = subtractive_lowpass(video, wl, po, reorthogonalize=False)

    Tslice = interior_slice(T, wl)
    if Tslice.start < Tslice.stop:
        # Expect U_out approx U_in in interior
        diff = np.abs(out.U[Tslice, :] - U[Tslice, :])
        assert np.all(diff < (1/wl) + 1e-6)


def test_sub_dff_orthonormal():
    # T3: Random rank-4 -> orthonormal=True and U.T @ U ≈ I
    rng = np.random.RandomState(2)
    T = 250
    r = 4
    nx, ny = 6, 6

    U = rng.randn(T, r)
    # make U orthonormal-ish by QR
    Q, _ = np.linalg.qr(U)
    U = Q[:, :r]
    S = np.linspace(5, 1, r)
    Vt = rng.randn(r, nx, ny)

    video = make_video(U, S, Vt)

    wl = 15
    po = 3

    out = subtractive_lowpass(video, wl, po, reorthogonalize=True)

    assert out.orthonormal is True
    # Check U^T U ≈ I
    I = np.eye(r)
    prod = out.U.T @ out.U
    assert np.allclose(prod, I, atol=1e-6)


def test_sub_dff_match_scipy():
    # T4: For small video, compare reconstruct vs scipy savgol on full pixels
    rng = np.random.RandomState(3)
    T = 120
    r = 3
    nx, ny = 5, 4

    # structured input video
    frames_gen = np.random.randn(T, nx * ny)
    frames_gen = 0.2 * np.cumsum(frames_gen, axis=0) + frames_gen
    U_gen, S_gen, Vt_gen = np.linalg.svd(frames_gen, full_matrices=False)

    video_gen = make_video(U_gen[:, :r], S_gen[:r], Vt_gen[:r, :].reshape((r, nx, ny)))

    U, S, Vt = video_gen.U, video_gen.S, video_gen.Vt
    video = make_video(U, S, Vt)

    wl = 11
    po = 3

    out = subtractive_lowpass(video, wl, po, reorthogonalize=True)

    # Reconstruct full pixel video
    frames_in = video[:]
    # scipy savgol along time (axis=0) applied to full pixels
    frames_lpf = savgol_filter(frames_in, wl, po, axis=0)
    frames_expected = frames_in - frames_lpf

    frames_out = out[:]
    Tslice = interior_slice(T, wl)
    if Tslice.start < Tslice.stop:
        maxdiff = np.max(np.abs(frames_out[Tslice, :] - frames_expected[Tslice, :]))
        assert maxdiff < 1e-6
