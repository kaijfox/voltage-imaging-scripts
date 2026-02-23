import numpy as np
import pytest
from numpy.fft import fft2, ifft2

from imaging_scripts.io.svd_video import SVDVideo
from imaging_scripts.io.motion_correction import (
    preprocess_basis_for_alignment,
    compute_basis_ffts,
    compute_reference_spectrum,
    estimate_shifts,
    compute_shifted_mean_image,
    roll_basis,
    expand_and_orthogonalize_1d,
    apply_shifts_lowrank,
    inpaint_basis,
    motion_correct_svd,
)


def _measure_shifts(video):
    """Helper: measure L1 shift norm using the same preprocessing as motion_correct_svd."""
    Vt_proc = preprocess_basis_for_alignment(np.asarray(video.Vt))
    Vhat = compute_basis_ffts(Vt_proc)
    W = np.asarray(video.U) * np.asarray(video.S)[None, :]
    R = compute_reference_spectrum(Vhat, np.mean(W, axis=0))
    shifts = estimate_shifts(W, Vhat, R)
    return int(np.sum(np.abs(shifts)))


# Fixtures
@pytest.fixture
def small_svd_video():
    """Small synthetic SVDVideo: rank=4, T=50, H=32, W=32 (numpy backend).

    Constructed from random bases (deterministic RNG) so tests are repeatable.
    """
    rng = np.random.RandomState(0)
    T, H, W, rank = 50, 32, 32, 4
    U = rng.normal(size=(T, rank))
    S = np.linspace(1.0, 0.5, rank)
    Vt = rng.normal(size=(rank, H, W))
    return SVDVideo(U, S, Vt, orthonormal=False)


@pytest.fixture
def shifted_svd_video():
    """Synthetic shifted video: each frame is an integer roll of a mean image.

    Returns (SVDVideo, shifts_array) where shifts_array is shape (T,2) of (dy,dx).
    """
    rng = np.random.RandomState(1)
    T, H, W, rank = 50, 32, 32, 4
    mean_img = np.zeros((H, W), dtype=float)
    mean_img[H // 2, W // 2] = 1.0

    frames = np.zeros((T, H, W), dtype=float)
    shifts = []
    for i in range(T):
        dy = (i % 7) - 3
        dx = ((i // 7) % 7) - 3
        shifts.append((dy, dx))
        frames[i] = np.roll(mean_img, shift=(dy, dx), axis=(0, 1))

    flat = frames.reshape(T, -1)
    U, s, Vh = np.linalg.svd(flat, full_matrices=False)
    r = min(rank, U.shape[1], Vh.shape[0])
    U_r = U[:, :r]
    s_r = s[:r]
    Vt_r = Vh[:r].reshape(r, H, W)
    return SVDVideo(U_r, s_r, Vt_r, orthonormal=False), np.array(shifts, dtype=int)


# Tests per-function, one TestClass each as requested
class TestPreprocessBasisForAlignment:
    """Preprocess spatial bases for alignment.

    Main: Voltage-mode highpass + unsharp should reduce low-freq (mean closer to zero)
    Edge cases: mode variations, identity/no-op cases
    """

    def test_output_shape_and_highpass(self, small_svd_video):
        Vt = np.asarray(small_svd_video.Vt)
        Vt_proc = preprocess_basis_for_alignment(Vt, mode="voltage")
        assert Vt_proc.shape == Vt.shape
        # high-pass should reduce the mean magnitude
        assert np.abs(np.mean(Vt_proc)) <= np.abs(np.mean(Vt)) + 1e-12
        # and not be identical to the input
        assert not np.allclose(Vt_proc, Vt)


class TestComputeBasisFFTs:
    """Compute 2D FFTs of spatial basis vectors.

    Main: output is complex with same shape; round-trip ifft should recover input.
    Edge cases: constant image -> delta at DC.
    """

    def test_fft_shape_and_roundtrip(self, small_svd_video):
        Vt = np.asarray(small_svd_video.Vt)
        Vhat = compute_basis_ffts(Vt)
        assert Vhat.shape == Vt.shape
        assert np.iscomplexobj(Vhat)
        Vt_rec = np.stack([np.real(ifft2(Vhat[j])) for j in range(Vhat.shape[0])])
        assert np.allclose(Vt_rec, Vt, atol=1e-8)

    def test_constant_image_delta_dc(self):
        H, W = 8, 8
        Vt_const = np.ones((1, H, W), dtype=float)
        Vhat = compute_basis_ffts(Vt_const)
        # only DC should be non-zero for a perfectly constant image
        nonzero = np.count_nonzero(np.abs(Vhat[0]) > 1e-12)
        assert nonzero == 1


class TestComputeReferenceSpectrum:
    """Reference spectrum construction from basis FFTs and weights.

    Main: R = sum_m w_ref[m] * conj(Vhat[m]); unit selection & linearity.
    Edge cases: zero weights, negative/complex weights.
    """

    def test_unit_vector_and_linearity(self):
        rank, H, W = 3, 8, 8
        rng = np.random.RandomState(2)
        Vhat = rng.normal(size=(rank, H, W)) + 1j * rng.normal(size=(rank, H, W))
        w_ref = np.zeros(rank)
        w_ref[1] = 1.0
        R = compute_reference_spectrum(Vhat, w_ref)
        assert np.allclose(R, np.conj(Vhat[1]))
        # linearity
        w_lin = 2.0 * np.array([1.0, 0.0, 0.0]) + 3.0 * np.array([0.0, 1.0, 0.0])
        R_lin = compute_reference_spectrum(Vhat, w_lin)
        assert np.allclose(R_lin, 2.0 * np.conj(Vhat[0]) + 3.0 * np.conj(Vhat[1]))


class TestEstimateShifts:
    """Estimate integer shifts using latent-space phase correlation.

    Main: zero shifts for self-alignment; known shift recovered for a shifted frame;
    Edge cases: clipping via max_shift.
    """

    def test_self_alignment_zero_shifts(self):
        H, W = 32, 32
        img = np.zeros((H, W), dtype=float)
        img[H // 2, W // 2] = 1.0
        Vhat = np.stack([fft2(img)])
        W = np.ones((10, 1), dtype=float)
        R = np.conj(fft2(img))
        shifts = estimate_shifts(W, Vhat, R)
        assert shifts.shape == (10, 2)
        assert np.all(shifts == 0)

    def test_known_shift_recovered(self):
        H, W = 32, 32
        img = np.zeros((H, W), dtype=float)
        img[H // 2, W // 2] = 1.0
        dy, dx = 3, -2
        img_shifted = np.roll(img, shift=(dy, dx), axis=(0, 1))
        Vhat = np.stack([fft2(img_shifted)])
        W = np.ones((1, 1), dtype=float)
        R = np.conj(fft2(img))
        shifts = estimate_shifts(W, Vhat, R)
        assert shifts.shape == (1, 2)
        assert int(shifts[0, 0]) == dy and int(shifts[0, 1]) == dx

    def test_max_shift_clipping(self):
        H, W = 32, 32
        img = np.zeros((H, W), dtype=float)
        img[H // 2, W // 2] = 1.0
        dy, dx = 10, 0
        img_shifted = np.roll(img, shift=(dy, dx), axis=(0, 1))
        Vhat = np.stack([fft2(img_shifted)])
        W = np.ones((1, 1), dtype=float)
        R = np.conj(fft2(img))
        shifts = estimate_shifts(W, Vhat, R, max_shift=5)
        # when true shift exceeds max_shift the search should clip and return zero
        assert shifts.shape == (1, 2)
        assert int(shifts[0, 0]) == 0 and int(shifts[0, 1]) == 0


class TestComputeShiftedMeanImage:
    """Compute mean image after applying per-frame integer shifts.

    Main: zero shifts => original mean; uniform shift => rolled mean.
    Edge cases: many unique shifts, weighting correctness.
    """

    def test_zero_shifts_returns_mean(self, small_svd_video):
        video = small_svd_video
        T = video.U.shape[0]
        W = np.asarray(video.U) * np.asarray(video.S)[None, :]
        Vt_flat = np.asarray(video.Vt).reshape(video.rank, -1)
        frames_flat = W.dot(Vt_flat)
        mean_img = frames_flat.mean(axis=0).reshape(video.Vt.shape[1], video.Vt.shape[2])
        shifts = np.zeros((T, 2), dtype=int)
        mean_shifted = compute_shifted_mean_image(video, shifts)
        assert mean_shifted.shape == mean_img.shape
        assert np.allclose(mean_shifted, mean_img)

    def test_uniform_shift(self, small_svd_video):
        video = small_svd_video
        T = video.U.shape[0]
        shifts = np.tile(np.array([2, -1], dtype=int), (T, 1))
        mean_shifted = compute_shifted_mean_image(video, shifts)
        W = np.asarray(video.U) * np.asarray(video.S)[None, :]
        Vt_flat = np.asarray(video.Vt).reshape(video.rank, -1)
        frames_flat = W.dot(Vt_flat)
        mean_img = frames_flat.mean(axis=0).reshape(video.Vt.shape[1], video.Vt.shape[2])
        mean_roll = np.roll(mean_img, shift=(2, -1), axis=(0, 1))
        assert np.allclose(mean_shifted, mean_roll)


class TestRollBasis:
    """Shift spatial bases by integer pixels.

    Main: zero shift = identity; basic shift equals np.roll.
    """

    def test_zero_shift_identity(self, small_svd_video):
        Vt = np.asarray(small_svd_video.Vt)
        assert np.allclose(roll_basis(Vt, 0, 0), Vt)

    def test_dy1_dx0(self, small_svd_video):
        Vt = np.asarray(small_svd_video.Vt)
        rolled = roll_basis(Vt, 1, 0)
        assert np.allclose(rolled, np.roll(Vt, shift=(1, 0), axis=(1, 2)))


class TestExpandAndOrthogonalize1D:
    """Expand SVD basis for 1D integer shifts and orthogonalize.

    Main: zero shifts round-trip exactly; mixed shifts => orthonormal U and
    reconstruction matches naive per-frame shifted reconstruction.
    """

    def test_zero_shifts_roundtrip(self, small_svd_video):
        video = small_svd_video
        T = video.U.shape[0]
        shifts_1d = np.zeros(T, dtype=int)
        out = expand_and_orthogonalize_1d(video, shifts_1d, axis=0)
        # Reconstruct frames from out and compare to original frames
        W_new = np.asarray(out.U) * np.asarray(out.S)[None, :]
        Vt_new_flat = np.asarray(out.Vt).reshape(out.rank, -1)
        frames_new = W_new.dot(Vt_new_flat)

        W_orig = np.asarray(video.U) * np.asarray(video.S)[None, :]
        Vt_orig_flat = np.asarray(video.Vt).reshape(video.rank, -1)
        frames_orig = W_orig.dot(Vt_orig_flat)
        assert np.allclose(frames_new, frames_orig, atol=1e-10)

    def test_mixed_shifts_properties(self, shifted_svd_video):
        video, shifts = shifted_svd_video
        shifts_1d = shifts[:, 0]  # vertical shifts
        out = expand_and_orthogonalize_1d(video, shifts_1d, axis=0)
        # U should be (approximately) orthonormal
        U = np.asarray(out.U)
        assert np.allclose(U.T.dot(U), np.eye(out.rank), atol=1e-8)
        # Rank bound
        n_unique = np.unique(shifts_1d).size
        assert out.rank <= video.rank * n_unique
        # Naive per-frame apply: compare reconstructions
        W = np.asarray(video.U) * np.asarray(video.S)[None, :]
        H, Wpix = video.Vt.shape[1], video.Vt.shape[2]
        frames_naive = np.zeros((video.U.shape[0], H * Wpix))
        for i in range(video.U.shape[0]):
            s = shifts_1d[i]
            Vt_rolled = np.roll(np.asarray(video.Vt), shift=(s, 0), axis=(1, 2))
            frames_naive[i] = W[i].dot(Vt_rolled.reshape(video.rank, -1))
        W_new = np.asarray(out.U) * np.asarray(out.S)[None, :]
        frames_out = W_new.dot(np.asarray(out.Vt).reshape(out.rank, -1))
        assert np.allclose(frames_out, frames_naive, atol=1e-8)


class TestApplyShiftsLowrank:
    """Apply 2D integer shifts in low-rank form via sequential 1D expansion.

    Main: zero shifts identity; mixed 2D shifts match naive application.
    """

    def test_zero_shifts_identity(self, small_svd_video):
        video = small_svd_video
        T = video.U.shape[0]
        shifts = np.zeros((T, 2), dtype=int)
        out = apply_shifts_lowrank(video, shifts)
        # reconstruct and compare
        W_new = np.asarray(out.U) * np.asarray(out.S)[None, :]
        frames_new = W_new.dot(np.asarray(out.Vt).reshape(out.rank, -1))
        W_orig = np.asarray(video.U) * np.asarray(video.S)[None, :]
        frames_orig = W_orig.dot(np.asarray(video.Vt).reshape(video.rank, -1))
        assert np.allclose(frames_new, frames_orig, atol=1e-10)

    def test_mixed_2d_shifts_match_naive(self, shifted_svd_video):
        video, shifts = shifted_svd_video
        out = apply_shifts_lowrank(video, shifts)
        # Naive per-frame reconstruction
        W = np.asarray(video.U) * np.asarray(video.S)[None, :]
        H, Wpix = video.Vt.shape[1], video.Vt.shape[2]
        frames_naive = np.zeros((video.U.shape[0], H * Wpix))
        for i in range(video.U.shape[0]):
            dy, dx = shifts[i]
            Vt_rolled = np.roll(np.asarray(video.Vt), shift=(dy, dx), axis=(1, 2))
            frames_naive[i] = W[i].dot(Vt_rolled.reshape(video.rank, -1))
        W_new = np.asarray(out.U) * np.asarray(out.S)[None, :]
        frames_out = W_new.dot(np.asarray(out.Vt).reshape(out.rank, -1))
        assert np.allclose(frames_out, frames_naive, atol=1e-8)
        assert getattr(out, "orthonormal", True) is True


class TestInpaintBasis:
    """Inpaint masked pixels using cv2.inpaint when available.

    Main: no-mask is identity; masked pixels are changed while unmasked remain.
    """

    def test_inpaint_behavior(self):
        cv2 = pytest.importorskip("cv2")
        rng = np.random.RandomState(3)
        rank, H, W = 3, 16, 16
        Vt = rng.normal(size=(rank, H, W)).astype(np.float32)
        mask = np.zeros((H, W), dtype=np.uint8)
        # single masked pixel in center
        mask[H // 2, W // 2] = 1
        out = inpaint_basis(Vt.copy(), mask, method="ns")
        # no-mask case -> identical
        out_nomask = inpaint_basis(Vt.copy(), np.zeros_like(mask), method="ns")
        assert np.allclose(out_nomask, Vt)
        # masked pixel changed for at least one basis vector
        center_idx = (H // 2, W // 2)
        changed = any(not np.isclose(out[j, center_idx[0], center_idx[1]], Vt[j, center_idx[0], center_idx[1]]) for j in range(rank))
        assert changed
        # unmasked pixel unchanged
        y, x = 0, 0
        assert np.allclose(out[:, y, x], Vt[:, y, x])


class TestMotionCorrectSVD:
    """Integration-level checks for the full motion correction pipeline.

    Main: motion_correct_svd should reduce apparent frame-to-reference shifts
    after 1 and 2 passes for a synthetically shifted video.
    """

    def test_motion_correct_reduces_shifts(self, shifted_svd_video):
        video, shifts = shifted_svd_video
        # Measure shifts using the same preprocessing pipeline as motion_correct_svd
        orig_norm = _measure_shifts(video)

        corrected1 = motion_correct_svd(video, n_passes=1)
        norm1 = _measure_shifts(corrected1)

        corrected2 = motion_correct_svd(video, n_passes=2)
        norm2 = _measure_shifts(corrected2)

        # Both passes should reduce apparent shift magnitude (or be equal)
        assert norm1 <= orig_norm + 1e-8
        assert norm2 <= norm1 + 1e-8
