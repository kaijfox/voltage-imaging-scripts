"""SVD-domain motion correction via latent-space FFT phase correlation."""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from typing import Tuple

from .svd_video import SVDVideo
from ..cli.common import configure_logging

logger, (error, warning, info, debug) = configure_logging("motion")


def preprocess_basis_for_alignment(Vt: np.ndarray, hpf_px = 50., sharp_px=1., amount=100.) -> np.ndarray:
    """
    Apply spatial preprocessing to each basis vector for shift estimation.

    spatial highpass (subtract Gaussian sigma=50) + unsharp filter

    Parameters
    ----------
    Vt : array (rank, H, W)

    Returns
    -------
    array (rank, H, W)
    """
    Vt = np.asarray(Vt)
    highpass = Vt - gaussian_filter(Vt, sigma=hpf_px, axes=(1, 2))
    sharpen_wide  = gaussian_filter(highpass, sigma=2*sharp_px, axes=(1, 2))
    sharpen_small = gaussian_filter(highpass, sigma=  sharp_px, axes=(1, 2))
    sharpened = highpass + amount * (sharpen_wide - sharpen_small)
    return sharpened


def compute_basis_ffts(Vt: np.ndarray) -> np.ndarray:
    """
    2D FFT of each spatial basis vector.

    Parameters
    ----------
    Vt : array (rank, H, W) real

    Returns
    -------
    array (rank, H, W) complex
    """
    Vt = np.asarray(Vt)
    return np.stack([fft2(Vt[j]) for j in range(Vt.shape[0])])


def compute_reference_spectrum(Vhat: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
    """
    Conjugate reference spectrum: R = sum_m w_ref[m] * conj(Vhat[m]).

    Parameters
    ----------
    Vhat : array (rank, H, W) complex
    w_ref : array (rank,) real
        Reference temporal weights, e.g. w_bar[j] = mean_t(U[:,j] * S[j]).

    Returns
    -------
    array (H, W) complex
    """
    Vhat = np.asarray(Vhat)
    w_ref = np.asarray(w_ref)
    # sum_m w_ref[m] * conj(Vhat[m])
    return np.sum(w_ref[:, None, None] * np.conj(Vhat), axis=0)


def estimate_shifts(
    codes: np.ndarray,
    Vhat: np.ndarray,
    R: np.ndarray,
    max_shift: int | None = None,
    batch_limit: int = 4000 * 4000,
) -> np.ndarray:
    """
    Per-frame integer shifts via latent-space phase correlation.

    Parameters
    ----------
    codes : array (T, rank)
        Temporal weights per frame: U * S[None, :].
    Vhat : array (rank, H, W) complex
        Preprocessed basis FFTs.
    R : array (H, W) complex
        Conjugate reference spectrum (from compute_reference_spectrum or
        conj(fft2(preprocessed_pixel_reference))).
    max_shift : int or None
        Clip cross-correlation search window to ±max_shift pixels.
    batch_limit : int
        Maximum number of complex elements in the cross-spectrum batch
        (T_batch, H, W). batch_size is set so the batch stays within
        this limit; minimum batch_size is 1.

    Returns
    -------
    array (T, 2) int
        Per-frame (dy, dx) integer pixel shifts.
    """
    rank = Vhat.shape[0]
    H = Vhat.shape[1]
    Wpix = Vhat.shape[2]
    T = codes.shape[0]
    HW = H * Wpix
    center = (H // 2, Wpix // 2)

    batch_size = max(1, batch_limit // HW)

    Vhat_flat = Vhat.reshape(rank, -1)  # (rank, H*W)
    shifts_out = np.empty((T, 2), dtype=int)

    Vhat_R = Vhat_flat * R.reshape(1, -1) # (rank, H*W)
    space_basis = ifft2(Vhat_R.reshape(rank, H, Wpix)).reshape(rank, -1)
    for start in range(0, T, batch_size):
        import time
        info(f"Batch: {start} / {T}, {time.time()}")
        t0 = time.time()
        end = min(start + batch_size, T)
        W_batch = codes[start:end]  # (B, rank)
        # (B, H, W)
        info(f"Step 0: {time.time() - t0}")
        info(f"{W_batch.shape} {Vhat_flat.shape} {R.shape}")
        cc = W_batch.dot(space_basis.reshape(end-start, H * Wpix)).reshape(end - start, H, Wpix)
        info(f"Step 1: {time.time() - t0}")
        info(f"Step 2: {time.time() - t0}")
        cc_shifted = fftshift(np.abs(cc), axes=(-2, -1))
        info(f"Step 3: {time.time() - t0}")
        flat = cc_shifted.reshape(end - start, -1)
        idx = np.argmax(flat, axis=1)
        peak_rows, peak_cols = np.unravel_index(idx, (H, Wpix))
        shifts_out[start:end, 0] = peak_rows - center[0]
        shifts_out[start:end, 1] = peak_cols - center[1]

    if max_shift is not None:
        # Zero out shifts whose magnitude exceeds max_shift (unreliable / too large)
        magnitude = np.sqrt(shifts_out[:, 0] ** 2 + shifts_out[:, 1] ** 2)
        shifts_out[magnitude > max_shift] = 0
    return shifts_out


def compute_shifted_mean_image(video: SVDVideo, shifts: np.ndarray) -> np.ndarray:
    """
    Mean pixel image of SVDVideo after applying integer shifts, grouped efficiently.

    Parameters
    ----------
    video : SVDVideo
        Vt must be (rank, H, W).
    shifts : array (T, 2) int
        Per-frame (dy, dx) integer shifts.

    Returns
    -------
    array (H, W)
        Mean image of the shifted video.
    """
    Wmat = np.asarray(video.U) * np.asarray(video.S)[None, :]
    T = Wmat.shape[0]
    rank = video.rank
    H = video.Vt.shape[1]
    Wpix = video.Vt.shape[2]
    Vt_arr = np.asarray(video.Vt)
    unique_shifts, inverse = np.unique(shifts, axis=0, return_inverse=True)
    mean_flat = np.zeros(H * Wpix, dtype=float)
    for u_idx, s in enumerate(unique_shifts):
        frames_idx = np.where(inverse == u_idx)[0]
        if frames_idx.size == 0:
            continue
        mean_W_u = np.mean(Wmat[frames_idx], axis=0) * (frames_idx.size / T)
        rolled = roll_basis(Vt_arr, int(s[0]), int(s[1]))
        mean_flat += mean_W_u.dot(rolled.reshape(rank, -1))
    return mean_flat.reshape(H, Wpix)


def roll_basis(Vt: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Shift each spatial basis vector by (dy, dx) pixels.

    Parameters
    ----------
    Vt : array (rank, H, W)
    dy : int
        Row shift (positive = down, wraps).
    dx : int
        Column shift (positive = right, wraps).

    Returns
    -------
    array (rank, H, W)
    """
    return np.roll(Vt, (dy, dx), axis=(1, 2))


def expand_and_orthogonalize_1d(
    video: SVDVideo,
    shifts_1d: np.ndarray,
    axis: int,
    max_rank=None,
) -> SVDVideo:
    """
    Apply 1D integer shifts along one axis by expanding the SVD basis, then orthogonalize.

    Parameters
    ----------
    video : SVDVideo
        Vt must be (rank, H, W).
    shifts_1d : array (T,) int
        Per-frame shift along the specified axis.
    axis : 'row' or 'col'
        'row' for rows (vertical), 'col' for columns (horizontal).

    Returns
    -------
    SVDVideo
        Orthogonalized; rank up to original_rank * n_unique_shifts.
    """
    from scipy.linalg import qr, svd as scipy_svd

    Vt_arr = np.asarray(video.Vt)
    U_arr = np.asarray(video.U)
    S_arr = np.asarray(video.S)
    T = U_arr.shape[0]
    r = video.rank
    H, W = Vt_arr.shape[1], Vt_arr.shape[2]
    unique_s, inverse = np.unique(shifts_1d, return_inverse=True)
    K = unique_s.size
    # If all shifts are zero (single unique shift zero), return original
    if K == 1 and int(unique_s[0]) == 0:
        return SVDVideo(
            U_arr.copy(), S_arr.copy(), Vt_arr.copy(), orthonormal=video.orthonormal
        )

    # Build expanded Vt by rolling for each unique shift
    Vt_blocks = []
    for s in unique_s:
        dy = int(s) if axis == 'row' else 0
        dx = int(s) if axis == 'col' else 0
        Vt_blocks.append(roll_basis(Vt_arr, dy, dx))
    Vt_expanded = np.concatenate(Vt_blocks, axis=0).astype(np.float64)  # (r*K, H, W)
    Vt_flat = Vt_expanded.reshape(r * K, -1)  # (r*K, H*W)

    # Build expanded U (block-sparse: frame i contributes to group k only)
    U_expanded = np.zeros((T, r * K), dtype=np.float64)
    for k in range(K):
        mask_k = (inverse == k)[:, None]
        U_expanded[:, k * r : (k + 1) * r] = U_arr * mask_k
    S_expanded = np.tile(S_arr, K)

    # QR-SVD: numerically stable orthogonalization via double QR
    # M = U_expanded @ diag(S_expanded) @ Vt_flat = A @ Vt_flat
    A = U_expanded * S_expanded[None, :]  # (T, r*K), absorb S into U
    Q_A, R_A = qr(A, mode="economic")  # Q_A (T, r*K), R_A (r*K, r*K)
    C = R_A @ Vt_flat  # (r*K, H*W)
    Q_B, R_B = qr(C.T, mode="economic")  # Q_B (H*W, r*K), R_B (r*K, r*K)
    
    # Limit to max_rank
    if max_rank is not None:
        rA, rB = np.linalg.matrix_rank(R_A), np.linalg.matrix_rank(R_B)
        new_rank = min(max_rank, min(rA, rB))
        Q_A, R_A = Q_A[:, :new_rank], R_A[:new_rank]
        Q_B, R_B = Q_B[:, :new_rank], R_B[:new_rank]
    new_rank = R_B.shape[0]
    
    # M = Q_A @ R_B.T @ Q_B.T  →  SVD of R_B'
    U_small, Sigma, Vt_small = scipy_svd(R_B.T, full_matrices=False)
    U_new = Q_A @ U_small  # (T, new_r) orthonormal
    Vt_flat_new = Vt_small @ Q_B.T  # (new_r, H*W) orthonormal
    Vt_new = Vt_flat_new.reshape(new_rank, H, W)
    
    # Drop near-zero singular value components
    thresh = Sigma[0] * np.finfo(np.float64).eps * max(T, new_rank)
    keep = Sigma > thresh
    return SVDVideo(U_new[:, keep], Sigma[keep], Vt_new[keep], orthonormal=True)


def apply_shifts_lowrank(video: SVDVideo, shifts: np.ndarray) -> SVDVideo:
    """
    Apply 2D integer shifts to SVDVideo.

    Parameters
    ----------
    video : SVDVideo
        Vt must be (rank, H, W).
    shifts : array (T, 2) int
        Per-frame (dy, dx) shifts.

    Returns
    -------
    SVDVideo
        Orthogonalized.
    """
    result = expand_and_orthogonalize_1d(video, shifts[:, 1], axis='col')
    result = expand_and_orthogonalize_1d(result, shifts[:, 0], axis='row')
    return result


def inpaint_basis(
    Vt: np.ndarray,
    mask: np.ndarray,
    method: str = "ns",
) -> np.ndarray:
    """
    Inpaint masked pixels in each spatial basis vector using cv2.inpaint.

    Parameters
    ----------
    Vt : array (rank, H, W)
    mask : array (H, W) bool or uint8
        True/1 where pixels should be inpainted.
    method : str
        'ns' (Navier-Stokes) or 'telea'.

    Returns
    -------
    array (rank, H, W)
    """
    import cv2

    cv2_flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    mask_u8 = mask.astype(np.uint8)
    # If no masked pixels, return original unchanged
    if not mask_u8.any():
        return Vt
    Vt_arr = np.asarray(Vt)
    out = np.empty_like(Vt_arr, dtype=Vt_arr.dtype)
    for j in range(Vt_arr.shape[0]):
        v = Vt_arr[j]
        minv = float(np.min(v))
        maxv = float(np.max(v))
        if maxv == minv:
            out[j] = v.copy()
            continue
        v_norm = (v - minv) / (maxv - minv)
        v_in = v_norm.astype(np.float32)
        inpainted = cv2.inpaint(v_in, mask_u8, 3, cv2_flag)
        out[j] = (inpainted.astype(np.float64) * (maxv - minv) + minv).astype(
            Vt_arr.dtype
        )
    return out


def motion_correct_svd(
    video: SVDVideo,
    max_shift: int | None = None,
    n_passes: int = 2,
    mask: np.ndarray | None = None,
    hpf_px: float = 10.,
    sharp_px: float = 1.,
    sharp_amount: float = 100.,
    max_rank: int | None = None,
) -> Tuple[SVDVideo, np.ndarray]:
    """
    Full two-pass SVD motion correction pipeline.

    Parameters
    ----------
    video : SVDVideo
        Vt must be (rank, H, W).
    mode : str
        Preprocessing mode for alignment ('voltage').
    max_shift : int or None
        Maximum shift in pixels; clips cross-correlation search window.
    n_passes : int
        Number of alignment passes (1 or 2).
    mask : array (H, W) bool or None
        Occluding-region mask; inpainted before alignment if provided.

    Returns
    -------
    SVDVideo
        Orthogonalized motion-corrected video.
    """
    vid = video

    # Optionally truncate rank
    if max_rank is not None:
        info(f"Truncating to rank {max_rank}.")
        vid = SVDVideo(
            vid.U[:, :max_rank],
            vid.S[:max_rank],
            vid.Vt[:max_rank],
            orthonormal=vid.orthonormal,
        )

    # Optionally inpaint occluding mask first
    if mask is not None:
        info(f"Inpainting {mask.sum()} pixels.")
        Vt_inp = inpaint_basis(np.asarray(video.Vt), mask)
        vid = SVDVideo(
            np.asarray(video.U).copy(),
            np.asarray(video.S).copy(),
            Vt_inp,
            orthonormal=False,
        )
        

    # Preprocess bases and compute FFTs
    info("Computing spatial basis.")
    preproc_kw = dict(hpf_px=hpf_px, sharp_px=sharp_px, amount=sharp_amount)
    Vt_hp = preprocess_basis_for_alignment(vid.Vt, **preproc_kw)
    Vhat = compute_basis_ffts(Vt_hp)
    frame_codes = np.asarray(vid.U) * np.asarray(vid.S)[None, :]

    # Always first pass: uncorrected temporal mean as reference image
    code_ref = np.mean(frame_codes, axis=0)
    R = compute_reference_spectrum(Vhat, code_ref)
    info("Estimating shifts.")
    shifts = estimate_shifts(frame_codes, Vhat, R, max_shift)
    all_shifts = [shifts]

    mean_shift = shifts.mean(axis=0)
    info(f"Avg.: row={mean_shift[0]:.2f}, col={mean_shift[1]:.2f}")

    # Multi-pass motion correction:
    # Recompute reference from shifted mean image
    for i in range(n_passes - 1):
        info(f"Multi-pass iteration {i + 2}:")
        all_shifts.append(shifts)
        mean_img = compute_shifted_mean_image(vid, shifts[-1])
        mean_proc = preprocess_basis_for_alignment(mean_img[None, :, :])[0]
        R = np.conj(fft2(mean_proc))
        info("Estimating shifts.")
        shifts = estimate_shifts(frame_codes, Vhat, R, max_shift)

        mean_shift = shifts.mean(axis=0)
        info(f"Avg.: row={mean_shift[0]:.2f}, col={mean_shift[1]:.2f}")

    # Negate: estimate_shifts returns the displacement of each frame;
    # apply_shifts_lowrank rolls bases forward, so we roll by -shifts to
    # correct.
    corrected = apply_shifts_lowrank(vid, -shifts)

    if n_passes == 1:
        return corrected, all_shifts[0]
    else:
        return corrected, all_shifts
