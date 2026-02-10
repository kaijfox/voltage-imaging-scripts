from array_api_compat import array_namespace

from .svd_video import SVDVideo, _backend
from ..cli.common import configure_logging

logger, (error, warning, info, debug) = configure_logging("ops")


def _savgol_temporal(video: SVDVideo, window_length: int, polyorder: int):
    """
    Build Savitzky-Golay kernel and apply temporally to video.U returning filtered U
    with shape (time, rank).
    """
    from scipy.signal import savgol_coeffs

    xp = array_namespace(video.U)
    debug(
        f"Applying savgol temporal filter (window={window_length}, order={polyorder})"
    )

    # Apply temporally via static convolve on U.T -> convolve along time axis
    if _backend(video.U, "numpy"):
        from scipy.signal import savgol_filter

        U_filtered_T = savgol_filter(
            video.U.T, window_length, polyorder, axis=1, mode="nearest"
        )
    else:
        # Build savgol kernel
        kernel = savgol_coeffs(window_length, polyorder, deriv=0)
        debug(f"Savgol kernel: {kernel.shape}")
        kernel = xp.asarray(kernel, dtype=video.U.dtype)
        U_filtered_T = SVDVideo.convolve(video.U.T, kernel, dim="t", pad_mode="nearest")

    U_filtered = xp.permute_dims(U_filtered_T, (1, 0))  # (time, rank)

    return U_filtered


def divisive_lowpass(
    video: SVDVideo,
    window_length: int,
    polyorder: int,
    min_baseline: float = 1e-8,
    reorthogonalize: bool = True,
) -> SVDVideo:
    """
    Apply divisive temporal low-pass filter to an SVD video.

    Measures low-frequency activity using a Savitzky-Golay filter,
    computes the spatial mean at each timepoint, and divides the
    original video by this baseline.

    Parameters
    ----------
    video : SVDVideo
        Input video in SVD form.
    window_length : int
        Length of the Savitzky-Golay filter window (must be odd).
    polyorder : int
        Order of the polynomial used in the Savitzky-Golay filter.
    min_baseline : float
        Minimum baseline value to avoid division by very small numbers.
        Default: 1e-8.

    Returns
    -------
    SVDVideo
        Filtered video with orthonormal bases.
    """

    xp = array_namespace(video.U)
    debug(f"Video shape: U={video.U.shape}, S={video.S.shape}, Vt={video.Vt.shape}")

    # 1. Build savgol kernel and apply temporally via helper
    U_filtered = _savgol_temporal(video, window_length, polyorder)

    # 2. Compute spatial mean baseline
    debug("Computing spatial mean baseline")
    spatial_sum = video.Vt
    for _ in range(video.ndim_spatial):
        spatial_sum = xp.sum(spatial_sum, axis=-1)
    # spatial_sum is now (rank,)

    n_pixels = 1
    for dim in video.Vt.shape[1:]:
        n_pixels *= dim
    spatial_mean = spatial_sum / n_pixels  # (rank,)

    # baseline(t) = U_filtered[t, :] @ (S * spatial_mean)
    baseline = U_filtered @ (video.S * spatial_mean)  # (time,)
    vmin, vmax = xp.min(baseline), xp.max(baseline)
    debug(f"Baseline range: [{float(vmin):.4g}, {float(vmax):.4g}]")

    # 4. Clamp baseline to avoid division by small values
    clamp_mask = abs(baseline) < min_baseline
    baseline_clamped = baseline.copy()
    baseline_clamped[clamp_mask] = min_baseline * xp.sign(baseline[clamp_mask])
    n_clamped = int(xp.sum(clamp_mask))
    if n_clamped > 0:
        warning(f"Clamped {n_clamped} baseline values below {min_baseline}")

    # Scale U by 1/baseline (equivalent to diag(1/baseline) @ U)
    F = video.U / baseline_clamped[:, None]

    # 5. Reorthogonalize: F = (filter) @ U, Gt = Vt (no spatial filter)
    if reorthogonalize:
        debug("Reorthogonalizing.")
        U_new, S_new, Vt_new = SVDVideo.orthogonal(video.S, F, video.Vt)

        # plt.tight_layout()
        # plt.show()

        info(f"Divisive lowpass complete: output rank={len(S_new)}")
        return SVDVideo(U_new, S_new, Vt_new, orthonormal=True)
    else:
        info(f"Divisive lowpass complete: output rank={video.S}")
        return SVDVideo(F, video.S, video.Vt, orthonormal=False)


def subtractive_lowpass(
    video: SVDVideo,
    window_length: int,
    polyorder: int,
    reorthogonalize: bool = True,
) -> SVDVideo:
    """
    Subtractive low-pass: subtract Savitzky-Golay temporal lowpass from U,
    optionally reorthogonalize result and return new SVDVideo.
    """
    xp = array_namespace(video.U)

    debug(f"Applying subtractive lowpass (window={window_length}, order={polyorder})")

    # 1. Low-pass in temporal domain on U
    F = _savgol_temporal(video, window_length, polyorder)

    # 2. Subtract
    G = video.U - F

    # 3. Optionally reorthogonalize
    if reorthogonalize:
        debug("Reorthogonalizing subtractive result.")
        U_new, S_new, Vt_new = SVDVideo.orthogonal(video.S, G, video.Vt)
        info(f"Subtractive lowpass complete: output rank={len(S_new)}")
        return SVDVideo(U_new, S_new, Vt_new, orthonormal=True)
    else:
        info("Subtractive lowpass complete: returning non-orthonormal SVDVideo")
        return SVDVideo(G, video.S, video.Vt, orthonormal=False)
