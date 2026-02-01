from array_api_compat import array_namespace

from .svd_video import SVDVideo
from ..cli.common import configure_logging

logger, (error, warning, info, debug) = configure_logging("ops")


def divisive_lowpass(
    video: SVDVideo,
    window_length: int,
    polyorder: int,
    min_baseline: float = 1e-8,
    reorthogonalize: bool = True
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
    from scipy.signal import savgol_coeffs

    xp = array_namespace(video.U)
    debug(f"Video shape: U={video.U.shape}, S={video.S.shape}, Vt={video.Vt.shape}")

    # 1. Build savgol kernel
    kernel = savgol_coeffs(window_length, polyorder, deriv=0)
    kernel = xp.asarray(kernel, dtype=video.U.dtype)

    # 2. Apply temporally via static convolve on U.T
    #    U.T has shape (rank, time), convolve along time axis
    debug("Applying temporal convolution")
    U_filtered_T = SVDVideo.convolve(video.U.T, kernel, dim="t", pad_mode='nearest')
    U_filtered = xp.permute_dims(U_filtered_T, (1, 0))  # (time, rank)

    # import matplotlib.pyplot as plt
    # import mplutil.util as vu
    # from scipy.signal import savgol_filter
    # print(video.U.shape, U_filtered.shape)
    # f, a = vu.subplots((4, 4), (2, 2))
    # i = 10
    # a[0, 0].plot(video.U[:,i])
    # a[0, 0].plot(U_filtered[:,i])
    # u_manual = savgol_filter(video.U[:, i], window_length, polyorder)
    # a[0, 0].plot(u_manual, 'k--', lw=1)
    # vu.label(a[0, 0], "time", "magnitude", "Smoothing")

    # 3. Compute spatial mean baseline
    #    Vt has shape (rank, spatial...), reduce over all spatial dims
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
    

    
