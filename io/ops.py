from array_api_compat import array_namespace

from .svd_video import SVDVideo


def divisive_lowpass(
    video: SVDVideo,
    window_length: int,
    polyorder: int,
    min_baseline: float = 1e-8,
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

    # 1. Build savgol kernel
    kernel = savgol_coeffs(window_length, polyorder, deriv=0)
    kernel = xp.asarray(kernel, dtype=video.U.dtype)

    # 2. Apply temporally via static convolve on U.T
    #    U.T has shape (rank, time), convolve along time axis
    U_filtered_T = SVDVideo.convolve(video.U.T, kernel, dim="t")
    U_filtered = xp.permute_dims(U_filtered_T, (1, 0))  # (time, rank)

    # 3. Compute spatial mean baseline
    #    Vt has shape (rank, spatial...), reduce over all spatial dims
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

    # 4. Clamp baseline to avoid division by small values
    baseline_clamped = xp.maximum(baseline, min_baseline)

    # Scale U by 1/baseline (equivalent to diag(1/baseline) @ U)
    F = video.U / baseline_clamped[:, None]

    # 5. Reorthogonalize: F = (filter) @ U, Gt = Vt (no spatial filter)
    U_new, S_new, Vt_new = SVDVideo.orthogonal(video.S, F, video.Vt)

    return SVDVideo(U_new, S_new, Vt_new, orthonormal=True)
