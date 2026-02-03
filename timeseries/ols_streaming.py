from contextlib import contextmanager
from pathlib import Path
from typing import Union
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter

import h5py

from .types import Traces
from .rois import ROI, ROICollection
from ..cli.common import configure_logging

logger, (error, warning, info, debug) = configure_logging("extract")


def extract_traces(
    source,
    rois,
    neuropil_range,
    *,
    all_roi_mask=None,
    normalize=False,
    bg_smooth_size=0,
    exclusion_threshold=0.4,
    ids=None,
    fs=None,
    batch_size=None,
    ols=True,
    weighted=True,
):
    """
    Extract ROI traces from video source.

    Parameters
    ----------
    source : str | Path | SVDVideo
        One of:
        - Path to H5 movie file (with "video" dataset)
        - Path to H5 SVD file (with "U", "S", "Vh" datasets)
        - SVDVideo object in memory
    rois : ndarray | ROICollection | str | Path
        ROI weights as:
        - (H, W, K) array of weight masks
        - ROICollection object
        - Path to ROI file (.mat, .h5, .npz)
    neuropil_range : tuple
        (r_inner, r_outer) for neuropil annulus dilation.
    all_roi_mask : ndarray, optional
        Combined mask of all ROIs for exclusion.
    normalize : bool
        If True, return dF/F as percentage.
    bg_smooth_size : int
        Savitzky-Golay window for neuropil smoothing.
    exclusion_threshold : float
        Fraction threshold for excluding other ROIs from annulus.
    ids : list[str], optional
        ROI identifiers. If None, generates "ROI 0", "ROI 1", etc.
    fs : float, optional
        Sampling rate in Hz.
    batch_size : int, optional
        Batch size for streaming extraction. Required for H5 movie sources.
    ols : bool, default True
        If True, use OLS regression to unmix ROI signals from background.
        If False, use simple weighted summation (still performs neuropil subtraction).
    weighted : bool, default True
        If True, use ROI weights from the collection.
        If False, use uniform weights normalized by L2 norm.

    Returns
    -------
    Traces
        Extracted traces with shape (K, T).
    """
    # Import SVDVideo here to avoid circular import
    from ..io.svd_video import SVDVideo
    from ..io.svd import SRSVD

    # Resolve ROIs to W array and get image shape
    W, image_shape = _resolve_rois(rois)
    K = W.shape[2]

    # Generate ids if not provided
    if ids is None:
        ids = [f"ROI {i}" for i in range(K)]

    # Detect source type and dispatch
    if isinstance(source, SVDVideo):
        F = extract_from_svd(
            source, W, neuropil_range,
            all_roi_mask=all_roi_mask,
            normalize=normalize,
            bg_smooth_size=bg_smooth_size,
            exclusion_threshold=exclusion_threshold,
            ols=ols,
            weighted=weighted,
        )
    elif isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Check what kind of H5 file it is
        with h5py.File(source_path, "r") as f:
            has_svd = "U" in f and "S" in f and "Vh" in f
            has_video = "video" in f

        if has_svd:
            # Load SVD and extract
            srsvd = SRSVD(str(source_path))
            svd_video = srsvd.to_loaded_svd()
            F = extract_from_svd(
                svd_video, W, neuropil_range,
                all_roi_mask=all_roi_mask,
                normalize=normalize,
                bg_smooth_size=bg_smooth_size,
                exclusion_threshold=exclusion_threshold,
                ols=ols,
                weighted=weighted,
            )
        elif has_video:
            # Streaming extraction from raw video
            if batch_size is None:
                raise ValueError("batch_size required for H5 movie extraction")
            F = extract_from_h5_video(
                source_path, W, neuropil_range,
                batch_size=batch_size,
                all_roi_mask=all_roi_mask,
                normalize=normalize,
                bg_smooth_size=bg_smooth_size,
                exclusion_threshold=exclusion_threshold,
                ols=ols,
                weighted=weighted,
            )
        else:
            raise ValueError(
                f"H5 file must contain either 'video' dataset or 'U','S','Vh' datasets"
            )
    else:
        raise TypeError(
            f"source must be SVDVideo, or path to H5 file, got {type(source)}"
        )

    return Traces(data=F, ids=ids, fs=fs)


def _resolve_rois(rois):
    """
    Convert ROI input to (H, W, K) weight array.

    Returns
    -------
    W : ndarray
        (H, W, K) weight masks.
    image_shape : tuple or None
        (H, W) if known from ROICollection.
    """
    if isinstance(rois, np.ndarray):
        if rois.ndim != 3:
            raise ValueError("W array must be (H, W, K)")
        return rois, (rois.shape[0], rois.shape[1])

    if isinstance(rois, (str, Path)):
        rois = ROICollection.load(rois)

    if isinstance(rois, ROICollection):
        if rois.image_shape is None:
            raise ValueError(
                "ROICollection must have image_shape set to build weight array"
            )
        W = _roi_collection_to_weights(rois)
        return W, rois.image_shape

    raise TypeError(f"rois must be ndarray, ROICollection, or path, got {type(rois)}")


def _roi_collection_to_weights(roi_collection):
    """Convert ROICollection to (H, W, K) weight array."""
    H, W_dim = roi_collection.image_shape
    K = len(roi_collection.rois)
    W = np.zeros((H, W_dim, K), dtype=np.float32)

    for k, roi in enumerate(roi_collection.rois):
        if len(roi.footprint) > 0:
            rows = roi.footprint[:, 0]
            cols = roi.footprint[:, 1]
            W[rows, cols, k] = roi.weights

    return W


def extract_from_h5_video(
    h5_path,
    W,
    neuropil_range,
    batch_size,
    all_roi_mask=None,
    normalize=False,
    bg_smooth_size=0,
    exclusion_threshold=0.4,
    ols=True,
    weighted=True,
):
    """
    Extract ROI traces from H5 video file using streaming.

    Parameters
    ----------
    h5_path : str | Path
        Path to H5 file with "video" dataset of shape (T, H, W).
    W : ndarray
        ROI weight masks, shape (H, W, K).
    neuropil_range : tuple
        (r_inner, r_outer) for neuropil annulus dilation.
    batch_size : int
        Number of frames per batch.
    all_roi_mask : ndarray, optional
        Combined mask of all ROIs for exclusion.
    normalize : bool
        If True, return dF/F as percentage.
    bg_smooth_size : int
        Savitzky-Golay window for neuropil smoothing.
    exclusion_threshold : float
        Fraction threshold for excluding other ROIs from annulus.
    ols : bool
        If True, use OLS regression. If False, use weighted summation.
    weighted : bool
        If True, use ROI weights. If False, use uniform weights.

    Returns
    -------
    F : ndarray
        ROI traces, shape (K, T).
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        video = f["video"]
        T, H, W_dim = video.shape

        # Validate dimensions match
        if W.shape[0] != H or W.shape[1] != W_dim:
            raise ValueError(
                f"W shape {W.shape[:2]} doesn't match video shape ({H}, {W_dim})"
            )

        extractor = StreamingTimeSeriesExtractor(
            W,
            neuropil_range,
            all_roi_mask=all_roi_mask,
            normalize=normalize,
            bg_smooth_size=bg_smooth_size,
            exclusion_threshold=exclusion_threshold,
            ols=ols,
            weighted=weighted,
        )

        # Precompute phase: accumulate mean image and neuropil
        with extractor.precompute():
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                info(f"Precomputation: batch {start}-{end} out of 0-{T}.")
                # video is (T, H, W), need (H, W, B) for extractor
                batch = video[start:end, :, :].transpose(1, 2, 0)
                extractor.receive_batch(batch)

        # Extract phase: compute traces
        traces_list = []
        with extractor.extract():
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                info(f"Extraction: batch {start}-{end} out of 0-{T}.")
                batch = video[start:end, :, :].transpose(1, 2, 0)
                F_batch = extractor.receive_batch(batch)
                traces_list.append(F_batch)

    F = np.concatenate(traces_list, axis=1)
    return F.astype(np.float32)


def extract_from_svd(
    svd_video,
    W,
    neuropil_range,
    all_roi_mask=None,
    normalize=False,
    bg_smooth_size=0,
    exclusion_threshold=0.4,
    ols=True,
    weighted=True,
):
    """
    Extract ROI traces from SVDVideo without reconstructing full frames.

    Parameters
    ----------
    svd_video : SVDVideo
        SVD-compressed video with U (T, R), S (R,), Vt (R, H, W) or (R, P).
    W : ndarray
        ROI weight masks, shape (H, W, K) or (P, K).
    neuropil_range : tuple
        (r_inner, r_outer) for neuropil annulus dilation.
    all_roi_mask : ndarray, optional
        Combined mask of all ROIs for exclusion, shape (H, W).
    normalize : bool
        If True, return dF/F as percentage.
    bg_smooth_size : int
        Savitzky-Golay window for smoothing neuropil (must be odd or 0).
    exclusion_threshold : float
        Fraction threshold for excluding other ROIs from annulus.
    ols : bool
        If True, use OLS regression. If False, use weighted summation.
    weighted : bool
        If True, use ROI weights. If False, use uniform weights.

    Returns
    -------
    F : ndarray
        ROI traces, shape (K, T).
    """
    _validate_params(neuropil_range, bg_smooth_size, exclusion_threshold)

    U, S, Vt = svd_video.U, svd_video.S, svd_video.Vt
    R = U.shape[0]

    # Flatten Vt spatial dimensions
    Vt_flat = Vt.reshape(Vt.shape[0], -1) if Vt.ndim > 2 else Vt  # (R, P)
    P = Vt_flat.shape[1]

    # Ensure W is float and validate shape
    W = W.astype(np.float32)
    if W.ndim == 3:
        K = W.shape[2]
    elif W.ndim == 2:
        K = W.shape[1]
    else:
        raise ValueError("W must be (H,W,K) or (P,K)")
    W_flat = W.reshape(-1, K)

    if W_flat.shape[0] != P:
        raise ValueError(f"W spatial size {W_flat.shape[0]} != SVD spatial size {P}")

    if all_roi_mask is None and W.ndim == 3:
        all_roi_mask = np.any(W > 0, axis=2)

    # Reuse shared helpers
    active_idx, W_act = _active_and_flatten(W)

    # Apply uniform weights if requested (normalize by L2 norm)
    if not weighted:
        W_act = _apply_uniform_weights(W_act)

    if W.ndim != 3:
        raise ValueError("Neuropil annuli require W as (H,W,K) to perform dilation")
    annuli = _build_annuli(W, neuropil_range, all_roi_mask, exclusion_threshold)

    # Compute neuropil from SVD (needed for both OLS and simple summation)
    if annuli is not None:
        F_np = _compute_neuropil_svd(Vt_flat, S, U, annuli, K, bg_smooth_size)

    if ols:
        # Mean image from SVD: mean(Y, axis=0) = mean(U, axis=0) @ diag(S) @ Vt
        U_mean = U.mean(axis=0)  # (R,)
        mean_image_full = (U_mean * S) @ Vt_flat  # (P,)
        mean_image_norm = _normalize_mean_image(mean_image_full[active_idx])

        # Build design matrix and pseudo-inverse
        design, weights_dict = _finalize_design(W_act, mean_image_norm)
        design_pinv = np.linalg.pinv(design).astype(np.float32)

        # Project design through SVD spatial basis
        # C = design_pinv @ Y_act.T, where Y_act.T = Vt_act.T @ diag(S) @ U.T
        Vt_act = Vt_flat[:, active_idx]  # (R, P_active)
        proj = design_pinv @ Vt_act.T  # (K+1, R)
        C = (proj * S) @ U.T  # (K+1, T)

        # Rescale and subtract neuropil
        F = _rescale_coefficients(C, weights_dict, F_np if annuli is not None else None)
    else:
        # Simple weighted summation (still with neuropil subtraction)
        # F[k, t] = sum_p(W_act[p, k] * Y[t, p]) - F_np[k, t]
        # Using SVD: Y = U @ diag(S) @ Vt, so sum over active pixels:
        # F[k, :] = W_act[:, k].T @ Vt_act.T @ diag(S) @ U.T
        Vt_act = Vt_flat[:, active_idx]  # (R, P_active)
        proj = W_act.T @ Vt_act.T  # (K, R)
        F = (proj * S) @ U.T  # (K, T)
        F = F - F_np

    if normalize:
        baseline = np.median(F, axis=1, keepdims=True)
        baseline[baseline == 0] = 1.0
        F = 100.0 * (F - baseline) / baseline

    return F.astype(np.float32)


def _validate_params(neuropil_range, bg_smooth_size, exclusion_threshold):
    """Validate common parameters for both streaming and SVD extraction."""
    if not isinstance(neuropil_range, tuple) or len(neuropil_range) != 2:
        raise ValueError("neuropil_range must be a tuple (r_inner, r_outer)")
    r_inner, r_outer = neuropil_range
    if r_outer < r_inner:
        raise ValueError("neuropil_range must have outer >= inner")
    if bg_smooth_size < 0:
        raise ValueError("bg_smooth_size must be >= 0")
    if bg_smooth_size and bg_smooth_size % 2 == 0:
        raise ValueError("bg_smooth_size must be odd when nonzero")
    if not (0.0 < exclusion_threshold <= 1.0):
        raise ValueError("exclusion_threshold must be in (0,1]")


def _active_and_flatten(W):
    """Select active pixels and flatten masks to (P_active, K)."""
    if W.ndim == 3:
        H, Ww, K = W.shape
        W_flat = W.reshape(-1, K)
    elif W.ndim == 2:
        W_flat = W
    else:
        raise ValueError("W must be (H,W,K) or (P,K)")
    active_idx = np.sum(W_flat, axis=1) > 0
    W_act = W_flat[active_idx, :].astype(np.float32)
    return active_idx, W_act


def _apply_uniform_weights(W_act):
    """Replace weights with uniform weights normalized by L2 norm.

    For each ROI k, sets all nonzero weights to 1/sqrt(n_pixels_k).
    This makes the weighted sum equivalent to a normalized average.
    """
    W_uniform = np.zeros_like(W_act)
    K = W_act.shape[1]
    for k in range(K):
        mask = W_act[:, k] > 0
        n_pixels = np.sum(mask)
        if n_pixels > 0:
            W_uniform[mask, k] = 1.0 / np.sqrt(n_pixels)
    return W_uniform


def _build_annuli(W, neuropil_range, all_roi_mask, exclusion_threshold):
    """Precompute annulus indices per ROI with exclusion logic."""
    if W.ndim != 3:
        raise ValueError("_build_annuli requires W as (H,W,K)")
    H, Ww, K = W.shape
    r_inner, r_outer = neuropil_range
    if r_inner < 0:
        return None
    if all_roi_mask is None:
        all_roi_mask = np.any(W > 0, axis=2)
    annuli = []
    for k in range(K):
        roi_mask = W[:, :, k] > 0
        dilated_outer = binary_dilation(roi_mask, iterations=r_outer)
        dilated_inner = binary_dilation(roi_mask, iterations=r_inner)
        annulus = dilated_outer & ~dilated_inner
        annulus_no_roi = annulus & ~all_roi_mask
        if annulus.sum() == 0:
            annuli.append(np.array([], dtype=np.int64))
            continue
        if annulus_no_roi.sum() / annulus.sum() >= exclusion_threshold:
            annulus = annulus_no_roi
        annuli.append(np.flatnonzero(annulus.ravel()))
    return annuli


def _finalize_design(W_act, mean_image_norm):
    """Construct design matrix and rescale weights."""
    mu = mean_image_norm[:, None]  # (P_active, 1)
    design = np.hstack([W_act, mu]).astype(np.float32)  # (P_active, K+1)
    mask_weight = np.sum(W_act, axis=0).astype(np.float32)  # (K,)
    mean_im_weight = (mu.T @ W_act).ravel().astype(np.float32)  # (K,)
    weights = {"mask_weight": mask_weight, "mean_im_weight": mean_im_weight}
    return design, weights


def _normalize_mean_image(mean_image):
    """Normalize mean image to [0, 1] range."""
    mean_im = mean_image.astype(np.float32)
    max_val = np.max(mean_im)
    if max_val == 0:
        max_val = 1.0
    return mean_im / max_val


def _compute_neuropil_svd(Vt_flat, S, U, annuli, K, bg_smooth_size):
    """Compute neuropil traces from SVD components."""
    T = U.shape[0]
    F_np = np.zeros((K, T), dtype=np.float32)

    for k in range(K):
        idx = annuli[k]
        if idx.size == 0:
            continue
        # mean(Y[:, idx], axis=1) = U @ diag(S) @ mean(Vt[:, idx], axis=1)
        Vt_annulus_mean = Vt_flat[:, idx].mean(axis=1)
        F_np[k, :] = U @ (S * Vt_annulus_mean)

    if bg_smooth_size > 0:
        for k in range(K):
            F_np[k, :] = savgol_filter(F_np[k, :], window_length=bg_smooth_size, polyorder=2)

    return F_np


def _rescale_coefficients(C, weights, neuropil=None):
    """Rescale OLS coefficients to absolute fluorescence."""
    roi_coeffs = C[:-1, :]  # (K, T)
    bg_coeff = C[-1:, :]    # (1, T)

    mw = weights["mask_weight"][:, None]
    miw = weights["mean_im_weight"][:, None]

    F = roi_coeffs * mw + miw * bg_coeff

    if neuropil is not None:
        F = F - neuropil

    return F.astype(np.float32)


class StreamingTimeSeriesExtractor:
    """Streaming OLS extractor with precompute/extract phases."""

    def __init__(
        self,
        W,  # (H, W, K) or (P, K)
        neuropil_range,
        all_roi_mask=None,  # (H, W) or None
        normalize=False,
        bg_smooth_size=0,
        exclusion_threshold=0.4,
        ols=True,
        weighted=True,
    ):
        _validate_params(neuropil_range, bg_smooth_size, exclusion_threshold)
        r_inner, r_outer = neuropil_range

        self.normalize = normalize
        self.bg_smooth_size = bg_smooth_size
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.exclusion_threshold = exclusion_threshold
        self.ols = ols
        self.weighted = weighted

        # Determine shapes and store mask W (require float masks for OLS)
        self.W = W.astype(np.float32)
        if self.W.ndim == 3:
            H, Ww, K = self.W.shape
            self._spatial_shape = (H, Ww)
        elif self.W.ndim == 2:
            P, K = self.W.shape
            self._spatial_shape = None
        else:
            raise ValueError("W must be (H,W,K) or (P,K)")
        self.K = K

        # Build all_roi_mask if not provided (only meaningful if W is 3D)
        if all_roi_mask is None and self.W.ndim == 3:
            self.all_roi_mask = np.any(self.W > 0, axis=2)
        else:
            self.all_roi_mask = all_roi_mask

        # Active pixels and flattened masks
        self.active_idx, self.W_act = _active_and_flatten(self.W)

        # Apply uniform weights if requested
        if not self.weighted:
            self.W_act = _apply_uniform_weights(self.W_act)

        # Precompute annuli indices per ROI in init (requires 2D masks)
        if self._spatial_shape is None:
            raise ValueError("Neuropil annuli require W as (H,W,K) to perform dilation")
        self.annuli = _build_annuli(
            self.W, (self.r_inner, self.r_outer), self.all_roi_mask, self.exclusion_threshold
        )

        # Internal state for phases
        self._in_precompute = False
        self._in_extract = False
        self._sum_per_pixel = None
        self._count_frames = 0
        self._mean_image_norm = None
        self._design = None
        self._design_pinv = None
        self._weights = None
        self._baseline = None  # for normalization
        # Precomputed neuropil timeseries (T-long); we accumulate across precompute
        self._neuropil_accum = None  # shape (K, T_precompute) or streaming buffer

    @contextmanager
    def precompute(self):
        """Accumulate mean image and neuropil timeseries across batches; build design (no pinv yet)."""
        if self._in_precompute or self._in_extract:
            raise RuntimeError("Cannot start precompute while another phase is active")
        # Initialize accumulators
        P_active = self.W_act.shape[0]
        self._sum_per_pixel = np.zeros((P_active,), dtype=np.float64)
        self._count_frames = 0
        # Neuropil accumulation per ROI (we will append per-batch)
        self._neuropil_accum = []  # list of (K, B) arrays
        self._in_precompute = True
        try:
            yield self
        finally:
            # Finalize mean image normalization (needed for OLS)
            mean_image = self._finalize_mean(self._sum_per_pixel, self._count_frames)
            self._mean_image_norm = _normalize_mean_image(mean_image)

            if self.ols:
                # Build design and weights for OLS
                self._design, self._weights = _finalize_design(self.W_act, self._mean_image_norm)
                # Compute pseudo-inverse now that design is fixed
                self._design_pinv = np.linalg.pinv(self._design).astype(np.float32)
            else:
                # For simple summation, just need the weights for weighted sum
                self._design = None
                self._design_pinv = None
                self._weights = None

            # Concatenate neuropil across batches and optionally smooth across time
            if len(self._neuropil_accum) > 0:
                F_np = np.concatenate(self._neuropil_accum, axis=1)
                if self.bg_smooth_size > 0:
                    # Apply Savitzky-Golay per ROI
                    F_np = np.vstack([
                        savgol_filter(F_np[k], window_length=self.bg_smooth_size, polyorder=2)
                        for k in range(self.K)
                    ])
                self._neuropil_ts = F_np.astype(np.float32)
            else:
                self._neuropil_ts = None
            # Cleanup precompute flags
            self._in_precompute = False

    @contextmanager
    def extract(self):
        """Consume batches and output OLS-unmixed traces using finalized design and neuropil."""
        if self._in_precompute or self._in_extract:
            raise RuntimeError("Cannot start extract while another phase is active")
        if self.ols:
            if self._design_pinv is None or self._weights is None or self._mean_image_norm is None:
                raise RuntimeError("precompute must be completed before extract")
        else:
            if self._mean_image_norm is None:
                raise RuntimeError("precompute must be completed before extract")
        # Prepare normalization baseline if enabled (defer if streaming baseline is needed)
        self._in_extract = True
        try:
            yield self
        finally:
            # Optionally finalize normalization baseline if deferred (no-op here)
            self._in_extract = False

    def receive_batch(self, Y_batch):
        """Single entry point for batches, behavior depends on current phase."""
        # Flatten batch to (P, B)
        if Y_batch.ndim == 3:
            H, Ww, B = Y_batch.shape
            if self._spatial_shape is None or (H, Ww) != self._spatial_shape:
                raise ValueError("Y_batch spatial shape must match W's spatial shape")
            Y_flat = Y_batch.reshape(-1, B)
        elif Y_batch.ndim == 2:
            Y_flat = Y_batch
            B = Y_flat.shape[1]
        else:
            raise ValueError("Y_batch must be (H,W,B) or (P,B)")
        # Select active pixels
        Y_act = Y_flat[self.active_idx, :]  # (P_active, B)

        if self._in_precompute:
            # Accumulate mean image stats and neuropil per batch
            self._accumulate_stats(Y_act, self.active_idx, Y_flat, self.annuli, self.bg_smooth_size)
            return None

        if self._in_extract:
            neuropil_batch = self._neuropil_slice(B)
            if self.ols:
                # OLS solve and rescale
                F_batch = self._solve_and_rescale(
                    Y_act, self._design_pinv, self._weights, neuropil_batch
                )
            else:
                # Simple weighted summation with neuropil subtraction
                F_batch = self._weighted_sum(Y_act, neuropil_batch)

            # Optional normalization: compute baseline on first call if not set
            if self.normalize:
                if self._baseline is None:
                    # Use robust median per ROI over the first batch as baseline (simple choice)
                    self._baseline = np.median(F_batch, axis=1, keepdims=True)
                    # Avoid divide-by-zero
                    self._baseline[self._baseline == 0] = 1.0
                F_batch = 100.0 * (F_batch - self._baseline) / self._baseline
            return F_batch

        raise RuntimeError("receive_batch must be called inside precompute() or extract()")

    def _accumulate_stats(self, Y_act, active_idx, Y_flat, annuli, bg_smooth_size):
        """Update running sums/counts for mean image and accumulate neuropil per ROI.
        Operates on flat active pixels; computes annulus means with optional smoothing.
        Mutates internal state holding sums, counts, and neuropil buffers.
        """
        # Mean image accumulation over active pixels
        self._sum_per_pixel += Y_act.sum(axis=1)
        self._count_frames += Y_act.shape[1]
        if annuli is not None:
            # Neuropil per-ROI for this batch
            K = self.K
            B = Y_act.shape[1]
            F_np_batch = np.zeros((K, B), dtype=np.float32)
            for k in range(K):
                idx = annuli[k]
                if idx.size == 0:
                    # Fallback: zero neuropil if no annulus pixels
                    F_np_batch[k, :] = 0.0
                    continue
                F_np_batch[k, :] = Y_flat[idx, :].mean(axis=0)
            self._neuropil_accum.append(F_np_batch)

    def _solve_and_rescale(self, Y_batch_act, design_pinv, weights, neuropil_batch):
        """Apply pseudo-inverse to get coefficients and rescale to absolute fluorescence."""
        C = design_pinv @ Y_batch_act  # (K+1, B)
        return _rescale_coefficients(C, weights, neuropil_batch)

    def _weighted_sum(self, Y_batch_act, neuropil_batch):
        """Compute simple weighted sum of pixels with neuropil subtraction."""
        # F[k, :] = W_act[:, k].T @ Y_batch_act - neuropil[k, :]
        F = self.W_act.T @ Y_batch_act  # (K, B)
        if neuropil_batch is not None:
            F = F - neuropil_batch
        return F.astype(np.float32)

    def _neuropil_slice(self, B):
        """Return neuropil segment matching the batch width, if precomputed."""
        if self._neuropil_ts is None:
            return None
        # Track how much consumed; align with accumulated frames
        # For simplicity, assume batches during extract match precompute sequence
        # We maintain a pointer to slice neuropil_ts per batch.
        if not hasattr(self, "_np_ptr"):
            self._np_ptr = 0
        start = self._np_ptr
        end = start + B
        if end > self._neuropil_ts.shape[1]:
            # If extract has more frames than precompute, pad with zeros
            pad = end - self._neuropil_ts.shape[1]
            np_seg = self._neuropil_ts[:, start:] if start < self._neuropil_ts.shape[1] else None
            if np_seg is None or np_seg.shape[1] == 0:
                seg = np.zeros((self.K, B), dtype=np.float32)
            else:
                seg = np.hstack([np_seg, np.zeros((self.K, pad), dtype=np.float32)])
        else:
            seg = self._neuropil_ts[:, start:end]
        self._np_ptr = end
        return seg

    @staticmethod
    def _finalize_mean(sum_pixels, count_frames):
        """Compute mean image from accumulators (unnormalized)."""
        if count_frames == 0:
            raise RuntimeError("No frames added during precompute")
        return (sum_pixels / float(count_frames)).astype(np.float32)