from contextlib import contextmanager
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter

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
    ):
        # Validate inputs
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

        self.normalize = normalize
        self.bg_smooth_size = bg_smooth_size
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.exclusion_threshold = exclusion_threshold

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
        self.active_idx, self.W_act = self._active_and_flatten(self.W)

        # Precompute annuli indices per ROI in init (requires 2D masks)
        if self._spatial_shape is None:
            # If W provided flattened, cannot do dilation; require 3D W for annulus
            raise ValueError("Neuropil annuli require W as (H,W,K) to perform dilation")
        self.annuli = self._build_annuli(
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
            # Finalize mean image normalization
            mean_image = self._finalize_mean(self._sum_per_pixel, self._count_frames)
            self._mean_image_norm = mean_image.astype(np.float32)
            # Build design and weights (no pinv yet)
            self._design, self._weights = self._finalize_design(self.W_act, self._mean_image_norm)
            # Compute pseudo-inverse now that design is fixed
            self._design_pinv = np.linalg.pinv(self._design).astype(np.float32)
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
        if self._design_pinv is None or self._weights is None or self._mean_image_norm is None:
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
            # OLS solve and rescale
            F_batch = self._solve_and_rescale(
                Y_act, self._design_pinv, self._weights, self._neuropil_slice(B)
            )
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

    # Helpers (max 5)

    @staticmethod
    def _active_and_flatten(W):
        """Select active pixels and flatten masks to (P_active, K).
        Encapsulates active index computation and reshaping for consistent batching.
        Returns (active_idx, W_act).
        """
        if W.ndim == 3:
            H, Ww, K = W.shape
            W_flat = W.reshape(-1, K)
        elif W.ndim == 2:
            W_flat = W
            K = W.shape[1]
        else:
            raise ValueError("W must be (H,W,K) or (P,K)")
        active_idx = np.sum(W_flat, axis=1) > 0
        W_act = W_flat[active_idx, :].astype(np.float32)
        return active_idx, W_act

    @staticmethod
    def _build_annuli(W, neuropil_range, all_roi_mask, exclusion_threshold):
        """Precompute annulus indices per ROI with exclusion logic.
        Handles dilation, exclusion of other ROIs, and thresholding of remaining area.
        Returns list of flat indices per ROI.
        """
        if W.ndim != 3:
            raise ValueError("_build_annuli requires W as (H,W,K)")
        H, Ww, K = W.shape
        r_inner, r_outer = neuropil_range
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

    def _accumulate_stats(self, Y_act, active_idx, Y_flat, annuli, bg_smooth_size):
        """Update running sums/counts for mean image and accumulate neuropil per ROI.
        Operates on flat active pixels; computes annulus means with optional smoothing.
        Mutates internal state holding sums, counts, and neuropil buffers.
        """
        # Mean image accumulation over active pixels
        self._sum_per_pixel += Y_act.sum(axis=1)
        self._count_frames += Y_act.shape[1]
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
        if bg_smooth_size > 0:
            for k in range(K):
                F_np_batch[k, :] = savgol_filter(F_np_batch[k, :], window_length=bg_smooth_size, polyorder=2)
        self._neuropil_accum.append(F_np_batch)

    @staticmethod
    def _finalize_design(W_act, mean_image_norm):
        """Construct design components and rescale weights (without pinv).
        Precomputes background regressor, weights per ROI, and the design matrix.
        Returns (design, weights_dict).
        """
        # Background regressor μ
        mu = mean_image_norm[:, None]  # (P_active, 1)
        # Design matrix [W | μ]
        design = np.hstack([W_act, mu]).astype(np.float32)  # (P_active, K+1)
        # Rescale weights
        mask_weight = np.sum(W_act, axis=0).astype(np.float32)  # (K,)
        mean_im_weight = (mu.T @ W_act).ravel().astype(np.float32)  # (K,)
        weights = {"mask_weight": mask_weight, "mean_im_weight": mean_im_weight}
        return design, weights

    def _solve_and_rescale(self, Y_batch_act, design_pinv, weights, neuropil_batch):
        """Apply pseudo-inverse to get coefficients and rescale to absolute fluorescence.
        Combines ROI and background components; subtracts provided neuropil batch.
        Returns per-ROI traces for the batch.
        """
        # Coefficients (K+1, B)
        C = design_pinv @ Y_batch_act  # (K+1, B)
        roi_coeffs = C[:-1, :]  # (K, B)
        bg_coeff = C[-1, :][None, :]  # (1, B)
        # Rescale
        mw = weights["mask_weight"][:, None]       # (K,1)
        miw = weights["mean_im_weight"][:, None]   # (K,1)
        F = roi_coeffs * mw + miw * bg_coeff       # (K, B)
        # Subtract neuropil if available
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
        """Compute normalized mean image from accumulators."""
        if count_frames == 0:
            raise RuntimeError("No frames added during precompute")
        mean_im = (sum_pixels / float(count_frames)).astype(np.float32)
        max_val = np.max(mean_im)
        if max_val == 0:
            max_val = 1.0
        mean_im_norm = mean_im / max_val
        return mean_im_norm