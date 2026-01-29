"""Data source abstraction for ROI analysis."""

from typing import Protocol, Optional, Tuple
import numpy as np
from scipy import ndimage


class DataSource(Protocol):
    """Protocol for data sources supporting ROI analysis.

    Abstracts over SVD-compressed data, raw video arrays, etc.
    """

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (height, width) of the data."""
        ...

    @property
    def n_components(self) -> int:
        """Number of temporal components (SVD rank or timepoints)."""
        ...

    def mean_image(self) -> np.ndarray:
        """Return mean image (h, w)."""
        ...

    def correlation_map(self, seed_row: int, seed_col: int) -> np.ndarray:
        """Compute correlation map from seed pixel (h, w).

        Returns correlation of each pixel's temporal code with seed pixel's code.
        """
        ...

    def correlation_map_from_code(self, code: np.ndarray) -> np.ndarray:
        """Compute correlation map from a code vector (h, w).

        Returns correlation of each pixel's temporal code with the given code.
        """
        ...

    def extract_code(self, footprint: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract temporal code for ROI defined by footprint.

        Args:
            footprint: (n_pixels, 2) array of (row, col) coordinates
            weights: Optional (n_pixels,) weights. If None, uniform weights used.

        Returns:
            code: (n_components,) temporal code vector
        """
        ...

    def extract_trace(self, footprint: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract temporal trace for ROI.

        Args:
            footprint: (n_pixels, 2) array of (row, col) coordinates
            weights: Optional (n_pixels,) weights. If None, uniform weights used.

        Returns:
            trace: (n_timepoints,) fluorescence trace
        """
        ...

    def get_pixel_codes(self, footprint: np.ndarray) -> np.ndarray:
        """Get code vectors for all pixels in footprint.

        Args:
            footprint: (n_pixels, 2) array of (row, col) coordinates

        Returns:
            codes: (n_pixels, n_components) code vectors
        """
        ...

    def local_correlation_map(self) -> np.ndarray:
        """Compute local correlation map (h, w).

        Returns correlation of each pixel with its annular neighborhood.
        """
        ...


class SVDDataSource:
    """Base data source backed by SVD components.

    Stores spatial loadings U (h, w, k) and temporal components u (t, k).
    The temporal activity at pixel (r, c) is approximately: U[r, c, :] @ u.T
    """

    def __init__(self, U: np.ndarray, u: np.ndarray, mean: Optional[np.ndarray] = None):
        """Initialize from SVD components.

        Args:
            U: (h, w, k) spatial loadings (with S absorbed)
            u: (t, k) temporal components
            mean: (h, w) mean image. If None, computed from components.
        """
        if U.ndim != 3:
            raise ValueError(f"Expected 3D spatial loadings (h, w, k), got {U.ndim}D")
        if u.ndim != 2:
            raise ValueError(f"Expected 2D temporal components (t, k), got {u.ndim}D")

        h, w, k = U.shape
        self._shape = (h, w)
        self._n_components = k
        self._U = U.astype(np.float32)
        self._u = u.astype(np.float32)

        # Compute mean from components if not provided
        if mean is None:
            u_mean = self._u.mean(axis=0)  # (k,)
            self._mean = (self._U @ u_mean).astype(np.float32)
        else:
            self._mean = mean.astype(np.float32)

        # Precompute norms for correlation
        self._U_norms = np.linalg.norm(self._U, axis=2)
        self._U_norms = np.maximum(self._U_norms, 1e-10)

        self._local_corr_cache = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def spatial_loadings(self) -> np.ndarray:
        """Return U (h, w, n_components)."""
        return self._U

    @property
    def temporal_components(self) -> np.ndarray:
        """Return u (t, n_components)."""
        return self._u

    def mean_image(self) -> np.ndarray:
        return self._mean

    def correlation_map(self, seed_row: int, seed_col: int) -> np.ndarray:
        """Compute normalized correlation map from seed pixel."""
        seed_code = self._U[seed_row, seed_col, :]
        return self.correlation_map_from_code(seed_code)

    def correlation_map_from_code(self, code: np.ndarray) -> np.ndarray:
        """Compute normalized correlation map from code vector."""
        corr = np.tensordot(self._U, code, axes=(2, 0))
        code_norm = np.linalg.norm(code)
        if code_norm > 1e-10:
            corr = corr / (self._U_norms * code_norm)
        return corr

    def extract_code(self, footprint: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract weighted average code for ROI."""
        if len(footprint) == 0:
            return np.zeros(self._n_components, dtype=np.float32)
        codes = self._U[footprint[:, 0], footprint[:, 1], :]
        if weights is None:
            weights = np.ones(len(footprint), dtype=np.float32)
        weights = weights / (weights.sum() + 1e-10)
        return (weights[:, np.newaxis] * codes).sum(axis=0)

    def extract_trace(self, footprint: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract temporal trace by projecting code through temporal components."""
        code = self.extract_code(footprint, weights)
        return self._u @ code

    def get_pixel_codes(self, footprint: np.ndarray) -> np.ndarray:
        """Get code vectors for all pixels in footprint."""
        if len(footprint) == 0:
            return np.zeros((0, self._n_components), dtype=np.float32)
        return self._U[footprint[:, 0], footprint[:, 1], :]

    def local_correlation_map(self) -> np.ndarray:
        """Compute local correlation map (cached)."""
        if self._local_corr_cache is not None:
            return self._local_corr_cache

        Z = self._U / self._U_norms[:, :, np.newaxis]
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=np.float32) / 8.0

        C = ndimage.convolve(Z, kernel, mode='nearest', axes=(0, 1))
        self._local_corr_cache = (Z * C).sum(axis=2)
        return self._local_corr_cache


class ArraySVDDataSource(SVDDataSource):
    """SVD data source computed from raw video array."""

    def __init__(self, video: np.ndarray, n_components: Optional[int] = None,
                 high_pass_width: int = 200, spatial_smooth_sigma: float = 1.0,
                 tmin: Optional[int] = None, tmax: Optional[int] = None):
        """Initialize from raw video array.

        Args:
            video: (h, w, t) video array
            n_components: Number of SVD components. If None, uses min(t/2, 500).
            high_pass_width: Width for temporal high-pass filter (frames)
            spatial_smooth_sigma: Sigma for spatial Gaussian smoothing
            tmin: Start frame index (inclusive). If None, starts at 0.
            tmax: End frame index (exclusive). If None, uses all frames.
        """
        if video.ndim != 3:
            raise ValueError(f"Expected 3D video (h, w, t), got shape {video.shape}")

        video = video[:, :, tmin:tmax]
        h, w, t = video.shape
        mean = video.mean(axis=2)

        if n_components is None:
            n_components = min(t // 2, 500)

        U, u = self._compute_svd(video, n_components, high_pass_width, spatial_smooth_sigma)
        super().__init__(U, u, mean)

    @staticmethod
    def _compute_svd(video: np.ndarray, n_components: int,
                     high_pass_width: int, spatial_smooth_sigma: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SVD decomposition from video."""
        from scipy.ndimage import gaussian_filter, uniform_filter1d

        h, w, t = video.shape
        mov = video.astype(np.float32).copy()

        # Temporal high-pass filter
        if high_pass_width > 0:
            low_pass = uniform_filter1d(mov, size=high_pass_width, axis=2, mode='reflect')
            mov = mov - low_pass

        # Spatial smoothing
        if spatial_smooth_sigma > 0:
            for i in range(t):
                mov[:, :, i] = gaussian_filter(mov[:, :, i], spatial_smooth_sigma)

        # Normalize by temporal std
        std_map = np.maximum(mov.std(axis=2), 1e-10)
        mov = mov / std_map[:, :, np.newaxis]

        # SVD via temporal covariance
        mov_flat = mov.reshape(h * w, t).T  # (t, h*w)
        cov = (mov_flat @ mov_flat.T) / mov_flat.shape[1]

        n_svd = min(n_components, t // 2)
        u_t, _, _ = np.linalg.svd(cov)
        u_t = u_t[:, :n_svd]

        # Project to get spatial loadings
        U_flat = u_t.T @ mov_flat
        U = U_flat.T.reshape(h, w, n_svd)

        return U.astype(np.float32), u_t.astype(np.float32)


class PrecomputedSVDDataSource(SVDDataSource):
    """SVD data source from pre-computed SVDVideo or HDF5 file."""

    def __init__(self, svd, n_components: Optional[int] = None,
                 tmin: Optional[int] = None, tmax: Optional[int] = None):
        """Initialize from SVDVideo.

        Args:
            svd: SVDVideo with U (t, k), S (k,), Vt (k, h, w)
            n_components: Max components to use. If None, uses all.
            tmin: Start frame index (inclusive). If None, starts at 0.
            tmax: End frame index (exclusive). If None, uses all frames.
        """
        U, S, Vt = svd.U, svd.S, svd.Vt
        if Vt.ndim != 3:
            raise ValueError(f"Expected 3D spatial components (k, h, w), got {svd.Vt.ndim}D")

        k_full, h, w = Vt.shape
        k = min(k_full, n_components) if n_components else k_full

        # Absorb S into spatial, transpose to (h, w, k)
        spatial = (Vt[:k] * S[:k, None, None]).transpose(1, 2, 0)
        temporal = U[tmin:tmax, :k]

        super().__init__(spatial, temporal)


class MeanImageDataSource:
    """Minimal data source from just a mean image (no temporal data).

    Useful for testing or when only static visualization is needed.
    """

    def __init__(self, mean_image: np.ndarray):
        if mean_image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {mean_image.shape}")
        self._mean = mean_image.astype(np.float32)
        self._shape = mean_image.shape

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def n_components(self) -> int:
        return 0

    def mean_image(self) -> np.ndarray:
        return self._mean

    def correlation_map(self, seed_row: int, seed_col: int) -> np.ndarray:
        # No temporal data, return zeros
        return np.zeros(self._shape, dtype=np.float32)

    def correlation_map_from_code(self, code: np.ndarray) -> np.ndarray:
        return np.zeros(self._shape, dtype=np.float32)

    def extract_code(self, footprint: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        return np.array([], dtype=np.float32)

    def extract_trace(self, footprint: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        return np.array([], dtype=np.float32)

    def get_pixel_codes(self, footprint: np.ndarray) -> np.ndarray:
        return np.zeros((len(footprint), 0), dtype=np.float32)

    def local_correlation_map(self) -> np.ndarray:
        # No temporal data, return zeros
        return np.zeros(self._shape, dtype=np.float32)
