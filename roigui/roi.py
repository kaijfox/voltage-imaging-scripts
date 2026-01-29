"""ROI dataclass and geometry utilities."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..timeseries.rois import ROI as BaseROI


def get_pixel_edges(r: int, c: int) -> tuple:
    """Return the 4 edges of pixel (r, c) as corner coordinate pairs.

    Corners of pixel (r, c): (r, c), (r, c+1), (r+1, c), (r+1, c+1)
    Returns edges as (start_corner, end_corner) tuples, normalized so start < end.
    """
    return (
        ((r, c), (r, c + 1)),      # top
        ((r + 1, c), (r + 1, c + 1)),  # bottom
        ((r, c), (r + 1, c)),      # left
        ((r, c + 1), (r + 1, c + 1)),  # right
    )


def compute_boundary_edges(pixel_set: set) -> set:
    """Compute all boundary edges from a set of pixel coordinates.

    An edge is on the boundary iff exactly one of its adjacent pixels is in the set.
    """
    edge_counts = {}
    for r, c in pixel_set:
        for edge in get_pixel_edges(r, c):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    # Boundary edges appear exactly once (interior edges appear twice)
    return {edge for edge, count in edge_counts.items() if count == 1}


def footprint_to_mask(footprint: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert footprint (n_pixels, 2) to boolean mask of given shape."""
    mask = np.zeros(shape, dtype=bool)
    if len(footprint) > 0:
        mask[footprint[:, 0], footprint[:, 1]] = True
    return mask


def mask_to_footprint(mask: np.ndarray) -> np.ndarray:
    """Convert boolean mask to footprint (n_pixels, 2)."""
    return np.argwhere(mask)


def footprint_to_crop_region(footprint: np.ndarray, padding: int = 0) -> tuple:
    """Get bounding box (r_min, r_max, c_min, c_max) for footprint with optional padding."""
    if len(footprint) == 0:
        return (0, 0, 0, 0)
    r_min, c_min = footprint.min(axis=0)
    r_max, c_max = footprint.max(axis=0)
    return (r_min - padding, r_max + 1 + padding, c_min - padding, c_max + 1 + padding)


@dataclass
class ROI(BaseROI):
    """ROI with cached pixel set for efficient GUI operations.

    Extends BaseROI with O(1) membership lookup via cached pixel_set.
    """
    _pixel_set_cache: Optional[set] = field(default=None, repr=False, compare=False)
    _footprint_hash: Optional[int] = field(default=None, repr=False, compare=False)

    @property
    def pixel_set(self) -> set:
        """Cached set of (row, col) tuples for O(1) membership lookup."""
        current_hash = hash(self.footprint.tobytes())
        if self._pixel_set_cache is None or self._footprint_hash != current_hash:
            self._pixel_set_cache = set(map(tuple, self.footprint))
            self._footprint_hash = current_hash
        return self._pixel_set_cache

    def invalidate_cache(self):
        """Call after modifying footprint to force cache rebuild."""
        self._pixel_set_cache = None
        self._footprint_hash = None

    @classmethod
    def from_mask(cls, mask: np.ndarray, weights: Optional[np.ndarray] = None,
                  code: Optional[np.ndarray] = None) -> "ROI":
        """Create ROI from boolean mask."""
        footprint = mask_to_footprint(mask)
        if weights is None:
            weights = np.ones(len(footprint))
        elif weights.shape == mask.shape:
            weights = weights[mask]
        if code is None:
            code = np.array([])
        return cls(footprint=footprint, weights=weights, code=code)

    @classmethod
    def empty(cls) -> "ROI":
        """Create an empty ROI."""
        return cls(
            footprint=np.zeros((0, 2), dtype=int),
            weights=np.zeros(0),
            code=np.array([])
        )


class ROIGeometry:
    """Efficient geometry operations on an ROI with incremental boundary tracking.

    Wraps an ROI and maintains boundary edges for fast outline rendering.
    """

    def __init__(self, roi: ROI):
        self.roi = roi
        self._boundary_edges = compute_boundary_edges(roi.pixel_set)

    @property
    def boundary_edges(self) -> set:
        """Current set of boundary edges."""
        return self._boundary_edges

    def add_pixel(self, r: int, c: int, weight: float = 1.0):
        """Add a pixel to the ROI. O(1) boundary update."""
        if (r, c) in self.roi.pixel_set:
            return

        # Update boundary edges (symmetric difference)
        edges = get_pixel_edges(r, c)
        self._boundary_edges.symmetric_difference_update(edges)

        # Update ROI arrays
        self.roi.footprint = np.vstack([self.roi.footprint, [[r, c]]])
        self.roi.weights = np.append(self.roi.weights, weight)
        self.roi.invalidate_cache()

    def remove_pixel(self, r: int, c: int):
        """Remove a pixel from the ROI. O(1) boundary update."""
        if (r, c) not in self.roi.pixel_set:
            return

        # Update boundary edges (symmetric difference)
        edges = get_pixel_edges(r, c)
        self._boundary_edges.symmetric_difference_update(edges)

        # Update ROI arrays - find and remove the pixel
        mask = ~((self.roi.footprint[:, 0] == r) & (self.roi.footprint[:, 1] == c))
        self.roi.footprint = self.roi.footprint[mask]
        self.roi.weights = self.roi.weights[mask]
        self.roi.invalidate_cache()

    def contains(self, r: int, c: int) -> bool:
        """Check if pixel is in ROI. O(1)."""
        return (r, c) in self.roi.pixel_set

    def rebuild_boundary(self):
        """Rebuild boundary edges from scratch (e.g., after bulk footprint change)."""
        self._boundary_edges = compute_boundary_edges(self.roi.pixel_set)


class RefineState:
    """State for refine mode with efficient boundary checkpointing.

    Holds pixels ranked by correlation with precomputed boundary checkpoints
    for efficient large slider jumps.

    The slider walks through ranked_pixels. Moving from index i to j:
    - If |j - i| is small, walk incrementally (O(|j-i|))
    - If |j - i| is large, snap to nearest checkpoint then walk (O(|j - nearest| + checkpoint_interval))
    """

    def __init__(self, ranked_pixels: list, correlations: np.ndarray,
                 initial_boundary: set, checkpoint_interval: int = 100):
        """Initialize refine state.

        Args:
            ranked_pixels: List of (r, c) tuples ordered by correlation (highest first)
            correlations: Array of correlation values matching ranked_pixels order
            initial_boundary: Boundary edges when all ranked_pixels are included
            checkpoint_interval: Build checkpoints every N pixels
        """
        self.ranked_pixels = ranked_pixels  # Highest correlation first
        self.correlations = correlations
        self.checkpoint_interval = checkpoint_interval
        self.n_pixels = len(ranked_pixels)

        # Start with all pixels included
        self._current_index = self.n_pixels

        # Build checkpoints: boundary state at indices 0, interval, 2*interval, ...
        # checkpoint[i] = boundary edges when first i pixels are included
        self._checkpoints = self._build_checkpoints(initial_boundary)

    def _build_checkpoints(self, full_boundary: set) -> dict:
        """Build boundary checkpoints by walking backwards from full ROI."""
        checkpoints = {self.n_pixels: full_boundary.copy()}

        # Walk backwards, removing pixels and saving checkpoints
        current_boundary = full_boundary.copy()
        current_pixels = set(self.ranked_pixels)

        for i in range(self.n_pixels - 1, -1, -1):
            # Remove pixel at index i
            r, c = self.ranked_pixels[i]
            edges = get_pixel_edges(r, c)
            current_boundary.symmetric_difference_update(edges)
            current_pixels.discard((r, c))

            # Save checkpoint at interval boundaries
            if i % self.checkpoint_interval == 0:
                checkpoints[i] = current_boundary.copy()

        return checkpoints

    @property
    def current_index(self) -> int:
        """Current number of pixels included (0 to n_pixels)."""
        return self._current_index

    @property
    def current_threshold(self) -> float:
        """Correlation threshold at current index."""
        if self._current_index == 0:
            return float('inf')
        if self._current_index >= self.n_pixels:
            return 0.0
        # Threshold is correlation of the last included pixel
        return self.correlations[self._current_index - 1]

    def set_index(self, target_index: int, roi_geometry: "ROIGeometry"):
        """Move to target index, updating ROI geometry.

        Uses checkpoints for large jumps.
        """
        target_index = max(0, min(self.n_pixels, target_index))

        if target_index == self._current_index:
            return

        delta = abs(target_index - self._current_index)

        # Decide whether to walk or snap to checkpoint
        # Find nearest checkpoint to target
        nearest_checkpoint = (target_index // self.checkpoint_interval) * self.checkpoint_interval
        if nearest_checkpoint > self.n_pixels:
            nearest_checkpoint = self.n_pixels

        walk_from_checkpoint = abs(target_index - nearest_checkpoint)
        walk_from_current = delta

        if walk_from_checkpoint + self.checkpoint_interval // 2 < walk_from_current:
            # Snap to checkpoint, then walk
            self._snap_to_checkpoint(nearest_checkpoint, roi_geometry)
            self._walk_to(target_index, roi_geometry)
        else:
            # Walk directly
            self._walk_to(target_index, roi_geometry)

    def _snap_to_checkpoint(self, checkpoint_index: int, roi_geometry: "ROIGeometry"):
        """Snap to a checkpoint, rebuilding ROI state."""
        if checkpoint_index not in self._checkpoints:
            # Find closest available checkpoint
            available = sorted(self._checkpoints.keys())
            checkpoint_index = min(available, key=lambda x: abs(x - checkpoint_index))

        # Rebuild ROI footprint from ranked pixels up to checkpoint
        included_pixels = self.ranked_pixels[:checkpoint_index]

        if len(included_pixels) == 0:
            roi_geometry.roi.footprint = np.zeros((0, 2), dtype=int)
            roi_geometry.roi.weights = np.zeros(0)
        else:
            roi_geometry.roi.footprint = np.array(included_pixels, dtype=int)
            roi_geometry.roi.weights = np.ones(len(included_pixels))

        roi_geometry.roi.invalidate_cache()
        roi_geometry._boundary_edges = self._checkpoints[checkpoint_index].copy()
        self._current_index = checkpoint_index

    def _walk_to(self, target_index: int, roi_geometry: "ROIGeometry"):
        """Walk incrementally from current index to target."""
        if target_index > self._current_index:
            # Adding pixels (lower correlation)
            for i in range(self._current_index, target_index):
                r, c = self.ranked_pixels[i]
                roi_geometry.add_pixel(r, c)
        else:
            # Removing pixels (higher correlation threshold)
            for i in range(self._current_index - 1, target_index - 1, -1):
                r, c = self.ranked_pixels[i]
                roi_geometry.remove_pixel(r, c)

        self._current_index = target_index

    @classmethod
    def from_roi_and_correlations(cls, roi: "ROI", pixel_correlations: np.ndarray,
                                  checkpoint_interval: int = 100) -> "RefineState":
        """Create RefineState from existing ROI and correlation values.

        Args:
            roi: Current ROI
            pixel_correlations: (n_pixels,) correlation of each pixel in roi.footprint
            checkpoint_interval: Checkpoint every N pixels
        """
        # Sort pixels by correlation (highest first)
        order = np.argsort(-pixel_correlations)
        ranked_pixels = [tuple(roi.footprint[i]) for i in order]
        sorted_correlations = pixel_correlations[order]

        # Compute initial boundary (all pixels included)
        initial_boundary = compute_boundary_edges(roi.pixel_set)

        return cls(ranked_pixels, sorted_correlations, initial_boundary, checkpoint_interval)
