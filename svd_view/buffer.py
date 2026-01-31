"""LRU frame buffer with look-ahead prefetching."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np


@dataclass
class CacheKey:
    """Key for cached frames, capturing all reconstruction parameters."""
    frame_idx: int
    rank: int
    spatial_roi: Optional[Tuple[int, int, int, int]]

    def __hash__(self):
        return hash((self.frame_idx, self.rank, self.spatial_roi))

    def __eq__(self, other):
        if not isinstance(other, CacheKey):
            return False
        return (
            self.frame_idx == other.frame_idx
            and self.rank == other.rank
            and self.spatial_roi == other.spatial_roi
        )


class FrameBuffer:
    """LRU cache for reconstructed frames with prefetch support.

    Features:
    - LRU eviction policy
    - Configurable capacity
    - Parameter-aware caching (rank, ROI)
    - Prefetch request generation for look-ahead

    Usage:
        buffer = FrameBuffer(capacity=30)

        # Try to get frame from cache
        frame = buffer.get(frame_idx, rank, roi)
        if frame is None:
            # Request from worker
            ...

        # Store frame when ready
        buffer.put(frame_idx, rank, roi, frame_data)

        # Get prefetch requests for playback
        requests = buffer.get_prefetch_requests(current_frame, rank, roi, n_ahead=5)
    """

    def __init__(self, capacity: int = 30):
        """Initialize frame buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of frames to cache.
        """
        self._capacity = capacity
        self._cache: OrderedDict[CacheKey, np.ndarray] = OrderedDict()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def fill_level(self) -> int:
        """Number of frames currently in cache."""
        return len(self._cache)

    def get(
        self,
        frame_idx: int,
        rank: int,
        spatial_roi: Optional[Tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        """Get frame from cache if available.

        Parameters
        ----------
        frame_idx : int
            Frame index.
        rank : int
            Truncation rank used for reconstruction.
        spatial_roi : tuple or None
            Spatial ROI (r0, r1, c0, c1) or None.

        Returns
        -------
        frame : ndarray or None
            Cached frame or None if not in cache.
        """
        key = CacheKey(frame_idx, rank, spatial_roi)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(
        self,
        frame_idx: int,
        rank: int,
        spatial_roi: Optional[Tuple[int, int, int, int]],
        frame_data: np.ndarray,
    ):
        """Store frame in cache.

        Parameters
        ----------
        frame_idx : int
            Frame index.
        rank : int
            Truncation rank used for reconstruction.
        spatial_roi : tuple or None
            Spatial ROI (r0, r1, c0, c1) or None.
        frame_data : ndarray
            Reconstructed frame data.
        """
        key = CacheKey(frame_idx, rank, spatial_roi)

        # If already cached, update and move to end
        if key in self._cache:
            self._cache[key] = frame_data
            self._cache.move_to_end(key)
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)

        # Insert new entry
        self._cache[key] = frame_data

    def has(
        self,
        frame_idx: int,
        rank: int,
        spatial_roi: Optional[Tuple[int, int, int, int]],
    ) -> bool:
        """Check if frame is in cache without updating LRU order."""
        key = CacheKey(frame_idx, rank, spatial_roi)
        return key in self._cache

    def get_prefetch_requests(
        self,
        current_frame: int,
        rank: int,
        spatial_roi: Optional[Tuple[int, int, int, int]],
        n_frames: int,
        n_ahead: int = 5,
    ) -> list:
        """Get list of frames to prefetch for smooth playback.

        Parameters
        ----------
        current_frame : int
            Current frame index.
        rank : int
            Truncation rank.
        spatial_roi : tuple or None
            Spatial ROI.
        n_frames : int
            Total number of frames (for wrapping).
        n_ahead : int
            Number of frames to look ahead.

        Returns
        -------
        requests : list of int
            Frame indices to prefetch (not in cache).
        """
        requests = []
        for i in range(1, n_ahead + 1):
            frame_idx = (current_frame + i) % n_frames if n_frames > 0 else 0
            if not self.has(frame_idx, rank, spatial_roi):
                requests.append(frame_idx)
        return requests

    def invalidate_for_params(
        self,
        rank: Optional[int] = None,
        spatial_roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Invalidate cached frames that don't match new parameters.

        Call this when rank or ROI changes to clear stale cached frames.

        Parameters
        ----------
        rank : int, optional
            New rank. If provided, evict frames with different rank.
        spatial_roi : tuple or None, optional
            New ROI. If provided, evict frames with different ROI.
        """
        keys_to_remove = []
        for key in self._cache:
            if rank is not None and key.rank != rank:
                keys_to_remove.append(key)
            elif spatial_roi is not None and key.spatial_roi != spatial_roi:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

    def clear(self):
        """Clear all cached frames."""
        self._cache.clear()

    def memory_usage_mb(self) -> float:
        """Estimate memory usage of cached frames in MB."""
        total_bytes = sum(
            frame.nbytes for frame in self._cache.values()
        )
        return total_bytes / (1024 * 1024)
