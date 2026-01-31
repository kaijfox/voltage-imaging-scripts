"""Frame reconstruction worker using napari threading."""

import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from napari.qt.threading import thread_worker

if TYPE_CHECKING:
    from .data_source import LazyHDF5SVDSource


@dataclass
class FrameRequest:
    """Request for frame reconstruction."""
    frame_idx: int
    rank: int
    spatial_roi: Optional[Tuple[int, int, int, int]]
    priority: int = 0  # Higher = more important (for scrubbing)
    prefetch: bool = False  # True if this is a prefetch request


@dataclass
class FrameResult:
    """Result of frame reconstruction."""
    frame_idx: int
    rank: int
    spatial_roi: Optional[Tuple[int, int, int, int]]
    frame_data: np.ndarray
    io_latency_ms: float
    compute_time_ms: float


class FrameWorkerController:
    """Controller for the frame reconstruction worker.

    Manages the request queue and provides an interface for the main thread
    to submit frame requests. The actual worker runs in a separate thread.

    Usage:
        controller = FrameWorkerController(data_source)
        controller.start(on_result_callback)

        # Request a frame
        controller.request_frame(frame_idx, rank, roi)

        # Clear pending requests (e.g., when scrubbing)
        controller.clear_queue()

        # Stop worker
        controller.stop()
    """

    def __init__(self, data_source: "LazyHDF5SVDSource"):
        self._data_source = data_source
        self._request_queue: Queue[Optional[FrameRequest]] = Queue()
        self._worker = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, on_result):
        """Start the worker thread.

        Parameters
        ----------
        on_result : callable
            Callback invoked with FrameResult when a frame is ready.
            Called from the main Qt thread.
        """
        if self._running:
            return

        self._running = True

        @thread_worker(connect={"yielded": on_result})
        def _worker_loop():
            while self._running:
                try:
                    request = self._request_queue.get(timeout=0.1)
                except Empty:
                    continue

                if request is None:
                    # Poison pill - stop worker
                    break

                # Reconstruct frame with timing
                t0 = time.perf_counter()
                frame = self._data_source.reconstruct_frame(
                    request.frame_idx,
                    rank=request.rank,
                    spatial_roi=request.spatial_roi,
                )
                t1 = time.perf_counter()

                # We can't easily separate I/O from compute in this design,
                # so report total time as compute time
                result = FrameResult(
                    frame_idx=request.frame_idx,
                    rank=request.rank,
                    spatial_roi=request.spatial_roi,
                    frame_data=frame,
                    io_latency_ms=0.0,  # Embedded in compute
                    compute_time_ms=(t1 - t0) * 1000,
                )

                yield result

        self._worker = _worker_loop()

    def stop(self):
        """Stop the worker thread."""
        if not self._running:
            return

        self._running = False
        # Send poison pill
        self._request_queue.put(None)
        self._worker = None

    def request_frame(
        self,
        frame_idx: int,
        rank: int,
        spatial_roi: Optional[Tuple[int, int, int, int]],
        priority: int = 0,
        prefetch: bool = False,
    ):
        """Submit a frame reconstruction request.

        Parameters
        ----------
        frame_idx : int
            Frame index.
        rank : int
            Truncation rank.
        spatial_roi : tuple or None
            Spatial ROI (r0, r1, c0, c1) or None.
        priority : int
            Request priority (higher = more important).
        prefetch : bool
            True if this is a prefetch request.
        """
        if not self._running:
            return

        request = FrameRequest(
            frame_idx=frame_idx,
            rank=rank,
            spatial_roi=spatial_roi,
            priority=priority,
            prefetch=prefetch,
        )
        self._request_queue.put(request)

    def request_frames(
        self,
        frame_indices: list,
        rank: int,
        spatial_roi: Optional[Tuple[int, int, int, int]],
        prefetch: bool = False,
    ):
        """Submit multiple frame reconstruction requests.

        Parameters
        ----------
        frame_indices : list of int
            Frame indices.
        rank : int
            Truncation rank.
        spatial_roi : tuple or None
            Spatial ROI.
        prefetch : bool
            True if these are prefetch requests.
        """
        for frame_idx in frame_indices:
            self.request_frame(
                frame_idx,
                rank,
                spatial_roi,
                priority=0,
                prefetch=prefetch,
            )

    def clear_queue(self):
        """Clear all pending requests.

        Call this when the user scrubs to a new position to prioritize
        the new frame over queued prefetch requests.
        """
        try:
            while True:
                self._request_queue.get_nowait()
        except Empty:
            pass

    @property
    def queue_size(self) -> int:
        """Number of pending requests."""
        return self._request_queue.qsize()
