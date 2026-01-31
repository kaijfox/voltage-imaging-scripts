"""Main SVD Viewer class integrating all components."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PySide6.QtCore import QTimer, Slot

from .buffer import FrameBuffer
from .data_source import LazyHDF5SVDSource
from .state import PlaybackState, TelemetryData, ViewerState
from .telemetry import TelemetryWidget
from .widgets import ControlPanel
from .worker import FrameResult, FrameWorkerController


class SVDViewer:
    """Napari-based viewer for HDF5 SVD video files.

    Reconstructs frames on-the-fly via U @ diag(S) @ Vh with async worker thread.

    Usage:
        # From HDF5 file
        viewer = SVDViewer("/path/to/svd.h5")
        viewer.show()

        # With existing napari viewer
        viewer = SVDViewer("/path/to/svd.h5", napari_viewer=existing_viewer)

        # With initial parameters
        viewer = SVDViewer(path, rank=50, spatial_roi=(100, 200, 100, 200))
    """

    def __init__(
        self,
        source: Union[str, Path, LazyHDF5SVDSource],
        napari_viewer=None,
        rank: Optional[int] = None,
        spatial_roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Initialize the SVD viewer.

        Parameters
        ----------
        source : str, Path, or LazyHDF5SVDSource
            Path to HDF5 file or existing data source.
        napari_viewer : napari.Viewer, optional
            Existing napari viewer to use. If None, creates a new one.
        rank : int, optional
            Initial truncation rank. If None, uses full rank.
        spatial_roi : tuple, optional
            Initial spatial ROI as (r0, r1, c0, c1).
        """
        import napari

        # Initialize data source
        if isinstance(source, (str, Path)):
            self._data_source = LazyHDF5SVDSource(source)
            self._owns_data_source = True
        else:
            self._data_source = source
            self._owns_data_source = False

        # Initialize state
        self._state = ViewerState()
        self._state.set_data_source(self._data_source)

        if rank is not None:
            self._state.set_rank(rank, emit=False)
        if spatial_roi is not None:
            self._state.set_spatial_roi(spatial_roi, emit=False)

        # Initialize frame buffer
        self._buffer = FrameBuffer(capacity=30)

        # Initialize worker
        self._worker = FrameWorkerController(self._data_source)

        # Create or use napari viewer
        if napari_viewer is None:
            self._viewer = napari.Viewer(title="SVD Video Viewer")
            self._owns_viewer = True
        else:
            self._viewer = napari_viewer
            self._owns_viewer = False

        # Add image layer
        self._image_layer = None
        self._setup_image_layer()

        # Setup widgets
        self._setup_widgets()

        # Setup playback timer
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._on_playback_tick)

        # Connect state signals
        self._connect_signals()

        # Start worker
        self._worker.start(self._on_frame_result)

        # Request initial frame
        self._request_current_frame()

    def _setup_image_layer(self):
        """Create the napari image layer."""
        # Get initial frame shape
        roi = self._state.spatial_roi
        if roi is not None:
            r0, r1, c0, c1 = roi
            shape = (r1 - r0, c1 - c0)
        else:
            shape = self._state.frame_shape

        # Create placeholder
        placeholder = np.zeros(shape, dtype=np.float32)
        self._image_layer = self._viewer.add_image(
            placeholder,
            name="SVD Video",
            colormap="gray",
        )

    def _setup_widgets(self):
        """Create and dock control widgets."""
        # Control panel
        self._control_panel = ControlPanel(self._state)
        self._control_panel.play_clicked.connect(self._on_play)
        self._control_panel.pause_clicked.connect(self._on_pause)
        self._control_panel.step_forward.connect(self._on_step_forward)
        self._control_panel.step_backward.connect(self._on_step_backward)
        self._control_panel.update_from_data_source()

        self._viewer.window.add_dock_widget(
            self._control_panel,
            name="Controls",
            area="right",
        )

        # Telemetry widget
        self._telemetry_widget = TelemetryWidget(self._state)
        self._viewer.window.add_dock_widget(
            self._telemetry_widget,
            name="Performance",
            area="right",
        )

    def _connect_signals(self):
        """Connect state signals to handlers."""
        self._state.frame_changed.connect(self._on_frame_changed)
        self._state.rank_changed.connect(self._on_params_changed)
        self._state.roi_changed.connect(self._on_roi_changed)
        self._state.frame_ready.connect(self._on_frame_ready)
        self._state.fps_changed.connect(self._on_fps_changed)

    @Slot(int)
    def _on_frame_changed(self, frame_idx: int):
        """Handle frame index change."""
        self._request_current_frame()

    @Slot(int)
    def _on_params_changed(self, _):
        """Handle rank change - invalidate cache and request new frame."""
        self._buffer.invalidate_for_params(rank=self._state.rank)
        self._request_current_frame()

    @Slot(object)
    def _on_roi_changed(self, roi):
        """Handle ROI change - invalidate cache, resize layer, request new frame."""
        self._buffer.invalidate_for_params(spatial_roi=roi)

        # Resize image layer placeholder
        if roi is not None:
            r0, r1, c0, c1 = roi
            shape = (r1 - r0, c1 - c0)
        else:
            shape = self._state.frame_shape

        placeholder = np.zeros(shape, dtype=np.float32)
        self._image_layer.data = placeholder

        self._request_current_frame()

    @Slot(float)
    def _on_fps_changed(self, fps: float):
        """Handle FPS change."""
        if self._playback_timer.isActive():
            interval_ms = int(1000 / fps)
            self._playback_timer.setInterval(interval_ms)

    def _request_current_frame(self):
        """Request current frame from cache or worker."""
        frame_idx = self._state.current_frame
        rank = self._state.rank
        roi = self._state.spatial_roi

        # Check cache first
        cached = self._buffer.get(frame_idx, rank, roi)
        if cached is not None:
            self._display_frame(cached)
            self._trigger_prefetch()
            return

        # Request from worker
        self._worker.request_frame(frame_idx, rank, roi, priority=10)

    def _trigger_prefetch(self):
        """Request prefetch of upcoming frames if playing."""
        if not self._state.is_playing:
            return

        prefetch = self._buffer.get_prefetch_requests(
            self._state.current_frame,
            self._state.rank,
            self._state.spatial_roi,
            self._state.n_frames,
            n_ahead=5,
        )
        self._worker.request_frames(
            prefetch,
            self._state.rank,
            self._state.spatial_roi,
            prefetch=True,
        )

    def _on_frame_result(self, result: FrameResult):
        """Handle frame result from worker (called from main thread)."""
        # Store in buffer
        self._buffer.put(
            result.frame_idx,
            result.rank,
            result.spatial_roi,
            result.frame_data,
        )

        # Update telemetry
        telemetry = TelemetryData(
            io_latency_ms=result.io_latency_ms,
            compute_time_ms=result.compute_time_ms,
            memory_usage_mb=self._buffer.memory_usage_mb(),
            buffer_fill=self._buffer.fill_level,
            buffer_capacity=self._buffer.capacity,
        )
        self._state.update_telemetry(telemetry)

        # Display if this is the current frame
        if (
            result.frame_idx == self._state.current_frame
            and result.rank == self._state.rank
            and result.spatial_roi == self._state.spatial_roi
        ):
            self._state.notify_frame_ready(result.frame_idx, result.frame_data)

    @Slot(int, object)
    def _on_frame_ready(self, frame_idx: int, frame_data: np.ndarray):
        """Handle frame ready signal from state."""
        self._display_frame(frame_data)
        self._trigger_prefetch()

    def _display_frame(self, frame_data: np.ndarray):
        """Display frame in napari layer."""
        self._image_layer.data = frame_data

    @Slot()
    def _on_play(self):
        """Start playback."""
        self._state.set_playback_state(PlaybackState.PLAYING)
        interval_ms = int(1000 / self._state.fps)
        self._playback_timer.start(interval_ms)
        self._control_panel.playback.set_playing(True)

    @Slot()
    def _on_pause(self):
        """Pause playback."""
        self._state.set_playback_state(PlaybackState.PAUSED)
        self._playback_timer.stop()
        self._control_panel.playback.set_playing(False)

    @Slot()
    def _on_step_forward(self):
        """Step to next frame."""
        self._state.set_current_frame(self._state.next_frame())

    @Slot()
    def _on_step_backward(self):
        """Step to previous frame."""
        self._state.set_current_frame(self._state.prev_frame())

    @Slot()
    def _on_playback_tick(self):
        """Handle playback timer tick."""
        # Try to get next frame from cache
        next_frame = self._state.next_frame()
        rank = self._state.rank
        roi = self._state.spatial_roi

        cached = self._buffer.get(next_frame, rank, roi)
        if cached is not None:
            # Advance frame
            self._state.set_current_frame(next_frame)
        else:
            # Frame not ready - request it with high priority
            self._worker.clear_queue()
            self._worker.request_frame(next_frame, rank, roi, priority=10)

    def show(self):
        """Show the napari viewer."""
        # napari.run() blocks, so just ensure viewer is visible
        pass

    def close(self):
        """Close the viewer and clean up resources."""
        self._playback_timer.stop()
        self._worker.stop()

        if self._owns_data_source:
            self._data_source.close()

        if self._owns_viewer:
            self._viewer.close()

    @property
    def viewer(self):
        """The napari viewer instance."""
        return self._viewer

    @property
    def state(self) -> ViewerState:
        """The viewer state."""
        return self._state

    @property
    def data_source(self) -> LazyHDF5SVDSource:
        """The data source."""
        return self._data_source
