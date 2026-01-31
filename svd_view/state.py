"""Application state with Qt signals for reactive updates."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from .data_source import LazyHDF5SVDSource


class PlaybackState(Enum):
    PAUSED = auto()
    PLAYING = auto()
    SCRUBBING = auto()


@dataclass
class TelemetryData:
    """Performance metrics from frame reconstruction."""
    io_latency_ms: float = 0.0
    compute_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    buffer_fill: int = 0
    buffer_capacity: int = 30


class ViewerState(QObject):
    """Central application state with signals for reactive UI updates.

    Components connect to signals they care about:
        state.frame_changed.connect(self.on_frame_changed)
        state.rank_changed.connect(self.on_rank_changed)
    """

    # Signals
    frame_changed = Signal(int)  # Current frame index changed
    rank_changed = Signal(int)  # Truncation rank changed
    roi_changed = Signal(object)  # Spatial window (r0, r1, c0, c1) or None
    playback_state_changed = Signal(object)  # PlaybackState enum
    frame_ready = Signal(int, object)  # Worker delivers frame (index, ndarray)
    telemetry_updated = Signal(object)  # TelemetryData
    fps_changed = Signal(float)  # Playback FPS changed

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._data_source: Optional["LazyHDF5SVDSource"] = None
        self._current_frame: int = 0
        self._rank: int = 0  # 0 means use full rank
        self._spatial_roi: Optional[Tuple[int, int, int, int]] = None  # (r0, r1, c0, c1)
        self._playback_state: PlaybackState = PlaybackState.PAUSED
        self._fps: float = 30.0
        self._telemetry: TelemetryData = TelemetryData()

        # Cached metadata from data source
        self._n_frames: int = 0
        self._max_rank: int = 0
        self._frame_shape: Tuple[int, ...] = ()

    # Data source
    @property
    def data_source(self) -> Optional["LazyHDF5SVDSource"]:
        return self._data_source

    def set_data_source(self, source: "LazyHDF5SVDSource"):
        """Set the data source and initialize metadata."""
        self._data_source = source
        self._n_frames = source.n_frames
        self._max_rank = source.max_rank
        self._frame_shape = source.frame_shape

        # Reset to valid defaults
        if self._rank == 0 or self._rank > self._max_rank:
            self._rank = self._max_rank
        if self._current_frame >= self._n_frames:
            self._current_frame = 0

    # Frame index
    @property
    def current_frame(self) -> int:
        return self._current_frame

    def set_current_frame(self, frame: int, emit: bool = True):
        """Set current frame index, clamping to valid range."""
        frame = max(0, min(frame, self._n_frames - 1)) if self._n_frames > 0 else 0
        if self._current_frame != frame:
            self._current_frame = frame
            if emit:
                self.frame_changed.emit(frame)

    @property
    def n_frames(self) -> int:
        return self._n_frames

    # Rank
    @property
    def rank(self) -> int:
        return self._rank

    def set_rank(self, rank: int, emit: bool = True):
        """Set truncation rank, clamping to valid range."""
        rank = max(1, min(rank, self._max_rank)) if self._max_rank > 0 else 1
        if self._rank != rank:
            self._rank = rank
            if emit:
                self.rank_changed.emit(rank)

    @property
    def max_rank(self) -> int:
        return self._max_rank

    # Spatial ROI
    @property
    def spatial_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._spatial_roi

    def set_spatial_roi(self, roi: Optional[Tuple[int, int, int, int]], emit: bool = True):
        """Set spatial ROI as (r0, r1, c0, c1) or None for full frame."""
        if self._spatial_roi != roi:
            self._spatial_roi = roi
            if emit:
                self.roi_changed.emit(roi)

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._frame_shape

    # Playback state
    @property
    def playback_state(self) -> PlaybackState:
        return self._playback_state

    def set_playback_state(self, state: PlaybackState, emit: bool = True):
        if self._playback_state != state:
            self._playback_state = state
            if emit:
                self.playback_state_changed.emit(state)

    @property
    def is_playing(self) -> bool:
        return self._playback_state == PlaybackState.PLAYING

    @property
    def is_scrubbing(self) -> bool:
        return self._playback_state == PlaybackState.SCRUBBING

    # FPS
    @property
    def fps(self) -> float:
        return self._fps

    def set_fps(self, fps: float, emit: bool = True):
        fps = max(1.0, min(fps, 120.0))
        if self._fps != fps:
            self._fps = fps
            if emit:
                self.fps_changed.emit(fps)

    # Telemetry
    @property
    def telemetry(self) -> TelemetryData:
        return self._telemetry

    def update_telemetry(self, data: TelemetryData, emit: bool = True):
        self._telemetry = data
        if emit:
            self.telemetry_updated.emit(data)

    # Convenience methods
    def notify_frame_ready(self, frame_idx: int, frame_data: np.ndarray):
        """Called by worker when a frame is ready."""
        self.frame_ready.emit(frame_idx, frame_data)

    def next_frame(self) -> int:
        """Return next frame index (wrapping)."""
        return (self._current_frame + 1) % self._n_frames if self._n_frames > 0 else 0

    def prev_frame(self) -> int:
        """Return previous frame index (wrapping)."""
        return (self._current_frame - 1) % self._n_frames if self._n_frames > 0 else 0
