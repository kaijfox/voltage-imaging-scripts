"""Control widgets for SVD viewer."""

from typing import TYPE_CHECKING, Optional, Tuple

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, Slot, Signal

if TYPE_CHECKING:
    from .state import ViewerState


class PlaybackWidget(QWidget):
    """Widget for frame navigation and playback control.

    Contains:
    - Frame slider
    - Frame number display
    - Play/Pause button
    - Step forward/backward buttons
    """

    play_clicked = Signal()
    pause_clicked = Signal()
    step_forward = Signal()
    step_backward = Signal()

    def __init__(self, state: "ViewerState", parent=None):
        super().__init__(parent)
        self._state = state
        self._updating = False  # Prevent signal loops

        self._setup_ui()
        self._connect_signals()
        self._update_from_state()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Frame slider
        slider_layout = QHBoxLayout()

        self._frame_label = QLabel("Frame:")
        slider_layout.addWidget(self._frame_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        slider_layout.addWidget(self._frame_slider, stretch=1)

        self._frame_spinbox = QSpinBox()
        self._frame_spinbox.setMinimum(0)
        self._frame_spinbox.setMaximum(0)
        self._frame_spinbox.setFixedWidth(80)
        slider_layout.addWidget(self._frame_spinbox)

        self._total_label = QLabel("/ 0")
        slider_layout.addWidget(self._total_label)

        layout.addLayout(slider_layout)

        # Playback controls
        controls_layout = QHBoxLayout()

        self._step_back_btn = QPushButton("<")
        self._step_back_btn.setFixedWidth(30)
        self._step_back_btn.setToolTip("Previous frame")
        controls_layout.addWidget(self._step_back_btn)

        self._play_btn = QPushButton("Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(60)
        controls_layout.addWidget(self._play_btn)

        self._step_fwd_btn = QPushButton(">")
        self._step_fwd_btn.setFixedWidth(30)
        self._step_fwd_btn.setToolTip("Next frame")
        controls_layout.addWidget(self._step_fwd_btn)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

    def _connect_signals(self):
        # Widget -> state
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        self._frame_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self._play_btn.clicked.connect(self._on_play_clicked)
        self._step_back_btn.clicked.connect(self.step_backward.emit)
        self._step_fwd_btn.clicked.connect(self.step_forward.emit)

        # State -> widget
        self._state.frame_changed.connect(self._on_state_frame_changed)

    def _update_from_state(self):
        """Update widget from current state."""
        n_frames = self._state.n_frames
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._frame_spinbox.setMaximum(max(0, n_frames - 1))
        self._total_label.setText(f"/ {n_frames}")

        self._updating = True
        self._frame_slider.setValue(self._state.current_frame)
        self._frame_spinbox.setValue(self._state.current_frame)
        self._updating = False

    @Slot(int)
    def _on_slider_changed(self, value: int):
        if not self._updating:
            self._updating = True
            self._frame_spinbox.setValue(value)
            self._state.set_current_frame(value)
            self._updating = False

    @Slot(int)
    def _on_spinbox_changed(self, value: int):
        if not self._updating:
            self._updating = True
            self._frame_slider.setValue(value)
            self._state.set_current_frame(value)
            self._updating = False

    @Slot()
    def _on_play_clicked(self):
        if self._play_btn.isChecked():
            self._play_btn.setText("Pause")
            self.play_clicked.emit()
        else:
            self._play_btn.setText("Play")
            self.pause_clicked.emit()

    @Slot(int)
    def _on_state_frame_changed(self, frame: int):
        if not self._updating:
            self._updating = True
            self._frame_slider.setValue(frame)
            self._frame_spinbox.setValue(frame)
            self._updating = False

    def set_playing(self, playing: bool):
        """Update play button state."""
        self._play_btn.setChecked(playing)
        self._play_btn.setText("Pause" if playing else "Play")

    def update_limits(self, n_frames: int):
        """Update frame limits when data source changes."""
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._frame_spinbox.setMaximum(max(0, n_frames - 1))
        self._total_label.setText(f"/ {n_frames}")


class RankWidget(QWidget):
    """Widget for truncation rank control."""

    def __init__(self, state: "ViewerState", parent=None):
        super().__init__(parent)
        self._state = state
        self._updating = False

        self._setup_ui()
        self._connect_signals()
        self._update_from_state()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        layout.addWidget(QLabel("Rank:"))

        self._rank_slider = QSlider(Qt.Orientation.Horizontal)
        self._rank_slider.setMinimum(1)
        self._rank_slider.setMaximum(100)
        layout.addWidget(self._rank_slider, stretch=1)

        self._rank_spinbox = QSpinBox()
        self._rank_spinbox.setMinimum(1)
        self._rank_spinbox.setMaximum(100)
        self._rank_spinbox.setFixedWidth(80)
        layout.addWidget(self._rank_spinbox)

        self._max_label = QLabel("/ 100")
        layout.addWidget(self._max_label)

    def _connect_signals(self):
        self._rank_slider.valueChanged.connect(self._on_slider_changed)
        self._rank_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self._state.rank_changed.connect(self._on_state_rank_changed)

    def _update_from_state(self):
        max_rank = self._state.max_rank if self._state.max_rank > 0 else 100
        self._rank_slider.setMaximum(max_rank)
        self._rank_spinbox.setMaximum(max_rank)
        self._max_label.setText(f"/ {max_rank}")

        self._updating = True
        rank = self._state.rank if self._state.rank > 0 else max_rank
        self._rank_slider.setValue(rank)
        self._rank_spinbox.setValue(rank)
        self._updating = False

    @Slot(int)
    def _on_slider_changed(self, value: int):
        if not self._updating:
            self._updating = True
            self._rank_spinbox.setValue(value)
            self._state.set_rank(value)
            self._updating = False

    @Slot(int)
    def _on_spinbox_changed(self, value: int):
        if not self._updating:
            self._updating = True
            self._rank_slider.setValue(value)
            self._state.set_rank(value)
            self._updating = False

    @Slot(int)
    def _on_state_rank_changed(self, rank: int):
        if not self._updating:
            self._updating = True
            self._rank_slider.setValue(rank)
            self._rank_spinbox.setValue(rank)
            self._updating = False

    def update_limits(self, max_rank: int):
        """Update rank limits when data source changes."""
        self._rank_slider.setMaximum(max_rank)
        self._rank_spinbox.setMaximum(max_rank)
        self._max_label.setText(f"/ {max_rank}")


class ROIWidget(QWidget):
    """Widget for spatial ROI control."""

    def __init__(self, state: "ViewerState", parent=None):
        super().__init__(parent)
        self._state = state
        self._updating = False

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        group = QGroupBox("Spatial ROI")
        group_layout = QVBoxLayout(group)

        # Row range
        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Rows:"))

        self._r0_spin = QSpinBox()
        self._r0_spin.setMinimum(0)
        self._r0_spin.setMaximum(9999)
        row_layout.addWidget(self._r0_spin)

        row_layout.addWidget(QLabel("to"))

        self._r1_spin = QSpinBox()
        self._r1_spin.setMinimum(0)
        self._r1_spin.setMaximum(9999)
        row_layout.addWidget(self._r1_spin)

        group_layout.addLayout(row_layout)

        # Col range
        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("Cols:"))

        self._c0_spin = QSpinBox()
        self._c0_spin.setMinimum(0)
        self._c0_spin.setMaximum(9999)
        col_layout.addWidget(self._c0_spin)

        col_layout.addWidget(QLabel("to"))

        self._c1_spin = QSpinBox()
        self._c1_spin.setMinimum(0)
        self._c1_spin.setMaximum(9999)
        col_layout.addWidget(self._c1_spin)

        group_layout.addLayout(col_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._clear_btn = QPushButton("Clear")
        btn_layout.addWidget(self._apply_btn)
        btn_layout.addWidget(self._clear_btn)
        group_layout.addLayout(btn_layout)

        layout.addWidget(group)

    def _connect_signals(self):
        self._apply_btn.clicked.connect(self._on_apply)
        self._clear_btn.clicked.connect(self._on_clear)
        self._state.roi_changed.connect(self._on_state_roi_changed)

    @Slot()
    def _on_apply(self):
        r0 = self._r0_spin.value()
        r1 = self._r1_spin.value()
        c0 = self._c0_spin.value()
        c1 = self._c1_spin.value()

        if r1 > r0 and c1 > c0:
            self._state.set_spatial_roi((r0, r1, c0, c1))

    @Slot()
    def _on_clear(self):
        self._state.set_spatial_roi(None)

    @Slot(object)
    def _on_state_roi_changed(self, roi: Optional[Tuple[int, int, int, int]]):
        if not self._updating:
            self._updating = True
            if roi is not None:
                r0, r1, c0, c1 = roi
                self._r0_spin.setValue(r0)
                self._r1_spin.setValue(r1)
                self._c0_spin.setValue(c0)
                self._c1_spin.setValue(c1)
            self._updating = False

    def update_limits(self, frame_shape: Tuple[int, ...]):
        """Update ROI limits when data source changes."""
        if len(frame_shape) >= 2:
            h, w = frame_shape[:2]
            self._r0_spin.setMaximum(h)
            self._r1_spin.setMaximum(h)
            self._r1_spin.setValue(h)
            self._c0_spin.setMaximum(w)
            self._c1_spin.setMaximum(w)
            self._c1_spin.setValue(w)


class FPSWidget(QWidget):
    """Widget for playback FPS control."""

    def __init__(self, state: "ViewerState", parent=None):
        super().__init__(parent)
        self._state = state
        self._updating = False

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        layout.addWidget(QLabel("FPS:"))

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setMinimum(1.0)
        self._fps_spin.setMaximum(120.0)
        self._fps_spin.setValue(30.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setSingleStep(5.0)
        self._fps_spin.setFixedWidth(70)
        layout.addWidget(self._fps_spin)

        layout.addStretch()

    def _connect_signals(self):
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        self._state.fps_changed.connect(self._on_state_fps_changed)

    @Slot(float)
    def _on_fps_changed(self, value: float):
        if not self._updating:
            self._state.set_fps(value)

    @Slot(float)
    def _on_state_fps_changed(self, fps: float):
        if not self._updating:
            self._updating = True
            self._fps_spin.setValue(fps)
            self._updating = False


class ControlPanel(QWidget):
    """Combined control panel containing all widgets."""

    play_clicked = Signal()
    pause_clicked = Signal()
    step_forward = Signal()
    step_backward = Signal()

    def __init__(self, state: "ViewerState", parent=None):
        super().__init__(parent)
        self._state = state

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Playback controls
        self.playback = PlaybackWidget(self._state)
        self.playback.play_clicked.connect(self.play_clicked.emit)
        self.playback.pause_clicked.connect(self.pause_clicked.emit)
        self.playback.step_forward.connect(self.step_forward.emit)
        self.playback.step_backward.connect(self.step_backward.emit)
        layout.addWidget(self.playback)

        # FPS control
        self.fps = FPSWidget(self._state)
        layout.addWidget(self.fps)

        # Rank control
        self.rank = RankWidget(self._state)
        layout.addWidget(self.rank)

        # ROI control
        self.roi = ROIWidget(self._state)
        layout.addWidget(self.roi)

        layout.addStretch()

    def update_from_data_source(self):
        """Update all widgets when data source changes."""
        self.playback.update_limits(self._state.n_frames)
        self.rank.update_limits(self._state.max_rank)
        self.roi.update_limits(self._state.frame_shape)
