"""Performance telemetry display widget."""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox
from PySide6.QtCore import Slot

if TYPE_CHECKING:
    from .state import ViewerState, TelemetryData


class TelemetryWidget(QWidget):
    """Widget displaying real-time performance metrics.

    Displays:
    - I/O latency (ms)
    - Compute time (ms)
    - Memory usage (MB)
    - Buffer fill level
    """

    def __init__(self, state: "ViewerState", parent=None):
        super().__init__(parent)
        self._state = state

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Performance group
        group = QGroupBox("Performance")
        group_layout = QVBoxLayout(group)

        self._io_label = QLabel("I/O: -- ms")
        self._compute_label = QLabel("Compute: -- ms")
        self._memory_label = QLabel("Memory: -- MB")
        self._buffer_label = QLabel("Buffer: 0/30")

        for label in [
            self._io_label,
            self._compute_label,
            self._memory_label,
            self._buffer_label,
        ]:
            group_layout.addWidget(label)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self):
        self._state.telemetry_updated.connect(self._on_telemetry_updated)

    @Slot(object)
    def _on_telemetry_updated(self, data: "TelemetryData"):
        """Update display with new telemetry data."""
        self._io_label.setText(f"I/O: {data.io_latency_ms:.1f} ms")
        self._compute_label.setText(f"Compute: {data.compute_time_ms:.1f} ms")
        self._memory_label.setText(f"Memory: {data.memory_usage_mb:.1f} MB")
        self._buffer_label.setText(f"Buffer: {data.buffer_fill}/{data.buffer_capacity}")
