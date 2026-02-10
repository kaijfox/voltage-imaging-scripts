"""View panel widgets extracted from main_window: MapDetailWidget and related helpers."""
from typing import Optional
import numpy as np
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider
from PySide6.QtCore import Qt
import pyqtgraph as pg

from .state import AppState, ViewMode
from .compositing import COLOR_MAP


class MapDetailWidget(QFrame):
    """Controls for the active map layer: color, range, gamma, intensity."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self.setMinimumHeight(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Color selector
        row = QHBoxLayout()
        label = QLabel("Color")
        label.setStyleSheet("color: #aaa;")
        row.addWidget(label)

        self.color_combo = QComboBox()
        for k in COLOR_MAP.keys():
            self.color_combo.addItem(k)
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        row.addWidget(self.color_combo)
        row.addStretch()
        layout.addLayout(row)

        # Histogram plot (zoomable/pannable x-axis)
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setFixedHeight(60)
        self.hist_plot.setBackground("#1a1a1a")
        self.hist_plot.hideAxis("left")
        self.hist_plot.hideAxis("bottom")
        self.hist_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self.hist_curve = None
        self.hist_fill = None
        layout.addWidget(self.hist_plot)

        # Vertical lo/hi lines
        self.lo_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 150), width=1))
        self.hi_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 150), width=1))
        self.hist_plot.addItem(self.lo_line)
        self.hist_plot.addItem(self.hi_line)
        self.lo_line.hide()
        self.hi_line.hide()

        # Track visible x range for slider mapping
        self._slider_xmin: float = 0.0
        self._slider_xmax: float = 1.0
        self.hist_plot.getViewBox().sigRangeChanged.connect(self._on_hist_range_changed)

        # Lo slider row (L)
        lrow = QHBoxLayout()
        llabel = QLabel("L")
        llabel.setStyleSheet("color: #aaa;")
        lrow.addWidget(llabel)
        self.lo_slider = QSlider(Qt.Horizontal)
        self.lo_slider.setRange(0, 1000)
        self.lo_slider.valueChanged.connect(self._on_lo_changed)
        lrow.addWidget(self.lo_slider)
        layout.addLayout(lrow)

        # Hi slider row (H)
        hrow = QHBoxLayout()
        hlabel = QLabel("H")
        hlabel.setStyleSheet("color: #aaa;")
        hrow.addWidget(hlabel)
        self.hi_slider = QSlider(Qt.Horizontal)
        self.hi_slider.setRange(0, 1000)
        self.hi_slider.valueChanged.connect(self._on_hi_changed)
        hrow.addWidget(self.hi_slider)
        layout.addLayout(hrow)

        # Gamma row (G) — now controls target geometric mean (0.01–0.99)
        grow = QHBoxLayout()
        glabel = QLabel("G")
        glabel.setStyleSheet("color: #aaa;")
        grow.addWidget(glabel)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 99)  # maps to 0.01–0.99 target gmean
        self.gamma_slider.setValue(50)
        self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        # Install event filter to capture right-clicks
        self.gamma_slider.installEventFilter(self)
        grow.addWidget(self.gamma_slider)
        layout.addLayout(grow)

        # Intensity row (I)
        irow = QHBoxLayout()
        ilabel = QLabel("I")
        ilabel.setStyleSheet("color: #aaa;")
        irow.addWidget(ilabel)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 200)  # maps to 0.0 - 2.0
        self.intensity_slider.setValue(100)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        irow.addWidget(self.intensity_slider)
        layout.addLayout(irow)

        # Cached raw image for slider→absolute value conversions
        self._current_raw: Optional[np.ndarray] = None
        self._current_raw_mn: float = 0.0
        self._current_raw_mx: float = 1.0

        # Connect to state signals
        # _repopulate only on active map switch (not on every slider-driven layer change)
        self.state.active_detail_map_changed.connect(self._repopulate)
        # Update lines when the underlying layer parameters change
        self.state.map_layer_changed.connect(self._on_map_layer_changed)

        # Initial populate
        self._repopulate()

    def eventFilter(self, obj, event):
        # Capture right-clicks on gamma slider to toggle gamma_enabled
        try:
            from PySide6.QtCore import QEvent
            from PySide6.QtGui import Qt as QGQt
        except Exception:
            QEvent = None
            QGQt = None
        if obj is self.gamma_slider and event is not None and hasattr(event, 'type'):
            if QEvent is not None and event.type() == QEvent.MouseButtonPress:
                # Use button() attribute for mouse press
                btn = getattr(event, 'button', lambda: None)()
                if btn == QGQt.RightButton:
                    # Toggle gamma_enabled
                    mode = self.state.active_detail_map
                    layer = self.state.map_layers.get(mode)
                    if layer is not None:
                        new = not getattr(layer, 'gamma_enabled', True)
                        self.state.update_map_layer(mode, gamma_enabled=new)
                        # Update visual immediately
                        self._update_gamma_visual(layer=layer)
                        return True
        return super().eventFilter(obj, event)

    def _update_gamma_visual(self, layer=None):
        # Adjust slider appearance to look dimmed when gamma disabled
        mode = self.state.active_detail_map
        if layer is None:
            layer = self.state.map_layers.get(mode)
        enabled = True
        if layer is not None:
            enabled = getattr(layer, 'gamma_enabled', True)
        if not enabled:
            # Dim via stylesheet (keep enabled so events still arrive)
            self.gamma_slider.setStyleSheet('QSlider { background: #2e2e2e; } QSlider::groove:horizontal { background: #333; } QSlider::handle:horizontal { background: #888; }')
            self.gamma_slider.setToolTip('Gamma disabled (right-click to toggle)')
        else:
            self.gamma_slider.setStyleSheet('')
            self.gamma_slider.setToolTip('Right-click to toggle gamma on/off')

    def _repopulate(self):
        """Load all controls from state for active_detail_map."""
        mode = self.state.active_detail_map
        layer = self.state.map_layers.get(mode)
        raw = self.state.get_raw_map(mode) if self.state.data_source is not None else None
        self._current_raw = raw

        if raw is not None:
            mn, mx = float(np.nanmin(raw)), float(np.nanmax(raw))
            if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                mn, mx = 0.0, 1.0
            self._current_raw_mn = mn
            self._current_raw_mx = mx
        else:
            self._current_raw_mn = 0.0
            self._current_raw_mx = 1.0

        if layer is not None:
            # Color
            idx = list(COLOR_MAP.keys()).index(layer.color) if layer.color in COLOR_MAP else 0
            self.color_combo.blockSignals(True)
            self.color_combo.setCurrentIndex(idx)
            self.color_combo.blockSignals(False)

            # Lo/Hi: initialize from raw range if uninitialized
            if raw is not None:
                mn, mx = self._current_raw_mn, self._current_raw_mx
                if (layer.lo >= layer.hi) or (layer.lo == 0.0 and layer.hi == 1.0):
                    self.state.update_map_layer(mode, lo=mn, hi=mx)
                    layer = self.state.map_layers[mode]  # re-read after update

                # Initialize slider range to full data range (will update on zoom/pan)
                self._slider_xmin = mn
                self._slider_xmax = mx

                span = mx - mn
                if span > 1e-12:
                    lo_pos = int(1000 * ((layer.lo - mn) / span))
                    hi_pos = int(1000 * ((layer.hi - mn) / span))
                else:
                    lo_pos, hi_pos = 0, 1000

                self.lo_slider.blockSignals(True)
                self.hi_slider.blockSignals(True)
                self.lo_slider.setValue(max(0, min(1000, lo_pos)))
                self.hi_slider.setValue(max(0, min(1000, hi_pos)))
                self.lo_slider.blockSignals(False)
                self.hi_slider.blockSignals(False)
            else:
                self.lo_slider.blockSignals(True)
                self.hi_slider.blockSignals(True)
                self.lo_slider.setValue(0)
                self.hi_slider.setValue(1000)
                self.lo_slider.blockSignals(False)
                self.hi_slider.blockSignals(False)

            # Gamma (target gmean: 0.01–0.99 mapped to slider 1–99)
            self.gamma_slider.blockSignals(True)
            self.gamma_slider.setValue(int(max(1, min(99, layer.gamma * 100))))
            self.gamma_slider.blockSignals(False)

            # Intensity
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(int(max(0, min(200, layer.intensity * 100))))
            self.intensity_slider.blockSignals(False)

        # Update histogram from cache
        self._update_histogram()
        # Ensure range lines reflect current layer after histogram autoscale
        self._update_range_lines()
        # Update gamma visual
        self._update_gamma_visual()

    def _update_histogram(self):
        """Update histogram display with quantile-based density line + fill."""
        mode = self.state.active_detail_map
        x, density = self.state.get_histogram(mode)

        # Remove old curve/fill but keep lo/hi lines
        if self.hist_curve is not None:
            self.hist_plot.removeItem(self.hist_curve)
            self.hist_curve = None
        if self.hist_fill is not None:
            self.hist_plot.removeItem(self.hist_fill)
            self.hist_fill = None

        if x.size == 0:
            return

        self.hist_curve = pg.PlotDataItem(x, density, pen=pg.mkPen("w", width=1))
        zero_line = pg.PlotDataItem(x, np.zeros_like(density))
        self.hist_fill = pg.FillBetweenItem(self.hist_curve, zero_line, brush=pg.mkBrush(255, 255, 255, 40))
        self.hist_plot.addItem(self.hist_fill)
        self.hist_plot.addItem(self.hist_curve)

        self.hist_plot.getViewBox().autoRange()

    def _update_range_lines(self):
        """Position and style the lo/hi vertical lines for the active layer.

        Hides the lines if no raw data or no active layer.
        """
        mode = self.state.active_detail_map
        layer = self.state.map_layers.get(mode)
        raw = self._current_raw
        if layer is None or raw is None:
            try:
                self.lo_line.hide()
                self.hi_line.hide()
            except Exception:
                pass
            return

        # Use absolute lo/hi in data coordinates
        lo = float(layer.lo)
        hi = float(layer.hi)

        # Map color vec (0..1) to 0..255 tuple for pen, add alpha
        color_vec = COLOR_MAP.get(getattr(layer, "color", "w"), COLOR_MAP["w"]).astype(np.float32)
        r, g, b = (color_vec * 255).astype(int)
        pen = pg.mkPen((int(r), int(g), int(b), 220), width=1)

        try:
            self.lo_line.setPen(pen)
            self.hi_line.setPen(pen)
            self.lo_line.setValue(lo)
            self.hi_line.setValue(hi)
            self.lo_line.show()
            self.hi_line.show()
        except Exception:
            # Be conservative on any unexpected viewbox/widget state
            pass

    def _on_hist_range_changed(self):
        """Update slider mapping when user zooms/pans histogram."""
        vb = self.hist_plot.getViewBox()
        x_range = vb.viewRange()[0]
        self._slider_xmin = x_range[0]
        self._slider_xmax = x_range[1]
        # Reposition sliders to reflect same absolute lo/hi in new range
        mode = self.state.active_detail_map
        layer = self.state.map_layers.get(mode)
        if layer is None:
            return
        span = self._slider_xmax - self._slider_xmin
        if span <= 1e-12:
            return
        lo_pos = int(1000 * ((layer.lo - self._slider_xmin) / span))
        hi_pos = int(1000 * ((layer.hi - self._slider_xmin) / span))
        self.lo_slider.blockSignals(True)
        self.hi_slider.blockSignals(True)
        self.lo_slider.setValue(max(0, min(1000, lo_pos)))
        self.hi_slider.setValue(max(0, min(1000, hi_pos)))
        self.lo_slider.blockSignals(False)
        self.hi_slider.blockSignals(False)

    def _on_color_changed(self, text: str):
        mode = self.state.active_detail_map
        self.state.update_map_layer(mode, color=text)

    def _on_lo_changed(self, val: int):
        mode = self.state.active_detail_map
        layer = self.state.map_layers.get(mode)
        if layer is None or self._current_raw is None:
            return
        mn, mx = self._slider_xmin, self._slider_xmax
        span = mx - mn
        if span <= 1e-12:
            return
        lo = mn + (val / 1000.0) * span
        hi = max(layer.hi, lo + 1e-6)
        self.state.update_map_layer(mode, lo=float(lo), hi=float(hi))

    def _on_hi_changed(self, val: int):
        mode = self.state.active_detail_map
        layer = self.state.map_layers.get(mode)
        if layer is None or self._current_raw is None:
            return
        mn, mx = self._slider_xmin, self._slider_xmax
        span = mx - mn
        if span <= 1e-12:
            return
        hi = mn + (val / 1000.0) * span
        lo = min(layer.lo, hi - 1e-6)
        self.state.update_map_layer(mode, lo=float(lo), hi=float(hi))

    def _on_gamma_changed(self, val: int):
        mode = self.state.active_detail_map
        gamma = float(val) / 100.0  # 0.01–0.99 target gmean
        self.state.update_map_layer(mode, gamma=gamma)
        # Keep visual in sync if necessary
        self._update_gamma_visual()

    def _on_intensity_changed(self, val: int):
        mode = self.state.active_detail_map
        intensity = float(val) / 100.0
        self.state.update_map_layer(mode, intensity=intensity)

    def _on_map_layer_changed(self, mode_changed: object):
        # If the changed layer is the currently-active one, refresh the range lines
        if mode_changed == self.state.active_detail_map:
            self._update_range_lines()
            self._update_gamma_visual()
