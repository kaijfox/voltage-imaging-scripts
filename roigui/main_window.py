"""Qt widget hierarchy for ROI visualization and editing."""

from typing import Optional, Set, Tuple, List
import numpy as np
from scipy import ndimage
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QButtonGroup,
    QFrame,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainterPath, QPen, QColor, QBrush
import pyqtgraph as pg

from .state import AppState, EditMode, ViewMode
from .keybindings import DEFAULT_BINDINGS
from .roi import ROI, RefineState
from .operations import extend_roi_watershed


class OutlineItem(pg.GraphicsObject):
    """Renders ROI boundary as a QPainterPath overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path = QPainterPath()
        self._pen = QPen(QColor(255, 255, 0), 1)  # Yellow, 1px
        self._pen.setCosmetic(True)  # Constant screen width regardless of zoom

    def set_boundary_edges(self, edges: set):
        """Update path from boundary edge set."""
        self._path = QPainterPath()
        for (r1, c1), (r2, c2) in edges:
            # Convert row/col to image coordinates (col=x, row=y)
            self._path.moveTo(c1, r1)
            self._path.lineTo(c2, r2)
        self.prepareGeometryChange()
        self.update()

    def clear(self):
        """Clear the outline."""
        self._path = QPainterPath()
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        return self._path.boundingRect()

    def paint(self, painter, option, widget):
        painter.setPen(self._pen)
        painter.drawPath(self._path)


class FillItem(pg.GraphicsObject):
    """Renders ROI pixels as semi-transparent fill overlay."""

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._pixels: set = set()
        self._color = color
        self._path = QPainterPath()
        self._visible = True

    def set_pixels(self, pixels: set):
        """Update fill from pixel set."""
        self._pixels = pixels
        self._rebuild_path()

    def clear(self):
        self._pixels = set()
        self._path = QPainterPath()
        self.prepareGeometryChange()
        self.update()

    def _rebuild_path(self):
        self._path = QPainterPath()
        for r, c in self._pixels:
            self._path.addRect(c, r, 1, 1)
        self.prepareGeometryChange()
        self.update()

    def set_fill_visible(self, visible: bool):
        self._visible = visible
        self.update()

    def boundingRect(self):
        return self._path.boundingRect()

    def paint(self, painter, option, widget):
        if not self._visible or not self._pixels:
            return
        painter.setPen(Qt.NoPen)
        brush = QBrush(self._color, Qt.FDiagPattern)
        painter.setBrush(brush)
        painter.drawPath(self._path)


class ImageView(pg.GraphicsLayoutWidget):
    """Main image display with ROI outline overlay and editing support."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        # Create plot and image item
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Outline overlays
        self.roi_outline = OutlineItem()  # Committed ROI (yellow)
        self.roi_outline._pen = QPen(QColor(255, 255, 0), 1)
        self.roi_outline._pen.setCosmetic(True)
        self.plot.addItem(self.roi_outline)

        self.candidate_outline = OutlineItem()  # Candidate (cyan)
        self.candidate_outline._pen = QPen(QColor(0, 255, 255), 1)
        self.candidate_outline._pen.setCosmetic(True)
        self.plot.addItem(self.candidate_outline)

        # Fill overlays (rendered below outlines)
        self.roi_fill = FillItem(QColor(255, 255, 0, 100))  # Yellow, semi-transparent
        self.plot.addItem(self.roi_fill)
        self.additions_fill = FillItem(QColor(0, 255, 0, 100))  # Green
        self.plot.addItem(self.additions_fill)
        self.subtractions_fill = FillItem(QColor(255, 0, 0, 100))  # Red
        self.plot.addItem(self.subtractions_fill)
        self._show_fills = False

        # Other ROI outlines and labels (dimmed, for Show All mode)
        self._other_roi_outlines: list[OutlineItem] = []
        self._roi_labels: list[pg.TextItem] = []

        # Lasso state
        self._lasso_stroke: Set[Tuple[int, int]] = set()
        self._is_dragging = False

        # Connect to state signals
        self.state.image_changed.connect(self._on_image_changed)
        self.state.roi_changed.connect(self._on_roi_changed)
        self.state.candidate_changed.connect(self._on_candidate_changed)
        self.state.edit_mode_changed.connect(self._on_edit_mode_changed)
        self.state.show_all_rois_changed.connect(self._on_show_all_changed)
        self.state.roi_list_changed.connect(self._on_roi_list_changed)
        self.state.current_roi_index_changed.connect(self._on_roi_list_changed)
        self.state.roi_id_changed.connect(self._on_roi_id_changed)

        # Enable mouse tracking for editing
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        # Track initial load for autoRange
        self._needs_autorange = True

        # Display initial state
        self._on_image_changed()
        self._on_roi_changed()
        self._on_candidate_changed()

    def _on_image_changed(self):
        """Update displayed image."""
        image = self.state.current_image
        if image is not None:
            # pyqtgraph expects (width, height) so transpose
            self.image_item.setImage(image.T)
            # Only autoRange on first load
            if self._needs_autorange:
                self.plot.autoRange()
                self._needs_autorange = False

    def _on_roi_changed(self):
        """Update committed ROI outline."""
        geom = self.state.roi_geometry
        if geom is not None:
            self.roi_outline.set_boundary_edges(geom.boundary_edges)
        else:
            self.roi_outline.clear()
        self._update_fills()

    def _on_candidate_changed(self):
        """Update candidate outline and fill overlays."""
        geom = self.state.candidate_geometry
        if geom is not None:
            self.candidate_outline.set_boundary_edges(geom.boundary_edges)
        else:
            self.candidate_outline.clear()
        self._update_fills()

    def _update_fills(self):
        """Update fill overlays for ROI, additions, and subtractions."""
        if not self._show_fills:
            self.roi_fill.clear()
            self.additions_fill.clear()
            self.subtractions_fill.clear()
            return

        # ROI fill
        if self.state.roi is not None:
            self.roi_fill.set_pixels(self.state.roi.pixel_set)
        else:
            self.roi_fill.clear()

        # Additions/subtractions
        self.additions_fill.set_pixels(self.state.additions)
        self.subtractions_fill.set_pixels(self.state.subtractions)

    def set_show_fills(self, show: bool):
        """Toggle fill visibility."""
        self._show_fills = show
        self._update_fills()

    def _on_edit_mode_changed(self, mode: EditMode):
        """Lock pan when in painting modes (scroll zoom still works)."""
        pan_locked = mode in (EditMode.ADD, EditMode.ERASE, EditMode.LASSO)
        self.plot.vb.setMouseEnabled(x=not pan_locked, y=not pan_locked)

    def _on_show_all_changed(self, show: bool):
        """Toggle visibility of other ROI outlines."""
        self._update_other_roi_outlines()

    def _on_roi_list_changed(self):
        """Update other ROI outlines when list changes."""
        if self.state.show_all_rois:
            self._update_other_roi_outlines()

    def _on_roi_id_changed(self, index: int, new_id: str):
        """Update labels when an ROI ID changes."""
        if self.state.show_all_rois:
            self._update_other_roi_outlines()

    def _update_other_roi_outlines(self):
        """Rebuild outlines and labels for all ROIs in Show All mode."""
        from .roi import ROIGeometry

        # Clear existing outlines and labels
        for outline in self._other_roi_outlines:
            self.plot.removeItem(outline)
        self._other_roi_outlines.clear()

        for label in self._roi_labels:
            self.plot.removeItem(label)
        self._roi_labels.clear()

        if not self.state.show_all_rois:
            return

        # Create outlines and labels for all ROIs
        current_idx = self.state.current_roi_index
        roi_ids = self.state.roi_ids
        for i, roi in enumerate(self.state.rois):
            if len(roi.footprint) == 0:
                continue

            # Compute centroid for label placement
            centroid = roi.footprint.mean(axis=0)
            # Note: image coords are (row, col) but pyqtgraph uses (x=col, y=row)
            label_x, label_y = centroid[1], centroid[0]

            # Get display text (use ID if available, otherwise index)
            label_text = roi_ids[i] if i < len(roi_ids) else str(i + 1)

            # Create label
            label = pg.TextItem(
                text=label_text,
                color="white" if i != current_idx else "yellow",
                anchor=(0.5, 0.5),
            )
            label.setPos(label_x, label_y)
            self.plot.addItem(label)
            self._roi_labels.append(label)

            # Create outline (skip current ROI - it has its own yellow outline)
            if i == current_idx:
                continue

            outline = OutlineItem()
            outline._pen = QPen(QColor(128, 128, 128), 1)  # Gray, dimmed
            outline._pen.setCosmetic(True)

            geom = ROIGeometry(roi)
            outline.set_boundary_edges(geom.boundary_edges)

            self.plot.addItem(outline)
            self._other_roi_outlines.append(outline)

    def _get_image_coords(self, scene_pos: QPointF) -> Optional[Tuple[int, int]]:
        """Convert scene position to image row, col coordinates."""
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        col, row = int(view_pos.x()), int(view_pos.y())

        # Check bounds
        if self.state.data_source is None:
            return None
        h, w = self.state.data_source.shape
        if 0 <= row < h and 0 <= col < w:
            return (row, col)
        return None

    def _get_brush_pixels(self, center_row: int, center_col: int) -> list:
        """Get list of pixels in brush centered at (row, col)."""
        pen_size = self.state.pen_size
        radius = pen_size // 2
        pixels = []

        if self.state.data_source is None:
            return pixels

        h, w = self.state.data_source.shape

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    r, c = center_row + dr, center_col + dc
                    if 0 <= r < h and 0 <= c < w:
                        pixels.append((r, c))
        return pixels

    def _on_mouse_clicked(self, event):
        """Handle mouse click for editing."""
        if event.button() != Qt.LeftButton:
            return

        mode = self.state.edit_mode
        if mode == EditMode.NONE:
            return

        coords = self._get_image_coords(event.scenePos())
        if coords is None:
            return

        if mode == EditMode.ADD:
            self._start_add_drag(coords)
        elif mode == EditMode.ERASE:
            self._start_erase_drag(coords)
        elif mode == EditMode.LASSO:
            self._start_lasso_drag(coords)

    def _on_mouse_moved(self, scene_pos):
        """Handle mouse move for dragging."""
        if not self._is_dragging:
            return

        coords = self._get_image_coords(scene_pos)
        if coords is None:
            return

        mode = self.state.edit_mode
        if mode == EditMode.ADD:
            self._continue_add_drag(coords)
        elif mode == EditMode.ERASE:
            self._continue_erase_drag(coords)
        elif mode == EditMode.LASSO:
            self._continue_lasso_drag(coords)

    def mousePressEvent(self, event):
        """Track drag start."""
        if event.button() == Qt.LeftButton and self.state.edit_mode != EditMode.NONE:
            self._is_dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle drag end."""
        if event.button() == Qt.LeftButton and self._is_dragging:
            self._is_dragging = False
            if self.state.edit_mode == EditMode.LASSO:
                self._finish_lasso_drag()
        super().mouseReleaseEvent(event)

    # Add mode
    def _start_add_drag(self, coords: Tuple[int, int]):
        self._ensure_candidate_exists()
        self._add_brush_pixels(coords)

    def _continue_add_drag(self, coords: Tuple[int, int]):
        self._add_brush_pixels(coords)

    def _add_brush_pixels(self, coords: Tuple[int, int]):
        geom = self.state.candidate_geometry
        if geom is None:
            return
        for r, c in self._get_brush_pixels(*coords):
            geom.add_pixel(r, c)
        self.state.notify_candidate_modified()

    # Erase mode
    def _start_erase_drag(self, coords: Tuple[int, int]):
        self._remove_brush_pixels(coords)

    def _continue_erase_drag(self, coords: Tuple[int, int]):
        self._remove_brush_pixels(coords)

    def _remove_brush_pixels(self, coords: Tuple[int, int]):
        geom = self.state.candidate_geometry
        if geom is None:
            return
        for r, c in self._get_brush_pixels(*coords):
            geom.remove_pixel(r, c)
        self.state.notify_candidate_modified()

    # Lasso mode
    def _start_lasso_drag(self, coords: Tuple[int, int]):
        self._ensure_candidate_exists()
        self._lasso_stroke.clear()
        self._lasso_stroke.update(self._get_brush_pixels(*coords))

    def _continue_lasso_drag(self, coords: Tuple[int, int]):
        self._lasso_stroke.update(self._get_brush_pixels(*coords))

    def _finish_lasso_drag(self):
        """Apply lasso stroke with morphological fill."""
        if not self._lasso_stroke:
            return

        geom = self.state.candidate_geometry
        if geom is None:
            return

        # Get bounding box of stroke with padding
        stroke_arr = np.array(list(self._lasso_stroke))
        r_min, c_min = stroke_arr.min(axis=0)
        r_max, c_max = stroke_arr.max(axis=0)
        pad = 2

        # Create small mask for morphological fill
        h = r_max - r_min + 1 + 2 * pad
        w = c_max - c_min + 1 + 2 * pad
        mask = np.zeros((h, w), dtype=bool)

        for r, c in self._lasso_stroke:
            mask[r - r_min + pad, c - c_min + pad] = True

        # Morphological fill: fill holes in the mask
        filled = ndimage.binary_fill_holes(mask)

        # Add all filled pixels to candidate
        filled_pixels = np.argwhere(filled)
        for local_r, local_c in filled_pixels:
            r = local_r + r_min - pad
            c = local_c + c_min - pad
            if self.state.data_source is not None:
                img_h, img_w = self.state.data_source.shape
                if 0 <= r < img_h and 0 <= c < img_w:
                    geom.add_pixel(r, c)

        self._lasso_stroke.clear()
        self.state.notify_candidate_modified()

    def _ensure_candidate_exists(self):
        """Create empty candidate if none exists."""
        self.state.ensure_candidate_exists()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Escape:
            self.state.set_edit_mode(EditMode.NONE)
        else:
            super().keyPressEvent(event)


class ToolButton(QPushButton):
    """Styled tool button for sidebar."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setMinimumHeight(28)
        self.setStyleSheet(
            """
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #ddd;
                padding: 4px 6px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:checked {
                background-color: #5a7a9a;
                border-color: #7a9aba;
            }
        """
        )


class ToolsSection(QFrame):
    """Tool selection buttons."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        label = QLabel("Tools")
        label.setStyleSheet("color: #aaa; font-weight: bold;")
        layout.addWidget(label)

        # Tool buttons in two rows
        self.btn_add = ToolButton("Add")
        self.btn_erase = ToolButton("Erase")
        self.btn_lasso = ToolButton("Lasso")
        self.btn_extend = ToolButton("Extend")
        self.btn_refine = ToolButton("Refine")

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(False)  # Allow unchecking

        self._mode_to_btn = {
            EditMode.ADD: self.btn_add,
            EditMode.ERASE: self.btn_erase,
            EditMode.LASSO: self.btn_lasso,
            EditMode.EXTEND: self.btn_extend,
            EditMode.REFINE: self.btn_refine,
        }

        # Row 1: Add, Erase, Lasso
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        for btn in [self.btn_add, self.btn_erase, self.btn_lasso]:
            row1.addWidget(btn)
        layout.addLayout(row1)

        # Row 2: Extend, Refine
        row2 = QHBoxLayout()
        row2.setSpacing(2)
        for btn in [self.btn_extend, self.btn_refine]:
            row2.addWidget(btn)
        row2.addStretch()
        layout.addLayout(row2)

        for mode, btn in self._mode_to_btn.items():
            self.button_group.addButton(btn)
            btn.clicked.connect(
                lambda checked, m=mode: self._on_tool_clicked(m, checked)
            )

        # Connect to state
        self.state.edit_mode_changed.connect(self._on_edit_mode_changed)

    def _on_tool_clicked(self, mode: EditMode, checked: bool):
        # Toggle behavior: if clicking already active mode, deselect
        if self.state.edit_mode == mode:
            self.state.set_edit_mode(EditMode.NONE)
        else:
            self.state.set_edit_mode(mode)

    def _on_edit_mode_changed(self, mode: EditMode):
        """Update button states when mode changes externally."""
        for m, btn in self._mode_to_btn.items():
            btn.setChecked(m == mode)


class ToolDetailSection(QFrame):
    """Context-dependent tool settings."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)

        # Pen size controls (for Add/Erase/Lasso)
        self.pen_size_widget = QWidget()
        pen_layout = QVBoxLayout(self.pen_size_widget)
        pen_layout.setContentsMargins(0, 0, 0, 0)

        pen_label = QLabel("Pen Size")
        pen_label.setStyleSheet("color: #aaa;")
        pen_layout.addWidget(pen_label)

        self.pen_slider = QSlider(Qt.Horizontal)
        self.pen_slider.setRange(1, 50)
        self.pen_slider.setValue(state.pen_size)
        self.pen_slider.valueChanged.connect(self._on_pen_size_changed)
        pen_layout.addWidget(self.pen_slider)

        self.pen_value_label = QLabel(str(state.pen_size))
        self.pen_value_label.setStyleSheet("color: #ddd;")
        pen_layout.addWidget(self.pen_value_label)

        self.layout.addWidget(self.pen_size_widget)

        # Refine slider (for Refine mode)
        self.refine_widget = QWidget()
        refine_layout = QVBoxLayout(self.refine_widget)
        refine_layout.setContentsMargins(0, 0, 0, 0)

        refine_label = QLabel("Threshold")
        refine_label.setStyleSheet("color: #aaa;")
        refine_layout.addWidget(refine_label)

        self.refine_slider = QSlider(Qt.Horizontal)
        self.refine_slider.setRange(0, 100)
        self.refine_slider.setValue(100)
        self.refine_slider.valueChanged.connect(self._on_refine_slider_changed)
        refine_layout.addWidget(self.refine_slider)

        self.refine_value_label = QLabel("All pixels")
        self.refine_value_label.setStyleSheet("color: #ddd;")
        refine_layout.addWidget(self.refine_value_label)

        self.layout.addWidget(self.refine_widget)

        self.layout.addStretch()

        # Connect to state
        self.state.edit_mode_changed.connect(self._on_edit_mode_changed)
        self.state.pen_size_changed.connect(self._on_pen_size_state_changed)
        self.state.refine_index_changed.connect(self._on_refine_index_changed)

        # Initial visibility
        self._update_visibility(state.edit_mode)

    def _update_visibility(self, mode: EditMode):
        """Show/hide controls based on edit mode."""
        show_pen = mode in (EditMode.ADD, EditMode.ERASE, EditMode.LASSO)
        show_refine = mode == EditMode.REFINE

        self.pen_size_widget.setVisible(show_pen)
        self.refine_widget.setVisible(show_refine)

    def _on_edit_mode_changed(self, mode: EditMode):
        self._update_visibility(mode)

        # Handle extend mode: run watershed then switch to refine
        if mode == EditMode.EXTEND:
            self._run_extend()
            return

        # Initialize refine slider when entering refine mode
        if mode == EditMode.REFINE:
            self._init_refine_mode()

    def _run_extend(self):
        """Run watershed extension on candidate and switch to refine mode."""
        if self.state.candidate is None or len(self.state.candidate.footprint) == 0:
            self.state.set_edit_mode(EditMode.NONE)
            return
        if self.state.data_source is None:
            self.state.set_edit_mode(EditMode.NONE)
            return

        # Run watershed extension on candidate using current view image
        extended_footprint = extend_roi_watershed(
            self.state.candidate,
            self.state.data_source,
            expansion_pixels=30,
            image=self.state.current_image,
        )

        # Update candidate with extended footprint
        from .roi import ROIGeometry

        extended_roi = ROI(
            footprint=extended_footprint,
            weights=np.ones(len(extended_footprint)),
            code=self.state.candidate.code.copy(),
        )
        self.state._candidate_geometry = ROIGeometry(extended_roi)
        self.state.notify_candidate_modified()

        # Switch to refine mode
        self.state.set_edit_mode(EditMode.REFINE)

    def _init_refine_mode(self):
        """Initialize refine state when entering refine mode."""
        if self.state.candidate is None or len(self.state.candidate.footprint) == 0:
            return
        if self.state.data_source is None:
            return

        candidate = self.state.candidate

        # Get pixel correlations with candidate code
        code = self.state.data_source.extract_code(
            candidate.footprint, candidate.weights
        )
        pixel_codes = self.state.data_source.get_pixel_codes(candidate.footprint)

        # Compute correlation of each pixel with the candidate code
        code_norm = np.linalg.norm(code)
        if code_norm > 1e-10:
            pixel_norms = np.linalg.norm(pixel_codes, axis=1)
            pixel_norms = np.maximum(pixel_norms, 1e-10)
            correlations = (pixel_codes @ code) / (pixel_norms * code_norm)
        else:
            correlations = np.ones(len(candidate.footprint))

        # Create refine state from candidate
        refine_state = RefineState.from_roi_and_correlations(
            candidate, correlations, checkpoint_interval=100
        )
        self.state.set_refine_state(refine_state)

        # Update slider
        self.refine_slider.setRange(0, refine_state.n_pixels)
        self.refine_slider.setValue(refine_state.n_pixels)

    def _on_pen_size_changed(self, value: int):
        self.state.set_pen_size(value)

    def _on_pen_size_state_changed(self, value: int):
        self.pen_slider.setValue(value)
        self.pen_value_label.setText(str(value))

    def _on_refine_slider_changed(self, value: int):
        self.state.set_refine_index(value)

    def _on_refine_index_changed(self, index: int):
        if self.state.refine_state is not None:
            n = self.state.refine_state.n_pixels
            self.refine_slider.setValue(index)
            self.refine_value_label.setText(f"{index}/{n} pixels")


class CandidateActionsSection(QFrame):
    """Accept/Reject buttons for candidate changes."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        label = QLabel("Changes")
        label.setStyleSheet("color: #aaa; font-weight: bold;")
        layout.addWidget(label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_accept = QPushButton("Accept")
        self.btn_accept.setStyleSheet(
            """
            QPushButton {
                background-color: #3a5a3a;
                border: 1px solid #5a7a5a;
                border-radius: 4px;
                color: #ddd;
                padding: 6px;
            }
            QPushButton:hover { background-color: #4a6a4a; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
        """
        )
        self.btn_accept.clicked.connect(self._on_accept)
        btn_layout.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setStyleSheet(
            """
            QPushButton {
                background-color: #5a3a3a;
                border: 1px solid #7a5a5a;
                border-radius: 4px;
                color: #ddd;
                padding: 6px;
            }
            QPushButton:hover { background-color: #6a4a4a; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
        """
        )
        self.btn_reject.clicked.connect(self._on_reject)
        btn_layout.addWidget(self.btn_reject)

        layout.addLayout(btn_layout)

        # Update button states
        self.state.candidate_changed.connect(self._update_buttons)
        self.state.roi_changed.connect(self._update_buttons)
        self._update_buttons()

    def _update_buttons(self):
        has_changes = self.state.has_uncommitted_changes
        self.btn_accept.setEnabled(has_changes)
        self.btn_reject.setEnabled(has_changes)

    def _on_accept(self):
        self.state.accept_candidate()

    def _on_reject(self):
        self.state.reject_candidate()


class ViewModeSection(QFrame):
    """View mode selection."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        label = QLabel("View")
        label.setStyleSheet("color: #aaa; font-weight: bold;")
        layout.addWidget(label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)

        self.btn_mean = ToolButton("Mean")
        self.btn_corr = ToolButton("Corr")
        self.btn_local_corr = ToolButton("Local")

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        for btn, mode in [
            (self.btn_mean, ViewMode.MEAN),
            (self.btn_corr, ViewMode.CORRELATION),
            (self.btn_local_corr, ViewMode.LOCAL_CORRELATION),
        ]:
            self.button_group.addButton(btn)
            btn.clicked.connect(
                lambda checked, m=mode: self._on_view_clicked(m, checked)
            )
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

        # Set initial state
        self.btn_mean.setChecked(True)

        self.state.view_mode_changed.connect(self._on_view_mode_changed)

    def _on_view_clicked(self, mode: ViewMode, checked: bool):
        if checked:
            self.state.set_view_mode(mode)

    def _on_view_mode_changed(self, mode: ViewMode):
        self.btn_mean.setChecked(mode == ViewMode.MEAN)
        self.btn_corr.setChecked(mode == ViewMode.CORRELATION)
        self.btn_local_corr.setChecked(mode == ViewMode.LOCAL_CORRELATION)


class ROISelectorSection(QFrame):
    """ROI selection and management."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        label = QLabel("ROIs")
        label.setStyleSheet("color: #aaa; font-weight: bold;")
        layout.addWidget(label)

        # ROI selector dropdown
        from PySide6.QtWidgets import QComboBox

        self.roi_combo = QComboBox()
        self.roi_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #ddd;
                padding: 4px;
            }
        """
        )
        self.roi_combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self.roi_combo)

        # Button row 1: New, Propose
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_new = QPushButton("New")
        self.btn_new.setStyleSheet(
            """
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #ddd;
                padding: 4px 8px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """
        )
        self.btn_new.clicked.connect(self._on_new_clicked)
        btn_layout.addWidget(self.btn_new)

        self.btn_propose = QPushButton("Propose")
        self.btn_propose.setStyleSheet(self.btn_new.styleSheet())
        self.btn_propose.clicked.connect(self._on_propose_clicked)
        btn_layout.addWidget(self.btn_propose)

        layout.addLayout(btn_layout)

        # Button row 2: Rename
        btn_layout2 = QHBoxLayout()
        btn_layout2.setSpacing(4)

        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setStyleSheet(self.btn_new.styleSheet())
        self.btn_rename.clicked.connect(self._on_rename_clicked)
        btn_layout2.addWidget(self.btn_rename)
        btn_layout2.addStretch()

        layout.addLayout(btn_layout2)

        # Show All toggle
        from PySide6.QtWidgets import QCheckBox

        self.show_all_checkbox = QCheckBox("Show All")
        self.show_all_checkbox.setStyleSheet("color: #ddd;")
        self.show_all_checkbox.toggled.connect(self._on_show_all_toggled)
        layout.addWidget(self.show_all_checkbox)

        # Status label for feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #f88; font-size: 11px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Connect to state
        self.state.roi_list_changed.connect(self._update_combo)
        self.state.current_roi_index_changed.connect(self._on_roi_index_changed)
        self.state.show_all_rois_changed.connect(self._on_show_all_state_changed)
        self.state.roi_id_changed.connect(self._on_roi_id_changed)

        self._update_combo()

    def _update_combo(self):
        """Rebuild combo box from ROI list."""
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        roi_ids = self.state.roi_ids
        for i in range(len(self.state.rois)):
            # Use ID if available, otherwise fall back to "ROI n"
            label = roi_ids[i] if i < len(roi_ids) else f"ROI {i + 1}"
            self.roi_combo.addItem(label)
        if self.state.current_roi_index >= 0:
            self.roi_combo.setCurrentIndex(self.state.current_roi_index)
        self.roi_combo.blockSignals(False)

    def _on_combo_changed(self, index: int):
        if index >= 0:
            self.state.set_current_roi_index(index)

    def _on_roi_index_changed(self, index: int):
        self.roi_combo.blockSignals(True)
        if index >= 0:
            self.roi_combo.setCurrentIndex(index)
        self.roi_combo.blockSignals(False)
        self._update_combo()  # Refresh pixel counts

    def _on_new_clicked(self):
        self.state.new_empty_roi()

    def _on_rename_clicked(self):
        """Open dialog to rename current ROI."""
        from PySide6.QtWidgets import QInputDialog

        index = self.state.current_roi_index
        if index < 0:
            return

        current_id = self.state.roi_ids[index] if index < len(self.state.roi_ids) else f"ROI {index + 1}"
        new_id, ok = QInputDialog.getText(
            self, "Rename ROI", "Enter new name:", text=current_id
        )
        if ok and new_id.strip():
            self.state.rename_roi(index, new_id.strip())

    def _on_roi_id_changed(self, index: int, new_id: str):
        """Update combo box when ROI ID changes."""
        self._update_combo()

    def _on_propose_clicked(self):
        """Find peak in current image not in any ROI, extend, enter refine."""
        self.status_label.setText("")

        if self.state.data_source is None:
            self.status_label.setText("No data source")
            return

        # Get current view image
        image = self.state.current_image
        if image is None:
            self.status_label.setText("No image available")
            return

        # Find all pixels covered by existing ROIs
        covered = self.state.all_roi_pixels()

        # Check if all pixels are covered
        h, w = image.shape
        if len(covered) >= h * w:
            self.status_label.setText("All pixels covered by existing ROIs")
            return

        # Mask covered pixels
        masked_image = image.copy()
        for r, c in covered:
            if 0 <= r < h and 0 <= c < w:
                masked_image[r, c] = -np.inf

        # Find peak
        peak_val = np.nanmax(masked_image[masked_image != -np.inf])
        if not np.isfinite(peak_val):
            self.status_label.setText("No valid peak found")
            return

        peak_idx = np.unravel_index(np.argmax(masked_image), masked_image.shape)
        r, c = int(peak_idx[0]), int(peak_idx[1])

        # Check peak value threshold (correlation maps range ~0-1)
        if peak_val < 0.1:
            self.status_label.setText(f"Peak too weak ({peak_val:.2f})")
            return

        # Create single-pixel ROI at peak
        footprint = np.array([[r, c]], dtype=np.int32)
        code = self.state.data_source.extract_code(footprint)
        roi = ROI(
            footprint=footprint, weights=np.array([1.0], dtype=np.float32), code=code
        )

        # Add and select
        self.state.add_and_select_roi(roi)

        # Run extend, then enter refine mode
        self.state.set_edit_mode(EditMode.EXTEND)

        # Check if extend produced valid result
        if self.state.candidate is None or len(self.state.candidate.footprint) == 0:
            self.status_label.setText("Extend found no correlated region")
            return

        self.status_label.setText("")  # Success - clear any message

    def _on_show_all_toggled(self, checked: bool):
        self.state.set_show_all_rois(checked)

    def _on_show_all_state_changed(self, show: bool):
        self.show_all_checkbox.blockSignals(True)
        self.show_all_checkbox.setChecked(show)
        self.show_all_checkbox.blockSignals(False)


class MainViewDetail(QFrame):
    """Main view details: minimap, dynamic range, view mode, ROI selector."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumHeight(350)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Minimap
        minimap_label = QLabel("Overview")
        minimap_label.setStyleSheet("color: #aaa; font-weight: bold;")
        layout.addWidget(minimap_label)

        self.minimap_widget = pg.GraphicsLayoutWidget()
        self.minimap_widget.setFixedHeight(100)
        self.minimap_widget.setBackground("#1a1a1a")
        self.minimap_plot = self.minimap_widget.addPlot()
        self.minimap_plot.hideAxis("left")
        self.minimap_plot.hideAxis("bottom")
        self.minimap_plot.setAspectLocked(True)
        self.minimap_plot.setMouseEnabled(x=False, y=False)

        self.minimap_image = pg.ImageItem()
        self.minimap_plot.addItem(self.minimap_image)

        # Viewport rectangle overlay
        self.viewport_rect = pg.RectROI(
            [0, 0], [10, 10],
            pen=pg.mkPen("y", width=1),
            movable=True,
            resizable=False,
        )
        self.viewport_rect.removeHandle(0)  # Remove resize handles
        self.minimap_plot.addItem(self.viewport_rect)
        self.viewport_rect.sigRegionChanged.connect(self._on_viewport_rect_moved)

        layout.addWidget(self.minimap_widget)

        # Dynamic range controls
        range_label = QLabel("Dynamic Range")
        range_label.setStyleSheet("color: #aaa; font-weight: bold;")
        layout.addWidget(range_label)

        # Histogram widget
        self.histogram = pg.HistogramLUTWidget()
        self.histogram.setFixedHeight(80)
        self.histogram.setBackground("#1a1a1a")
        self.histogram.item.sigLevelsChanged.connect(self._on_levels_changed)
        layout.addWidget(self.histogram)

        # View mode section (reuse existing)
        self.view_section = ViewModeSection(state)
        layout.addWidget(self.view_section)

        # ROI selector section (reuse existing)
        self.roi_selector = ROISelectorSection(state)
        layout.addWidget(self.roi_selector)

        # Store reference to ImageView (set later)
        self._image_view = None
        self._updating_rect = False

        # Per-view-mode dynamic range storage
        self._levels_cache: dict[ViewMode, tuple] = {}
        self._current_view_mode = state.view_mode

        # Connect to state
        self.state.image_changed.connect(self._on_image_changed)
        self.state.view_mode_changed.connect(self._on_view_mode_changed)

    def set_image_view(self, image_view: "ImageView"):
        """Connect to main ImageView for viewport tracking."""
        self._image_view = image_view
        image_view.plot.vb.sigRangeChanged.connect(self._on_main_view_range_changed)
        self._on_image_changed()

    def _on_view_mode_changed(self, mode: ViewMode):
        """Save current levels and restore cached levels for new mode."""
        # Save current levels for previous mode
        if hasattr(self, '_current_view_mode'):
            try:
                self._levels_cache[self._current_view_mode] = self.histogram.getLevels()
            except:
                pass
        self._current_view_mode = mode

    def _on_image_changed(self):
        """Update minimap and histogram when image changes."""
        image = self.state.current_image
        if image is not None:
            # Update the histogram without overwriting the levels cache
            mode = self.state.view_mode
            retained_levels = self._levels_cache.get(mode, None)

            self.minimap_image.setImage(image.T)
            self.minimap_plot.autoRange()
            self.histogram.setImageItem(self.minimap_image)

            if retained_levels is not None:
                self._levels_cache[mode] = retained_levels

            # Restore cached levels or autoscale
            
            if mode in self._levels_cache:
                levels = self._levels_cache[mode]
                self.histogram.setLevels(*levels)
                self.histogram.setHistogramRange(levels[0], levels[1])
            else:
                # Autoscale to image range
                mn, mx = float(image.min()), float(image.max())
                self.histogram.setLevels(mn, mx)
                self.histogram.setHistogramRange(mn, mx)

            self._update_viewport_rect()

    def _on_main_view_range_changed(self):
        """Update viewport rect when main view pans/zooms."""
        if self._updating_rect or self._image_view is None:
            return
        self._update_viewport_rect()

    def _update_viewport_rect(self):
        """Sync viewport rect to main view's visible range."""
        if self._image_view is None:
            return

        self._updating_rect = True
        vb = self._image_view.plot.vb
        view_range = vb.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]

        self.viewport_rect.setPos([x_min, y_min])
        self.viewport_rect.setSize([x_max - x_min, y_max - y_min])
        self._updating_rect = False

    def _on_viewport_rect_moved(self):
        """Pan main view when viewport rect is dragged."""
        if self._updating_rect or self._image_view is None:
            return

        self._updating_rect = True
        pos = self.viewport_rect.pos()
        size = self.viewport_rect.size()
        center_x = pos.x() + size.x() / 2
        center_y = pos.y() + size.y() / 2

        vb = self._image_view.plot.vb
        vb.setRange(
            xRange=[pos.x(), pos.x() + size.x()],
            yRange=[pos.y(), pos.y() + size.y()],
            padding=0,
        )
        self._updating_rect = False

    def _on_levels_changed(self):
        """Update main image levels when histogram changes."""
        if self._image_view is None:
            return
        levels = self.histogram.getLevels()
        self._image_view.image_item.setLevels(levels)
        # Cache levels for current view mode
        self._levels_cache[self.state.view_mode] = levels


class ROIDetailSection(QFrame):
    """ROI detail display: weight map and trace plot."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        label = QLabel("ROI Detail")
        label.setStyleSheet("color: #aaa; font-weight: bold;")
        layout.addWidget(label)

        # Weight map display
        self.weight_view = pg.GraphicsLayoutWidget()
        self.weight_view.setFixedHeight(80)
        self.weight_view.setBackground("#2a2a2a")
        self.weight_plot = self.weight_view.addPlot()
        self.weight_plot.hideAxis("left")
        self.weight_plot.hideAxis("bottom")
        self.weight_plot.setAspectLocked(True)
        self.weight_image = pg.ImageItem()
        self.weight_plot.addItem(self.weight_image)
        layout.addWidget(self.weight_view)

        # Trace plot
        self.trace_view = pg.PlotWidget()
        self.trace_view.setFixedHeight(120)
        self.trace_view.setBackground("#1a1a1a")
        # Show axes with tick numbers only (no labels)
        plot_item = self.trace_view.getPlotItem()
        plot_item.showAxis("bottom")
        plot_item.showAxis("left")
        plot_item.getAxis("bottom").setStyle(tickLength=-5, showValues=True)
        plot_item.getAxis("left").setStyle(tickLength=-5, showValues=True)
        plot_item.getAxis("left").setWidth(40)
        plot_item.setContentsMargins(5, 5, 5, 5)
        # X-axis only mouse controls
        plot_item.getViewBox().setMouseEnabled(x=True, y=False)

        # Trace lines
        self.trace_committed = self.trace_view.plot(pen=pg.mkPen("#ffffff", width=1))
        self.trace_candidate = self.trace_view.plot(pen=pg.mkPen("#00ffff", width=1))
        self.trace_added = self.trace_view.plot(pen=pg.mkPen("#00ff00", width=1))
        self.trace_removed = self.trace_view.plot(pen=pg.mkPen("#ff0000", width=1))

        layout.addWidget(self.trace_view)

        # Toggle buttons
        from PySide6.QtWidgets import QCheckBox

        self.simple_view_checkbox = QCheckBox("Hide proposal")
        self.simple_view_checkbox.setStyleSheet("color: #ddd;")
        self.simple_view_checkbox.toggled.connect(self._on_simple_view_toggled)
        layout.addWidget(self.simple_view_checkbox)

        self.show_fill_checkbox = QCheckBox("Show fill")
        self.show_fill_checkbox.setStyleSheet("color: #ddd;")
        self.show_fill_checkbox.toggled.connect(self._on_show_fill_toggled)
        layout.addWidget(self.show_fill_checkbox)

        self._simple_view = False
        self._image_view = None  # Set by MainWindow

        # Debounced update
        from .debounce import CallbackDebouncer

        self._update_debouncer = CallbackDebouncer(
            self._do_update, delay_ms=200, parent=self
        )

        # Connect to state
        self.state.candidate_changed.connect(self._request_update)
        self.state.roi_changed.connect(self._request_update)
        self.state.current_roi_index_changed.connect(self._request_update)

    def _request_update(self):
        self._update_debouncer.request()

    def set_image_view(self, image_view: "ImageView"):
        """Store reference to ImageView for fill toggle."""
        self._image_view = image_view

    def _on_simple_view_toggled(self, checked: bool):
        self._simple_view = checked
        self._update_trace_visibility()
        self._request_update()

    def _on_show_fill_toggled(self, checked: bool):
        if self._image_view is not None:
            self._image_view.set_show_fills(checked)

    def set_show_fill(self, show: bool):
        """Programmatic fill toggle (for keyboard shortcut)."""
        self.show_fill_checkbox.setChecked(show)

    def _update_trace_visibility(self):
        self.trace_candidate.setVisible(not self._simple_view)
        self.trace_added.setVisible(not self._simple_view)
        self.trace_removed.setVisible(not self._simple_view)

    def _do_update(self):
        """Update weight map and trace plots."""
        self._update_weight_map()
        self._update_traces()

    def _update_weight_map(self):
        """Update weight map display."""
        candidate = self.state.candidate
        if candidate is None or len(candidate.footprint) == 0:
            self.weight_image.clear()
            return

        # Get bounding box
        footprint = candidate.footprint
        r_min, c_min = footprint.min(axis=0)
        r_max, c_max = footprint.max(axis=0)

        # Create weight image
        h = r_max - r_min + 1
        w = c_max - c_min + 1
        weight_img = np.zeros((h, w), dtype=np.float32)
        weight_min = candidate.weights.min()
        for i, (r, c) in enumerate(footprint):
            weight_img[r - r_min, c - c_min] = candidate.weights[i] - weight_min

        self.weight_image.setImage(
            weight_img.T, levels=(0, candidate.weights.max() - weight_min)
        )
        self.weight_plot.autoRange()

    def _update_traces(self):
        """Update trace plots."""
        ds = self.state.data_source
        if ds is None:
            self.trace_committed.setData([], [])
            self.trace_candidate.setData([], [])
            self.trace_added.setData([], [])
            self.trace_removed.setData([], [])
            return

        # Committed ROI trace
        roi = self.state.roi
        if roi is not None and len(roi.footprint) > 0:
            trace = ds.extract_trace(roi.footprint, roi.weights)
            self.trace_committed.setData(trace)
        else:
            self.trace_committed.setData([], [])

        if self._simple_view:
            return

        # Candidate trace
        candidate = self.state.candidate
        if candidate is not None and len(candidate.footprint) > 0:
            trace = ds.extract_trace(candidate.footprint, candidate.weights)
            self.trace_candidate.setData(trace)
        else:
            self.trace_candidate.setData([], [])

        # Added pixels trace
        additions = self.state.additions
        if additions:
            add_fp = np.array(list(additions))
            trace = ds.extract_trace(add_fp)
            self.trace_added.setData(trace)
        else:
            self.trace_added.setData([], [])

        # Removed pixels trace
        subtractions = self.state.subtractions
        if subtractions:
            sub_fp = np.array(list(subtractions))
            trace = ds.extract_trace(sub_fp)
            self.trace_removed.setData(trace)
        else:
            self.trace_removed.setData([], [])


class Sidebar(QFrame):
    """Sidebar containing tools and settings."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setFixedWidth(200)
        self.setStyleSheet("background-color: #2a2a2a;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main view detail panel (minimap, range, view mode, ROI selector)
        self.main_view_detail = MainViewDetail(state)
        layout.addWidget(self.main_view_detail)

        # Separator
        self._add_separator(layout)

        # Tools section
        self.tools_section = ToolsSection(state)
        layout.addWidget(self.tools_section)

        # Separator
        self._add_separator(layout)

        # Tool detail section
        self.tool_detail = ToolDetailSection(state)
        layout.addWidget(self.tool_detail)

        # Separator
        self._add_separator(layout)

        # Accept/Reject section
        self.candidate_actions = CandidateActionsSection(state)
        layout.addWidget(self.candidate_actions)

        # Separator
        self._add_separator(layout)

        # ROI detail section
        self.roi_detail = ROIDetailSection(state)
        layout.addWidget(self.roi_detail)

        layout.addStretch()

    def _add_separator(self, layout):
        """Add horizontal separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #444;")
        sep.setFixedHeight(1)
        layout.addWidget(sep)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._keybindings = DEFAULT_BINDINGS.copy()

        self.setWindowTitle("ROI Editor")
        self.resize(1200, 1000)
        self.setMinimumSize(800, 1000)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main image view
        self.image_view = ImageView(state)
        layout.addWidget(self.image_view, stretch=1)

        # Sidebar
        self.sidebar = Sidebar(state)
        layout.addWidget(self.sidebar)

        # Wire up cross-component connections
        self.sidebar.main_view_detail.set_image_view(self.image_view)
        self.sidebar.roi_detail.set_image_view(self.image_view)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()
        action = self._keybindings.get(key)

        if action is None:
            super().keyPressEvent(event)
            return

        # Tool actions
        if action == "tool_add":
            self.state.set_edit_mode(EditMode.ADD)
        elif action == "tool_erase":
            self.state.set_edit_mode(EditMode.ERASE)
        elif action == "tool_lasso":
            self.state.set_edit_mode(EditMode.LASSO)
        elif action == "tool_extend":
            self.state.set_edit_mode(EditMode.EXTEND)
        elif action == "tool_refine":
            self.state.set_edit_mode(EditMode.REFINE)
        elif action == "tool_none":
            self.state.set_edit_mode(EditMode.NONE)

        # Accept/Reject
        elif action == "accept":
            self.state.accept_candidate()
        elif action == "reject":
            self.state.reject_candidate()

        # View modes
        elif action == "view_mean":
            self.state.set_view_mode(ViewMode.MEAN)
        elif action == "view_correlation":
            self.state.set_view_mode(ViewMode.CORRELATION)
        elif action == "view_local_correlation":
            self.state.set_view_mode(ViewMode.LOCAL_CORRELATION)

        # ROI management
        elif action == "new_roi":
            self.state.new_empty_roi()
        elif action == "propose_roi":
            self.sidebar.main_view_detail.roi_selector._on_propose_clicked()
        elif action == "next_roi":
            n = self.state.n_rois
            if n > 0:
                next_idx = (self.state.current_roi_index + 1) % n
                self.state.set_current_roi_index(next_idx)

        # Display toggles
        elif action == "toggle_show_all":
            self.state.set_show_all_rois(not self.state.show_all_rois)
        elif action == "toggle_fill":
            checkbox = self.sidebar.roi_detail.show_fill_checkbox
            checkbox.setChecked(not checkbox.isChecked())

        # Pen size
        elif action == "pen_smaller":
            self.state.set_pen_size(self.state.pen_size - 2)
        elif action == "pen_larger":
            self.state.set_pen_size(self.state.pen_size + 2)

        else:
            super().keyPressEvent(event)
