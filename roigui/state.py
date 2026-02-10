"""Application state with Qt signals for reactive updates."""

from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, List
import numpy as np
from PySide6.QtCore import QObject, Signal

from .roi import ROI, ROIGeometry
from .compositing import MapLayerState

if TYPE_CHECKING:
    from .data_source import DataSource
    from .roi import RefineState


class ViewMode(Enum):
    MEAN = auto()
    CORRELATION = auto()
    LOCAL_CORRELATION = auto()
    # Future: VIDEO, TSNE, etc.


class EditMode(Enum):
    NONE = auto()
    ADD = auto()
    ERASE = auto()
    LASSO = auto()
    EXTEND = auto()
    REFINE = auto()


class AppState(QObject):
    """Central application state with signals for reactive UI updates.

    Components connect to signals they care about:
        state.roi_changed.connect(self.on_roi_changed)
        state.view_mode_changed.connect(self.on_view_mode_changed)

    Multi-ROI support:
        - `rois` property returns list of all ROIs
        - `current_roi_index` indicates which ROI is being edited
        - `roi` / `roi_geometry` refer to the current ROI
        - `roi_list_changed` emitted when ROIs added/removed
        - `current_roi_index_changed` emitted when switching ROIs

    ROI IDs:
        - `roi_ids` returns list of string IDs for all ROIs
        - IDs default to "ROI 1", "ROI 2", etc. but can be renamed
        - `_persist_ids` controls whether IDs are saved to file
    """

    # Signals
    roi_changed = Signal()  # Committed ROI modified
    candidate_changed = Signal()  # Candidate ROI modified
    roi_list_changed = Signal()  # ROI list modified (add/remove)
    current_roi_index_changed = Signal(int)  # Switched to different ROI
    show_all_rois_changed = Signal(bool)  # Toggle for showing all ROI outlines
    roi_id_changed = Signal(int, str)  # ROI ID changed (index, new_id)
    view_mode_changed = Signal(ViewMode)
    edit_mode_changed = Signal(EditMode)
    image_changed = Signal()  # Underlying image data changed
    # New signals for multi-layer map overlay
    map_layer_changed = Signal(object)          # emits ViewMode
    active_detail_map_changed = Signal(object)  # emits ViewMode
    pen_size_changed = Signal(int)
    refine_index_changed = Signal(int)  # Refine slider position changed
    correlation_seed_changed = Signal()  # Seed pixel for correlation map changed

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        # Map layer states (per ViewMode)
        self._map_layers: dict[ViewMode, MapLayerState] = {
            ViewMode.MEAN: MapLayerState(enabled=True, color="w"),
            ViewMode.CORRELATION: MapLayerState(enabled=False, color="g"),
            ViewMode.LOCAL_CORRELATION: MapLayerState(enabled=False, color="r"),
        }
        self._active_detail_map: ViewMode = ViewMode.MEAN
        self._summary_label: str = "Mean"
        # Multi-ROI storage
        self._rois: list[ROI] = []  # List of committed ROIs
        self._roi_ids: List[str] = []  # List of ROI IDs (parallel to _rois)
        self._persist_ids: bool = False  # Whether to save IDs to file
        self._current_roi_index: int = -1  # -1 means no ROI selected
        self._show_all_rois: bool = False

        # Current ROI editing state (for the selected ROI)
        self._roi_geometry: Optional[ROIGeometry] = None  # Committed state
        self._candidate_geometry: Optional[ROIGeometry] = None  # Working copy

        self._view_mode = ViewMode.MEAN
        self._edit_mode = EditMode.NONE
        self._data_source: Optional["DataSource"] = None
        self._pen_size: int = 3
        self._refine_state: Optional["RefineState"] = None
        self._correlation_seed: Optional[tuple] = None  # (row, col) for correlation map
        self._corr_map_cache: Optional[np.ndarray] = None  # cached correlation map
        # Per-mode caches for histogram and gmean (invalidated with corr cache or on set_data_source)
        self._hist_cache: dict = {}    # ViewMode -> (centers, heights)
        self._gmean_cache: dict = {}   # ViewMode -> float
        self._output_path = None  # Set by launch()

    # Data source
    @property
    def data_source(self) -> Optional["DataSource"]:
        return self._data_source

    def set_data_source(self, source: "DataSource"):
        self._data_source = source
        self._corr_map_cache = None
        self._hist_cache.clear()
        self._gmean_cache.clear()
        self.image_changed.emit()

    # Multi-ROI management
    @property
    def rois(self) -> list[ROI]:
        """List of all committed ROIs."""
        return self._rois

    @property
    def n_rois(self) -> int:
        return len(self._rois)

    @property
    def roi_ids(self) -> List[str]:
        """List of ROI IDs (parallel to rois list)."""
        return self._roi_ids

    @property
    def persist_ids(self) -> bool:
        """Whether ROI IDs should be saved to file."""
        return self._persist_ids

    def set_persist_ids(self, persist: bool):
        """Set whether ROI IDs should be saved to file."""
        self._persist_ids = persist

    @property
    def current_roi_index(self) -> int:
        """Index of currently selected ROI, or -1 if none."""
        return self._current_roi_index

    @property
    def show_all_rois(self) -> bool:
        return self._show_all_rois

    def set_show_all_rois(self, show: bool):
        if self._show_all_rois != show:
            self._show_all_rois = show
            self.show_all_rois_changed.emit(show)

    def set_current_roi_index(self, index: int):
        """Switch to a different ROI by index."""
        if index < -1 or index >= len(self._rois):
            index = -1

        if self._current_roi_index == index:
            return

        # Save current candidate back to list if we have uncommitted changes
        # (Actually, we'll reject uncommitted changes on switch for simplicity)
        if self._current_roi_index >= 0 and self._roi_geometry is not None:
            # Update the list with current committed state
            self._rois[self._current_roi_index] = self._roi_geometry.roi

        self._current_roi_index = index

        if index >= 0:
            # Load the selected ROI
            roi = self._rois[index]
            self._roi_geometry = ROIGeometry(roi)
            candidate_roi = ROI(
                footprint=roi.footprint.copy(),
                weights=roi.weights.copy(),
                code=roi.code.copy()
            )
            self._candidate_geometry = ROIGeometry(candidate_roi)
        else:
            self._roi_geometry = None
            self._candidate_geometry = None

        # Clear refine state when switching
        self._refine_state = None
        if self._edit_mode == EditMode.REFINE:
            self._edit_mode = EditMode.NONE
            self.edit_mode_changed.emit(EditMode.NONE)

        self.current_roi_index_changed.emit(index)
        self._invalidate_corr_caches()
        self.roi_changed.emit()
        self.candidate_changed.emit()

    def add_roi(self, roi: ROI, roi_id: Optional[str] = None) -> int:
        """Add a new ROI to the list and return its index.

        Args:
            roi: The ROI to add.
            roi_id: Optional ID for the ROI. If None, auto-generates "ROI n".
        """
        self._rois.append(roi)
        index = len(self._rois) - 1
        if roi_id is None:
            roi_id = f"ROI {index + 1}"
        self._roi_ids.append(roi_id)
        self.roi_list_changed.emit()
        return index

    def add_and_select_roi(self, roi: ROI):
        """Add a new ROI and switch to it."""
        index = self.add_roi(roi)
        self.set_current_roi_index(index)

    def remove_roi(self, index: int):
        """Remove ROI at index."""
        if 0 <= index < len(self._rois):
            self._rois.pop(index)
            self._roi_ids.pop(index)

            # Adjust current index
            if self._current_roi_index == index:
                # Removed the current ROI
                if len(self._rois) == 0:
                    self.set_current_roi_index(-1)
                else:
                    self.set_current_roi_index(min(index, len(self._rois) - 1))
            elif self._current_roi_index > index:
                self._current_roi_index -= 1

            self.roi_list_changed.emit()

    def rename_roi(self, index: int, new_id: str):
        """Rename an ROI and enable ID persistence.

        Args:
            index: Index of the ROI to rename.
            new_id: New ID string for the ROI.
        """
        if 0 <= index < len(self._roi_ids):
            self._roi_ids[index] = new_id
            self._persist_ids = True  # User manually set an ID, so persist
            self.roi_id_changed.emit(index, new_id)

    def new_empty_roi(self):
        """Create a new empty ROI and switch to it."""
        empty = ROI.empty()
        if self._data_source is not None:
            empty = ROI(
                footprint=np.empty((0, 2), dtype=np.int32),
                weights=np.empty(0, dtype=np.float32),
                code=np.zeros(self._data_source.n_components, dtype=np.float32)
            )
        self.add_and_select_roi(empty)

    def all_roi_pixels(self) -> set:
        """Return set of all pixels covered by any ROI."""
        pixels = set()
        for roi in self._rois:
            pixels.update(map(tuple, roi.footprint))
        return pixels

    def _sync_current_roi_to_list(self):
        """Sync current ROI geometry back to the list."""
        if self._current_roi_index >= 0 and self._roi_geometry is not None:
            self._rois[self._current_roi_index] = self._roi_geometry.roi

    # ROI (committed state) - wraps current ROI
    @property
    def roi_geometry(self) -> Optional[ROIGeometry]:
        return self._roi_geometry

    @property
    def roi(self) -> Optional[ROI]:
        return self._roi_geometry.roi if self._roi_geometry else None

    def set_roi(self, roi: Optional[ROI]):
        """Set committed ROI and initialize candidate as copy.

        If no ROI is selected, this creates a new ROI and selects it.
        """
        if roi is None:
            self._roi_geometry = None
            self._candidate_geometry = None
            if self._current_roi_index >= 0:
                self._rois[self._current_roi_index] = ROI.empty()
        else:
            self._roi_geometry = ROIGeometry(roi)
            # Initialize candidate as copy
            candidate_roi = ROI(
                footprint=roi.footprint.copy(),
                weights=roi.weights.copy(),
                code=roi.code.copy()
            )
            self._candidate_geometry = ROIGeometry(candidate_roi)

            # Sync to list
            if self._current_roi_index >= 0:
                self._rois[self._current_roi_index] = roi
            else:
                # No ROI selected, add this as new
                self.add_and_select_roi(roi)
                return  # add_and_select_roi already emits signals

        self._invalidate_corr_caches()
        self.roi_changed.emit()
        self.candidate_changed.emit()

    # Candidate (working copy)
    @property
    def candidate_geometry(self) -> Optional[ROIGeometry]:
        return self._candidate_geometry

    @property
    def candidate(self) -> Optional[ROI]:
        return self._candidate_geometry.roi if self._candidate_geometry else None

    def ensure_candidate_exists(self):
        """Create empty candidate if none exists."""
        if self._candidate_geometry is None:
            empty_roi = ROI.empty()
            self._candidate_geometry = ROIGeometry(empty_roi)
            self.candidate_changed.emit()

    def notify_candidate_modified(self):
        """Call after modifying candidate through candidate_geometry.

        Recomputes candidate weights based on alignment with code.
        """
        if self._candidate_geometry is not None and self._data_source is not None:
            cand = self._candidate_geometry.roi
            if len(cand.footprint) > 0:
                from .operations import recompute_weights
                # Recompute code from current footprint
                code = self._data_source.extract_code(cand.footprint, cand.weights)
                # Recompute weights based on alignment with code
                new_weights = recompute_weights(cand.footprint, code, self._data_source)
                # Update candidate ROI with new weights and code
                self._candidate_geometry.roi._weights = new_weights
                self._candidate_geometry.roi._code = code
        self.candidate_changed.emit()

    # Deltas (computed on demand)
    @property
    def additions(self) -> set:
        """Pixels in candidate but not in committed ROI."""
        roi_pixels = self._roi_geometry.roi.pixel_set if self._roi_geometry else set()
        cand_pixels = self._candidate_geometry.roi.pixel_set if self._candidate_geometry else set()
        return cand_pixels - roi_pixels

    @property
    def subtractions(self) -> set:
        """Pixels in committed ROI but not in candidate."""
        roi_pixels = self._roi_geometry.roi.pixel_set if self._roi_geometry else set()
        cand_pixels = self._candidate_geometry.roi.pixel_set if self._candidate_geometry else set()
        return roi_pixels - cand_pixels

    @property
    def has_uncommitted_changes(self) -> bool:
        """True if candidate differs from committed ROI."""
        return bool(self.additions or self.subtractions)

    def accept_candidate(self):
        """Commit candidate to ROI, recomputing code and weights."""
        if self._candidate_geometry is None:
            return

        cand = self._candidate_geometry.roi
        footprint = cand.footprint.copy()

        # Recompute code and weights from data source
        if self._data_source is not None and len(footprint) > 0:
            from .operations import recompute_weights
            code = self._data_source.extract_code(footprint)
            weights = recompute_weights(footprint, code, self._data_source)
        else:
            code = cand.code.copy()
            weights = cand.weights.copy()

        committed_roi = ROI(
            footprint=footprint,
            weights=weights,
            code=code
        )
        self._roi_geometry = ROIGeometry(committed_roi)

        # Also update candidate to match
        self._candidate_geometry = ROIGeometry(ROI(
            footprint=footprint.copy(),
            weights=weights.copy(),
            code=code.copy()
        ))

        # Sync to list
        self._sync_current_roi_to_list()

        self._invalidate_corr_caches()
        self.roi_changed.emit()
        self.candidate_changed.emit()

    def reject_candidate(self):
        """Reset candidate to match committed ROI."""
        if self._roi_geometry is None:
            self._candidate_geometry = None
        else:
            roi = self._roi_geometry.roi
            candidate_roi = ROI(
                footprint=roi.footprint.copy(),
                weights=roi.weights.copy(),
                code=roi.code.copy()
            )
            self._candidate_geometry = ROIGeometry(candidate_roi)
        self.candidate_changed.emit()

    # View mode
    @property
    def view_mode(self) -> ViewMode:
        return self._view_mode

    def set_view_mode(self, mode: ViewMode):
        if self._view_mode != mode:
            self._view_mode = mode
            self.view_mode_changed.emit(mode)
            self.image_changed.emit()  # Image display needs update

    # Edit mode
    @property
    def edit_mode(self) -> EditMode:
        return self._edit_mode

    def set_edit_mode(self, mode: EditMode):
        if self._edit_mode != mode:
            old_mode = self._edit_mode
            self._edit_mode = mode

            # Clear refine state when leaving refine mode
            if old_mode == EditMode.REFINE and mode != EditMode.REFINE:
                self._refine_state = None

            self.edit_mode_changed.emit(mode)

    # Pen size
    @property
    def pen_size(self) -> int:
        return self._pen_size

    def set_pen_size(self, size: int):
        size = max(1, min(50, size))  # Clamp to reasonable range
        if self._pen_size != size:
            self._pen_size = size
            self.pen_size_changed.emit(size)

    # Refine state
    @property
    def refine_state(self) -> Optional["RefineState"]:
        return self._refine_state

    def set_refine_state(self, state: Optional["RefineState"]):
        self._refine_state = state
        if state is not None:
            self.refine_index_changed.emit(state.current_index)

    def set_refine_index(self, index: int):
        """Update refine slider position, applying changes to candidate."""
        if self._refine_state is None:
            return
        self._refine_state.set_index(index, self._candidate_geometry)
        self.refine_index_changed.emit(index)
        self.candidate_changed.emit()

    # Correlation seed
    @property
    def correlation_seed(self) -> Optional[tuple]:
        return self._correlation_seed

    def set_correlation_seed(self, row: int, col: int):
        self._correlation_seed = (row, col)
        self._invalidate_corr_caches()
        self.correlation_seed_changed.emit()
        if self._view_mode == ViewMode.CORRELATION:
            self.image_changed.emit()

    # Image data
    @property
    def mean_image(self) -> Optional[np.ndarray]:
        if self._data_source is not None:
            return self._data_source.mean_image()
        return None

    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Get image for current view mode.
        If map-layer machinery is active return the raw image for active_detail_map
        (used by extend/propose and minimap). Falls back to legacy view_mode behavior.
        """
        if self._data_source is None:
            return None

        # Prefer active_detail_map if available
        if hasattr(self, "_map_layers") and self._active_detail_map is not None:
            return self.get_raw_map(self._active_detail_map)

        # Legacy behavior (fallback)
        if self._view_mode == ViewMode.MEAN:
            return self._data_source.mean_image()
        elif self._view_mode == ViewMode.CORRELATION:
            if self._correlation_seed is not None:
                return self._data_source.correlation_map(*self._correlation_seed)
            # If no seed, use ROI centroid or return mean
            if self.roi is not None and len(self.roi.footprint) > 0:
                centroid = self.roi.footprint.mean(axis=0).astype(int)
                return self._data_source.correlation_map(centroid[0], centroid[1])
            return self._data_source.mean_image()
        elif self._view_mode == ViewMode.LOCAL_CORRELATION:
            return self._data_source.local_correlation_map()

        return self._data_source.mean_image()

    # --- Map layer accessors / mutators ---
    @property
    def map_layers(self) -> dict:
        """Return dict[ViewMode, MapLayerState]."""
        return self._map_layers

    @property
    def active_detail_map(self) -> ViewMode:
        return self._active_detail_map

    @property
    def summary_label(self) -> str:
        return self._summary_label

    def set_active_detail_map(self, mode: ViewMode):
        if self._active_detail_map == mode:
            return
        self._active_detail_map = mode
        self.active_detail_map_changed.emit(mode)
        # Update minimap / single-image consumers
        self.image_changed.emit()

    def set_map_enabled(self, mode: ViewMode, enabled: bool):
        state = self._map_layers.get(mode)
        if state is None:
            return
        if state.enabled == enabled:
            # Still make active when toggled on via button click
            self.map_layer_changed.emit(mode)
            self.image_changed.emit()
            return
        state.enabled = enabled
        self.map_layer_changed.emit(mode)
        self.image_changed.emit()

    def update_map_layer(self, mode: ViewMode, **kwargs):
        """Update MapLayerState fields and emit change signals."""
        state = self._map_layers.get(mode)
        if state is None:
            return
        for k, v in kwargs.items():
            if hasattr(state, k):
                setattr(state, k, v)
        # Invalidate gmean cache when lo/hi changes (gmean depends on normalization)
        if "lo" in kwargs or "hi" in kwargs:
            self._gmean_cache = {k: v for k, v in self._gmean_cache.items()
                                 if k[0] != mode}
        self.map_layer_changed.emit(mode)
        self.image_changed.emit()

    def get_raw_map(self, mode: ViewMode) -> np.ndarray:
        """Return raw (h,w) image for a given ViewMode. Correlation map is cached."""
        if self._data_source is None:
            return None

        if mode == ViewMode.MEAN:
            return self._data_source.mean_image()
        elif mode == ViewMode.CORRELATION:
            if self._corr_map_cache is not None:
                return self._corr_map_cache
            if self._correlation_seed is not None:
                ret = self._data_source.correlation_map(*self._correlation_seed)
            elif self.roi is not None and len(self.roi.footprint) > 0:
                centroid = self.roi.footprint.mean(axis=0).astype(int)
                ret = self._data_source.correlation_map(centroid[0], centroid[1])
            else:
                return self._data_source.mean_image()
            self._corr_map_cache = ret
            return ret
        elif mode == ViewMode.LOCAL_CORRELATION:
            return self._data_source.local_correlation_map()
        else:
            return self._data_source.mean_image()

    def get_histogram(self, mode: ViewMode, n_quantiles: int = 128) -> tuple:
        """Return cached (x, density) for quantile-based density estimate.

        Uses n_quantiles evenly-spaced quantiles; density estimated as
        1/diff(quantiles) (higher density where quantiles are close together).
        """
        if mode in self._hist_cache:
            return self._hist_cache[mode]
        raw = self.get_raw_map(mode)
        if raw is None:
            return (np.array([]), np.array([]))
        flat = np.ravel(raw)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return (np.array([]), np.array([]))
        probs = np.linspace(0, 1, n_quantiles)
        q = np.quantile(flat, probs)
        # Midpoints and local density
        x = (q[:-1] + q[1:]) / 2.0
        dx = np.diff(q)
        dx = np.maximum(dx, 1e-12)
        density = 1.0 / dx
        self._hist_cache[mode] = (x, density)
        return (x, density)

    def get_gmean(self, mode: ViewMode) -> float:
        """Return cached geometric mean of normalized map. Computes on miss.

        The gmean is computed on the lo/hi-normalized image clipped to (0,1],
        so it depends on the current lo/hi. We cache it keyed by (mode, lo, hi).
        """
        layer = self._map_layers.get(mode)
        if layer is None:
            return 0.5
        cache_key = (mode, layer.lo, layer.hi)
        if cache_key in self._gmean_cache:
            return self._gmean_cache[cache_key]
        raw = self.get_raw_map(mode)
        if raw is None:
            return 0.5
        denom = layer.hi - layer.lo
        if denom <= 1e-12:
            return 0.5
        norm = np.clip((raw - layer.lo) / denom, 0.0, 1.0)
        # Geometric mean via exp(mean(log(x))) on positive values only
        pos = norm[norm > 1e-10]
        if pos.size == 0:
            gmean = 0.5
        else:
            gmean = float(np.exp(np.mean(np.log(pos))))
        self._gmean_cache[cache_key] = gmean
        return gmean

    def _invalidate_corr_caches(self):
        """Clear correlation map, histogram, and gmean caches for CORRELATION mode."""
        self._corr_map_cache = None
        self._hist_cache.pop(ViewMode.CORRELATION, None)
        # Clear all gmean entries for CORRELATION (any lo/hi combo)
        self._gmean_cache = {k: v for k, v in self._gmean_cache.items()
                             if k[0] != ViewMode.CORRELATION}

    # Legacy compatibility
    def set_mean_image(self, image: np.ndarray):
        """Legacy method - creates MeanImageDataSource."""
        from .data_source import MeanImageDataSource
        self._data_source = MeanImageDataSource(image)
        self.image_changed.emit()
