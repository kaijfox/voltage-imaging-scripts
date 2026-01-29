"""Entry point for ROI GUI application."""

import sys
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject

from .state import AppState
from .roi import ROI
from .main_window import MainWindow
from .data_source import (
    ArraySVDDataSource,
    MeanImageDataSource,
    PrecomputedSVDDataSource,
    SVDDataSource
)
from .debounce import CallbackDebouncer
from ..timeseries.rois import ROICollection


class Autosaver(QObject):
    """Debounced autosave of ROI data to file."""

    def __init__(
        self, state: AppState, output_path: Path, parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.state = state
        self.output_path = output_path

        self._debouncer = CallbackDebouncer(self._save, delay_ms=500, parent=self)

        # Connect to relevant signals
        self.state.roi_changed.connect(self._request_save)
        self.state.roi_list_changed.connect(self._request_save)

    def _request_save(self):
        self._debouncer.request()

    def _save(self):
        """Save all ROIs to output file."""
        rois = self.state.rois
        if not rois:
            return

        # Get image shape from data source for .mat format
        shape = None
        if self.state.data_source is not None:
            shape = self.state.data_source.shape[:2]

        collection = ROICollection(rois=list(rois))
        collection.save(self.output_path, shape=shape)


def launch(
    data: Union[np.ndarray, "SVDVideo", str, Path],  # type: ignore
    output_path: Optional[Union[str, Path]] = None,
    roi: Optional[ROI] = None,
    n_components: Optional[int] = None,
    tmin: Optional[int] = None,
    tmax: Optional[int] = None,
) -> None:
    """Launch the ROI editor GUI.

    Args:
        data: One of:
            - (h, w, t) video array: SVD computed automatically
            - (h, w) mean image: static visualization only
            - SVDVideo: use pre-computed SVD
            - str/Path: load SVD from HDF5 file (SRSVD format)
        output_path: Path to save ROI data. Format determined by extension
                     (.npz, .mat, .h5). Autosave enabled if provided.
        roi: Optional initial ROI to display/edit.
        n_components: Number of SVD components to use. If None, auto-selected.
        tmin: Start frame index (inclusive). If None, starts at 0.
        tmax: End frame index (exclusive). If None, uses all frames.
    """
    from ..io.svd_video import SVDVideo
    from ..io.svd import SRSVD

    # Create or get QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Initialize state
    state = AppState()

    # Handle data input
    if isinstance(data, SVDVideo):
        print(f"Using pre-computed SVD.")
        data_source = PrecomputedSVDDataSource(
            data, n_components=n_components, tmin=tmin, tmax=tmax
        )
        print(f"Loaded {data_source.n_components} components")
    elif isinstance(data, (str, Path)):
        if not Path(data).exists():
            raise ValueError("Data source appears to be a nonexistent file.")
        print(f"Loading SVD from {data}...")
        svd = SRSVD(data).to_loaded_svd()
        data_source = PrecomputedSVDDataSource(
            svd, n_components=n_components, tmin=tmin, tmax=tmax
        )
        print(f"Loaded {data_source.n_components} components")
    elif isinstance(data, np.ndarray):
        if data.ndim == 3:
            print(f"Computing SVD from video {data.shape}...")
            data_source = ArraySVDDataSource(
                data, n_components=n_components, tmin=tmin, tmax=tmax
            )
            print(f"SVD complete: {data_source.n_components} components")
        elif data.ndim == 2:
            data_source = MeanImageDataSource(data)
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
    elif isinstance(data, SVDDataSource):
        data_source = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
    print("spatial", data_source._U.shape, "temporal", data_source._u.shape)

    state.set_data_source(data_source)

    if roi is not None:
        state.set_roi(roi)

    # Store output path and setup autosave
    state._output_path = Path(output_path) if output_path else None

    autosaver = None
    if state._output_path is not None:
        autosaver = Autosaver(state, state._output_path)
        print(f"Autosave enabled: {state._output_path}")

    # Create and show main window
    window = MainWindow(state)
    window.show()

    # Keep autosaver alive
    window._autosaver = autosaver

    # Run event loop (only if we created the app)
    try:
        # In notebook context?
        from IPython import get_ipython

        ipy = get_ipython()

        # Require %gui qt or run normal event loop
        gui_active = getattr(ipy, "active_eventloop", None) == "qt"
        if not gui_active:
            raise NameError("Throwing to normal event loop mode.")

        print("Running in notebook-embedded event loop mode.")
        return app, window

    except (NameError, ImportError):
        print("running standalone gui")
        app.exec()  # Standalone - block
        return app, window


def main():
    """CLI entry point for testing."""
    # Generate test data: synthetic video with a cell-like signal
    h, w, t = 256, 256, 500
    y, x = np.ogrid[:h, :w]

    # Background with some structure
    background = (np.sin(x / 30) * np.cos(y / 30) * 20 + 100).astype(np.float32)

    # Create a "cell" with time-varying fluorescence
    cy, cx, r = 128, 128, 25
    cell_mask = ((y - cy) ** 2 + (x - cx) ** 2) < r**2

    # Time-varying signal
    time = np.arange(t)
    signal = (
        np.sin(time / 50) * 30 + np.random.randn(t) * 5  # Slow oscillation  # Noise
    ).astype(np.float32)

    # Build video
    video = np.broadcast_to(background[:, :, np.newaxis], (h, w, t)).copy()
    video = video + np.random.randn(h, w, t).astype(np.float32) * 5  # Pixel noise

    # Add cell signal
    for ti in range(t):
        video[:, :, ti][cell_mask] += signal[ti]

    # Create test ROI (slightly offset from true cell)
    test_cy, test_cx, test_r = 130, 125, 20
    test_mask = ((y - test_cy) ** 2 + (x - test_cx) ** 2) < test_r**2
    test_roi = ROI.from_mask(test_mask)

    launch(video, roi=test_roi, n_components=50)


if __name__ == "__main__":
    main()
