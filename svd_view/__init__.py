"""Napari-based SVD video viewer with async frame reconstruction.

This module provides a viewer for HDF5 SVD video files created by SRSVD.
Frames are reconstructed on-the-fly via U @ diag(S) @ Vh, with a worker
thread for responsive UI.

Usage:
    from imaging_scripts.svd_view import launch

    # From HDF5 file
    viewer = launch("/path/to/svd.h5")

    # From SVDVideo object (loads into memory)
    viewer = launch(svd_video_instance)

    # With initial parameters
    viewer = launch(path, rank=50, spatial_roi=(100, 200, 100, 200))

    # Integrate with existing napari viewer
    viewer = launch(path, napari_viewer=existing_viewer)
"""

from pathlib import Path
from typing import Optional, Tuple, Union

from .data_source import LazyHDF5SVDSource
from .state import PlaybackState, TelemetryData, ViewerState
from .viewer import SVDViewer


def launch(
    source: Union[str, Path, "SVDVideo", LazyHDF5SVDSource], #type: ignore
    napari_viewer=None,
    rank: Optional[int] = None,
    spatial_roi: Optional[Tuple[int, int, int, int]] = None,
) -> SVDViewer:
    """Launch the SVD video viewer.

    Parameters
    ----------
    source : str, Path, SVDVideo, or LazyHDF5SVDSource
        Data source. Can be:
        - Path to HDF5 file (lazy loading, recommended for large files)
        - SVDVideo object (will write to temp file, then lazy load)
        - LazyHDF5SVDSource (existing data source)
    napari_viewer : napari.Viewer, optional
        Existing napari viewer to use. If None, creates a new viewer.
    rank : int, optional
        Initial truncation rank. If None, uses full rank.
    spatial_roi : tuple, optional
        Initial spatial ROI as (r0, r1, c0, c1).

    Returns
    -------
    viewer : SVDViewer
        The SVD viewer instance.

    Examples
    --------
    >>> from imaging_scripts.svd_view import launch
    >>> viewer = launch("data/svd.h5")

    >>> # With truncated rank for faster playback
    >>> viewer = launch("data/svd.h5", rank=50)

    >>> # With spatial ROI
    >>> viewer = launch("data/svd.h5", spatial_roi=(100, 200, 100, 200))
    """
    import napari

    # Handle SVDVideo input by writing to temp file
    if hasattr(source, "U") and hasattr(source, "S") and hasattr(source, "Vt"):
        import tempfile
        import h5py

        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        with h5py.File(temp_path, "w") as f:
            f.create_dataset("U", data=source.U)
            f.create_dataset("S", data=source.S)
            f.create_dataset("Vh", data=source.Vt)

        source = temp_path

    # Create viewer
    viewer = SVDViewer(
        source,
        napari_viewer=napari_viewer,
        rank=rank,
        spatial_roi=spatial_roi,
    )

    # If we created the napari viewer, start the event loop
    if napari_viewer is None:
        napari.run()

    return viewer


__all__ = [
    "launch",
    "SVDViewer",
    "ViewerState",
    "PlaybackState",
    "TelemetryData",
    "LazyHDF5SVDSource",
]
