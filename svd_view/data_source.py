"""Lazy HDF5 data source with hyperslab selection for efficient SVD reconstruction."""

import threading
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np


class LazyHDF5SVDSource:
    """Lazy HDF5 data source with hyperslab selection for efficient SVD frame reconstruction.

    HDF5 file structure (from SRSVD):
        - U: (n_frames, rank) - temporal components
        - S: (rank,) - singular values
        - Vh: (rank, height, width) - spatial components

    Frame reconstruction: frame[t] = U[t, :k] * S[:k] @ Vh[:k, r0:r1, c0:c1]

    Thread safety: Single file handle with threading.Lock for multi-threaded access.
    """

    def __init__(self, path: Union[str, Path]):
        """Open HDF5 file and cache metadata.

        Parameters
        ----------
        path : str or Path
            Path to HDF5 file containing U, S, Vh datasets.
        """
        self._path = Path(path)
        self._lock = threading.Lock()
        self._file: Optional[h5py.File] = None

        # Open file and cache metadata
        self._file = h5py.File(self._path, "r")

        # Validate required datasets
        for name in ("U", "S", "Vh"):
            if name not in self._file:
                raise ValueError(f"Missing required dataset '{name}' in {self._path}")

        # Cache dataset references
        self._U_dset = self._file["U"]
        self._S_dset = self._file["S"]
        self._Vh_dset = self._file["Vh"]

        # Cache metadata
        self._n_frames = self._U_dset.shape[0]
        self._max_rank = self._S_dset.shape[0]
        self._frame_shape = self._Vh_dset.shape[1:]  # (height, width) or (height, width, ...)

        # Validate shapes
        if self._U_dset.shape[1] != self._max_rank:
            raise ValueError(f"U rank mismatch: U.shape[1]={self._U_dset.shape[1]} vs S.shape[0]={self._max_rank}")
        if self._Vh_dset.shape[0] != self._max_rank:
            raise ValueError(f"Vh rank mismatch: Vh.shape[0]={self._Vh_dset.shape[0]} vs S.shape[0]={self._max_rank}")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def max_rank(self) -> int:
        return self._max_rank

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._frame_shape

    def reconstruct_frame(
        self,
        frame_idx: int,
        rank: Optional[int] = None,
        spatial_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Reconstruct a single frame using hyperslab selection.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based).
        rank : int, optional
            Truncation rank. If None, use full rank.
        spatial_roi : tuple, optional
            Spatial window as (r0, r1, c0, c1). If None, use full frame.

        Returns
        -------
        frame : ndarray
            Reconstructed frame with shape (r1-r0, c1-c0) or full frame_shape.
        """
        if rank is None:
            rank = self._max_rank
        rank = min(rank, self._max_rank)

        with self._lock:
            # Read U row for this frame: shape (k,)
            U_row = self._U_dset[frame_idx, :rank]

            # Read S: shape (k,)
            S_k = self._S_dset[:rank]

            # Read Vh with spatial ROI if specified
            if spatial_roi is not None:
                r0, r1, c0, c1 = spatial_roi
                # Vh: (rank, height, width)
                Vh_sel = self._Vh_dset[:rank, r0:r1, c0:c1]
            else:
                Vh_sel = self._Vh_dset[:rank, ...]

        # Reconstruct: (k,) * (k,) @ (k, h', w') -> (h', w')
        # U_row * S_k gives scaled coefficients, then contract with Vh
        scaled = U_row * S_k  # (k,)
        frame = np.tensordot(scaled, Vh_sel, axes=(0, 0))

        return frame

    def close(self):
        """Close the HDF5 file."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()


def open_svd_source(path: Union[str, Path]) -> LazyHDF5SVDSource:
    """Open an HDF5 SVD file as a lazy data source.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.

    Returns
    -------
    source : LazyHDF5SVDSource
    """
    return LazyHDF5SVDSource(path)
