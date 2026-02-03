from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import os
import numpy as np


@dataclass
class ROI:
    """Region of interest with footprint, weights, and temporal code.

    Attributes:
        footprint: (n_pixels, 2) array of (row, col) pixel coordinates
        weights: (n_pixels,) array of pixel weights
        code: (n_components,) or (n_timepoints,) temporal representation
    """
    footprint: np.ndarray
    weights: np.ndarray
    code: np.ndarray

    @classmethod
    def from_mask(cls, mask: np.ndarray, weights: Optional[np.ndarray] = None,
                  code: Optional[np.ndarray] = None) -> "ROI":
        """Create ROI from boolean mask."""
        footprint = np.argwhere(mask)
        if weights is None:
            weights = np.ones(len(footprint))
        elif weights.shape == mask.shape:
            weights = weights[mask]
        if code is None:
            code = np.array([])
        return cls(footprint=footprint, weights=weights, code=code)

    @classmethod
    def from_linear_indices(cls, indices: np.ndarray, shape: Tuple[int, int],
                            weights: Optional[np.ndarray] = None,
                            code: Optional[np.ndarray] = None) -> "ROI":
        """Create ROI from linear indices (column-major/Fortran order, 1-indexed for MATLAB compat)."""
        # Convert 1-indexed to 0-indexed, then unravel
        indices_0 = np.asarray(indices).ravel() - 1
        rows, cols = np.unravel_index(indices_0, shape, order='F')
        footprint = np.column_stack([rows, cols])
        if weights is None:
            weights = np.ones(len(footprint))
        if code is None:
            code = np.array([])
        return cls(footprint=footprint, weights=weights, code=code)

    @classmethod
    def empty(cls) -> "ROI":
        """Create an empty ROI."""
        return cls(
            footprint=np.zeros((0, 2), dtype=int),
            weights=np.zeros(0),
            code=np.array([])
        )

    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Convert to boolean mask of given shape."""
        mask = np.zeros(shape, dtype=bool)
        if len(self.footprint) > 0:
            mask[self.footprint[:, 0], self.footprint[:, 1]] = True
        return mask

    def to_linear_indices(self, shape: Tuple[int, int]) -> np.ndarray:
        """Convert to linear indices (column-major, 1-indexed for MATLAB)."""
        if len(self.footprint) == 0:
            return np.array([], dtype=int)
        indices = np.ravel_multi_index(
            (self.footprint[:, 0], self.footprint[:, 1]), shape, order='F'
        )
        return indices + 1  # 1-indexed for MATLAB


@dataclass
class ROICollection:
    rois: List['ROI'] = field(default_factory=list)
    image_shape: Optional[Tuple[int, int]] = None  # Set when loaded from .mat
    ids: Optional[List[str]] = None
    colors: Optional[List[Tuple[int, int, int]]] = None

    def save(self, path: os.PathLike, shape: Optional[Tuple[int, int]] = None):
        """Save all ROIs to output file."""
        path = Path(path)

        # Collect data from all ROIs
        footprints = [roi.footprint for roi in self.rois]
        weights = [roi.weights for roi in self.rois]
        codes = [roi.code for roi in self.rois]

        # Save based on extension
        ext = path.suffix.lower()
        if ext == ".mat":
            # shape required for .mat (SemiSeg uses linear indices)
            shape = shape or self.image_shape
            if shape is None:
                raise ValueError("shape required for .mat format")
            self._save_mat(path, self.rois, shape)
        elif ext == ".h5":
            self._save_h5(path, footprints, weights, codes)
        else:
            # Default to npz
            self._save_npz(path, footprints, weights, codes)

    @staticmethod
    def _save_npz(path, footprints, weights, codes):
        """Save as numpy npz file."""
        import numpy as np

        np.savez(
            path,
            footprints=np.array(footprints, dtype=object),
            weights=np.array(weights, dtype=object),
            codes=np.array(codes, dtype=object),
            n_rois=len(footprints),
        )

    @staticmethod
    def _save_mat(path, rois: List['ROI'], shape: Tuple[int, int]):
        """Save as MATLAB mat file (SemiSeg-compatible).

        Creates roiList struct array with pixel_idx (linear indices, 1-indexed, column-major).
        """
        try:
            from scipy.io import savemat

            # Build struct array for SemiSeg compatibility
            roi_list = []
            for roi in rois:
                roi_list.append({
                    'pixel_idx': roi.to_linear_indices(shape),
                    'weights': roi.weights.astype(np.float64),
                    'code': roi.code.astype(np.float64) if len(roi.code) > 0 else np.array([]),
                })

            savemat(str(path), {
                'roiList': np.array(roi_list, dtype=object) if roi_list else np.array([], dtype=object),
                'image_shape': np.array(shape),
            })
        except ImportError:
            print("scipy not available, falling back to npz")
            footprints = [r.footprint for r in rois]
            weights = [r.weights for r in rois]
            codes = [r.code for r in rois]
            ROICollection._save_npz(path.with_suffix('.npz'), footprints, weights, codes)

    @staticmethod
    def _save_h5(path, footprints, weights, codes):
        """Save as HDF5 file."""
        try:
            import h5py

            with h5py.File(path, "w") as f:
                for i, (fp, w, c) in enumerate(zip(footprints, weights, codes)):
                    grp = f.create_group(f"roi_{i}")
                    grp.create_dataset("footprint", data=fp)
                    grp.create_dataset("weights", data=w)
                    grp.create_dataset("code", data=c)
                f.attrs["n_rois"] = len(footprints)
        except ImportError:
            print("h5py not available, falling back to npz")
            ROICollection._save_npz(path.with_suffix('.npz'), footprints, weights, codes)

    @classmethod
    def load(cls, path: os.PathLike, shape: Optional[Tuple[int, int]] = None) -> 'ROICollection':
        """Load ROIs from file. Shape required for .mat if not stored in file."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext == ".mat":
            return cls._load_mat(path, shape)
        elif ext == ".h5":
            return cls._load_h5(path)
        else:
            return cls._load_npz(path)

    @classmethod
    def _load_npz(cls, path) -> 'ROICollection':
        data = np.load(path, allow_pickle=True)
        footprints = data['footprints']
        weights = data['weights']
        codes = data['codes']
        rois = [
            ROI(footprint=fp, weights=w, code=c)
            for fp, w, c in zip(footprints, weights, codes)
        ]
        return cls(rois=rois)

    @classmethod
    def _load_mat(cls, path, shape: Optional[Tuple[int, int]] = None) -> 'ROICollection':
        from scipy.io import loadmat

        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)

        # Try to get shape from file
        if 'image_shape' in data:
            file_shape = tuple(data['image_shape'].astype(int))
            shape = shape or file_shape
        if shape is None:
            raise ValueError("shape required to load .mat (not stored in file)")

        roi_list = data.get('roiList', data.get('CellList', []))
        if not hasattr(roi_list, '__len__') or isinstance(roi_list, np.ndarray) and roi_list.ndim == 0:
            roi_list = [roi_list] if roi_list is not None else []

        rois = []
        for roi_data in roi_list:
            # Handle both Simon's (pixel_idx) and Kyle's (PixelIdxList) formats
            if hasattr(roi_data, 'pixel_idx'):
                indices = np.atleast_1d(roi_data.pixel_idx)
            elif hasattr(roi_data, 'PixelIdxList'):
                indices = np.atleast_1d(roi_data.PixelIdxList)
            else:
                continue
            # Ensure integer format
            indices = indices.astype(int)

            weights = getattr(roi_data, 'weights', None)
            if weights is not None:
                weights = np.atleast_1d(weights)
            code = getattr(roi_data, 'code', None)
            if code is not None:
                code = np.atleast_1d(code)

            rois.append(ROI.from_linear_indices(indices, shape, weights=weights, code=code))

        return cls(rois=rois, image_shape=shape)

    @classmethod
    def _load_h5(cls, path) -> 'ROICollection':
        import h5py

        rois = []
        with h5py.File(path, 'r') as f:
            n_rois = f.attrs.get('n_rois', 0)
            for i in range(n_rois):
                grp = f[f'roi_{i}']
                rois.append(ROI(
                    footprint=grp['footprint'][:],
                    weights=grp['weights'][:],
                    code=grp['code'][:],
                ))
        return cls(rois=rois)
