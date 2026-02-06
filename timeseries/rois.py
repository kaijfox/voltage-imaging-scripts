from dataclasses import dataclass, field
from typing import Tuple, Optional, Mapping, ClassVar, List, Sequence
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
    def from_mask(
        cls,
        mask: np.ndarray,
        weights: Optional[np.ndarray] = None,
        code: Optional[np.ndarray] = None,
    ) -> "ROI":
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
    def from_linear_indices(
        cls,
        indices: np.ndarray,
        shape: Tuple[int, int],
        weights: Optional[np.ndarray] = None,
        code: Optional[np.ndarray] = None,
    ) -> "ROI":
        """Create ROI from linear indices (column-major/Fortran order, 1-indexed for MATLAB compat)."""
        # Convert 1-indexed to 0-indexed, then unravel
        indices_0 = np.asarray(indices).ravel() - 1
        rows, cols = np.unravel_index(indices_0, shape, order="F")
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
            code=np.array([]),
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
            (self.footprint[:, 0], self.footprint[:, 1]), shape, order="F"
        )
        return indices + 1  # 1-indexed for MATLAB


@dataclass
class ROICollection:
    rois: List["ROI"] = field(default_factory=list)
    image_shape: Optional[Tuple[int, int]] = None  # Set when loaded from .mat
    ids: Optional[List[str]] = None
    ids_short: Optional[List[str]] = None
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
            self._save_mat(path, self.rois, shape, ids=self.ids)
        elif ext == ".h5":
            self._save_h5(
                path, footprints, weights, codes, ids=self.ids, ids_short=self.ids_short
            )
        else:
            # Default to npz
            self._save_npz(
                path, footprints, weights, codes, ids=self.ids, ids_short=self.ids_short
            )

    @staticmethod
    def _save_npz(path, footprints, weights, codes, ids=None, ids_short=None):
        """Save as numpy npz file."""
        import numpy as np

        np.savez(
            path,
            footprints=np.array(footprints, dtype=object),
            weights=np.array(weights, dtype=object),
            codes=np.array(codes, dtype=object),
            ids=(
                np.array(ids, dtype=object)
                if ids is not None
                else np.array([], dtype=object)
            ),
            ids_short=(
                np.array(ids_short, dtype=object)
                if ids_short is not None
                else np.array([], dtype=object)
            ),
            n_rois=len(footprints),
        )

    @staticmethod
    def _save_mat(
        path, rois: List["ROI"], shape: Tuple[int, int], ids: Optional[List[str]] = None
    ):
        """Save as MATLAB mat file (SemiSeg-compatible).

        Creates roiList struct array with pixel_idx (linear indices, 1-indexed, column-major).
        """
        try:
            from scipy.io import savemat

            # Build struct array for SemiSeg compatibility
            roi_list = []
            for roi in rois:
                roi_list.append(
                    {
                        "pixel_idx": roi.to_linear_indices(shape),
                        "weights": roi.weights.astype(np.float64),
                        "code": (
                            roi.code.astype(np.float64)
                            if len(roi.code) > 0
                            else np.array([])
                        ),
                    }
                )
                if ids is not None:
                    roi_list[-1]["id"] = ids[len(roi_list) - 1]

            savemat(
                str(path),
                {
                    "roiList": (
                        np.array(roi_list, dtype=object)
                        if roi_list
                        else np.array([], dtype=object)
                    ),
                    "image_shape": np.array(shape),
                },
            )
        except ImportError:
            print("scipy not available, falling back to npz")
            footprints = [r.footprint for r in rois]
            weights = [r.weights for r in rois]
            codes = [r.code for r in rois]
            ROICollection._save_npz(
                path.with_suffix(".npz"), footprints, weights, codes
            )

    @staticmethod
    def _save_h5(path, footprints, weights, codes, ids=None, ids_short=None):
        """Save as HDF5 file."""
        try:
            import h5py

            string_dt = h5py.string_dtype(encoding="utf-8")
            with h5py.File(path, "w") as f:
                for i, (fp, w, c) in enumerate(zip(footprints, weights, codes)):
                    grp = f.create_group(f"roi_{i}")
                    grp.create_dataset("footprint", data=fp)
                    grp.create_dataset("weights", data=w)
                    grp.create_dataset("code", data=c)
                f.attrs["n_rois"] = len(footprints)
                if ids is not None:
                    f.create_dataset(
                        "ids",
                        data=np.array(ids, dtype=object).astype("S"),
                        dtype=string_dt,
                    )
                if ids_short is not None:
                    f.create_dataset(
                        "ids_short",
                        data=np.array(ids_short, dtype=object).astype("S"),
                        dtype=string_dt,
                    )
        except ImportError:
            print("h5py not available, falling back to npz")
            ROICollection._save_npz(
                path.with_suffix(".npz"),
                footprints,
                weights,
                codes,
                ids=ids,
                ids_short=ids_short,
            )

    @classmethod
    def load(
        cls, path: os.PathLike, shape: Optional[Tuple[int, int]] = None
    ) -> "ROICollection":
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
    def _load_npz(cls, path) -> "ROICollection":
        data = np.load(path, allow_pickle=True)
        footprints = data["footprints"]
        weights = data["weights"]
        codes = data["codes"]
        ids = data["ids"].tolist() if "ids" in data.files else None
        ids_short = data["ids_short"].tolist() if "ids_short" in data.files else None
        rois = [
            ROI(footprint=fp, weights=w, code=c)
            for fp, w, c in zip(footprints, weights, codes)
        ]
        return cls(rois=rois, ids=ids, ids_short=ids_short)

    @classmethod
    def _load_mat(
        cls, path, shape: Optional[Tuple[int, int]] = None
    ) -> "ROICollection":
        from scipy.io import loadmat

        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)

        # Try to get shape from file
        if "image_shape" in data:
            file_shape = tuple(data["image_shape"].astype(int))
            shape = shape or file_shape
        if shape is None:
            raise ValueError("shape required to load .mat (not stored in file)")

        roi_list = data.get("roiList", data.get("CellList", []))
        if (
            not hasattr(roi_list, "__len__")
            or isinstance(roi_list, np.ndarray)
            and roi_list.ndim == 0
        ):
            roi_list = [roi_list] if roi_list is not None else []

        rois = []
        ids = []
        ids_short = []
        for roi_data in roi_list:
            # Handle both Simon's (pixel_idx) and Kyle's (PixelIdxList) formats
            if hasattr(roi_data, "pixel_idx"):
                indices = np.atleast_1d(roi_data.pixel_idx)
            elif hasattr(roi_data, "PixelIdxList"):
                indices = np.atleast_1d(roi_data.PixelIdxList)
            else:
                continue
            # Ensure integer format
            indices = indices.astype(int)

            weights = getattr(roi_data, "weights", None)
            if weights is not None:
                weights = np.atleast_1d(weights)
            code = getattr(roi_data, "code", None)
            if code is not None:
                code = np.atleast_1d(code)

            rois.append(
                ROI.from_linear_indices(indices, shape, weights=weights, code=code)
            )

            # ids stored as 'id' in _save_mat; tolerate other common names
            id_val = getattr(roi_data, "id", None)
            if id_val is None:
                id_val = getattr(roi_data, "ID", None)
            if id_val is not None:
                ids.append(str(id_val))

            # optional short id
            short_val = (
                getattr(roi_data, "id_short", None)
                or getattr(roi_data, "ids_short", None)
                or getattr(roi_data, "short_id", None)
            )
            if short_val is not None:
                ids_short.append(str(short_val))

        ids = ids if len(ids) > 0 else None
        ids_short = ids_short if len(ids_short) > 0 else None

        return cls(rois=rois, image_shape=shape, ids=ids, ids_short=ids_short)

    @classmethod
    def _load_h5(cls, path) -> "ROICollection":
        import h5py

        rois = []
        ids = None
        ids_short = None
        with h5py.File(path, "r") as f:
            n_rois = f.attrs.get("n_rois", 0)
            for i in range(n_rois):
                grp = f[f"roi_{i}"]
                rois.append(
                    ROI(
                        footprint=grp["footprint"][:],
                        weights=grp["weights"][:],
                        code=grp["code"][:],
                    )
                )
            if "ids" in f:
                ids = [str(x) for x in f["ids"][:]]
            if "ids_short" in f:
                ids_short = [str(x) for x in f["ids_short"][:]]
        return cls(rois=rois, ids=ids, ids_short=ids_short)


@dataclass(frozen=True)
class HierarchicalId:
    parts: Tuple[str, ...]
    meta: Optional[Mapping] = None

    @classmethod
    def from_string(cls, s: str):
        parts = tuple(p for p in (s or "").strip().split(".") if p != "")
        return cls(parts=parts, meta=None)

    def __str__(self):
        return ".".join(self.parts)

    def parent(self) -> Optional["HierarchicalId"]:
        if len(self.parts) <= 1:
            return None
        return HierarchicalId(parts=self.parts[:-1], meta=None)

    def child(self, token: str) -> "HierarchicalId":
        return HierarchicalId(parts=self.parts + (token,), meta=None)

    def depth(self) -> int:
        return len(self.parts)

    def is_ancestor_of(self, other: "HierarchicalId") -> bool:
        return (
            len(self.parts) < len(other.parts)
            and other.parts[: len(self.parts)] == self.parts
        )


@dataclass(frozen=True)
class ProcessROIId(HierarchicalId):
    ROOT: ClassVar[str] = "soma"
    NODE: ClassVar[str] = "branch"
    ROOT_SHORT: ClassVar[str] = "s"
    NODE_SHORT: ClassVar[str] = "b"
    SEP: ClassVar[str] = "."

    @classmethod
    def from_string(cls, s: str):
        # 1.b2 -> ('1', '2'), kind=branch
        # soma1.branch2 -> ('1', '2'), kind=branch

        parts = tuple(p for p in (s or "").strip().split(cls.SEP) if p != "")
        if not len(parts):
            raise ValueError("ProcessROIId string must contain a valid part")

        # Infer kind or error if not specified
        if parts[-1].startswith(cls.ROOT_SHORT) or parts[-1].startswith(cls.ROOT):
            kind = cls.ROOT
        elif parts[-1].startswith(cls.NODE_SHORT) or parts[-1].startswith(cls.NODE):
            kind = cls.NODE
        elif len(parts) > 1:
            # Infer kind=node if multiple parts and no prefix on last part
            kind = cls.NODE
        else:
            raise ValueError(
                f"Depth-1 {cls.__name__} must have prefix, one of",
                f"({cls.ROOT_SHORT}, {cls.ROOT}, {cls.NODE_SHORT}, {cls.NODE})",
            )

        # Remove kind prefixes
        remove = lambda k: lambda p: p if not p.startswith(k) else p[len(k) :]
        parts = filter(remove(cls.ROOT), parts)
        parts = filter(remove(cls.NODE), parts)
        parts = filter(remove(cls.ROOT_SHORT), parts)
        parts = filter(remove(cls.NODE_SHORT), parts)
        parts = tuple(parts)

        return cls(parts=parts, meta={"kind": kind})

    @staticmethod
    def resolve_kind(all_ids, parts):
        if all_ids is None:
            return None
        for it in all_ids:
            pid = (
                it
                if isinstance(it, ProcessROIId)
                else ProcessROIId.from_string(str(it))
            )
            if pid.parts == parts:
                return (pid.meta or {}).get("kind")
        return None

    def short(self, all_ids: Sequence["ProcessROIId"] = None) -> str:
        """Return shortform string representation; when provided, use kinds from
        all_ids for parents."""
        if "kind" not in (self.meta or {}):
            raise ValueError("Cannot determine kind for string representation.")

        out = []
        for i in range(len(self.parts)):

            p = self.parts[: i + 1]
            if i == len(self.parts) - 1:
                prefix = ProcessROIId.resolve_kind(all_ids, p) or ""
            else:
                prefix = self.meta.get("kind", "")
            prefix = self.ROOT_SHORT if prefix == self.ROOT else prefix
            prefix = self.NODE_SHORT if prefix == self.NODE else prefix
            out.append(f"{prefix}{self.parts[i]}")

        return self.SEP.join(out)

    def long(self, all_ids: Sequence["ProcessROIId"]) -> str:
        """Return longform string representation; when provided, use kinds from
        all_ids for parents."""
        if "kind" not in (self.meta or {}):
            raise ValueError("Cannot determine kind for string representation.")

        out = []
        for i in range(len(self.parts)):

            p = self.parts[: i + 1]
            if i == len(self.parts) - 1:
                prefix = ProcessROIId.resolve_kind(all_ids, p) or ""
            else:
                prefix = self.meta.get("kind", "")
            out.append(f"{prefix}{self.parts[i]}")

        return self.SEP.join(out)


class ROIHierarchy:
    """Minimal helper to work with dot-separated ROI ids as a forest.

    Accepts inputs as either `HierarchicalId` or string; works internally with
    `HierarchicalId` objects but returns canonical string ids for compatibility.
    """

    def __init__(self, ids):
        # build list of HierarchicalId objects
        hlist = []
        for it in ids:
            hid = (
                it
                if isinstance(it, HierarchicalId)
                else HierarchicalId.from_string(str(it))
            )
            hlist.append(hid)

        self._hids = hlist
        # canonical string keys
        self._keys = [str(h) for h in self._hids]
        self._id_to_index = {k: idx for idx, k in enumerate(self._keys)}

        self._children = {k: [] for k in self._keys}
        self._parent = {}

        for h in self._hids:
            k = str(h)
            p = h.parent()
            if p is not None and str(p) in self._children:
                pk = str(p)
                self._parent[k] = pk
                self._children[pk].append(k)
            else:
                self._parent[k] = None

        self._roots = [k for k, p in self._parent.items() if p is None]

    @classmethod
    def from_collection(cls, coll):
        ids = getattr(coll, "ids", None)
        if ids is None:
            raise ValueError("collection has no ids")
        return cls(ids)

    def _to_hid(self, id_or_str):
        return (
            id_or_str
            if isinstance(id_or_str, HierarchicalId)
            else HierarchicalId.from_string(str(id_or_str))
        )

    def _key(self, id_or_str):
        return str(self._to_hid(id_or_str))

    @property
    def ids(self):
        return list(self._keys)

    @property
    def roots(self):
        return list(self._roots)

    @property
    def size(self):
        return len(self._keys)

    def contains(self, id_str):
        return self._key(id_str) in self._id_to_index

    def index(self, id_str):
        return self._id_to_index[self._key(id_str)]

    def parent(self, id_str):
        return self._parent.get(self._key(id_str))

    def children(self, id_str):
        return list(self._children.get(self._key(id_str), []))

    def ancestors(self, id_str):
        out = []
        cur = self._key(id_str)
        while True:
            p = self._parent.get(cur)
            if not p:
                break
            out.append(p)
            cur = p
        return out

    def root_of(self, id_str):
        cur = self._key(id_str)
        while True:
            p = self._parent.get(cur)
            if not p:
                return cur
            cur = p

    def subtree(self, id_str):
        """Return list of id_str and all descendants in pre-order (strings)."""
        start = self._key(id_str)
        out = []
        stack = [start]
        while stack:
            node = stack.pop()
            out.append(node)
            ch = list(self._children.get(node, []))
            for c in reversed(ch):
                stack.append(c)
        return out

    def color_greedy(self):
        """Adjacent (parent-child) coloring. Forest -> 2 colors (0/1) via DFS.

        Returns mapping from canonical id string -> color int.
        """
        colors = {}
        for root in self._roots:
            if root in colors:
                continue
            stack = [(root, 0)]
            while stack:
                node, col = stack.pop()
                if node in colors:
                    continue
                colors[node] = col
                for c in self._children.get(node, []):
                    if c not in colors:
                        stack.append((c, 1 - col))
        return colors
