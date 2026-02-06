from dataclasses import dataclass, field
from typing import Tuple, Optional, Mapping, ClassVar, List, Sequence
from pathlib import Path
import os
import itertools
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
    _is_hid: ClassVar[bool] = True

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

    def repr_in_tree(self) -> str:
        """Return string representation suitable for tree display."""
        return str(self)


@dataclass(frozen=True)
class ProcessROIId(HierarchicalId):
    ROOT: ClassVar[str] = "soma"
    NODE: ClassVar[List[str]] = ["branch"]
    DEFAULT: ClassVar[str] = "unk"
    SHORT: ClassVar[dict[str, str]] = {
        "soma": "s",
        "branch": "b",
        "unk": "u",
    }
    SEP: ClassVar[str] = "."

    @classmethod
    def from_string(cls, s: str):
        # 1.b2 -> ('1', '2'), kind=branch
        # soma1.branch2 -> ('1', '2'), kind=branch

        parts = tuple(p for p in (s or "").strip().split(cls.SEP) if p != "")
        if not len(parts):
            raise ValueError("ProcessROIId string must contain a valid part")

        # Check for kind prefix on last part
        kind = None
        kinds = [cls.ROOT, cls.DEFAULT] + cls.NODE + list(cls.SHORT.values())
        for k in kinds:
            if parts[-1].startswith(k):
                kind = k
        # Convert short kind to full name
        if kind in cls.SHORT.values():
            kind = next(long for long, short in cls.SHORT.items() if short == kind)
        # DEFAULT kind if none found
        if kind is None:
            kind = cls.DEFAULT

        # Remove kind prefixes to get ids
        remove = lambda k: lambda p: p if not p.startswith(k) else p[len(k) :]
        for k in [cls.ROOT, cls.DEFAULT] + cls.NODE:
            parts = tuple(map(remove(k), parts))
            parts = tuple(map(remove(cls.SHORT[k]), parts))
        parts = tuple(parts)

        return cls(parts=parts, meta={"kind": kind})

    @classmethod
    def resolve_kind(cls, all_ids, parts):
        if all_ids is None:
            return None
        for it in all_ids:
            pid = (
                it
                if isinstance(it, ProcessROIId)
                else ProcessROIId.from_string(str(it))
            )
            if pid.parts == parts:
                return (pid.meta or {}).get("kind", cls.DEFAULT)
        return None

    def short(self, all_ids: Sequence["ProcessROIId"] = None) -> str:
        """Return shortform string representation; when provided, use kinds from
        all_ids for parents."""
        if "kind" not in (self.meta or {}):
            raise ValueError("Cannot determine kind for string representation.")

        out = []
        for i in range(len(self.parts)):

            p = self.parts[: i + 1]
            if i < len(self.parts) - 1:
                prefix = ""
                if all_ids is not None:
                    prefix = ProcessROIId.resolve_kind(all_ids, p) or self.DEFAULT
                prefix = ProcessROIId.SHORT.get(prefix, prefix)
            else:
                prefix = self.meta.get("kind", self.DEFAULT)
                prefix = ProcessROIId.SHORT.get(prefix, prefix)

            out.append(f"{prefix}{self.parts[i]}")

        return self.SEP.join(out)

    def long(self, all_ids: Sequence["ProcessROIId"] = None) -> str:
        """Return longform string representation; when provided, use kinds from
        all_ids for parents."""
        if "kind" not in (self.meta or {}):
            raise ValueError("Cannot determine kind for string representation.")

        out = []
        for i in range(len(self.parts)):

            p = self.parts[: i + 1]
            if i < len(self.parts) - 1:
                prefix = ""
                if all_ids is not None:
                    prefix = ProcessROIId.resolve_kind(all_ids, p) or self.DEFAULT
            else:
                prefix = self.meta.get("kind", "")
            out.append(f"{prefix}{self.parts[i]}")

        return self.SEP.join(out)
    
    def __str__(self):
        return self.short()

    def repr_in_tree(self) -> str:
        """Return string representation suitable for tree display."""
        return self.SEP + self.meta.get("kind", self.DEFAULT) + str(self.parts[-1])


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
                if hasattr(it, "_is_hid") and it._is_hid
                else HierarchicalId.from_string(str(it))
            )
            hlist.append(hid)

        self._hids = hlist
        # canonical string keys
        self._keys = [self._key(h) for h in self._hids]
        self._id_to_index = {k: idx for idx, k in enumerate(self._keys)}

        self._children = {k: [] for k in self._keys}
        self._parent = {}

        for h in self._hids:
            k = self._key(h)
            p = h.parent()
            if p is not None and self._key(p) in self._children:
                pk = self._key(p)
                self._parent[k] = pk
                self._children[pk].append(k)
            else:
                self._parent[k] = None

        self._roots = [k for k, p in self._parent.items() if p is None]

    @classmethod
    def from_collection(cls, coll, hid_type=HierarchicalId):
        ids = getattr(coll, "ids", None)
        if ids is None:
            raise ValueError("collection has no ids")
        hids = []
        for i in ids:
            if isinstance(i, str):
                hid = hid_type.from_string(str(i))
                hids.append(hid)
            elif isinstance(i, HierarchicalId):
                hids.append(i)
            else:
                raise ValueError("ids must be str or HierarchicalId")
        return cls(hids)

    def _to_hid(self, id_or_str):
        if hasattr(id_or_str, "_is_hid") and id_or_str._is_hid:
            return id_or_str
        elif isinstance(id_or_str, tuple):
            try:
                return next(
                    h
                    for h in self._hids
                    if id_or_str == h.parts
                )
            except StopIteration:
                raise ValueError(f"Unknown id: {id_or_str}")

        else:
            try:
                return next(
                    h
                    for h in self._hids
                    if type(h).from_string(id_or_str).parts == h.parts
                )
            except StopIteration:
                raise ValueError(f"Unknown id: {id_or_str}")

    def _key(self, id_or_str):
        return self._to_hid(id_or_str).parts

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

    def color_greedy(self, colors=None):
        """Adjacent (parent-child) coloring.

        Returns mapping from canonical id string -> color int.
        """
        if colors is None:
            import matplotlib.pyplot as plt
            colors = plt.cm.get_cmap("tab10").colors

        # make colors an infinite iterator
        colors = itertools.cycle(colors)

        # Set up for search
        node_colors = {}
        visited = set()

        for i_root, root in enumerate(self._roots):
            # Invariant: if a node is in the stack, it has a color
            node_colors[root] = next(colors)
            stack = [root]
            # BFS
            while stack:
                node = stack.pop()
                # Invariant if node in visited, its children have been stacked
                if node in visited:
                    continue
                visited.add(node)

                # Bypass parent color if we made it back to that point in the
                # cycle
                color = next(colors)
                if node_colors[node] == color:
                    color = next(colors)
                for c in self._children.get(node, []):
                    stack.append(c)
                    node_colors[c] = color
                    color = next(colors)
        return node_colors

    def iter_topological(self, as_hid: bool = False):
        """Yield nodes in parent-before-children order using an explicit stack (DFS).

        If as_hid is True, yield HierarchicalId objects; otherwise yield canonical
        string ids.
        """
        visited = set()
        # Initialize stack so that roots are visited in the order of self._roots
        stack = list(reversed(self._roots))
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            yield self._to_hid(node) if as_hid else node
            # push children so that the first child is processed first
            children = list(self._children.get(node, []))
            for c in reversed(children):
                if c not in visited:
                    stack.append(c)

    def sorted_ids(self):
        """Return list of ids in topological (parent-before-children) order."""
        return list(self.iter_topological(as_hid=False))

    def __repr__(self):
        """Multi-line representation: topologically-sorted ids with indentation by depth."""
        lines = []
        for hid in self.iter_topological(as_hid=True):
            depth = hid.depth()
            indent = "  " * (depth - 1)
            lines.append(f"{indent}{hid.repr_in_tree()}")
        return "\n".join(lines)
