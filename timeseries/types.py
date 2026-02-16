"""Dataclasses for timeseries data with .mat file I/O."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.io as sio


@dataclass
class Traces:
    """Voltage/fluorescence traces.

    Attributes:
        data: (n_cells, n_frames) array of trace values
        ids: list of cell identifiers
        fs: sampling rate in Hz (optional)
    """

    data: np.ndarray
    ids: list[str]
    fs: float | None = None

    def save_mat(self, path: str | Path) -> None:
        """Save to .mat file."""
        mat_dict = {"data": self.data}
        if self.ids is not None:
            mat_dict["ids"] = np.array(self.ids, dtype=str)
        if self.fs is not None:
            mat_dict["fs"] = self.fs
        sio.savemat(str(path), mat_dict)

    @classmethod
    def from_mat(cls, path: str | Path) -> "Traces":
        """Load from .mat file."""
        mat = sio.loadmat(str(path), squeeze_me=True)
        ids = mat.get("ids", None)
        n = mat["data"].shape[0]
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        elif ids is None:
            ids = None
        elif isinstance(ids, str):  # matfile can flatten [str] to str
            ids = [ids]
        else:
            raise ValueError(f"Invalid `ids` field in Traces matfile: {type(ids)}")
        return cls(
            data=mat["data"],
            ids=ids,
            fs=mat.get("fs", None),
        )

    def __getitem__(self, key) -> "Traces":
        """Subset by cell index or id."""
        if isinstance(key, int):
            return Traces(
                data=self.data[key : key + 1], ids=[self.ids[key]], fs=self.fs
            )
        elif isinstance(key, str):
            idx = self.ids.index(key)
            return Traces(
                data=self.data[idx : idx + 1], ids=[self.ids[idx]], fs=self.fs
            )
        elif isinstance(key, slice):
            return Traces(data=self.data[key], ids=self.ids[key], fs=self.fs)
        elif isinstance(key, list) and all(isinstance(k, int) for k in key):
            return Traces(
                data=self.data[key], ids=[self.ids[k] for k in key], fs=self.fs
            )
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            return Traces(
                data=self.data[[self.ids.index(k) for k in key]], ids=key, fs=self.fs
            )
        else:
            raise TypeError(
                "Traces must be indexed by int, str, slice, or list of int/str"
            )

    @staticmethod
    def concatenate(traces_list: list["Traces"]) -> "Traces":
        """Concatenate multiple Traces along the cell axis."""
        data = np.concatenate([t.data for t in traces_list], axis=0)
        ids = sum([t.ids for t in traces_list], [])
        fs_values = set(t.fs for t in traces_list if t.fs is not None)
        if len(fs_values) > 1:
            raise ValueError("All Traces must have the same fs to concatenate")
        fs = fs_values.pop() if fs_values else None
        return Traces(data=data, ids=ids, fs=fs)


def _is_nested_indexer(key):
    """Check if key is an iterable of iterables (list of lists, ak.Array ndim=2, etc.)."""
    try:
        import awkward as ak #type: ignore
        if isinstance(key, ak.Array) and key.ndim == 2:
            return True
    except ImportError:
        pass
    if isinstance(key, (list, tuple)) and len(key) > 0 and isinstance(key[0], (list, tuple, np.ndarray)):
        return True
    return False


@dataclass
class Events:
    """Detected spike events.

    Attributes:
        spike_frames: list of arrays, each containing frame indices of spikes for one cell
        ids: list of cell identifiers (matching Traces)
        detection_params: dict of parameters used for detection
    """

    spike_frames: list[np.ndarray]
    ids: Optional[list[str]] = None
    detection_params: dict[str, Any] = field(default_factory=dict)

    def save_mat(self, path: str | Path) -> None:
        """Save to .mat file."""
        # Convert spike_frames list to object array for MATLAB cell array
        spike_frames_arr = np.empty(len(self.spike_frames), dtype=object)
        for i, frames in enumerate(self.spike_frames):
            spike_frames_arr[i] = frames

        mat_dict = {
            "spike_frames": spike_frames_arr,
            "detection_params": {},
        }
        if self.ids is not None:
            mat_dict["ids"] = np.array(self.ids, dtype=str)
        for key, value in self.detection_params.items():
            if value is not None:
                mat_dict["detection_params"][key] = value

        sio.savemat(str(path), mat_dict)

    @classmethod
    def from_mat(cls, path: str | Path) -> "Events":
        """Load from .mat file."""
        mat = sio.loadmat(str(path), squeeze_me=True)

        spike_frames_raw = mat["spike_frames"]
        if spike_frames_raw.ndim == 0:
            spike_frames = [spike_frames_raw.item()]
        else:
            spike_frames = [np.atleast_1d(sf) for sf in spike_frames_raw]

        ids = mat.get("ids", None)

        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        elif ids is None:
            ids = None
        else:
            raise ValueError(f"Invalid `ids` field Events matfile: {type(ids)}")

        detection_params = mat.get("detection_params", {})
        if isinstance(detection_params, np.ndarray):
            detection_params = {}

        return cls(
            spike_frames=spike_frames,
            ids=ids,
            detection_params=detection_params,
        )
    
    def __getitem__(self, key) -> "Events":
        """Subset by cell index or id."""
        if isinstance(key, int):
            return Events(
                spike_frames=[self.spike_frames[key]],
                ids=[self.ids[key]] if self.ids is not None else None,
                detection_params=self.detection_params,
            )
        elif isinstance(key, str):
            idx = self.ids.index(key)
            return Events(
                spike_frames=[self.spike_frames[idx]],
                ids=[self.ids[idx]] if self.ids is not None else None,
                detection_params=self.detection_params,
            )
        elif isinstance(key, slice):
            return Events(
                spike_frames=self.spike_frames[key],
                ids=self.ids[key] if self.ids is not None else None,
                detection_params=self.detection_params,
            )
        elif isinstance(key, list) and all(isinstance(k, int) for k in key):
            return Events(
                spike_frames=[self.spike_frames[k] for k in key],
                ids=[self.ids[k] for k in key] if self.ids is not None else None,
                detection_params=self.detection_params,
            )
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            indices = [self.ids.index(k) for k in key]
            return Events(
                spike_frames=[self.spike_frames[i] for i in indices],
                ids=key,
                detection_params=self.detection_params,
            )
        elif _is_nested_indexer(key):
            # Per-event bool mask or int fancy index. See spec: design/handoff/26-02-10_spike-fns.md
            # Bool: new_spike_frames[i] = spike_frames[i][key[i]], len(key[i]) must match
            # Int:  new_spike_frames[i] = spike_frames[i][key[i]]
            # Implement both cases supporting ak.Array (ndim==2) or nested lists/ndarrays
            import awkward as ak #type: ignore

            if isinstance(key, ak.Array) and key.ndim == 2:
                key_list = ak.to_list(key)
            else:
                key_list = [np.asarray(k) for k in key]

            new_spike_frames = []
            for orig, k in zip(self.spike_frames, key_list):
                orig_arr = np.asarray(orig)
                if k.dtype == bool:
                    if len(k) != len(orig_arr):
                        raise ValueError("Boolean mask length must match number of spikes for that cell")
                    new_spike_frames.append(orig_arr[k])
                else:
                    # integer fancy indexing
                    new_spike_frames.append(orig_arr[np.asarray(k)])

            return Events(
                spike_frames=[np.atleast_1d(s) for s in new_spike_frames],
                ids=self.ids,
                detection_params=self.detection_params,
            )
        else:
            raise TypeError(
                "Events must be indexed by int, str, slice, or list of int/str"
            )
