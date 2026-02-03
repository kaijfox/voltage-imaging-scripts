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
        mat_dict = {'data': self.data}
        if self.ids is not None:
            mat_dict['ids'] = np.array(self.ids, dtype=str)
        if self.fs is not None:
            mat_dict['fs'] = self.fs
        sio.savemat(str(path), mat_dict)

    @classmethod
    def from_mat(cls, path: str | Path) -> "Traces":
        """Load from .mat file."""
        mat = sio.loadmat(str(path), squeeze_me=True)
        ids = mat.get('ids', None)
        n = mat['data'].shape[0]
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        elif ids is None:
            ids = None
        elif isinstance(ids, str): # matfile can flatten [str] to str
            ids = [ids]
        else:
            raise ValueError(f"Invalid `ids` field in Traces matfile: {type(ids)}")
        return cls(
            data=mat['data'],
            ids=ids,
            fs=mat.get('fs', None),
        )


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
            'spike_frames': spike_frames_arr,
            'detection_params': {},
        }
        if self.ids is not None:
            mat_dict['ids'] = np.array(self.ids, dtype=str)
        for key, value in self.detection_params.items():
            if value is not None:
                mat_dict['detection_params'][key] = value

        sio.savemat(str(path), mat_dict)

    @classmethod
    def from_mat(cls, path: str | Path) -> "Events":
        """Load from .mat file."""
        mat = sio.loadmat(str(path), squeeze_me=True)

        spike_frames_raw = mat['spike_frames']
        if spike_frames_raw.ndim == 0:
            spike_frames = [spike_frames_raw.item()]
        else:
            spike_frames = [np.atleast_1d(sf) for sf in spike_frames_raw]

        ids = mat.get('ids', None)

        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        elif ids is None:
            ids = None
        else:
            raise ValueError(f"Invalid `ids` field Events matfile: {type(ids)}")

        detection_params = mat.get('detection_params', {})
        if isinstance(detection_params, np.ndarray):
            detection_params = {}

        return cls(
            spike_frames=spike_frames,
            ids=ids,
            detection_params=detection_params,
        )
