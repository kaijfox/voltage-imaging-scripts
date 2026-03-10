"""Pipeline configuration and session path resolution.

Provides:
- PipelineConfig: dataclass holding processing parameters, loadable from TOML
- SessionPaths: dataclass of resolved paths for a single session
- session_paths(): construct SessionPaths from config + session id
- load_config(): load PipelineConfig from a TOML file
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Processing parameters shared across pipeline stages.

    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    rank : int
        SVD rank used for video compression and trace extraction.
    neuropil_range : tuple[int, int]
        Inner and outer radii (pixels) of the neuropil annulus.
    neuropil_threshold : float
        Exclusion threshold for overlapping ROIs in annulus building.
    bg_smooth_size : int
        Savitzky-Golay smoothing window for neuropil background (0 = off).
    hpf_ms : float
        High-pass filter cutoff in milliseconds for dF/F.
    spike_hpf_ms : float
        High-pass filter cutoff in milliseconds for spike detection.
    spike_threshold_sd : float
        Spike detection threshold in standard deviations.
    window_stats_s : float
        Window size in seconds for binned statistics.
    """

    version: str
    neuropil_range: Tuple[int, int]
    neuropil_threshold: float
    bg_smooth_size: int
    hpf_ms: float
    spike_hpf_ms: float
    spike_threshold_sd: float
    window_stats_s: float


@dataclass
class SessionConfig:
    """Session-specific config parameters.

    Attributes
    ----------
    name: str
        Unique identifier for the session
    roi_filename: str
        Filename of the ROI file within the session directory.
    fs: float
        Sampling rate in Hz for this session.
    rank: int
        SVD rank used for video compression and trace extraction.
    """

    name: str
    roi_filename: str
    fs: float
    rank: int


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a TOML file.

    The TOML file should have a [pipeline] table whose keys match
    PipelineConfig fields. Missing keys fall back to defaults.

    Parameters
    ----------
    path : str or Path
        Path to the TOML config file.

    Returns
    -------
    PipelineConfig
    """
    path = Path(path)
    with open(path, "rb") as fh:
        cfg_data = tomllib.load(fh)

    # only use keys present in the TOML [pipeline] table; keep defaults otherwise
    keys = set(PipelineConfig.__dataclass_fields__.keys())
    kwargs: dict = {}
    for k, v in cfg_data.items():
        if k not in keys:
            continue
        if k == "neuropil_range" and isinstance(v, (list, tuple)):
            v = tuple(v)
        kwargs[k] = v

    missing = keys - set(kwargs.keys())
    if missing:
        raise ValueError(f"Missing required config keys in {path}: {missing}")

    return PipelineConfig(**kwargs)


# ---------------------------------------------------------------------------
# Session paths
# ---------------------------------------------------------------------------

@dataclass
class SessionPaths:
    """Resolved file paths for a single imaging session.

    All paths are absolute.
    """

    session_dir: Path
    roi_file: Path
    raw_video: Path
    dff_video: Path
    frames_h5: Path
    traces_raw: Path
    traces_dff: Path
    traces_vmhigh: Path
    traces_dff_baseline: Path
    traces_neuropil: Path
    events: Path


def session_paths(
    data_dir: str | Path,
    session: SessionConfig,
) -> SessionPaths:
    """Construct SessionPaths for a session given data root and config.

    Parameters
    ----------
    session_dir : str or Path
        Directory containing session data.
    roi_filename : str
        Filename (stem + suffix) of the ROI file within the session dir.
    config : PipelineConfig
        Pipeline config; rank is used to resolve video filenames.

    Returns
    -------
    SessionPaths
    """
    session_dir = (Path(data_dir) / session.name).resolve()

    roi_file = session_dir / session.roi_filename
    raw_video = session_dir / f"svd-rank{session.rank}.h5"
    dff_video = session_dir / f"dff-slow-rank{session.rank}.h5"
    frames_h5 = session_dir / "frames.h5"
    traces_raw = session_dir / "traces_raw.mat"
    traces_dff = session_dir / "traces_dff.mat"
    traces_vmhigh = session_dir / "traces_vmhigh.mat"
    traces_dff_baseline = session_dir / "traces_dff_baseline.mat"
    traces_neuropil = session_dir / "traces_neuropil.mat"
    events = session_dir / "spikes.mat"

    return SessionPaths(
        session_dir=session_dir,
        roi_file=roi_file,
        raw_video=raw_video,
        dff_video=dff_video,
        frames_h5=frames_h5,
        traces_raw=traces_raw,
        traces_dff=traces_dff,
        traces_vmhigh=traces_vmhigh,
        traces_dff_baseline=traces_dff_baseline,
        traces_neuropil=traces_neuropil,
        events=events,
    )
