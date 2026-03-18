import click
from pathlib import Path


from ..io.svd_video import SVDVideo
from ..io.pipeline import load_config, load_session_config, session_paths
from ..timeseries.rois import ROICollection
from ..timeseries.events import despike_impute
from ..timeseries.filtering import frequency_filter
from ..timeseries.spike_analysis import ms_to_samples
from ..timeseries.types import Events, Traces
from ..viz.filterbands import (
    robust_coherence,
    plot_filtered_spectra,
    plot_coherence_overlay,
)
from ..viz.qc import setup_plotter
from ..viz.rois import display_rois
from ..viz.motion_correction import compare_videos, plot_shifts
from .common import input_output_options
import numpy as np
from pathlib import Path


@click.command()
@input_output_options
@click.option(
    "--pipeline-cfg", type=str, required=True, help="Path to pipeline config TOML file."
)
@click.option(
    "--despike-window-ms",
    type=float,
    default=5.0,
    help="Window size in ms for despiking traces.",
)
@click.option(
    "--despike-savgol-ms",
    type=float,
    default=100.0,
    help="Window size in ms for Savitzky-Golay smoothing in despiked trace imputation.",
)
def inspect_filters_cmd(
    input_path: str,
    output_path: str,
    pipeline_cfg: str,
    despike_window_ms: float,
    despike_savgol_ms: float,
):
    # Path to directory containing session.toml
    input_path = Path(input_path)
    # Path to directory where plots should be saved; optionally containing {session.name}
    output_path = Path(output_path)
    
    session = load_session_config(input_path / "session.toml")
    # Constructs paths as <data_dir>/<session_name>/<data-file-name>
    paths = session_paths(input_path.parent, session)

    raw_video = SVDVideo.load(paths.raw_video)
    spatial_rois = ROICollection.load(paths.spatial_rois)
    spatial_traces = Traces.from_mat(paths.spatial_traces)
    spikes = Events.from_mat(paths.spatial_spikes)
    fs = spatial_traces.fs

    despiked = despike_impute(
        spatial_traces,
        spikes,
        window_size=ms_to_samples(despike_window_ms, fs),
        savgol_window_frames=ms_to_samples(despike_savgol_ms, fs),
    )

    filtered = Traces(
        frequency_filter(despiked.data, fs, session.band_cuts, order=2),
        ids=despiked.ids,
        fs=despiked.fs,
    )
    coh, freqs = robust_coherence(despiked.data, fs, 2 * fs)
    coh_filt, _ = robust_coherence(filtered.data, fs, 2 * fs)

    plotter = setup_plotter()
    plotter.plot_dir = str(output_path).format(session=session)
    fig, ax = display_rois(spatial_rois, raw_video, target_gamma=0.2)
    plotter.finalize(fig, name="rois")
    fig, ax = plot_filtered_spectra(despiked, filtered, fs)
    plotter.finalize(fig, name="bandcut-mean-spectrum")
    fig, ax = plot_coherence_overlay(coh, freqs)
    plotter.finalize(fig, name="coherence-raw")
    fig, ax = plot_coherence_overlay(coh_filt, freqs)
    plotter.finalize(fig, name="coherence-filtered")


@click.command()
@input_output_options
@click.option(
    "--target-fs",
    type=float,
    default=10.0,
    help='Target playback framerate for video comparison',
)
@click.option("--filename", nargs=2, multiple=True, type=(str, str))
def inspect_mc_cmd(
    input_path: str,
    output_path: str,
    target_fs: float,
    filename
):
    # Path to directory containing session.toml
    input_path = Path(input_path)
    # Path to directory where plots should be saved; optionally containing {session.name}
    output_path = Path(output_path)

    # Constructs paths as <data_dir>/<session_name>/<data-file-name>
    session = load_session_config(input_path / "session.toml")
    paths = session_paths(input_path.parent, session)

    print(paths.session_dir)
    print(paths.raw_video)
    print(paths.mc_video)
    print(paths.mc_shifts)
    overrides = {k: v for k, v in filename}
    raw_filename = overrides.get("raw_video", paths.raw_video)
    mc_filename = overrides.get("mc_video", paths.mc_video)
    shifts_filename = overrides.get("mc_shifts", paths.mc_shifts)
    print(raw_filename)
    print(mc_filename)
    print(shifts_filename)
    raw_video = SVDVideo.load(raw_filename)
    mc_video = SVDVideo.load(mc_filename)
    shifts = np.loadtxt(shifts_filename, delimiter=",", skiprows=1)

    plotter = setup_plotter()
    plotter.plot_dir = str(output_path).format(session=session)
    fig, ax = plot_shifts(shifts, fs=session.fs)
    plotter.finalize(fig, name="mc-shifts")

    compare_videos(
        raw_video,
        mc_video,
        fs=session.fs,
        target_fs=target_fs,
        output_path=str(Path(plotter.plot_dir) / "mc-comparison.mp4"),
    )
