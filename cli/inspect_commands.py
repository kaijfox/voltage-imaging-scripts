import click
from pathlib import Path


from ..io.svd_video import SVDVideo
from ..io.pipeline import load_config, load_session_config
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
    "--session_toml",
    type=click.Path(exists=True),
    required=True,
    help="Path to session.toml file (uses band_cuts for filtering)",
)
@click.option(
    "--rois", 
    type=click.Path(exists=True),
    required=True,
    help="Path to ROIs file for display",
)
@click.option(
    "--traces",
    type=click.Path(exists=True),
    required=True,
    help="Path to traces file for filtering and spectral analysis",
)
@click.option(
    "--events",
    type=click.Path(exists=True),
    required=True,
    help="Path to events file for despiking traces.",
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
    session_toml: str,
    rois: str,
    traces: str,
    events: str,
    output_path: str,
    despike_window_ms: float,
    despike_savgol_ms: float,
):
    # Path to directory where plots should be saved; optionally containing {session.name}
    output_path = Path(output_path)

    session = load_session_config(session_toml)
    raw_video = SVDVideo.load(input_path)
    spatial_rois = ROICollection.load(rois)
    spatial_traces = Traces.from_mat(traces)
    spikes = Events.from_mat(events)
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
    "--session_toml",
    type=click.Path(exists=True),
    required=True,
    help="Path to session.toml file (uses fs and session name)",
)
@click.option(
    "--raw_video",
    type=click.Path(exists=True),
    required=True,
    help="Path to raw video SVD file",
)
@click.option(
    "--mc_video",
    type=click.Path(exists=True),
    required=True,
    help="Path to motion-corrected video SVD file",
)
@click.option(
    "--mc_shifts",
    type=click.Path(exists=True),
    required=True,
    help="Path to motion correction shifts CSV file",
)
@click.option(
    "--target-fs",
    type=float,
    default=10.0,
    help='Target playback framerate for video comparison',
)
@click.option(
    "--target-gamma",
    type=float,
    default=0.2,
    help='Target mean [0, 1] for framewise luminance correction. If negative, no correction is applied.',
)
def inspect_mc_cmd(
    input_path: str,
    session_toml: str,
    raw_video: str,
    mc_video: str,
    mc_shifts: str,
    output_path: str,
    target_fs: float,
    target_gamma: float,
):
    # Path to directory containing session.toml
    input_path = Path(input_path)
    # Path to directory where plots should be saved; optionally containing {session.name}
    output_path = Path(output_path)

    session = load_session_config(session_toml)

    raw_video = SVDVideo.load(raw_video)
    mc_video = SVDVideo.load(mc_video)
    shifts = np.loadtxt(mc_shifts, delimiter=",", skiprows=1)

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
        target_gamma=target_gamma,
    )
