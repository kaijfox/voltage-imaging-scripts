"""CLI commands for io module: SVD conversion, slicing, filtering."""

import click
from pathlib import Path
import time
import numpy as np

from .common import (
    input_output_options,
    batch_progress_options,
    configure_logging,
)


@click.command()
@input_output_options
@batch_progress_options(1000)
@click.option("-r", "--rank", type=int, required=True, help="Target rank for SRSVD.")
@click.option(
    "--rank-spatial", type=int, default=None, help="Spatial rank if different."
)
@click.option("--no-second-pass", is_flag=True, help="Skip second pass.")
@click.option("--seed", type=int, default=None, help="Random seed.")
@click.option("--checkpoint-every", type=int, default=1, help="Checkpoint frequency.")
@click.option("--restart", is_flag=True, help="Delete existing output first.")
def convert(
    input_path,
    output_path,
    batch_size,
    no_progress,
    rank,
    rank_spatial,
    no_second_pass,
    seed,
    checkpoint_every,
    restart,
):
    """Convert video to SVD format."""
    import h5py
    from ..io.svd_conversion import (
        stream_framereader,
        stream_h5,
        delete_svd,
    )

    logger, (error, warning, info, debug) = configure_logging("converter")

    if batch_size <= 0:
        raise click.BadParameter("--batch-size must be > 0")
    if rank <= 0:
        raise click.BadParameter("--rank must be > 0")

    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    # Detect input type
    try:
        with h5py.File(input_path, "r") as f:
            input_type = "raw_h5" if "video" in f else "raw"
    except:
        input_type = "raw"

    if restart:
        warning(f"Restarting at {output_path} in 5s")
        time.sleep(5)
        delete_svd(output_path)

    try:
        if input_type == "raw":
            stream_framereader(
                input_path=input_path,
                output_path=output_path,
                batch_size=batch_size,
                rank=rank,
                rank_spatial=rank_spatial,
                second_pass=not no_second_pass,
                seed=seed,
                checkpoint_every=checkpoint_every,
                progress=not no_progress,
            )
        else:
            stream_h5(
                input_path=input_path,
                output_path=output_path,
                batch_size=batch_size,
                rank=rank,
                seed=seed,
                checkpoint_every=checkpoint_every,
                progress=not no_progress,
            )
    except Exception as exc:
        error(f"Conversion failed: {exc}")
        raise


@click.command()
@input_output_options
@click.option(
    "--rois",
    type=click.Path(exists=True),
    required=True,
    help="Path to ROICollection (.mat) file.",
)
@click.option("--max_rank", type=int, default=None, help="Target rank for guided SVD.")
@click.option(
    "--start-frame", type=int, default=0, help="Start frame index (inclusive)."
)
@click.option(
    "--end-frame", type=int, default=None, help="End frame index (exclusive)."
)
@click.option(
    "--n-clusters", type=int, default=10, help="Number of clusters for guided SVD."
)
@click.option("--batch-r", type=int, default=8, help="Batch size for rows.")
@click.option("--batch-c", type=int, default=8, help="Batch size for columns.")
@click.option(
    "--spatial",
    type=int,
    multiple=True,
    default=[21, 11, 5],
    help="Spatial window sizes for guide computation in pixels.",
)
@click.option(
    "--temporal",
    type=int,
    multiple=True,
    default=[5000, 1000, 200],
    help="Temporal window sizes for guide computation in frames.",
)
@click.option(
    "--hop",
    type=int,
    multiple=True,
    default=[10, 5, 2],
    help="Spatial hops for guide computation in pixels.",
)
def convert_guided(
    input_path,
    output_path,
    max_rank,
    rois,
    start_frame,
    end_frame,
    n_clusters,
    batch_r,
    batch_c,
    spatial,
    temporal,
    hop,
):
    """Convert h5 video to guided SVD format using ROI footprints."""
    from ..io.svd_conversion import stream_guided_svd
    from ..timeseries.rois import ROICollection

    logger, (error, warning, info, debug) = configure_logging("svd")

    info(
        f"Starting guided SVD conversion: input={input_path}, output={output_path}"
    )

    # Load inputs
    roc = ROICollection.load(rois)
    footprints = [r.footprint for r in roc.rois]
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    # Run SVD
    stream_guided_svd(
        input_path=input_path,
        footprints=footprints,
        output_path=output_path,
        max_rank=max_rank,
        start_frame=start_frame,
        end_frame=end_frame,
        n_clusters=n_clusters,
        spatial_sizes=spatial,
        hops=hop,
        window_sizes=temporal,
        batch_r=batch_r,
        batch_c=batch_c,
    )

    info("Guided SVD conversion complete")


def _parse_slice(s):
    """Parse a string like '0:100' or '-100:' into a slice object.

    Other valid arguments include
    - None -> slice(None)
    - 0:50,,0:100 -> (slice(0, 100), slice(None), slice(0, 100))"""

    # Empty or multiple slices
    if s is None or s == "":
        return slice(None)
    if "," in s:
        return tuple(_parse_slice(part) for part in s.split(","))

    parts = s.split(":")
    if len(parts) == 1:
        # Single index
        idx = int(parts[0])
        return slice(idx, idx + 1)
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return slice(start, stop, step)


@click.command()
@input_output_options
@batch_progress_options(None)
@click.option(
    "--temporal", type=str, default=None, help='Temporal slice, e.g. "0:100".'
)
@click.option("--spatial", type=str, multiple=True, help="Spatial slice(s).")
@click.option(
    "--component", type=str, default=None, help='Component slice, e.g. "0:10".'
)
@click.option("--stream-dim", type=int, default=0, help="Dim to stream for U.")
@click.option(
    "--stream-dim-spatial", type=int, default=None, help="Dim to stream for Vh."
)
def slice_cmd(
    input_path,
    output_path,
    batch_size,
    no_progress,
    temporal,
    spatial,
    component,
    stream_dim,
    stream_dim_spatial,
):
    """Slice an existing SVD file."""
    from ..io.svd_conversion import slice_svd_file

    logger, (error, warning, info, debug) = configure_logging("slice")

    temporal_slice = _parse_slice(temporal)
    component_slice = _parse_slice(component)
    spatial_slices = []
    for s in spatial or []:
        s = _parse_slice(s)
        spatial_slices.extend(s if isinstance(s, tuple) else [s])

    debug(f"Component slice: {component_slice}")
    debug(f"Temporal slice:  {temporal_slice}")
    debug(f"Spatial slice:   {spatial_slices}")

    slice_svd_file(
        input_path=input_path,
        output_path=output_path,
        temporal=temporal_slice,
        spatial=tuple(spatial_slices),
        component=component_slice,
        batch_size=batch_size,
        stream_dim=stream_dim,
        stream_dim_spatial=stream_dim_spatial,
        progress=not no_progress,
    )


@click.command()
@input_output_options
@batch_progress_options(1000)
@click.option("--start", type=int, default=0, help="Start frame for resume.")
def h5_convert(input_path, output_path, batch_size, no_progress, start):
    """Stream RAW video frames to HDF5 store."""
    from ..io.h5_conversion import stream_framereader

    logger, (error, *_) = configure_logging("converter")

    if batch_size <= 0:
        raise click.BadParameter("--batch-size must be > 0")

    try:
        stream_framereader(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            progress=not no_progress,
            start=start,
        )
    except Exception as exc:
        error(f"Conversion failed: {exc}")
        raise


@click.command()
@input_output_options
@click.option(
    "-w",
    "--window-length",
    type=int,
    required=True,
    help="Savitzky-Golay window length (must be odd).",
)
@click.option(
    "-p", "--polyorder", type=int, required=True, help="Polynomial order for filter."
)
@click.option(
    "--min-baseline",
    type=float,
    default=1e-8,
    help="Minimum baseline value (default: 1e-8).",
)
def divisive_lowpass_cmd(
    input_path, output_path, window_length, polyorder, min_baseline
):
    """Apply divisive temporal low-pass filter to SVD video."""
    from ..io.svd_video import SVDVideo
    from ..io.ops import divisive_lowpass

    logger, (error, warning, info, debug) = configure_logging("io")

    if window_length % 2 == 0:
        raise click.BadParameter("--window-length must be odd")
    if polyorder >= window_length:
        raise click.BadParameter("--polyorder must be < window-length")

    info(f"Loading SVD from {input_path}")
    video = SVDVideo.load(input_path)

    info(f"Applying divisive lowpass (window={window_length}, order={polyorder})")
    result = divisive_lowpass(video, window_length, polyorder, min_baseline)

    info(f"Saving result to {output_path}")
    result.save(output_path)
    info("Done")


@click.command()
@input_output_options
@click.option(
    "-w",
    "--window-length",
    type=int,
    required=True,
    help="Savitzky-Golay window length (must be odd).",
)
@click.option(
    "-p", "--polyorder", type=int, required=True, help="Polynomial order for filter."
)
def sub_lowpass_cmd(input_path, output_path, window_length, polyorder):
    """Apply subtractive temporal low-pass filter to SVD video."""
    from ..io.svd_video import SVDVideo
    from ..io.ops import subtractive_lowpass

    logger, (error, warning, info, debug) = configure_logging("io")

    if window_length % 2 == 0:
        raise click.BadParameter("--window-length must be odd")
    if polyorder >= window_length:
        raise click.BadParameter("--polyorder must be < window-length")

    info(f"Loading SVD from {input_path}")
    video = SVDVideo.load(input_path)

    info(f"Applying subtractive lowpass (window={window_length}, order={polyorder})")
    result = subtractive_lowpass(video, window_length, polyorder)

    info(f"Saving result to {output_path}")
    result.save(output_path)
    info("Done")


@click.command()
@input_output_options
@click.option(
    "--shifts_output",
    type=click.Path(),
    default=None,
    help="Optional path to save estimated shifts as .csv file.",
)
@click.option(
    "--rois",
    type=click.Path(exists=True),
    default=None,
    help="Path to ROICollection (.mat) file.",
)
@click.option("--max-shift", type=int, default=None, help="Maximum shift (px).")
@click.option("--n-passes", type=int, default=2, help="Number of passes.")
@click.option("--batch_lim", type=int, default=4000 * 4000, help="Pixels per batch.")
@click.option(
    "--hpf-px", type=float, default=10.0, help="High-pass filter width in px."
)
@click.option(
    "--sharp-px", type=float, default=1.0, help="Sharpening kernel size in px."
)
@click.option("--sharp-amount", type=float, default=100.0, help="Sharpen amount.")
@click.option("--max-rank", type=int, default=None, help="Maximum rank to retain.")
def motion_correct_cmd(
    input_path,
    output_path,
    shifts_output,
    rois,
    max_shift,
    n_passes,
    hpf_px,
    sharp_px,
    sharp_amount,
    max_rank,
    batch_lim,
):
    """Apply motion correction to SVD video using SVD-based algorithm."""
    from ..io.svd_video import SVDVideo
    from ..io.motion_correction import motion_correct_svd
    from ..timeseries.rois import ROICollection
    from ..viz.rois import footprint_mask

    logger, (error, warning, info, debug) = configure_logging("motion_correct")

    info(f"Loading SVD from {input_path}")
    video = SVDVideo.load(input_path)

    mask = None
    if rois:
        artifact_rois = ROICollection.load(rois)
        artifact_roi = artifact_rois.rois[0]
        mask = footprint_mask(artifact_rois.image_shape, artifact_roi.footprint)

    corrected, shifts = motion_correct_svd(
        video,
        max_shift=max_shift,
        n_passes=n_passes,
        mask=mask,
        hpf_px=hpf_px,
        sharp_px=sharp_px,
        sharp_amount=sharp_amount,
        max_rank=max_rank,
        batch_limit=batch_lim,
    )

    info(f"Saving video to {output_path}")
    corrected.save(output_path)
    if shifts_output:
        # Save only final shifts if multiple passes
        if n_passes > 1:
            shifts = shifts[-1]
        info(f"Saving shifts to {shifts_output}")
        np.savetxt(
            shifts_output,
            shifts,
            delimiter=",",
            header="est_row_shift,est_col_shift",
        )

    info("Done")


@click.command()
@click.argument("path", type=click.Path(exists=True))
def svd_info(path):
    """Print info about an SVD video file."""
    import h5py

    with h5py.File(path, "r") as f:
        click.echo(f"File: {path}")
        click.echo("\nDatasets:")
        for name in f.keys():
            dset = f[name]
            click.echo(f"  {name}: shape={dset.shape}, dtype={dset.dtype}")

        click.echo("\nAttributes:")
        for key, val in f.attrs.items():
            click.echo(f"  {key}: {val}")


@click.command()
@input_output_options
@click.option(
    "--no-plot",
    is_flag=True,
    help="Disable plotting of ROI footprints.",
)
def build_artifact_cmd(
    input_path,
    output_path,
    no_plot,
):
    from ..io.motion_correction import build_artifact_mask
    from ..timeseries.rois import ROICollection
    from ..roigui import launch
    from ..viz.rois import footprint_mask
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    logger, (error, warning, info, debug) = configure_logging("build_artifact")

    output_path = Path(output_path).with_suffix("")
    build_file = str(output_path) + "_build.mat"
    final_file = str(output_path) + ".mat"
    if Path(build_file).exists():
        input_collection = ROICollection.load(build_file)
    else:
        input_collection = None
    info(f"Starting from existing ROIs: {build_file}")

    # Uncomment to redraw for new video
    launch(
        iio.imread(input_path),
        n_components=200,
        roi_collection=input_collection,
        output_path=build_file,
        summary_mode="mean",
    )

    # Load drawn ROI
    _, artifact_rois = build_artifact_mask(
        ROICollection.load(build_file),
    )
    artifact_rois.save(final_file)
    artifact_roi = artifact_rois["artifact"].rois[0]
    info(f"Saved artifact ROIs to {final_file}")

    if not no_plot:
        mask = footprint_mask(
            artifact_rois.image_shape,
            artifact_roi.footprint,
        )
        plt.imshow(mask, cmap="gray")
        plt.show()


@click.command()
@input_output_options
def edit_rois_cmd(
    input_path,
    output_path,
):
    from ..io.motion_correction import build_artifact_mask
    from ..io.svd_video import SVDVideo
    from ..timeseries.rois import ROICollection
    from ..roigui import launch
    from ..viz.rois import footprint_mask
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    logger, (error, warning, info, debug) = configure_logging("build_artifact")
    output_path = Path(output_path)
    input_path = Path(input_path)

    if output_path.exists():
        input_collection = ROICollection.load(output_path)
    else:
        input_collection = None
    info(f"Starting from existing ROIs: {output_path}")

    if input_path.suffix in [".h5", ".hdf5"]:
        reference = SVDVideo.load(input_path)
    else:
        reference = iio.imread(input_path)

    # Uncomment to redraw for new video
    launch(
        reference,
        n_components=200,
        roi_collection=input_collection,
        output_path=output_path,
        summary_mode="mean",
    )
