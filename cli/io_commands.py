"""CLI commands for io module: SVD conversion, slicing, filtering."""

import click
from pathlib import Path
import time

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

def _parse_slice(s):
    """Parse a string like '0:100' or '-100:' into a slice object.
    
    Other valid arguments include
    - None -> slice(None)
    - 0:50,,0:100 -> (slice(0, 100), slice(None), slice(0, 100))"""

    # Empty or multiple slices
    if s is None or s == '':
        return slice(None)
    if ',' in s:
        return tuple(_parse_slice(part) for part in s.split(','))
    
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
    for s in (spatial or []):
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
