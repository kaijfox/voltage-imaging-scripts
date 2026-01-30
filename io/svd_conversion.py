from .FrameReader import FrameReader
from .svd import SRSVD
from .cli_tools import configure_logging, enable_logging

from pathlib import Path
import os
import h5py
import numpy as np
from numpy.typing import NDArray
import time
import tqdm
import click

logger, (error, warning, info, debug) = configure_logging("converter", 2)


def _first_pass_complete(svd):
    if not isinstance(svd, os.PathLike):
        svd = svd.h5_filename
    if not Path(svd).exists():
        return False
    with h5py.File(svd, "r") as f:
        # Once U and V written, first pass must be complete
        # (doesn't rule out that second pass also complete)
        return ("U" in f) and ("V" in f)


def delete_svd(path: os.PathLike):
    path = Path(path)
    base, ext = os.path.splitext(path)
    spaces_path = Path(base + ".spaces.h5" if ext == ".h5" else path + ".spaces.h5")
    if path.exists() and spaces_path.exists():
        os.remove(path)
        os.remove(spaces_path)
        return True
    else:
        return False


def _stream_conversion(
    get_frames_by_indices,
    n_frames,
    output_path: os.PathLike,
    batch_size: int,
    rank: int,
    rank_spatial: int = None,
    second_pass: bool = True,
    seed=None,
    checkpoint_every=1,
    progress=True,
    second_pass_type="row",
    second_pass_dim=0,
    get_frames_second_pass=None,
):
    if get_frames_second_pass is None:
        get_frames_second_pass = get_frames_by_indices

    # Normalize path
    output_path = str(Path(output_path))

    # Set up SVD object
    svd = SRSVD(
        output_path,
        n_rows=n_frames,
        n_rand_col=rank,
        n_rand_row=rank_spatial,
        seed=seed,
        checkpoint_every=checkpoint_every,
        second_pass_type=second_pass_type,
    )

    # Perform first pass if it's not already complete
    if not _first_pass_complete(svd):
        with svd.first_pass():
            # Find location to resume from
            mask = svd._spaces["mask_first"][:]
            next_idx = np.argmax(~mask.astype(bool))
            if next_idx == 0 and np.all(mask.astype(bool)):
                # Handle no frames processed
                next_idx = len(mask) - 1
            # skip iteration if already at end
            if next_idx < len(mask) - 1:

                # Load and process collections of frames
                batch_idxs = (
                    tqdm.trange(next_idx, len(mask), batch_size, desc="First pass")
                    if progress
                    else range(next_idx, len(mask), batch_size)
                )
                for batch_idx in batch_idxs:
                    frames = get_frames_by_indices(
                        batch_idx, min(batch_idx + batch_size, len(mask))
                    )
                    svd.receive_batch(frames, batch_idx)

    # Perform second pass
    if second_pass:
        with svd.second_pass(stream_dim=second_pass_dim):
            # Find location to resume from
            mask = svd._spaces["mask_second"][:].astype(bool)
            next_idx = np.argmax(~mask)
            if next_idx == 0 and np.all(mask):
                next_idx = len(mask) - 1
            if next_idx < len(mask) - 1:

                # Load and process collections of frames
                batch_idxs = (
                    tqdm.trange(next_idx, len(mask), batch_size, desc="Second pass")
                    if progress
                    else range(next_idx, len(mask), batch_size)
                )
                for batch_idx in batch_idxs:
                    frames = get_frames_second_pass(
                        batch_idx, min(batch_idx + batch_size, len(mask))
                    )
                    svd.receive_batch(frames, batch_idx)

    return svd


def stream_framereader(
    input_path: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
    rank: int,
    rank_spatial: int = None,
    second_pass: bool = True,
    seed=None,
    checkpoint_every=1,
    progress=True,
):
    reader = FrameReader(str(input_path))
    return _stream_conversion(
        # FrameReader indexes inclusively
        lambda i_start, i_end: reader.get_frames_by_indexes(
            i_start + 1, i_end
        ).transpose(2, 0, 1),
        n_frames=reader.max_frames,
        output_path=output_path,
        batch_size=batch_size,
        rank=rank,
        rank_spatial=rank_spatial,
        second_pass=second_pass,
        seed=seed,
        checkpoint_every=checkpoint_every,
        progress=progress,
    )


def stream_h5(
    input_path: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
    rank: int,
    seed=None,
    checkpoint_every=1,
    progress=True,
):
    with h5py.File(input_path, "r") as f:
        video = f["video"]
        return _stream_conversion(
            lambda i_start, i_end: video[i_start:i_end],
            get_frames_second_pass=lambda i_start, i_end: video[:, i_start:i_end],
            second_pass_dim=1,
            second_pass_type="inner",
            n_frames=f.attrs["frames"],
            output_path=output_path,
            batch_size=batch_size,
            rank=rank,
            second_pass=True,
            seed=seed,
            checkpoint_every=checkpoint_every,
            progress=progress,
        )


def array_to_svd(
    video_array: NDArray,
    output_path: os.PathLike,
    batch_size: int = None,
    rank: int = None,
    rank_spatial: int = None,
    second_pass: bool = True,
    seed=None,
    checkpoint_every=1,
    progress=True,
    second_pass_type="row",
):
    # Default to full rank in one pass
    if rank is None:
        rank = min(len(video_array), np.prod(video_array.shape[1:]))
    if batch_size is None:
        batch_size = len(video_array)

    if second_pass_type == "row":
        return _stream_conversion(
            lambda i_start, i_end: video_array[i_start:i_end],
            n_frames=len(video_array),
            output_path=output_path,
            batch_size=batch_size,
            rank=rank,
            rank_spatial=rank_spatial,
            second_pass=second_pass,
            seed=seed,
            checkpoint_every=checkpoint_every,
            progress=progress,
        )
    elif second_pass_type == "inner":
        return _stream_conversion(
            lambda i_start, i_end: video_array[i_start:i_end],
            get_frames_second_pass=lambda i_start, i_end: video_array[:, i_start:i_end],
            second_pass_dim=1,
            second_pass_type="inner",
            n_frames=len(video_array),
            output_path=output_path,
            batch_size=batch_size,
            rank=rank,
            rank_spatial=rank_spatial,
            second_pass=second_pass,
            seed=seed,
            checkpoint_every=checkpoint_every,
            progress=progress,
        )


def _parse_slice(s):
    """Parse a string like '0:100' or '-100:' into a slice object."""
    if s is None:
        return slice(None)
    parts = s.split(":")
    if len(parts) == 1:
        # Single index
        idx = int(parts[0])
        return slice(idx, idx + 1)
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return slice(start, stop, step)


def _slice_indices(s, size):
    """Get (start, stop, step, length) for a slice applied to dimension of given size."""
    start, stop, step = s.indices(size)
    length = len(range(start, stop, step))
    return start, stop, step, length


def _stream_slice(src_dset, dst_dset, idx, slice_params, stream_dim, batch_size, desc, progress):
    """
    Stream-copy a sliced dataset.

    Parameters
    ----------
    src_dset : h5py.Dataset
        Source dataset
    dst_dset : h5py.Dataset
        Destination dataset (already created with correct shape)
    idx : tuple of slices
        Full index tuple for the source
    slice_params : list of (start, stop, step, length)
        Slice parameters for each dimension
    stream_dim : int
        Dimension to stream over
    batch_size : int
        Batch size for streaming
    desc : str
        Description for progress bar
    progress : bool
        Whether to show progress bar
    """
    stream_start, stream_stop, stream_step, stream_len = slice_params[stream_dim]
    out_shape = dst_dset.shape

    batch_iter = range(0, stream_len, batch_size)
    if progress:
        batch_iter = tqdm.tqdm(batch_iter, desc=desc)

    for out_i in batch_iter:
        out_end = min(out_i + batch_size, stream_len)
        # Map output indices to source indices
        src_start = stream_start + out_i * stream_step
        src_end = stream_start + out_end * stream_step

        # Build source index - replace streaming dim with batch slice
        src_idx = list(idx)
        src_idx[stream_dim] = slice(src_start, src_end, stream_step)

        # Build dest index
        dst_idx = [slice(None)] * len(out_shape)
        dst_idx[stream_dim] = slice(out_i, out_end)

        dst_dset[tuple(dst_idx)] = src_dset[tuple(src_idx)]


def slice_svd_file(
    input_path,
    output_path,
    temporal=None,
    spatial=None,
    component=None,
    batch_size=None,
    stream_dim=0,
    stream_dim_spatial=None,
    progress=True,
):
    """
    Slice an existing SVD file to create a smaller one.

    Parameters
    ----------
    temporal : slice
        Slice for temporal dimension (dim 0 of U)
    spatial : tuple of slice objects
        Slices for spatial dimensions (dims 1+ of Vh)
    component : slice
        Slice for component/rank dimension (dim 1 of U, dim 0 of S, dim 0 of Vh)
    batch_size : int
        Batch size for streaming. If None or >= stream dim size, loads full array.
    stream_dim : int
        Dimension to stream over for U. 0=temporal, 1=component.
    stream_dim_spatial : int
        Dimension to stream over for Vh. If None, uses stream_dim.
    """
    temporal = temporal or slice(None)
    component = component or slice(None)
    spatial = spatial or ()
    if stream_dim_spatial is None:
        stream_dim_spatial = stream_dim

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        # Get source shapes
        U_src_shape = src["U"].shape  # (n_rows, rank)
        Vh_src_shape = src["Vh"].shape  # (rank, *spatial)

        # Build index tuples and compute output shapes
        U_idx = (temporal, component)
        U_slice_params = [
            _slice_indices(temporal, U_src_shape[0]),
            _slice_indices(component, U_src_shape[1]),
        ]
        U_out_shape = tuple(p[3] for p in U_slice_params)

        S_idx = (component,)

        # Pad spatial slices for Vh
        n_spatial_missing = len(Vh_src_shape) - len(spatial) - 1
        Vh_idx = (component, *spatial, *([slice(None)] * n_spatial_missing))
        Vh_slice_params = [
            _slice_indices(s, Vh_src_shape[dim]) for dim, s in enumerate(Vh_idx)
        ]
        Vh_out_shape = tuple(p[3] for p in Vh_slice_params)

        # S is always small - just slice directly
        dst.create_dataset("S", data=src["S"][S_idx])

        # U: check if we can load all at once
        U_out_stream_size = U_out_shape[stream_dim]
        if batch_size is None or batch_size >= U_out_stream_size:
            dst.create_dataset("U", data=src["U"][U_idx], chunks=True)
        else:
            dst.create_dataset("U", shape=U_out_shape, dtype=src["U"].dtype, chunks=True)
            _stream_slice(
                src["U"], dst["U"], U_idx, U_slice_params,
                stream_dim, batch_size, "Slicing U", progress
            )

        # Vh: check if we can load all at once
        Vh_out_stream_size = Vh_out_shape[stream_dim_spatial]
        if batch_size is None or batch_size >= Vh_out_stream_size:
            dst.create_dataset("Vh", data=src["Vh"][Vh_idx], chunks=True)
        else:
            dst.create_dataset("Vh", shape=Vh_out_shape, dtype=src["Vh"].dtype, chunks=True)
            _stream_slice(
                src["Vh"], dst["Vh"], Vh_idx, Vh_slice_params,
                stream_dim_spatial, batch_size, "Slicing Vh", progress
            )

        # Copy and adjust metadata
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["n_rows"] = dst["U"].shape[0]
        dst.attrs["n_inner"] = int(np.prod(dst["Vh"].shape[1:]))


@click.group()
@click.option(
    "-v", "--verbose", count=True, help="Increase verbosity (use -v, -vv, -vvv)"
)
def cli(verbose):
    """SVD conversion and slicing tools."""
    if verbose > 3:
        raise click.BadParameter("Verbose level must be 0-3")
    enable_logging(verbose)
    configure_logging("srsvd", verbose)
    configure_logging("converter", verbose)


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input video/source.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output file path.",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=1000,
    help="Number of frames to read per batch (default: 1000).",
)
@click.option(
    "-r", "--rank", type=int, required=True, help="Target rank (n_rand_col) for SRSVD."
)
@click.option(
    "-rs",
    "--rank-spatial",
    type=int,
    default=None,
    help="Row space reconstruction rank (n_rand_row) if different from row rank.",
)
@click.option(
    "--no-second-pass",
    is_flag=True,
    help="Disable the second pass; only perform the first pass.",
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option(
    "--checkpoint-every",
    type=int,
    default=1,
    help="Number of iterations between checkpoints.",
)
@click.option("--restart", is_flag=True, help="Delete extant output before converting.")
@click.option("--no-progress", is_flag=True, help="Disable tqdm progress bars.")
def convert(
    input_path,
    output_path,
    batch_size,
    rank,
    rank_spatial,
    no_second_pass,
    seed,
    checkpoint_every,
    restart,
    no_progress,
):
    """Convert video to SVD format."""
    # Validation
    if batch_size <= 0:
        raise click.BadParameter("--batch-size must be > 0")
    if rank <= 0:
        raise click.BadParameter("--rank must be > 0")
    if rank_spatial is not None and rank_spatial <= 0:
        raise click.BadParameter("--rank-spatial must be > 0 if provided")
    if checkpoint_every <= 0:
        raise click.BadParameter("--checkpoint-every must be > 0")

    second_pass = not no_second_pass
    progress = not no_progress

    # Normalize paths
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    # Check input type
    try:
        with h5py.File(input_path, "r") as f:
            if "video" in f:
                input_type = "raw_h5"
    except:
        input_type = "raw"

    # Optional force restart
    if restart:
        warning(f"Restarting svd at {output_path} in 5s")
        time.sleep(5)
        delete_svd(output_path)

    # Run conversion
    try:
        if input_type == "raw":
            result = stream_framereader(
                input_path=input_path,
                output_path=output_path,
                batch_size=batch_size,
                rank=rank,
                rank_spatial=rank_spatial,
                second_pass=second_pass,
                seed=seed,
                checkpoint_every=checkpoint_every,
                progress=progress,
            )
        elif input_type == "raw_h5":
            result = stream_h5(
                input_path=input_path,
                output_path=output_path,
                batch_size=batch_size,
                rank=rank,
                seed=seed,
                checkpoint_every=checkpoint_every,
                progress=progress,
            )
    except Exception as exc:
        error(f"Conversion failed: {exc}")
        raise

    return result


@cli.command("slice")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input SVD file.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output file path.",
)
@click.option(
    "--temporal",
    type=str,
    default=None,
    help='Temporal slice, e.g. "0:100" or "-100:".',
)
@click.option(
    "--spatial",
    type=str,
    default=None,
    multiple=True,
    help='Spatial slice(s), e.g. --spatial "10:50" --spatial "20:60".',
)
@click.option(
    "--component", type=str, default=None, help='Component/rank slice, e.g. "0:10".'
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for streaming. If not set, loads full arrays.",
)
@click.option(
    "--stream-dim",
    type=int,
    default=0,
    help="Dimension to stream over for U (0=temporal, 1=component).",
)
@click.option(
    "--stream-dim-spatial",
    type=int,
    default=None,
    help="Dimension to stream over for Vh. Defaults to --stream-dim.",
)
@click.option("--no-progress", is_flag=True, help="Disable tqdm progress bars.")
def slice_cmd(
    input_path,
    output_path,
    temporal,
    spatial,
    component,
    batch_size,
    stream_dim,
    stream_dim_spatial,
    no_progress,
):
    """Slice an existing SVD file."""
    # Parse slice strings
    temporal_slice = _parse_slice(temporal)
    component_slice = _parse_slice(component)
    spatial_slices = tuple(_parse_slice(s) for s in spatial) if spatial else ()

    slice_svd_file(
        input_path=input_path,
        output_path=output_path,
        temporal=temporal_slice,
        spatial=spatial_slices,
        component=component_slice,
        batch_size=batch_size,
        stream_dim=stream_dim,
        stream_dim_spatial=stream_dim_spatial,
        progress=not no_progress,
    )


if __name__ == "__main__":
    cli()
