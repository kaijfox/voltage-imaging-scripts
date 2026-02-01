from .FrameReader import FrameReader
from .svd import SRSVD
from ..cli.common import configure_logging

from pathlib import Path
import os
import h5py
import numpy as np
from numpy.typing import NDArray
import time
import tqdm


logger, (error, warning, info, debug) = configure_logging("converter")


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


