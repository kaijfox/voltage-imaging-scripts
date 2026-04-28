from ..cli.common import configure_logging

import os
import h5py
import tqdm
from pathlib import Path
import numpy as np
import argparse
import sys
from contextlib import nullcontext
from typing import Sequence

from .svd_video import SVDVideo
from .FrameReader import FrameReader

logger, (error, warning, info, debug) = configure_logging("converter")


def stream_framereader(
    input_path: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
    progress=True,
    start: int = 0,
    chunk: Sequence[int] = None,
):
    # Thin wrapper around stream_to_h5: construct FrameReader and a 0-indexed batch_loader
    input_path = str(Path(input_path))
    reader = FrameReader(input_path)

    def batch_loader(i, end_frame):
        # Translate 0-indexed [i, end_frame) to FrameReader's 1-indexed API
        frames = reader.get_frames_by_indexes(i + 1, end_frame) # (H, W, T)
        return frames.transpose(2, 0, 1) # (T, H, W)

    # Delegate to stream_to_h5
    stream_to_h5(
        batch_loader=batch_loader,
        n_frames=reader.max_frames,
        width=reader.w,
        height=reader.h,
        dtype=reader.dtype,
        output_path=output_path,
        batch_size=batch_size,
        progress=progress,
        start=start,
        chunk=chunk,
    )


def create_h5_video(
    path, frames, width, height, reference_frame_start, dtype, chunk=None
):
    """Create blank H5 video file."""
    # Apply to an open hdf5 file
    chunks = tuple(chunk) if chunk is not None else None
    if isinstance(path, h5py.File):
        f = path
        f.create_dataset("video", (frames, height, width), dtype=dtype, chunks=chunks)
        f.attrs["frames"] = frames
        f.attrs["width"] = width
        f.attrs["height"] = height
        f.attrs["reference_frame_start"] = reference_frame_start
    # Create and open, then apply
    else:
        with h5py.File(path, "w") as f:
            create_h5_video(f, frames, width, height, reference_frame_start, dtype, chunk=chunk)


def stream_to_h5(
    batch_loader,
    n_frames: int,
    width: int,
    height: int,
    dtype,
    output_path: os.PathLike,
    batch_size: int,
    progress=True,
    start: int = 0,
    chunk: Sequence[int] = None,
):
    """Write to HDF5 video, abstracted over the reading/generation function.

    batch_loader(i, end_frame) -> ndarray shape (T, H, W). Streams frames
    from `start` up to `n_frames` in `batch_size` chunks and writes them to an
    HDF5 file at `output_path` using `create_h5_video`.
    """
    output_path = str(Path(output_path))

    # Initialize hdf5 file and video dataset
    with h5py.File(output_path, "w") as f:
        create_h5_video(
            f,
            frames=n_frames - start,
            width=width,
            height=height,
            dtype=np.dtype(dtype),
            reference_frame_start=start,
            chunk=chunk,
        )

        # Set up optional progress indicator
        if progress:
            iterator = tqdm.trange(start, n_frames, batch_size)
        else:
            iterator = range(start, n_frames, batch_size)

        # Write batches of frames to the movie archive
        for i in iterator:
            end_frame = min(i + batch_size, n_frames)
            debug(f"Loading frames [{i}, {end_frame})")
            frames = batch_loader(i, end_frame)

            vid_start = i - start
            vid_end = vid_start + len(frames)
            debug(f"Writing {frames.shape} -> [{vid_start}, {vid_end}]")
            f["video"][vid_start:vid_end] = frames


def stream_svd_video(
    input_path: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
    progress=True,
    start: int = 0,
    chunk: Sequence[int] = None,
):
    """Stream reconstructed frames from an SVDVideo HDF5 to a framewise HDF5.

    - Loads the SVDVideo object (U, S, Vh) via SVDVideo.load(input_path).
    - Constructs a 0-indexed batch_loader that returns svd[i:end_frame].
    - Delegates to stream_to_h5 with derived n_frames/width/height/dtype.
    """
    input_path = str(Path(input_path))

    info(f"Loading SVDVideo from {input_path}")
    svd = SVDVideo.load(input_path)

    n_frames, height, width = len(svd.U), svd.Vt.shape[1], svd.Vt.shape[2]
    dtype = np.result_type(svd.U.dtype, svd.S.dtype, svd.Vt.dtype)

    def batch_loader(i, end_frame):
        return svd[i:end_frame] # returns (T, H, W)

    # Delegate to the core streaming engine
    stream_to_h5(
        batch_loader=batch_loader,
        n_frames=n_frames,
        width=width,
        height=height,
        dtype=dtype,
        output_path=output_path,
        batch_size=batch_size,
        progress=progress,
        start=start,
        chunk=chunk,
    )

