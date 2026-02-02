from .FrameReader import FrameReader
from ..cli.common import configure_logging

import os
import h5py
import tqdm
from pathlib import Path
import numpy as np
import argparse
import sys
from contextlib import nullcontext

logger, (error, warning, info, debug) = configure_logging("converter")


def stream_framereader(
    input_path: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
    progress=True,
    start: int = 0,
):
    # Normalize paths
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))
    reader = FrameReader(input_path)

    # Initialize hdf5 file and video datset
    with h5py.File(output_path, "w") as f:
        create_h5_video(
            f,
            frames=reader.max_frames - start,
            width=reader.w,
            height=reader.h,
            dtype=np.dtype(reader.dtype),
            reference_frame_start=start,
        )
        
        # Set up optional progress indicator
        if progress:
            iterator = tqdm.trange(start, reader.max_frames, batch_size)
        else:
            iterator = range(start, reader.max_frames, batch_size)

        # Write batches of frames to the movie archive
        for i in iterator:
            end_frame = min(i + batch_size, reader.max_frames)
            debug(f"Loading frames [{i+1}, {end_frame}]")
            frames = reader.get_frames_by_indexes(i + 1, end_frame)
            frames = frames.transpose(2, 0, 1)

            vid_start = i - start
            vid_end = vid_start + len(frames)
            debug(f"Writing {frames.shape} -> [{vid_start}, {vid_end}]")
            f["video"][vid_start:vid_end] = frames


def create_h5_video(path, frames, width, height, reference_frame_start, dtype):
    """Create blank H5 video file."""
    # Apply to an open hdf5 file
    if isinstance(path, h5py.File):
        f = path
        f.create_dataset("video", (frames, height, width), dtype=dtype)
        f.attrs["frames"] = frames
        f.attrs["width"] = width
        f.attrs["height"] = height
        f.attrs["reference_frame_start"] = reference_frame_start
    # Create and open, then apply
    else:
        with h5py.File(path, "w") as f:
            create_h5_video(f, frames, width, height, reference_frame_start, dtype)
