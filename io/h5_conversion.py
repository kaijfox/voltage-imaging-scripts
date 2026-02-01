from .FrameReader import FrameReader
from ..cli.common import configure_logging

import os
import h5py
import tqdm
from pathlib import Path
import numpy as np
import argparse
import sys

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
        f.create_dataset(
            "video",
            (reader.max_frames - start, reader.h, reader.w),
            dtype=np.dtype(reader.dtype),
        )
        f.attrs["frames"] = reader.max_frames - start
        f.attrs["width"] = reader.w
        f.attrs["height"] = reader.h
        f.attrs["reference_frame_start"] = start

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
            f["video"][vid_start : vid_end] = frames
