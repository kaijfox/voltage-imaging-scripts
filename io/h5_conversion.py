from .FrameReader import FrameReader
from .svd_conversion import (
    add_stream_conversion_args,
    configure_logging,
    enable_logging,
)

import os
import h5py
import tqdm
from pathlib import Path
import numpy as np
import argparse
import sys

logger, (error, warning, info, debug) = configure_logging("converter", 2)


def stream_framereader(
    input_path: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
    progress=True,
):
    # Normalize paths
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))
    reader = FrameReader(input_path)

    # Initialize hdf5 file and video datset
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "video",
            (reader.max_frames, reader.h, reader.w),
            dtype=np.dtype(reader.dtype),
        )
        f.attrs["frames"] = reader.max_frames
        f.attrs["width"] = reader.w
        f.attrs["height"] = reader.h

        # Set up optional progress indicator
        if progress:
            iterator = tqdm.trange(0, reader.max_frames, batch_size)
        else:
            iterator = range(0, reader.max_frames, batch_size)

        # Write batches of frames to the movie archive
        for i in iterator:
            end_frame = min(i + batch_size, reader.max_frames)
            frames = reader.get_frames_by_indexes(i + 1, end_frame)
            f["video"][i : i + batch_size, :, :] = frames.transpose(2, 0, 1)


def _cli(argv):

    parser = argparse.ArgumentParser(
        prog="stream_svd", description="Stream RAW video frames to HDF5 store."
    )

    # Mode + paths
    add_stream_conversion_args(parser)

    args = parser.parse_args(argv)

    input_path = args.input
    output_path = args.output
    batch_size = args.batch_size
    progress = not args.no_progress

    # Basic validation
    if batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.verbose > 3:
        parser.errror("Verbose level must be 0-3")

    # Set up logging
    enable_logging(args.verbose)
    configure_logging("srsvd", args.verbose)
    configure_logging("converter", args.verbose)

    # Call stream_framereader
    try:
        result = stream_framereader(
            input_path=input_path,
            output_path=str(output_path),
            batch_size=batch_size,
            progress=progress,
        )
    except Exception as exc:
        # surface a useful message and re-raise to keep traceback if running from CLI
        error(f"stream_framereader failed: {exc}")
        raise

    return result


if __name__ == "__main__":
    _cli(sys.argv[1:])
