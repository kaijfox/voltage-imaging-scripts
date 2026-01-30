from .FrameReader import FrameReader
from .svd import SRSVD
from .cli_tools import (add_stream_conversion_args, configure_logging, enable_logging)

from pathlib import Path
import os
import h5py
import numpy as np
from numpy.typing import NDArray
import time
import tqdm
import argparse
import sys

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
    spaces_path = Path(
        base + ".spaces.h5" if ext == ".h5" else path + ".spaces.h5"
    )
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
    second_pass_type='row',
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
        second_pass_type=second_pass_type
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
    with h5py.File(input_path, 'r') as f:
        video = f['video']
        return _stream_conversion(
            lambda i_start, i_end: video[i_start:i_end],
            get_frames_second_pass=lambda i_start, i_end: video[:, i_start:i_end],
            second_pass_dim = 1,
            second_pass_type = 'inner',
            n_frames=f.attrs['frames'],
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
    second_pass_type='row',
):
    # Default to full rank in one pass
    if rank is None:
        rank = min(len(video_array), np.prod(video_array.shape[1:]))
    if batch_size is None:
        batch_size = len(video_array)

    if second_pass_type == 'row':
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
    elif second_pass_type == 'inner':
        return _stream_conversion(
            lambda i_start, i_end: video_array[i_start:i_end],
            get_frames_second_pass=lambda i_start, i_end: video_array[:, i_start:i_end],
            second_pass_dim = 1,
            second_pass_type = 'inner',
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





def _cli(argv):

    parser = argparse.ArgumentParser(
        prog="stream_svd", description="Stream video frames and build SRSVD stores."
    )

    # Core straming & cli args
    add_stream_conversion_args(parser)
    
    # Core SVD parameterss
    parser.add_argument(
        "--rank",
        "-r",
        type=int,
        required=True,
        help="Target rank (n_rand_col) for SRSVD.",
    )
    parser.add_argument(
        "--rank-spatial",
        "-rs",
        type=int,
        default=None,
        help="Row space reconstruction rank (n_rand_row) if different from row rank.",
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Disable the second pass; only perform the first pass.",
    )

    # Random seed and checkpointing
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Number of iterations between checkpoints (passed to SRSVD).",
    )
    parser.add_argument(
        "--restart", type="store_true",
        help="Delete extant output before converting"
    )

    args = parser.parse_args(argv)

    input_path = args.input
    output_path = args.output
    batch_size = args.batch_size
    rank = args.rank
    rank_spatial = args.rank_spatial
    second_pass = not args.no_second_pass
    seed = args.seed
    checkpoint_every = args.checkpoint_every
    progress = not args.no_progress
    
    # Basic validation
    if batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if rank <= 0:
        parser.error("--rank must be > 0")
    if rank_spatial is not None and rank_spatial <= 0:
        parser.error("--rank-spatial must be > 0 if provided")
    if checkpoint_every <= 0:
        parser.error("--checkpoint-every must be > 0")
    if args.verbose > 3:
        parser.errror("Verbose level must be 0-3")
    
    # Set up logging
    enable_logging(args.verbose)
    configure_logging("srsvd", args.verbose)
    configure_logging("converter", args.verbose)

    # Normalize paths
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    # Check input type
    try:
        # Detect video generated with `.h5_conversion` module
        with h5py.File(input_path, 'r') as f:
            if "video" in f:
                input_type = "raw_h5"
    except:
        # Detect .raw video file
        input_type = 'raw'

    # Optional force restart
    if args.restart:
        warning(f"Restarting svd at {output_path} in 5s")
        time.sleep(5)
        delete_svd(output_path)

    # Call stream_framereader
    try:
        if input_type == 'raw':
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
        elif input_type == 'raw_h5':
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
        # surface a useful message and re-raise to keep traceback if running from CLI
        error(f"stream_framereader failed: {exc}")
        raise

    return result


if __name__ == "__main__":
    _cli(sys.argv[1:])
