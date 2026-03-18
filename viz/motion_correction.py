from ..io.svd_video import SVDVideo
from ..viz.rois import gamma_correct
from ..cli.common import configure_logging
import mplutil.util as vu
import numpy as np
import imageio.v3 as iio
import gc

logger, (error, warning, info, debug) = configure_logging("motion")

def compare_videos(
    a: SVDVideo,
    b: SVDVideo,
    fs: float,
    target_fs: float,
    max_rank: int = None,
    stack="v",
    output_path: str = None,
):
    frame_hop = int(fs / target_fs)
    output_fps = fs / frame_hop

    # Reconstruct video segment. (T, H, W)
    info(f"Reconstructing videos with frame hop: {frame_hop}")
    raw_slice = a.reconstruct(
        slice(0, None, frame_hop),
        (slice(None), slice(None)),
        rank_idx=slice(0, max_rank),
    )
    info("Reconstructing motion-corrected video")
    mc_slice = b.reconstruct(
        slice(0, None, frame_hop),
        (slice(None), slice(None)),
        rank_idx=slice(0, max_rank),
    )
    info(f"Reconstructed videos: {raw_slice.shape}")

    # Gamma-correct each framewise
    raw_slice = gamma_correct(raw_slice, target=0.2)
    mc_slice = gamma_correct(mc_slice, target=0.2)

    # Concatenate and save/return
    if stack == "v":
        combined = np.concatenate([raw_slice, mc_slice], axis=1)  # (T, 2*H, W)
    else:
        combined = np.concatenate([raw_slice, mc_slice], axis=2)  # (T, H, 2*W)
    del raw_slice, mc_slice

    if output_path is not None:
        info(f"Saving video to {output_path}")
        info(f"  fps={output_fps}")
        info(f"  shape={combined.shape}")

        with iio.imopen(output_path, "w", plugin="pyav") as writer:
            writer.init_video_stream("libx264", fps=output_fps)
            for i, frame in enumerate(combined):
                frame = (frame * 255).astype(np.uint8)
                writer.write_frame(frame[..., None])
                del frame
                gc.collect()
    return combined


def plot_shifts(shifts: np.ndarray, fs: float):
    fig, ax = vu.subplots((3, 3), (1, 1))
    ax.plot(shifts[:, 0], label="Estimated\nrow shift")
    ax.plot(shifts[:, 1], label="Estimated\ncol shift")
    vu.legend(ax)
    return fig, ax
