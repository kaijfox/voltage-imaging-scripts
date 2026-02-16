import numpy as np
from pathlib import Path
import tempfile
import os

from imaging_scripts.viz.psth_videos import extract_mean_videos, video_and_trace
from imaging_scripts.timeseries.rois import ROI, ROICollection


class MockSVDVideo:
    def __init__(self, U, S, Vt):
        self.U = U
        self.S = S
        self.Vt = Vt



def test_extract_mean_videos_shape_1d_batch():
    n_t = 10
    rank = 3
    rows = 4
    cols = 5
    rng = np.random.RandomState(0)
    U = rng.randn(n_t, rank)
    S = np.arange(1, rank + 1).astype(float)
    Vt = rng.randn(rank, rows, cols)
    svd = MockSVDVideo(U, S, Vt)
    fs = 10.0
    pre_ms = 100.0
    post_ms = 200.0
    n_pre = round(pre_ms * fs / 1000)
    n_post = round(post_ms * fs / 1000)
    window_t = n_pre + n_post + 1
    event_frames = [np.array([2, 5]), np.array([3])]
    out = extract_mean_videos(svd, event_frames, fs, pre_ms, post_ms)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(event_frames), window_t, rows, cols)


def test_extract_mean_videos_max_rank_clamping():
    n_t = 8
    rank = 2
    rows = 3
    cols = 3
    U = np.ones((n_t, rank))
    S = np.array([2.0, 1.0])
    Vt = np.ones((rank, rows, cols))
    svd = MockSVDVideo(U, S, Vt)
    fs = 10.0
    pre_ms = 0.0
    post_ms = 0.0
    event_frames = [np.array([1])]
    out = extract_mean_videos(svd, event_frames, fs, pre_ms, post_ms, max_rank=5)
    assert out.shape == (1, 1, rows, cols)


def test_extract_mean_videos_spatial_slicing():
    n_t = 6
    rank = 3
    rows = 6
    cols = 6
    U = np.ones((n_t, rank))
    S = np.ones(rank)
    Vt = np.zeros((rank, rows, cols))
    Vt[:, 1:4, 2:5] = 1.0
    svd = MockSVDVideo(U, S, Vt)
    fs = 1.0
    pre_ms = 0.0
    post_ms = 0.0
    event_frames = [np.array([1])]
    out = extract_mean_videos(
        svd, event_frames, fs, pre_ms, post_ms, row_slice=slice(1, 4), col_slice=slice(2, 5)
    )
    assert out.shape == (1, 1, 3, 3)
    assert out.sum() > 0



def test_video_and_trace_creates_file_with_roi(tmp_path=None):
    if tmp_path is None:
        tmpdir = tempfile.mkdtemp()
        cleanup = True
    else:
        tmpdir = str(tmp_path)
        cleanup = False
    try:
        out_path = Path(tmpdir) / "test_out_with_roi.mp4"
        window_t = 4
        rows = 3
        cols = 3
        trace_mean = np.zeros(window_t)
        frames = np.zeros((window_t, rows, cols))
        # construct single-ROI collection with a 1-pixel footprint and colors
        mask = np.zeros((rows, cols), dtype=bool)
        mask[0, 0] = True
        roi_obj = ROI.from_mask(mask)
        roi_collection = ROICollection(rois=[roi_obj], colors=[(1.0, 0.0, 0.0)])
        video_and_trace(out_path, trace_mean, frames, fs=10.0, pre_ms=0.0, roi_collection=roi_collection)
        assert out_path.exists()
    finally:
        if cleanup:
            try:
                os.remove(out_path)
                os.rmdir(tmpdir)
            except Exception:
                pass
