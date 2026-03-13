import numpy as np
from scipy import signal as scipy_signal
from sklearn.cluster import KMeans

from ..cli.common import configure_logging

logger, (error, warn, info, debug) = configure_logging("svd")

# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------


def _spatial_mixing(fp, sizes, hops):
    """fp: (P, 2); returns list of mixing arrays per scale, each (M_s, P)"""
    r0, r1 = fp[:, 0].min(), fp[:, 0].max()
    c_within_r = [sorted(fp[fp[:, 0] == r, 1]) for r in range(r0, r1 + 1)]
    mixings = []
    for s, h in zip(sizes, hops):
        scale = []
        for r in range(r0, r1 + 1, h):
            for c in c_within_r[r - r0][::h]:
                near = (
                    (fp[:, 0] - r >= -(s // 2))
                    & (fp[:, 0] - r <= s // 2)
                    & (fp[:, 1] - c >= -(s // 2))
                    & (fp[:, 1] - c <= s // 2)
                )
                scale.append(near)
        mixings.append(np.array(scale))
    return mixings


def _spatial_batches(H, W, batch_r, batch_c, footprints=None):
    """
    Yields (r_slice, c_slice, batch_masks) tiling H x W.
    batch_masks: list of (P_i,) bool per ROI if footprints given, else None.
    """
    for r0 in range(0, H, batch_r):
        for c0 in range(0, W, batch_c):
            r_sl = slice(r0, min(r0 + batch_r, H))
            c_sl = slice(c0, min(c0 + batch_c, W))
            if footprints is None:
                yield r_sl, c_sl, None
            else:
                masks = [
                    (fp[:, 0] >= r_sl.start)
                    & (fp[:, 0] < r_sl.stop)
                    & (fp[:, 1] >= c_sl.start)
                    & (fp[:, 1] < c_sl.stop)
                    for fp in footprints
                ]
                yield r_sl, c_sl, masks


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------


def _bandpass(px, window_sizes):
    """
    px: (T, P) — mean-subtracted in place before filtering.
    Returns (n_bands, T, P) with n_bands = len(window_sizes) + 1.
    Bands ordered slow -> fast: [slowpass, ..bandpasses.., highpass].
    """
    px = px - px.mean(axis=0, keepdims=True)
    prev_lpf = None
    bpfs = np.empty([len(window_sizes) + 1, *px.shape])
    for i, w in enumerate(window_sizes):
        alpha = 1 - np.exp(np.log(0.5) / (w / 10))
        curr_lpf = scipy_signal.filtfilt(
            [alpha],
            [1, alpha - 1],
            px,
            axis=0,
            method="gust",
        )
        if prev_lpf is None:
            bpfs[i] = curr_lpf
        else:
            bpfs[i] = prev_lpf - curr_lpf
        prev_lpf = curr_lpf
    bpfs[-1] = px - prev_lpf
    return bpfs  # (n_bands, T, P)


# ---------------------------------------------------------------------------
# Per-ROI mixture accumulation
# ---------------------------------------------------------------------------


def _update_roi_mixtures(bandpassed, fp_ix, mixing_matrix, state):
    """
    bandpassed: (n_bands, T, B) for B pixels of this ROI in this batch.
    fp_ix: (B,) indices into ROI footprint array.
    mixing_matrix: (M, P_fp).
    state: dict with 'signals' (M, n_bands, T) and 'weight_sums' (M,).
    """
    batch_mix = mixing_matrix[:, fp_ix]  # (M, B)
    active = np.unique(np.nonzero(batch_mix)[0])
    if len(active) == 0:
        return

    # (n_bands, T, B) @ (B, M_act) -> (n_bands, T, M_act)
    contrib = bandpassed @ batch_mix[active].T
    state["mixtures"][active] += contrib.transpose(2, 0, 1) # (M_act, n_bands, T)

    state["weight_sums"][active] += batch_mix[active].sum(axis=1)


def _cluster_roi_seeds(mixtures, n_clusters):
    """
    mixtures: (M, n_bands, T) normalized mixture signals for one ROI.
    Returns seeds: (n_bands * K, T).
    """
    M, n_bands, T = mixtures.shape
    K = min(n_clusters, M)
    seeds = []
    for b in range(n_bands):
        sig = mixtures[:, b, :]  # (M, T)
        labels = KMeans(n_clusters=K, n_init="auto").fit_predict(sig)
        for k in range(K):
            seeds.append(sig[labels == k].mean(axis=0))
    return np.array(seeds)  # (n_bands * K, T)


# ---------------------------------------------------------------------------
# Pass 1: compute guide vectors
# ---------------------------------------------------------------------------


def compute_guide_vectors(
    frames,
    footprints,
    spatial_sizes=(21, 11, 5),
    hops=(10, 5, 2),
    window_sizes=(2000, 200),
    n_clusters=10,
    max_rank=None,
    batch_r=20,
    batch_c=20,
    start_frame=0,
    end_frame=None,
):
    """
    frames: hdf5 dataset (T, H, W).
    footprints: list of (P_i, 2) arrays.
    Returns Q: (T_used, max_rank) orthogonal guide vectors (includes constant
    component), where T_used = end_frame - start_frame.
    """
    _, H, W = frames.shape
    end_frame = end_frame or frames.shape[0]
    T = end_frame - start_frame
    n_bands = len(window_sizes) + 1

    mixing_matrices = [
        np.concatenate(_spatial_mixing(fp, spatial_sizes, hops), axis=0)
        for fp in footprints
    ]  # list of (M_i, P_i)

    states = {}
    pixels_done = [np.zeros(len(fp), dtype=bool) for fp in footprints]
    finished = [False] * len(footprints)
    all_seeds = []

    batches = _spatial_batches(H, W, batch_r, batch_c, footprints)
    prev_row = -1
    for r_sl, c_sl, batch_masks in batches:
        if r_sl.start != prev_row:
            info(f"Rows [{r_sl.start}:{r_sl.stop}] of {H}")
            prev_row = r_sl.start
        # skip batch if no ROI has pixels here
        has_px_in_batch = [
            m.any() and not done for m, done in zip(batch_masks, finished)
        ]
        if not any(has_px_in_batch):
            continue

        batch_frames = np.array(frames[start_frame:end_frame, r_sl, c_sl])  # (T, H_b, W_b)

        for roi_i, (fp, mm, mask) in enumerate(
            zip(footprints, mixing_matrices, batch_masks)
        ):
            if not has_px_in_batch[roi_i]:
                continue
            ix = np.nonzero(mask)[0]
            px = batch_frames[
                :, fp[ix, 0] - r_sl.start, fp[ix, 1] - c_sl.start
            ]  # (T, B)
            bandpassed = _bandpass(px, window_sizes)  # (n_bands, T, B)

            # Start tracking ROI if not already
            if roi_i not in states:
                M = mm.shape[0]
                states[roi_i] = {
                    "mixtures": np.zeros((M, n_bands, T)),
                    "weight_sums": np.zeros(M),
                }

            # Apply signals from this spatial batch to the ROI mixtures
            _update_roi_mixtures(bandpassed, ix, mm, states[roi_i])
            pixels_done[roi_i][ix] = True

            # Finish ROI and clean up if all pixels processed
            if pixels_done[roi_i].all():
                state = states.pop(roi_i)
                w = state["weight_sums"][:, None, None].clip(1e-12)
                mixtures = state["mixtures"] / w  # (M, n_bands, T)
                all_seeds.append(_cluster_roi_seeds(mixtures, n_clusters))
                finished[roi_i] = True

    # SVD seeds from all ROIs & truncate
    seeds = np.concatenate(all_seeds, axis=0)  # (total_seeds, T)
    _, _, Vt = np.linalg.svd(seeds, full_matrices=False)
    if max_rank is not None:
        guide = Vt[: max_rank - 1].T  # (T, max_rank - 1)
    else:
        guide = Vt.T  # (T, total_seeds)

    # QR-augment with constant vector
    const = np.ones((T, 1)) / np.sqrt(T)
    Q, _ = np.linalg.qr(np.hstack([const, guide]))  # (T, rank)

    return Q


# ---------------------------------------------------------------------------
# Pass 2: row sketch + final SVD
# ---------------------------------------------------------------------------


def row_sketch(frames, Q, batch_r=20, batch_c=20, start_frame=0, end_frame=None):
    """
    frames: hdf5 dataset (T, H, W).
    Q: (T_used, n) guide vectors.
    Returns G: (n, H, W).
    """
    end_frame = end_frame or frames.shape[0]
    T = end_frame - start_frame
    _, H, W = frames.shape
    n = Q.shape[1]
    G = np.zeros((n, H, W))
    prev_row = -1
    for r_sl, c_sl, _ in _spatial_batches(H, W, batch_r, batch_c):
        if r_sl.start != prev_row:
            info(f"Rows [{r_sl.start}:{r_sl.stop}] of {H}")
            prev_row = r_sl.start
        batch = np.array(frames[start_frame:end_frame, r_sl, c_sl]).reshape(T, -1)  # (T, P)
        batch_H = r_sl.stop - r_sl.start
        batch_W = c_sl.stop - c_sl.start
        G_batch = (Q.T @ batch).reshape(n, batch_H, batch_W)
        G[:, r_sl, c_sl] = G_batch
    return G


class GuidedSVDResult:
    def __init__(self, U, S, Vt):
        """U: (T, rank), S: (rank,), Vt: (rank, H, W)"""
        self.U, self.S, self.Vt = U, S, Vt

    def save(self, path):
        import h5py  # optional dep; only needed if saving
        with h5py.File(path, "w") as f:
            f.create_dataset("U", data=self.U)
            f.create_dataset("S", data=self.S)
            f.create_dataset("Vh", data=self.Vt)
            f.attrs['dtype'] = str(self.U.dtype)
            f.attrs['n_inner'] = np.prod(self.Vt.shape[1:])
            f.attrs['n_frames'] = self.U.shape[0]
            f.attrs['n_rand_col'] = self.U.shape[1]
            f.attrs['orthogonal'] = True
            f.attrs['second_pass_type'] = "guided"



def guided_svd(
    frames,
    footprints,
    spatial_sizes=(21, 11, 5),
    hops=(10, 5, 2),
    window_sizes=(2000, 200),
    n_clusters=10,
    max_rank=None,
    batch_r=20,
    batch_c=20,
    start_frame=0,
    end_frame=None,
):
    """
    frames: hdf5 dataset (T, H, W).
    footprints: list of (P_i, 2) arrays.
    Returns GuidedSVDResult with U (T_used, max_rank), S (max_rank,), Vt (max_rank, H, W).
    """
    end_frame = end_frame or frames.shape[0]
    info(f"Computing guide vectors")
    Q = compute_guide_vectors(
        frames,
        footprints,
        spatial_sizes,
        hops,
        window_sizes,
        n_clusters,
        max_rank,
        batch_r,
        batch_c,
        start_frame,
        end_frame,
    )
    info(f"Computing row sketch")
    G = row_sketch(frames, Q, batch_r, batch_c, start_frame, end_frame)  # (rank, H, W)

    info(f"Orthogonalizing sketch")
    rank, H, W = G.shape
    G_flat = G.reshape(rank, -1)  # (rank, H*W)
    U_reduced, S, Vt_flat = np.linalg.svd(G_flat, full_matrices=False)
    U = Q @ U_reduced  # (T_used, rank)
    Vt = Vt_flat.reshape(rank, H, W)
    return GuidedSVDResult(U, S, Vt)
