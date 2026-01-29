import os
import numpy as np
import h5py
import tempfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from imaging_scripts.io.svd import SRSVD


def run_svd_experiments(
    A,
    n_repeats,
    n_rand_col,
    n_rand_row=None,
    batch_size=None,
    seeds=None,
    dtype="float32",
    checkpoint_every=1,
    tmpdir=None,
    cleanup=True,
):
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be 2D (n_rows × n_cols)")
    n_rows, n_cols = A.shape
    Lc = int(n_rand_col)
    Lr = int(n_rand_row) if n_rand_row is not None else Lc

    # Full SVD
    U_full, S_full, Vh_full = np.linalg.svd(A, full_matrices=False)
    V_full = Vh_full.T

    # Prepare seeds
    if seeds is None:
        ss = np.random.SeedSequence()
        seeds = ss.spawn(n_repeats)
        seeds = [int(s.generate_state(1)[0]) for s in seeds]
    else:
        if len(seeds) != n_repeats:
            raise ValueError("seeds length must equal n_repeats")

    # Randomized SVD (in-memory) repeats
    rsvd_results = []
    for i in range(n_repeats):
        rng = np.random.default_rng(seeds[i])
        Omega = rng.standard_normal((n_cols, Lc))
        Y = A @ Omega  # n_rows × Lc
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ A  # Lc × n_cols
        U_tilde, S_r, Vh_r = np.linalg.svd(B, full_matrices=False)
        U_r = Q @ U_tilde  # n_rows × r
        V_r = Vh_r.T  # n_cols × r
        rsvd_results.append(
            {
                "U": U_r.astype(dtype),
                "S": S_r.astype(dtype),
                "V": V_r.astype(dtype),
                "seed": seeds[i],
            }
        )

    # Streaming randomized SVD repeats (two-pass)
    s_stream_results = []
    # batching
    bsz = int(batch_size) if batch_size is not None and batch_size > 0 else n_rows

    # Temp directory handling
    tmp_ctx = None
    if tmpdir is None:
        tmp_ctx = tempfile.TemporaryDirectory()
        tmpdir_path = tmp_ctx.name
    else:
        tmpdir_path = tmpdir
        os.makedirs(tmpdir_path, exist_ok=True)

    try:
        for i in range(n_repeats):
            # file paths
            h5_main = os.path.join(tmpdir_path, f"srsvd_{i}.h5")
            # Instantiate
            svd = SRSVD(
                h5_main,
                n_rows=n_rows,
                n_rand_col=Lc,
                n_rand_row=Lr,
                seed=seeds[i],
                dtype=dtype,
                checkpoint_every=checkpoint_every,
            )

            # First pass
            with svd.first_pass():
                # stream batches
                for start in range(0, n_rows, bsz):
                    end = min(start + bsz, n_rows)
                    svd.receive_batch(A[start:end, :], index=start)

            # Second pass
            with svd.second_pass():
                for start in range(0, n_rows, bsz):
                    end = min(start + bsz, n_rows)
                    svd.receive_batch(A[start:end, :], index=start)

            # Load results from h5
            with h5py.File(h5_main, "r") as f:
                U = f["U"][:]
                S = f["S"][:]
                V = f["V"][:]
            s_stream_results.append({"U": U, "S": S, "V": V, "seed": seeds[i]})

        results = {
            "full": {
                "U": U_full.astype(dtype),
                "S": S_full.astype(dtype),
                "V": V_full.astype(dtype),
            },
            "rsvd": rsvd_results,
            "stream": s_stream_results,
            "params": {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_rand_col": Lc,
                "n_rand_row": Lr,
                "batch_size": bsz,
                "seeds": seeds,
                "dtype": str(dtype),
            },
        }
    finally:
        if cleanup and tmp_ctx is not None:
            tmp_ctx.cleanup()

    return results


def plot_nd_comparison(
    A, results, use_right_vectors=True, figsize=(12, 3), analysis_only=False
):
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    n_rows, n_cols = A.shape

    # Full SVD
    U_full = results["full"]["U"]
    S_full = results["full"]["S"]
    V_full = results["full"]["V"]

    # Collect repeats
    S_rsvd = [d["S"] for d in results["rsvd"]]
    S_stream = [d["S"] for d in results["stream"]]
    # Pad to common max rank with zeros
    r_max = max([len(S_full)] + [len(s) for s in S_rsvd] + [len(s) for s in S_stream])

    def pad_s_list(S_list):
        padded = []
        for s in S_list:
            if len(s) < r_max:
                s2 = np.zeros(r_max, dtype=float)
                s2[: len(s)] = s
                padded.append(s2)
            else:
                padded.append(np.asarray(s, dtype=float)[:r_max])
        return np.asarray(padded)

    S_rsvd_mat = pad_s_list(S_rsvd)
    S_stream_mat = pad_s_list(S_stream)
    S_full_pad = np.zeros(r_max, dtype=float)
    S_full_pad[: len(S_full)] = S_full

    # Select side
    if use_right_vectors:
        D = n_cols
        V_rsvd = [d["V"] for d in results["rsvd"]]
        V_stream = [d["V"] for d in results["stream"]]
        V_full_side = V_full
    else:
        D = n_rows
        V_rsvd = [d["U"] for d in results["rsvd"]]
        V_stream = [d["U"] for d in results["stream"]]
        V_full_side = U_full

    # Pad vectors to r_max with zeros
    def pad_vecs_list(V_list):
        out = []
        for V in V_list:
            r = V.shape[1]
            if r < r_max:
                V2 = np.zeros((D, r_max), dtype=float)
                V2[:, :r] = V
                out.append(V2)
            else:
                out.append(np.asarray(V, dtype=float)[:, :r_max])
        return out

    V_rsvd_pad = pad_vecs_list(V_rsvd)
    V_stream_pad = pad_vecs_list(V_stream)

    # Align signs to full vectors to reduce variance
    def align_to_full(V_list, V_full):
        V_out = []
        for V in V_list:
            V_aligned = V.copy()
            for k in range(min(V.shape[1], V_full.shape[1])):
                v = V[:, k]
                vf = V_full[:, k]
                dot = float(np.dot(v, vf))
                if dot < 0:
                    V_aligned[:, k] = -v
            V_out.append(V_aligned)
        return V_out

    V_rsvd_aligned = align_to_full(V_rsvd_pad, V_full_side)
    V_stream_aligned = align_to_full(V_stream_pad, V_full_side)

    # Compute per-dimension variance across repeats for each component
    # Ignoring dimensions with zero singular values
    def component_variances(V_list, S_list_mat):
        # V_list: list of arrays D×r_max; S_list_mat: repeats×r_max
        reps = len(V_list)
        var = np.zeros((r_max, D), dtype=float)
        for k in range(r_max):
            # mask repeats with nonzero singular value
            mask = S_list_mat[:, k] > 0
            if not np.any(mask):
                var[k, :] = 0
                continue
            Vk = np.stack(
                [V_list[i][:, k] for i in range(reps) if mask[i]], axis=0
            )  # R×D
            var[k, :] = Vk.var(axis=0)
        return var

    var_rsvd = component_variances(V_rsvd_aligned, S_rsvd_mat)
    var_stream = component_variances(V_stream_aligned, S_stream_mat)

    # Mask zeros in mean/std
    mean_rsvd = np.mean(np.where(S_rsvd_mat > 0, S_rsvd_mat, np.nan), axis=0)
    std_rsvd = np.std(np.where(S_rsvd_mat > 0, S_rsvd_mat, np.nan), axis=0)
    mean_stream = np.mean(np.where(S_stream_mat > 0, S_stream_mat, np.nan), axis=0)
    std_stream = np.std(np.where(S_stream_mat > 0, S_stream_mat, np.nan), axis=0)

    # Construct a dictionary summarizing analysis
    analysis = {
        "params": {
            "use_right_vectors": use_right_vectors,
            "r_max": r_max,
            "n_rows": n_rows,
            "n_cols": n_cols,
        },
        "full": {
            "S": S_full,
            "S_padded": S_full_pad,
            "U": U_full,
            "V": V_full,
            "vectors_side": V_full_side,
        },
        "randomized": {
            "S_mat": S_rsvd_mat,
            "mean_S": mean_rsvd,
            "std_S": std_rsvd,
            "vectors_aligned": V_rsvd_aligned,
            "var_components": var_rsvd,
        },
        "streaming": {
            "S_mat": S_stream_mat,
            "mean_S": mean_stream,
            "std_S": std_stream,
            "vectors_aligned": V_stream_aligned,
            "var_components": var_stream,
        },
    }

    if analysis_only:
        return analysis

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes

    # Panel 1: full vs randomized singular values
    k_idx = np.arange(r_max)
    ax1.plot(k_idx[: len(S_full)], S_full_pad[: len(S_full)], "k.-", label="Full S")
    ax1.errorbar(
        k_idx,
        mean_rsvd,
        yerr=std_rsvd,
        fmt="o",
        color="tab:blue",
        alpha=0.8,
        label="Rand SVD (mean±std)",
    )
    ax1.set_title("Randomized\n")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Singular value")

    # Panel 2: full vs streaming randomized singular values
    ax2.plot(k_idx[: len(S_full)], S_full_pad[: len(S_full)], "k.-", label="Full S")
    ax2.errorbar(
        k_idx,
        mean_stream,
        yerr=std_stream,
        fmt="o",
        color="tab:orange",
        alpha=0.8,
        label="Streaming RSVD (mean±std)",
    )
    ax2.set_title("Streaming\n")
    ax2.set_xlabel("Component")

    # Find common color axis for variance plots
    vmin = min(var_rsvd.min(), var_stream.min())
    vmax = max(var_rsvd.max(), var_stream.max())

    # Panel 3: variance of singular vector components (randomized)
    variance_im_kws = dict(aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    im3 = ax3.imshow(var_rsvd, **variance_im_kws)
    ax3.set_title(
        f"Variance of {('right' if use_right_vectors else 'left')} "
        "vectors\nRandomized\n"
    )
    ax3.set_xlabel("Data dimension")
    ax3.set_ylabel("Component")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Panel 4: variance of singular vector components (streaming)

    im4 = ax4.imshow(var_stream, **variance_im_kws)
    ax4.set_title(f"Streaming\n")
    ax4.set_xlabel("Data dimension")
    ax4.set_ylabel("Component")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Tight layout and add hanging components
    plt.tight_layout()
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5))

    return fig, axes, analysis


def plot_2d_comparison(A, results, figsize=(8, 3)):
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("plot_2d_comparison requires A with n_cols=2")

    U_full = results["full"]["U"]
    S_full = results["full"]["S"]
    V_full = results["full"]["V"]

    # Prepare singular vectors for full SVD
    full_points = []
    for k in range(min(2, len(S_full))):
        if S_full[k] > 0:
            full_points.append(V_full[:, k] * float(np.sqrt(S_full[k])))
        else:
            full_points.append(None)

    # Prepare singular vectors for randomized & streaming
    pts = dict()
    for alg in ["rsvd", "stream"]:
        pts[alg] = [[], []]  # [comp0 list, comp1 list]
        for d in results[alg]:
            V = d["V"]
            S = d["S"]
            r = min(V.shape[1], len(S), 2)
            for k in range(2):
                if k < r and S[k] > 0:
                    v = V[:, k]
                    if k < V_full.shape[1]:
                        # Sign-align vector to full
                        sign = np.sign(np.dot(v, V_full[:, k]))
                        v = v * (1.0 if sign == 0 else sign)
                    pts[alg][k].append(v * float(np.sqrt(S[k])))
                else:
                    pts[alg][k].append(None)

        # Filter out results from with zero singular value
        pts[alg] = [np.array([p for p in comp if p is not None]) for comp in pts[alg]]

    rsvd_pts = pts["rsvd"]
    stream_pts = pts["stream"]
    
    # Figure and layout: 2 rows × 4 cols; left two cols are the big data plot
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig)

    ax_data = fig.add_subplot(gs[:, :2])
    ax_k1_rand = fig.add_subplot(gs[0, 2])
    ax_k1_stream = fig.add_subplot(gs[0, 3])
    ax_k1_stream.sharex(ax_k1_rand)
    ax_k1_stream.sharey(ax_k1_rand)
    ax_k2_rand = fig.add_subplot(gs[1, 2])
    ax_k2_stream = fig.add_subplot(gs[1, 3])
    ax_k2_stream.sharex(ax_k2_rand)
    ax_k2_stream.sharey(ax_k2_rand)

    # 1. Data scatter with endpoints of singular vectors overlaid

    ax_data.scatter(A[:, 0], A[:, 1], s=5, alpha=0.6, marker=".", color=".3")
    # Overlay full endpoints
    for k, p in enumerate(full_points):
        if p is not None:
            ax_data.scatter(
                [p[0]],
                [p[1]],
                color="black",
                s=40,
                marker="x",
                label=f"full" if k == 0 else None,
            )
    # Overlay randomized endpoints
    if rsvd_pts[0].size > 0:
        ax_data.scatter(
            rsvd_pts[0][:, 0],
            rsvd_pts[0][:, 1],
            color="tab:blue",
            s=10,
            alpha=0.5,
            marker="o",
            label="rand",
        )
    if rsvd_pts[1].size > 0:
        ax_data.scatter(
            rsvd_pts[1][:, 0],
            rsvd_pts[1][:, 1],
            color="tab:blue",
            s=10,
            alpha=0.5,
            marker="o",
        )
    # Overlay streaming endpoints
    if stream_pts[0].size > 0:
        ax_data.scatter(
            stream_pts[0][:, 0],
            stream_pts[0][:, 1],
            color="tab:orange",
            s=10,
            alpha=0.5,
            marker="o",
            label="stream",
        )
    if stream_pts[1].size > 0:
        ax_data.scatter(
            stream_pts[1][:, 0],
            stream_pts[1][:, 1],
            color="tab:orange",
            s=10,
            alpha=0.5,
            marker="o",
        )
    ax_data.set_title("Data and singular vectors (v_k · $\\sigma$_k)")
    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    ax_data.axis("equal")
    ax_data.legend(loc="upper right")

    # Helper to draw component spread on small axes
    def draw_component_spread(ax, pts, full_pt, title, color):
        ax.cla()
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.7, color=color, marker="o")
        if full_pt is not None:
            ax.scatter([full_pt[0]], [full_pt[1]], color="black", s=40, marker="x")
        ax.set_title(title)
        # ax.axis("equal")

    # k=1 spreads
    draw_component_spread(
        ax_k1_rand, rsvd_pts[0], full_points[0], "k=1 randomized", "tab:blue"
    )
    draw_component_spread(
        ax_k1_stream, stream_pts[0], full_points[0], "k=1 streaming", "tab:orange"
    )
    # k=2 spreads
    draw_component_spread(
        ax_k2_rand, rsvd_pts[1], full_points[1], "k=2 randomized", "tab:blue"
    )
    draw_component_spread(
        ax_k2_stream, stream_pts[1], full_points[1], "k=2 streaming", "tab:orange"
    )

    fig.tight_layout()
    axes = {
        "data": ax_data,
        "k1_rand": ax_k1_rand,
        "k1_stream": ax_k1_stream,
        "k2_rand": ax_k2_rand,
        "k2_stream": ax_k2_stream,
    }
    return fig, axes
