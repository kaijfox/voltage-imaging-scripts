import numpy as np
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio.v3 as iio
from tqdm import trange


def generate_traces(N, T, tau=0.95, noise_std=0.5, initial="mean"):
    """Generate N positive drifting fluorescence traces.

    F(t) = tau * F(t-1) + eps(t), eps ~ |N(0, noise_std^2)|
    """
    traces = np.zeros((N, T))
    if initial == "mean":
        initial = noise_std * np.sqrt(2 / np.pi) / (1 - tau)
    traces[:, 0] = initial

    for t in range(1, T):
        eps = np.abs(np.random.randn(N) * noise_std)
        traces[:, t] = tau * traces[:, t - 1] + eps

    return traces


def generate_spatial_components(H, W, K, alpha=2.0, freq_range=(2, 20)):
    """Generate K spatial components with 1/f^alpha DCT spectrum.

    Returns: (H, W, K) array normalized to [0, 1] per component.
    """
    components = np.zeros((H, W, K))

    for k in range(K):
        # Create frequency grid
        fx = np.fft.fftfreq(H)[: H // 2 + 1]
        fy = np.fft.fftfreq(W)[: W // 2 + 1]
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        freq = np.sqrt(FX**2 + FY**2)

        # 1/f^alpha spectrum in frequency range
        power = np.zeros_like(freq)
        mask = (freq >= freq_range[0] / max(H, W)) & (freq <= freq_range[1] / max(H, W))
        power[mask] = 1.0 / (freq[mask] + 1e-10) ** alpha

        # Random phases
        phase = np.random.randn(*freq.shape) * np.sqrt(power)

        # Inverse DCT with random coefficients
        coeffs = np.zeros((H, W))
        coeffs[: H // 2 + 1, : W // 2 + 1] = phase * np.random.randn(*phase.shape)
        spatial = idctn(coeffs, norm="ortho")

        # Normalize to [0, 1]
        components[:, :, k] = (spatial - spatial.min()) / (
            spatial.max() - spatial.min()
        )

    return components


def sample_fluorescence(
    spatial_weights,
    T,
    n_neuropil=3,
    neuropil_alpha=2.,
    neuropil_freq_range=(2, 10),
    mean_spatial_alpha=1.,
    mean_spatial_freq_range=(2, 10),
    mean_decay_tau=0.01,
    mean_baseline=0.1,
    mean_amplitude=1.0,
    trace_tau=0.5,
    trace_noise_std=0.5,
    trace_initial="mean",
    neuropil_tau=0.5,
    neuropil_noise_std=0.2,
    neuropil_initial="mean",
    gain=100.0,
):
    """Generate and sample fluorescence with signal, neuropil, and decaying mean.

    Returns: dict with 'fluorescence', 'traces', 'neuropil_traces', 'mean_trace',
             'spatial_weights', 'neuropil_spatial', 'mean_spatial'
    """
    H, W, N = spatial_weights.shape

    # Generate traces
    traces = generate_traces(
        N, T, tau=trace_tau, noise_std=trace_noise_std, initial=trace_initial
    )

    # Generate neuropil components
    neuropil_spatial = generate_spatial_components(
        H, W, n_neuropil, alpha=neuropil_alpha, freq_range=neuropil_freq_range
    )
    neuropil_traces = generate_traces(
        n_neuropil,
        T,
        tau=neuropil_tau,
        noise_std=neuropil_noise_std,
        initial=neuropil_initial,
    )

    # Generate decaying mean spatial component
    mean_spatial = generate_spatial_components(
        H, W, 1, alpha=mean_spatial_alpha, freq_range=mean_spatial_freq_range
    )[:, :, 0]
    mean_trace = mean_amplitude * np.exp(-mean_decay_tau * np.arange(T)) + mean_baseline

    # Signal contribution: (H,W,N) @ (N,T) -> (H,W,T)
    signal = np.einsum("hwn,nt->hwt", spatial_weights, traces)

    # Neuropil contribution: (H,W,K) @ (K,T) -> (H,W,T)
    neuropil = np.einsum("hwk,kt->hwt", neuropil_spatial, neuropil_traces)

    # Mean spatial contribution: (H,W) * (T,) -> (H,W,T)
    mean_component = mean_spatial[:, :, None] * mean_trace[None, None, :]

    # Total intensity (ensure positive)
    intensity = np.maximum(signal + neuropil + mean_component, 0) * gain

    # Sample Poisson
    fluorescence = np.random.poisson(intensity).astype(np.float32)

    return {
        "fluorescence": fluorescence,
        "traces": traces,
        "neuropil_traces": neuropil_traces,
        "mean_trace": mean_trace,
        "spatial_weights": spatial_weights,
        "neuropil_spatial": neuropil_spatial,
        "mean_spatial": mean_spatial,
        "intensity": intensity,
    }


def save_fluorescence_video(
    fluorescence, filepath, fps=30, vmin=None, vmax=None, cmap="Greys_r"
):
    """Save (H, W, T) fluorescence as video file using imageio.

    Args:
        fluorescence: (H, W, T) array
        filepath: output path (e.g., 'movie.mp4')
        fps: frames per second
        vmin, vmax: intensity range (auto if None)
        cmap: matplotlib colormap name or object (default 'Greys')
    """
    H, W, T = fluorescence.shape

    if vmin is None:
        vmin = fluorescence.min()
    if vmax is None:
        vmax = np.percentile(fluorescence, 99)

    # Get colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Normalize to [0, 1]
    normalized = np.clip((fluorescence - vmin) / (vmax - vmin), 0, 1)

    # Write frames to file with progress bar
    with iio.imopen(filepath, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        for t in trange(T):
            frame_rgba = cmap(normalized[:, :, t])
            frame_rgb = (frame_rgba[:, :, :3] * 255).astype(np.uint8)
            writer.write_frame(frame_rgb)


def plot_sample_summary(data_dict, plot_to=None):
    """Plot fluorescence sample summary across provided axes.

    Args:
        data_dict: dict from sample_fluorescence()
        plot_to: Figure, List[Axes], or None
            If figure, generates layout with 2 neuropil example axes. If list of
            axes, expects [ax_traces, ax_image, ax_mean, ax_neuropil1, ...].
    """
    F = data_dict["fluorescence"]
    traces = data_dict["traces"]
    neuropil_traces = data_dict["neuropil_traces"]
    mean_trace = data_dict["mean_trace"]
    mean_spatial = data_dict["mean_spatial"]
    neuropil_spatial = data_dict["neuropil_spatial"]

    if plot_to is None:
        plot_to = plt.figure(figsize=(8, 3))
    if isinstance(plot_to, plt.Figure):
        fig = plot_to
        gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=1.0)
        ax_image = fig.add_subplot(gs[0:2, 0:2])  # Large: sample frame
        ax_traces = fig.add_subplot(gs[1, 2:5])  # Wide: traces
        ax_mean = fig.add_subplot(gs[0, -3])  # Small: mean spatial
        ax_neuropil = [
            fig.add_subplot(gs[0, -2]),  # Small: neuropil exmaples
            fig.add_subplot(gs[0, -1]),
        ]
    else:
        ax_traces, ax_image, ax_mean = plot_to[:2]
        ax_neuropil = plot_to[3:]

    # Traces with background
    ax_traces.plot(traces.T, alpha=0.5, lw=1.0, color="C0")
    ax_traces.plot(neuropil_traces.T, lw=1.0, color="C1")
    ax_traces.plot(mean_trace, lw=1, linestyle="--", color="black")
    ax_traces.set_title(
        f"Timeseries (N={traces.shape[0]}, K={neuropil_traces.shape[0]})"
    )
    ax_traces.set_xlabel("Time")
    ax_traces.set_ylabel("Intensity")

    # Sample frame
    ax_image.imshow(F[:, :, F.shape[2] // 2], cmap="gray")
    ax_image.set_title(f"Frame {F.shape[2]//2}")
    ax_image.axis("off")

    # Mean spatial component
    ax_mean.imshow(mean_spatial, cmap="viridis")
    ax_mean.set_title("Mean Spatial")
    ax_mean.axis("off")

    # First two neuropil components
    for i, ax in enumerate(ax_neuropil):
        ax.imshow(neuropil_spatial[:, :, i], cmap="viridis")
        ax.set_title(f"Neuropil axis {i}")
        ax.axis("off")
