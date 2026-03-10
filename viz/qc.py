"""QC helpers for timeseries analysis and plotting."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mplutil.util as vu
from matplotlib import ticker


def plot_distribution(ax, t, data, label=None, color=".7", mean_color="k"):
    """Plot a distribution summary (IQR, min/max, mean) across observations over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib Axes instance on which to draw.
    t : array-like, shape (T,)
        Time vector corresponding to the second axis of ``data``.
    data : ndarray, shape (n, T)
        Observations array where rows are independent samples and columns are
        timepoints. May contain ``np.nan`` values which are ignored in
        statistical reductions (``nanmean``, ``nanquantile``).
    label : str or None, optional
        Base label used for legend entries; if ``None``, legend labels are
        omitted or simplified.
    color : color-like, optional
        Color used for the IQR fill and min/max lines.
    mean_color : color-like, optional
        Color used for the mean line.

    Returns
    -------
    None
        Draws on ``ax`` and does not return a value.
    """
    # data: (n, time) array, may contain nan; reduce across axis=0
    mean = np.nanmean(data, axis=0)
    q0, q25, q75, q100 = np.nanquantile(data, [0, 0.25, 0.75, 1], axis=0)
    # labels: prefix with provided label if given
    iqr_label = f"{label} IQR" if label else "IQR"
    avg_label = f"{label} Avg." if label else "Avg."
    # draw IQR as filled area
    ax.fill_between(t, q25, q75, color=color, lw=0, label=iqr_label)
    # dashed min/max lines; only label the first (min) one
    ax.plot(t, q0, color=color, ls="--", lw=0.5, label="Min/Max")
    ax.plot(t, q100, color=color, ls="--", lw=0.5)
    # plot mean line
    ax.plot(t, mean, color=mean_color, label=avg_label)


def plot_binned_metric(
    binned_df: pd.DataFrame, col: str, ax=None, colors: dict = None, ylabel: str = None
):
    """Plot per-bin metric points colored by ROI with a smoothed rolling mean line per (session, roi_id).

    Parameters
    ----------
    binned_df : pandas.DataFrame
        DataFrame with columns including ['session','roi_id','bin_time_s', ...].
    col : str
        Column name in binned_df to plot on the y-axis.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. If None, a new figure/axes is created.
    colors : dict or None
        Mapping {(session, roi_id): color}. If None, colors are assigned from tab10.
    ylabel : str or None
        Y-axis label; defaults to col if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    df = binned_df.copy()
    if df.empty:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel or col)
        return ax

    session_roi_ids = df[['session', "roi_id"]].drop_duplicates()
    # build colors mapping if not provided
    if colors is None:
        n_colors = session_roi_ids.shape[0]
        pal = sns.color_palette("tab10", n_colors)
        colors = {
            (row['session'], row['roi_id']): pal[i]
            for i, (_, row) in enumerate(session_roi_ids.iterrows())
        }

    # lineplot per (session, roi_id)
    grouped = df.groupby(["session", "roi_id"])
    for (sess, rid), g in grouped:
        g_sorted = g.sort_values("bin_time_s")
        clr = colors.get((sess, rid), '.7')
        if g_sorted.shape[0] == 0:
            continue
        ax.plot(g_sorted["bin_time_s"], g_sorted[col], color=clr, lw=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel or col)
    return ax


def plot_spike_metric(
    spike_df: pd.DataFrame,
    col: str,
    ax=None,
    colors: dict = None,
    ylabel: str = None,
    window_size_s=5,
):
    """Plot per-spike metric points colored by ROI with a smoothed rolling mean per (session, roi_id).

    Parameters
    ----------
    spike_df : pandas.DataFrame
        DataFrame with columns including ['session','roi_id','spike_time_s', ...].
    col : str
        Column name in spike_df to plot on the y-axis.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. If None, a new figure/axes is created.
    colors : dict or None
        Mapping {roi_id: color}. If None, colors are assigned from tab10.
    ylabel : str or None
        Y-axis label; defaults to col if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    df = spike_df.copy()
    if df.empty:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel or col)
        return ax

    session_roi_ids = df[['session', "roi_id"]].drop_duplicates()
    # build colors mapping if not provided
    if colors is None:
        pal = sns.color_palette("tab10", len(session_roi_ids))
        colors = {
            (row['session'], row['roi_id']): pal[i]
            for i, (_, row) in enumerate(session_roi_ids.iterrows())
        }

    # rolling mean per (session, roi_id)
    grouped = df.groupby(["session", "roi_id"])
    for (sess, rid), g in grouped:
        # covert index to datetime (in seconds)
        seconds_index = pd.to_datetime(g["spike_time_s"], unit='s')
        g = g.set_index(seconds_index)
        clr = colors.get((sess, rid), '.7')
        if g.shape[0] == 0:
            continue

        # scatter: each spike
        g_smooth_s = g.index.astype(np.int64) / 1e9
        ax.scatter(g_smooth_s, g[col], color=clr, s=10, alpha=0.8)

        # lineplot: rolling mean
        y_smooth = (
            g[col]
            .rolling(f"{window_size_s}s", min_periods=2)
            .mean()
        )
        # convert index back to float seconds for plotting
        y_smooth_s = y_smooth.index.astype(np.int64) / 1e9
        ax.plot(y_smooth_s, y_smooth, color=clr, lw=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel or col)
    return ax


def plot_bleaching_rates(rates_df: pd.DataFrame, ax=None):
    """Plot bleaching rate swarm and histogram overlay for mean_F log fits.

    Parameters
    ----------
    rates_df : pandas.DataFrame
        DataFrame with columns ['session','roi_id','metric','fit_type','slope',...].
    ax : matplotlib.axes.Axes or None
        Axes to draw on. If None, a new figure/axes is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    sel = rates_df[(rates_df["metric"] == "mean_F") & (rates_df["fit_type"] == "log")]
    slopes = sel["slope"].dropna()
    if slopes.size == 0:
        return ax

    sns.swarmplot(x=slopes, orient="h", ax=ax, color="k", size=3)
    hist, bins = np.histogram(slopes, bins=20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(bin_centers, hist, width=(bins[1] - bins[0]), color=".8")

    ax.set_ylim(-0.5, max(hist) + 0.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=2))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Bleaching rate (slope)")

    return ax
