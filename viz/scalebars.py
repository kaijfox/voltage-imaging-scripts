"""Scale bar utilities for publication figures."""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

from .psth_grid import override_kws


# ── Constants ──────────────────────────────────────────────────────────

_LOC_TO_FRAC: dict[str, tuple[float, float]] = {
    "lower right": (1.0, 0.0),
    "lower left": (0.0, 0.0),
    "upper right": (1.0, 1.0),
    "upper left": (0.0, 1.0),
    "center right": (1.0, 0.5),
    "center left": (0.0, 0.5),
    "lower center": (0.5, 0.0),
    "upper center": (0.5, 1.0),
}

_OFFSET_SIGN: dict[tuple[str, str], tuple[int, int]] = {
    # (axis, edge) -> (dx_sign, dy_sign) for 'outside'
    # 'inside' negates these
    ("x", "lower"): (0, -1),
    ("x", "upper"): (0, +1),
    ("y", "left"): (-1, 0),
    ("y", "right"): (+1, 0),
}

# ha/va for rotation=0, keyed by offset direction sign
_ALIGN_ROT: dict[tuple[int, int, str], tuple[str, str]] = {
    # (dx_sign, dy_sign) -> (ha, va)
    # y offset => horizontal bar
    (0, -1, "along"): ("center", "top"),
    (0, +1, "along"): ("center", "bottom"),
    (0, -1, "perp"): ("center", "top"),
    (0, +1, "perp"): ("center", "bottom"),
    # x offset => vertical bar
    (-1, 0, "along"): ("right", "center"),
    (+1, 0, "along"): ("left", "center"),
    (-1, 0, "perp"): ("right", "center"),
    (+1, 0, "perp"): ("left", "center"),
}


# ── Public ─────────────────────────────────────────────────────────────


def choose_bar_size(data_range: float, max_nbins: int = 3) -> float:
    """Pick a nice round scale bar length for a given data range."""
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")
    if max_nbins < 1:
        raise ValueError(f"max_nbins must be >= 1, got {max_nbins}")
    loc = MaxNLocator(nbins=max_nbins, steps=[1, 2, 5, 10])
    ticks = loc.tick_values(0, data_range)
    return ticks[1] - ticks[0]


def scale_bar(
    ax: plt.Axes,
    axis: str,
    size: Optional[float] = None,
    max_nbins: int = 3,
    loc: Optional[str] = None,
    xy: Optional[tuple[float, float]] = None,
    fmt: Optional[str] = None,
    text_loc: str = "outside",
    text_orient: str = "along",
    text_pad_pts: float = 5.0,
    line_kw: dict = {},
    text_kw: dict = {},
) -> tuple[Line2D, Text]:
    """Draw a single scale bar on ax.

    Parameters
    ----------
    ax: plt.Axes
        Target axes to draw the scale bar on.
    axis: str
        'x' or 'y' to indicate bar orientation.
    size: float or None
        Bar length in data units. If None, auto-chosen from axis limits.
    max_nbins: int
        When auto-sizing, maximum bins passed to the locator.
    loc: str or None
        Anchor string (e.g. 'lower right') from _LOC_TO_FRAC, or None to use
        `xy` as axes-fraction placement.
    xy: tuple or None
        Data-unit offset when `loc` is a str, or axes-fraction coords when
        `loc` is None.
    fmt: str
        Format string applied as `fmt.format(size)`; empty string
        suppresses the label.
    text_loc: str
        'inside' or 'outside' placement relative to the bar.
    text_orient: str
        'along' to align text with the bar, 'perp' for perpendicular.
    line_kw, text_kw: dict
        Keyword overrides forwarded to matplotlib; merged with sensible
        defaults.

    Returns
    -------
    tuple(Line2D, Text)
        The created Line2D and Text artists. Text may be None if label is
        suppressed.
    """
    line_kw = dict(line_kw)
    text_kw = dict(text_kw)

    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    # ── Resolve size ──
    # If size is None, auto-choose from axis limits via choose_bar_size.
    # Raise if limits are default (0, 1) and no size given.
    if size is not None:
        if size == 0:
            raise ValueError("size must be non-zero")
    else:
        # read axis limits and compute data range
        if axis == "x":
            x0, x1 = ax.get_xlim()
            if np.allclose((x0, x1), (0.0, 1.0)):
                raise ValueError(
                    "Axis limits are default (0, 1); provide size or set axis limits"
                )
            data_range = abs(x1 - x0)
        else:
            y0, y1 = ax.get_ylim()
            if np.allclose((y0, y1), (0.0, 1.0)):
                raise ValueError(
                    "Axis limits are default (0, 1); provide size or set axis limits"
                )
            data_range = abs(y1 - y0)
        size = choose_bar_size(data_range, max_nbins=max_nbins)

    # ── Resolve position ──
    start, end = _resolve_position(ax, axis, size, loc, xy)

    # Preserve limits (spec: do NOT modify xlim/ylim)
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()

    # ── Draw line ──
    # ax.plot([start_x, end_x], [start_y, end_y], **merged_kw)
    # defaults: color='black', linewidth=1.5, solid_capstyle='butt'
    # Set clip_on=False on the Line2D artist
    line_defaults = {"color": "black", "linewidth": 1.5, "solid_capstyle": "butt"}
    merged_line_kw = override_kws(line_kw, **line_defaults)
    xs = (start[0], end[0])
    ys = (start[1], end[1])
    line_artists = ax.plot(xs, ys, **merged_line_kw)
    line = line_artists[0]
    try:
        line.set_clip_on(False)
    except Exception:
        pass

    # ── Draw text ──
    # label = fmt.format(value=size); skip if empty
    # offset, ha, va, rot = _resolve_text(...)
    # ax.annotate(label, midpoint, textcoords='offset points', ...)
    # Set clip_on=False on the Text artist

    # Format label
    label = str(size)
    if fmt is not None:
        try:
            label = fmt.format(size)
        except Exception:
            label = ""
    else:
        if int(size) == size:
            label = str(int(size))

    text = None
    if label != "":
        # Find text location
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        offset_pts, ha, va, rot = _resolve_text(
            ax,
            axis,
            start,
            end,
            text_loc=text_loc,
            text_orient=text_orient,
            pad_pts=text_pad_pts,
        )
        # Set up defaults & kwargs
        fontsize = plt.matplotlib.rcParams.get(f"{axis}tick.labelsize", 8)
        if not isinstance(fontsize, (int, float)):
            fontsize = plt.matplotlib.rcParams["font.size"]
        text_defaults = {
            "color": merged_line_kw.get("color", "black"),
            "fontsize": fontsize,
        }
        merged_text_kw = override_kws(text_kw, **text_defaults)
        # Create text artist
        text = ax.annotate(
            label,
            xy=midpoint,
            textcoords="offset points",
            xytext=offset_pts,
            ha=ha,
            va=va,
            rotation=rot,
            **merged_text_kw,
        )
        try:
            text.set_clip_on(False)
        except Exception:
            pass

    # Restore limits to avoid autoscale changes
    ax.set_xlim(old_xlim)
    ax.set_ylim(old_ylim)

    return line, text


# ── Helpers ────────────────────────────────────────────────────────────


def _resolve_position(
    ax: plt.Axes,
    axis: str,
    size: float,
    loc: Optional[str],
    xy: Optional[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute bar start & end in data coords.

    loc is str: anchor from _LOC_TO_FRAC, xy is data-unit offset.
    loc is None: xy is axes-fraction coords (required).
    Bar is centered on the resolved anchor.
    """
    if loc is None and xy is None:
        raise ValueError("xy is required when loc is None")

    if loc is not None:
        if loc not in _LOC_TO_FRAC:
            raise ValueError(f"Unknown loc {loc!r}. Valid: {list(_LOC_TO_FRAC)}")
        frac = _LOC_TO_FRAC[loc]
        # Convert axes-fraction to display, then to data coords
        disp = ax.transAxes.transform(frac)
        anchor_x, anchor_y = ax.transData.inverted().transform(disp)
        # Apply data-unit offset if provided
        if xy is not None:
            anchor_x += xy[0]
            anchor_y += xy[1]
    else:
        # xy is axes-fraction coords
        if xy is None:
            raise ValueError("xy (axes-fraction) required when loc is None")
        disp = ax.transAxes.transform(xy)
        anchor_x, anchor_y = ax.transData.inverted().transform(disp)

    # Center bar on anchor:
    if axis == "x":
        start = (anchor_x - size / 2.0, anchor_y)
        end = (anchor_x + size / 2.0, anchor_y)
    else:
        start = (anchor_x, anchor_y - size / 2.0)
        end = (anchor_x, anchor_y + size / 2.0)

    return start, end


def _resolve_text(
    ax: plt.Axes,
    axis: str,
    bar_start: tuple[float, float],
    bar_end: tuple[float, float],
    text_loc: str = "outside",
    text_orient: str = "along",
    pad_pts: float = 3.0,
) -> tuple[tuple[float, float], str, str, float]:
    """Compute (offset_pts, ha, va, rotation) for the label.

    Infers edge from bar midpoint vs axes center. Uses _OFFSET_SIGN,
    _ALIGN_ROT0, _ROT90_SWAP lookup tables.
    """
    if text_loc not in ("inside", "outside"):
        raise ValueError(f"text_loc must be 'inside'/'outside', got {text_loc!r}")
    if text_orient not in ("along", "perp"):
        raise ValueError(f"text_orient must be 'along'/'perp', got {text_orient!r}")

    # 1. Bar midpoint, axes center (from xlim/ylim midpoints)
    mid_x = (bar_start[0] + bar_end[0]) / 2.0
    mid_y = (bar_start[1] + bar_end[1]) / 2.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0

    # 2. Infer edge: axis='x' compare y, axis='y' compare x
    if axis == "x":
        edge = "lower" if (mid_y < center_y) else "upper"
    else:
        edge = "left" if (mid_x < center_x) else "right"

    # 3. offset_sign = _OFFSET_SIGN[(axis, edge)]; negate if inside
    try:
        dx_sign, dy_sign = _OFFSET_SIGN[(axis, edge)]
    except KeyError:
        dx_sign, dy_sign = (0, -1)
    if text_loc == "inside":
        dx_sign = -dx_sign
        dy_sign = -dy_sign

    offset_pts = (dx_sign * pad_pts, dy_sign * pad_pts)

    # 4. rotation = 0 if (axis,orient) in {(x,along),(y,perp)} else 90
    if (axis == "x" and text_orient == "along") or (
        axis == "y" and text_orient == "perp"
    ):
        rot = 0.0
    else:
        rot = 90.0

    # 5. ha, va = _ALIGN_ROT0[offset_sign]; if rot==90 apply _ROT90_SWAP
    ha, va = _ALIGN_ROT.get((dx_sign, dy_sign, text_orient), ("center", "center"))

    return offset_pts, ha, va, rot
