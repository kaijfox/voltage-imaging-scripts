"""Interactive trace/event visualization via Plotly Dash.

See design/26-03-16_trace-composer.md for full design rationale.

Public API
----------
TraceComposer       — builder for multi-layer trace/event figures
_resolve_selection  — shared helper (also used by rois.py)
_compute_positions  — shared helper (also used by rois.py)

Helpers to migrate from rois.py
--------------------------------
_infer_names_and_colors  — move here, re-export from rois.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, Dash

from ..timeseries.types import Events, Traces
from ..timeseries.rois import ROICollection


# ---------------------------------------------------------------------------
# Shared helpers (used by both rois.py matplotlib path and TraceComposer)
# ---------------------------------------------------------------------------


def _resolve_selection(
    data_obj,  # Traces or Events
    select: Optional[Sequence[int] | Sequence[str]],
    roi_collection: Optional[ROICollection] = None,
) -> Tuple[List[int], List[str]]:
    """Resolve a select argument to (sel_idx, roi_ids).

    Parameters
    ----------
    data_obj : Traces or Events
        Must have either .ids attribute or integer-indexed rows.
    select : list[int] | list[str] | None
        If None, selects all rows.
    roi_collection : ROICollection, optional
        Preferred source of ids; falls back to data_obj.ids then integer labels.

    Returns
    -------
    sel_idx : list[int]
        Integer row indices into data_obj.
    roi_ids : list[str]
        String id for each entry in sel_idx (used as y-position key).

    Pseudocode
    ----------
    # n = number of rows in data_obj (len(data_obj.spike_frames) for Events, data_obj.data.shape[0] for Traces)
    # ids_source = roi_collection.ids if available, else data_obj.ids if available, else [str(i) for i in range(n)]
    # if select is None: sel_idx = list(range(n))
    # elif select contains strings: sel_idx = [ids_source.index(s) for s in select]
    # else: sel_idx = list(select)
    # roi_ids = [ids_source[i] for i in sel_idx]
    # return sel_idx, roi_ids
    """
    # Determine number of rows in data_obj
    if hasattr(data_obj, "spike_frames"):
        n = len(data_obj.spike_frames)
    elif getattr(data_obj, "data", None) is not None:
        n = int(np.asarray(data_obj.data).shape[0])
    else:
        n = int(len(data_obj))

    # Determine source of ids
    if (roi_collection is not None) and (
        getattr(roi_collection, "ids", None) is not None
    ):
        ids_source = list(roi_collection.ids)
    elif getattr(data_obj, "ids", None) is not None:
        ids_source = list(data_obj.ids)
    else:
        ids_source = [str(i) for i in range(n)]

    # Resolve selection
    if select is None:
        sel_idx = list(range(n))
    else:
        if (len(select) > 0) and isinstance(select[0], str):
            sel_idx = [ids_source.index(s) for s in select]
        else:
            sel_idx = list(select)

    roi_ids = [ids_source[i] for i in sel_idx]
    return sel_idx, roi_ids


def _compute_positions(
    roi_ids: List[str],
    active_roi_ids: List[str],
    data: Optional[np.ndarray] = None,
    scale_to: Optional[float] = None,
    provided: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, float], Optional[List[float]]]:
    """Compute y-position for each active roi_id and optional scale factors.

    Parameters
    ----------
    roi_ids : list[str]
        Ids corresponding to rows of data (same length as data).
    active_roi_ids : list[str]
        Subset that should appear; determines spacing and ordering.
    data : ndarray shape (n_rows, n_frames), optional
        Used to compute span for offset. If None, offset defaults to 1.0.
    scale_to : float, optional
        If given, scale each row so max abs fits within ±(scale_to/2).
    provided : sequence of float, optional
        If given, use these positions directly (len must match active_roi_ids).

    Returns
    -------
    positions : dict[str, float]
        Maps roi_id → y-offset. Only contains active_roi_ids.
    scale_factors : list[float] | None
        One factor per roi_id in active_roi_ids, or None if scale_to is None.

    Pseudocode
    ----------
    # if provided: validate len, return {roi_id: p for roi_id, p in zip(active_roi_ids, provided)}, None
    # if scale_to: scale rows in data corresponding to active_roi_ids, record factors
    # spans = [2 * nanmax(abs(data[row_idx])) for each active roi_id's row in data]
    # offset = max(spans) * 1.2 if spans else 1.0
    # positions = {roi_id: k * offset for k, roi_id in enumerate(active_roi_ids)}
    # return positions, scale_factors
    """
    # If explicit positions provided, validate and return
    if provided is not None:
        if len(provided) != len(active_roi_ids):
            raise ValueError("Length of provided positions must match active_roi_ids")
        return ({rid: float(p) for rid, p in zip(active_roi_ids, provided)}, None)

    scale_factors: Optional[List[float]] = None

    # Optionally scale rows in-place when data and scale_to are provided
    if data is not None and scale_to is not None:
        if not (isinstance(scale_to, (int, float)) and scale_to > 0):
            raise ValueError("'scale_to' must be a positive number")
        half_range = float(scale_to) / 2.0
        scale_factors = []
        for rid in active_roi_ids:
            row_idx = roi_ids.index(rid)
            row = data[row_idx]
            max_abs = np.nanmax(np.abs(row))
            if max_abs > 0:
                factor = half_range / max_abs
                data[row_idx] = row * factor
            else:
                factor = 1.0
            scale_factors.append(factor)

    # Compute per-row spans to determine offset
    spans: List[float] = []
    if data is not None:
        for rid in active_roi_ids:
            row_idx = roi_ids.index(rid)
            spans.append(2 * float(np.nanmax(np.abs(data[row_idx]))))

    if len(spans) > 0:
        span = max(spans)
        offset = span * 1.2 if span > 0 else 1.0
    else:
        offset = 1.0

    positions = {rid: float(k * offset) for k, rid in enumerate(active_roi_ids)}
    return positions, scale_factors


def _infer_names_and_colors(
    roi_collection: ROICollection,
    sel_idx: Optional[Sequence[int]] = None,
) -> Tuple[List[str], List]:
    """Infer display names and per-ROI colors.

    Moved from rois.py — re-export from there to avoid breaking imports.
    """
    import matplotlib.pyplot as plt

    n = len(roi_collection.rois)
    if sel_idx is not None:
        n = len(sel_idx)

    # names
    if getattr(roi_collection, "ids", None) is not None:
        names_all = list(roi_collection.ids)
    else:
        names_all = [f"ROI {i}" for i in range(len(roi_collection.rois))]

    if sel_idx is None:
        names = names_all
    else:
        names = [names_all[i] for i in sel_idx]

    # colors
    if getattr(roi_collection, "colors", None) is not None:
        colors_all = list(roi_collection.colors)
    else:
        cmap = plt.get_cmap("tab10")
        colors_all = [cmap(i % cmap.N) for i in range(len(roi_collection.rois))]

    if sel_idx is None:
        colors = colors_all
    else:
        colors = [colors_all[i] for i in sel_idx]

    return names, colors


# ---------------------------------------------------------------------------
# Layer dataclass (internal)
# ---------------------------------------------------------------------------


@dataclass
class _Layer:
    """One show/hide unit in the TraceComposer.

    A layer may contain multiple Traces objects and/or events. The
    `traces_list` field stores a list of tuples
    (traces_obj, roi_ids, line_kws) where `roi_ids` are the string ids
    corresponding to rows in `traces_obj` (derived when the traces were
    added). This allows multiple Traces objects to share a single layer.

    Note: `events`, `event_traces`, `roi_ids`, and `event_kws` are
    preserved for compatibility with existing event-handling logic.
    """

    name: str
    traces_list: List[Tuple[Traces, List[str], Dict[str, Any]]] = field(
        default_factory=list
    )
    events: Optional[Events] = None
    # traces object associated with events (if any) — drives point vs bar rendering
    event_traces: Optional[Traces] = None
    # Per-layer roi_ids (unchanged semantics)
    roi_ids: List[str] = field(default_factory=list)
    event_kws: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraceComposer
# ---------------------------------------------------------------------------


class TraceComposer:
    """Builder for multi-layer interactive trace/event figures.

    Usage
    -----
    vc = TraceComposer()
    vc.add_traces(dff, roi_collection=rois, name="dff", line_kws={"color": "k"})
    vc.add_traces(neuropil, roi_collection=rois, name="neuropil", line_kws={"color": "r"})
    vc.add_events(events, name="dff")   # attaches to existing "dff" layer
    vc.show()

    See design/26-03-16_trace-composer.md for full design rationale.
    """

    _composers: List["TraceComposer"] = []
    _is_trace_composer: str = "TraceComposer"

    @staticmethod
    def composers() -> List["TraceComposer"]:
        """Return all existing TraceComposer instances."""
        return TraceComposer._composers

    @staticmethod
    def gather_composers():
        import gc

        for obj in gc.get_objects():
            try:
                if not obj._is_trace_composer == "TraceComposer":
                    continue
            # Missing _is_trace_composer *or* any other issue signals also an
            # object we aren't supposed to touch
            except:
                continue
            if obj not in TraceComposer._composers:
                TraceComposer._composers.append(obj)

    @staticmethod
    def clean_all():
        """Close & delete existing TraceComposer instances."""
        old_composers = [*TraceComposer._composers]
        TraceComposer._composers.clear()
        for composer in old_composers:
            del composer

    def __init__(self) -> None:
        self._layers: List[_Layer] = []
        # Union of all roi ids seen, in order of first appearance
        self.roi_ids: List[str] = []
        self.app = None
        TraceComposer._composers.append(self)

    def __del__(self):
        self.close()
        TraceComposer._composers.remove(self)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_traces(
        self,
        traces: Traces,
        name: str = "traces",
        layer_name: Optional[str] = None,
        roi_collection: Optional[ROICollection] = None,
        line_kws: Optional[Dict[str, Any]] = None,
        scale_to: Optional[float] = None,
    ) -> "TraceComposer":
        """Add a trace layer.

        Parameters
        ----------
        traces : Traces
        name : str
            Layer name to use when creating a new layer. Ignored if layer_name is provided.
        layer_name : str, optional
            If given, append these traces to the existing layer with this name.
        roi_collection : ROICollection, optional
            Used to resolve string ids and infer colors (kept for compatibility).
        line_kws : dict, optional
            Forwarded to go.Scatter line kwargs.
        scale_to : float, optional
            Scale each trace so max abs fits within ±(scale_to/2).

        Pseudocode
        ----------
        # sel_idx, ids = _resolve_selection(traces, select, roi_collection)
        # if name already in [l.name for l in self._layers]: raise ValueError
        # append _Layer(name, traces=traces, sel_idx=sel_idx, roi_ids=ids, line_kws=line_kws or {})
        # extend self.roi_ids with ids not already present (preserve order)
        # return self

        # New logic:
        # Infer roi_ids directly from traces.ids if present, else generate [str(i) for i in range(len(traces.data))]
        # If layer_name is provided: find existing layer by that name and append (traces, roi_ids, line_kws) to its traces_list
        #   and extend that layer.roi_ids with any new ids
        # Else: ensure 'name' is unique, create new _Layer with traces_list containing (traces, roi_ids, line_kws)
        # Extend the global self.roi_ids with any new ids preserving first-seen order
        # return self
        """

        # Infer roi ids directly from the traces object per spec
        if getattr(traces, "ids", None) is not None:
            ids = list(getattr(traces, "ids"))
        else:
            d = getattr(traces, "data", None)
            if d is not None:
                n = int(np.asarray(d).shape[0])
            else:
                # Fallback to len(traces) when data is not present
                n = int(len(traces))
            ids = [str(i) for i in range(n)]

        # If attaching to an existing layer by layer_name, append to its traces_list
        if layer_name is not None:
            for layer in self._layers:
                if layer.name == layer_name:
                    layer.traces_list.append((traces, ids, line_kws or {}))
                    # Update the layer's roi_ids to include any new ids
                    for rid in ids:
                        if rid not in layer.roi_ids:
                            layer.roi_ids.append(rid)
                    break
            else:
                raise ValueError(f"Layer with name '{layer_name}' not found")
        else:
            # Ensure the new layer name is unique
            if name in [l.name for l in self._layers]:
                raise ValueError(f"Layer with name '{name}' already exists")

            # Create and append the new layer
            layer = _Layer(
                name=name,
                traces_list=[(traces, ids, line_kws or {})],
                roi_ids=ids,
            )
            self._layers.append(layer)

        # Extend the global roi_ids list preserving first-seen order
        for rid in ids:
            if rid not in self.roi_ids:
                self.roi_ids.append(rid)

        return self

    def add_events(
        self,
        events: Events,
        traces: Optional[Traces] = None,
        name: str = "events",
        layer_name: Optional[str] = None,
        line_kws: Optional[Dict[str, Any]] = None,
    ) -> "TraceComposer":
        """Add events, optionally attaching to an existing layer.

        Parameters
        ----------
        events : Events
        traces : Traces, optional
            If provided, events are rendered as points sampled on the trace
            y-values at each spike frame. If None, events are rendered as
            vertical bars at the roi's y-position.
        name : str
            Layer name to create if layer_name is None.
        layer_name : str, optional
            If given, attach events to the existing layer with this name
            (they will share that layer's show/hide toggle).
        line_kws : dict, optional
            Forwarded to go.Scatter marker kwargs (e.g. marker_symbol).

        Pseudocode
        ----------
        # Infer roi_ids directly from events.ids if present, else generate [str(i) for i in range(n)] where n = len(events.spike_frames)
        # if layer_name given: find layer by name, attach events + traces ref + event_kws to it
        # else: create new _Layer(name, events=events, traces=traces, roi_ids=ids, ...)
        # extend self.roi_ids with new ids
        # return self
        """
        # Infer roi ids directly from the events object per spec
        if getattr(events, "ids", None) is not None:
            ids = list(getattr(events, "ids"))
        else:
            # Use length of spike_frames when available
            if getattr(events, "spike_frames", None) is not None:
                n = int(len(getattr(events, "spike_frames")))
            else:
                # Fallback to len(events)
                n = int(len(events))
            ids = [str(i) for i in range(n)]

        # If attaching to an existing layer, find it and attach event data
        if layer_name is not None:
            for layer in self._layers:
                if layer.name == layer_name:
                    layer.events = events
                    # Keep a reference to the traces object that should be used to render events
                    layer.event_traces = traces
                    layer.event_kws = line_kws or {}
                    # Update the layer's roi_ids to include any new ids
                    for rid in ids:
                        if rid not in layer.roi_ids:
                            layer.roi_ids.append(rid)
                    break
            else:
                raise ValueError(f"Layer with name '{layer_name}' not found")
        else:
            # Ensure the new layer name is unique
            if name in [l.name for l in self._layers]:
                raise ValueError(f"Layer with name '{name}' already exists")

            layer = _Layer(
                name=name,
                traces_list=[],
                events=events,
                event_traces=traces,
                roi_ids=ids,
                event_kws=line_kws or {},
            )
            self._layers.append(layer)

        # Extend the global roi_ids list preserving first-seen order
        for rid in ids:
            if rid not in self.roi_ids:
                self.roi_ids.append(rid)

        return self

    # ------------------------------------------------------------------
    # Figure building (pure — no side effects)
    # ------------------------------------------------------------------

    def build_figure(
        self,
        active_roi_ids: Optional[List[str]] = None,
        active_layer_names: Optional[List[str]] = None,
        scale_to: Optional[float] = None,
    ):
        """Build and return a go.Figure for the given active subset.

        Parameters
        ----------
        active_roi_ids : list[str] | None
            ROI ids to include. Defaults to all self.roi_ids.
        active_layer_names : list[str] | None
            Layer names to include. Defaults to all layers.
        scale_to : float, optional
            Passed through to _compute_positions.

        Returns
        -------
        go.Figure

        Pseudocode
        ----------
        # active_roi_ids = active_roi_ids or self.roi_ids
        # active_layer_names = active_layer_names or [l.name for l in self._layers]
        # positions = _compute_positions(self.roi_ids, active_roi_ids, data=combined_data, scale_to=scale_to)
        #   (combined_data spans all active trace layers to get a consistent offset)
        # fig = go.Figure()
        # for layer in self._layers where layer.name in active_layer_names:
        #   for roi_id in (layer.roi_ids ∩ active_roi_ids):
        #     row_data = layer.traces.data[sel_idx_for_roi]; y = row_data + positions[roi_id]
        #     fig.add_trace(go.Scatter(x=time_axis, y=y, name=f"{layer.name} — {roi_id}", **layer.line_kws))
        #   if layer.events: add marker traces at positions[roi_id] or on trace y-values
        # set yticks/yticklabels via fig.update_layout(yaxis_tickvals, yaxis_ticktext)
        # return fig
        """
        # Resolve defaults
        active_roi_ids = (
            list(active_roi_ids) if active_roi_ids is not None else list(self.roi_ids)
        )
        active_layer_names = (
            list(active_layer_names)
            if active_layer_names is not None
            else [l.name for l in self._layers]
        )

        # Filter layers by active names preserving add order
        active_layers = [l for l in self._layers if l.name in active_layer_names]

        # Build a combined data array (rows aligned with self.roi_ids) that captures the
        # maximum absolute amplitude across all active trace layers. This is used to
        # compute a consistent offset for all traces.
        trace_layers = [
            l for l in active_layers if getattr(l, "traces_list", None) and len(l.traces_list) > 0
        ]
        combined_data = None
        if len(trace_layers) > 0:
            # Determine max number of frames across all traces in trace_layers
            n_frames_max = 0
            for l in trace_layers:
                for t_obj, _, _ in l.traces_list:
                    d = np.asarray(getattr(t_obj, "data"))
                    if d.ndim >= 2:
                        n_frames_max = max(n_frames_max, d.shape[1])

            # Initialize combined abs array with NaNs
            combined_data = np.full(
                (len(self.roi_ids), n_frames_max), np.nan, dtype=float
            )

            # For each layer and each traces entry, take absolute of each row and accumulate elementwise max
            for l in trace_layers:
                for t_obj, t_ids, _ in l.traces_list:
                    d = np.asarray(getattr(t_obj, "data"))
                    # Per spec: row i in the traces object maps to t_ids[i]
                    for i, rid in enumerate(t_ids):
                        # Directly index row i (no sel_idx lookups)
                        row = np.asarray(d[i], dtype=float)
                        # pad if needed
                        padded = np.full((n_frames_max,), np.nan, dtype=float)
                        padded[: row.shape[0]] = np.abs(row)
                        global_row = self.roi_ids.index(rid)
                        if np.isnan(combined_data[global_row]).all():
                            combined_data[global_row] = padded
                        else:
                            # elementwise max ignoring NaNs
                            a = combined_data[global_row]
                            mask = ~np.isnan(padded)
                            a[mask] = np.maximum(a[mask], padded[mask])
                            combined_data[global_row] = a

        # Compute positions and optional scale factors
        positions, scale_factors = _compute_positions(
            self.roi_ids, active_roi_ids, data=combined_data, scale_to=scale_to
        )

        fig = go.Figure()

        # Iterate layers and add traces/events
        for layer in active_layers:
            # Build a mapping from roi_id -> (traces_obj, row_idx, data, time_axis, line_kws)
            trace_row_map = {}
            for t_obj, t_ids, t_line_kws in getattr(layer, "traces_list", []):
                d = np.asarray(getattr(t_obj, "data"))
                time_axis = self._time_axis(t_obj)
                # Per spec: row i maps to t_ids[i]
                for i, rid in enumerate(t_ids):
                    row_idx = i
                    # Prefer the first mapping encountered for a given rid
                    if rid not in trace_row_map:
                        trace_row_map[rid] = (t_obj, row_idx, d, time_axis, t_line_kws)

            # Add trace lines for roi_ids present in both layer and active set
            for t_obj, t_ids, t_line_kws in getattr(layer, "traces_list", []):
                d = np.asarray(getattr(t_obj, "data"))
                time_axis = self._time_axis(t_obj)
                for i, rid in enumerate(t_ids):
                    if rid not in active_roi_ids:
                        continue
                    # Per spec: use row i directly
                    row = np.asarray(d[i], dtype=float)
                    y = row + positions[rid]
                    # Forward line keyword args via the 'line' parameter
                    line_kws = t_line_kws or {}
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=y,
                            name=f"{layer.name} — {rid}",
                            mode="lines",
                            line=line_kws,
                        )
                    )

            # Events handling
            if getattr(layer, "events", None) is not None:
                # Use the layer's roi_ids to map rows: row i maps to roi_ids[i]
                event_ids = getattr(layer, "roi_ids", [])
                # Build mapping from roi_id -> row index in the events object
                ev_obj_ids = getattr(layer.events, "ids", None)
                event_row_idx_map = {}
                if ev_obj_ids is not None:
                    ev_ids_list = list(ev_obj_ids)
                    for i, rid in enumerate(event_ids):
                        if rid in ev_ids_list:
                            event_row_idx_map[rid] = ev_ids_list.index(rid)
                else:
                    # Assume order aligns
                    for i, rid in enumerate(event_ids):
                        event_row_idx_map[rid] = i

                # Determine time axis to use for event plotting when sampling on traces
                event_time_axis = None
                event_trace_row_map = {}
                event_trace_data = None
                if getattr(layer, "event_traces", None) is not None:
                    try:
                        event_time_axis = self._time_axis(layer.event_traces)
                        event_trace_data = np.asarray(
                            getattr(layer.event_traces, "data")
                        )
                        # Build a mapping from roi_id -> row index for event_traces when available
                        et_ids = getattr(layer.event_traces, "ids", None)
                        if et_ids is not None:
                            et_ids_list = list(et_ids)
                            for i, rid in enumerate(event_ids):
                                if rid in et_ids_list:
                                    event_trace_row_map[rid] = et_ids_list.index(rid)
                    except Exception:
                        event_time_axis = None
                        event_trace_data = None
                        event_trace_row_map = {}

                for i, rid in enumerate(event_ids):
                    if rid not in active_roi_ids:
                        continue
                    e_row_idx = event_row_idx_map.get(rid)
                    if e_row_idx is None:
                        continue
                    frames = np.asarray(layer.events.spike_frames[e_row_idx], dtype=int)
                    # Attempt to sample on event_trace_data when available.
                    t_row_idx = None
                    if event_trace_data is not None:
                        # Prefer explicit mapping from event_traces.ids if available
                        if rid in event_trace_row_map:
                            t_row_idx = event_trace_row_map[rid]
                        elif rid in trace_row_map:
                            # Only use the fallback mapping if it references the same traces object
                            mapped = trace_row_map[rid]
                            mapped_tobj = mapped[0]
                            mapped_row_idx = mapped[1]
                            if getattr(layer, "event_traces", None) is not None and mapped_tobj is layer.event_traces:
                                t_row_idx = mapped_row_idx
                            else:
                                t_row_idx = None
                    if event_trace_data is not None and t_row_idx is not None:
                        # Sample on trace y-values from event_traces
                        trace_row = np.asarray(event_trace_data[t_row_idx], dtype=float)
                        x_pts = (
                            event_time_axis[frames]
                            if event_time_axis is not None
                            else frames
                        )
                        y_pts = trace_row[frames] + positions[rid]
                        fig.add_trace(
                            go.Scatter(
                                x=x_pts,
                                y=y_pts,
                                mode="markers",
                                name=f"{layer.name} — {rid}",
                                marker=layer.event_kws or {},
                            )
                        )
                    else:
                        # Render vertical bar markers at the roi's y-position (frame indices on x)
                        x_pts = frames
                        y_pts = np.full_like(x_pts, positions[rid], dtype=float)
                        marker_kws = dict(
                            symbol=(layer.event_kws or {}).get("symbol", "line-ns"),
                            size=(layer.event_kws or {}).get("size", 10),
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_pts,
                                y=y_pts,
                                mode="markers",
                                name=f"{layer.name} — {rid}",
                                marker=marker_kws,
                            )
                        )

        # Set y-axis ticks to correspond to active_roi_ids and their positions
        tickvals = [positions[rid] for rid in active_roi_ids]
        ticktext = [str(rid) for rid in active_roi_ids]

        # Determine x-axis label: if any active trace layer has sampling rate 'fs' set, label in seconds
        has_fs = any(
            (getattr(t_obj, "fs", None) is not None)
            for l in active_layers
            for t_obj, _, _ in getattr(l, "traces_list", [])
        )
        x_label = "Time (s)" if has_fs else "Frame"

        fig.update_layout(
            yaxis=dict(tickvals=tickvals, ticktext=ticktext),
            xaxis=dict(title=x_label),
            showlegend=False,
            uirevision="static",
        )

        return fig

    def _time_axis(self, traces: Traces) -> np.ndarray:
        """Return x-axis values for a Traces object.

        Pseudocode
        ----------
        # if traces.fs: return np.arange(n_frames) / traces.fs
        # else: return np.arange(n_frames)
        """
        # Determine number of frames from traces.data
        n_frames = int(np.asarray(getattr(traces, "data")).shape[1])
        fs = getattr(traces, "fs", None)
        if fs:
            return np.arange(n_frames) / float(fs)
        return np.arange(n_frames)

    # ------------------------------------------------------------------
    # Dash app
    # ------------------------------------------------------------------

    def show(self, port: int = 8050, debug: bool = False, mode='inline') -> None:
        """Launch a Dash app with ROI selector and layer toggles.

        Layout
        ------
        - dcc.Checklist for layer show/hide (all layers checked by default)
        - dcc.Checklist for ROI selection (all roi_ids checked by default)
        - dcc.Graph displaying build_figure output

        Callback
        --------
        @app.callback(Output('graph','figure'),
                      Input('roi-checklist','value'),
                      Input('layer-checklist','value'))
        def update(active_roi_ids, active_layer_names):
            return self.build_figure(active_roi_ids, active_layer_names)

        Pseudocode
        ----------
        # build layout: html.Div([layer_checklist, roi_checklist, dcc.Graph(id='graph')])
        # register callback as above
        # app.run(debug=debug, port=port, jupyter_mode="external")
        """
        # Build checklist options and defaults
        layer_names = [l.name for l in self._layers]
        layer_options = [{"label": name, "value": name} for name in layer_names]
        roi_options = [{"label": rid, "value": rid} for rid in self.roi_ids]

        default_layers = layer_names.copy()
        default_rois = self.roi_ids.copy()

        self.app = Dash()

        sidebar_style = {
            "fontFamily": "'Helvetica Neue', Arial, sans-serif",
            "padding": "4px",
            "width": "180px",
            "overflowY": "auto",
            "maxHeight": "80vh",
            "flexShrink": 0,
            "paddingRight": "8px",
            "background": "#ffffff",
        }

        self.app.layout = html.Div(
            style={"display": "flex", "flexDirection": "row"},
            children=[
                html.Div(
                    style=sidebar_style,
                    children=[
                        html.Div("Layers"),
                        dcc.Checklist(
                            id="layer-checklist",
                            options=layer_options,
                            value=default_layers,
                        ),
                        html.Hr(),
                        html.Div("ROIs"),
                        dcc.Checklist(
                            id="roi-checklist",
                            options=roi_options,
                            value=default_rois,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": 1, "paddingLeft": "8px"},
                    children=[
                        dcc.Graph(
                            id="graph",
                            figure=self.build_figure(default_rois, default_layers),
                        )
                    ],
                ),
            ],
        )

        @self.app.callback(
            Output("graph", "figure"),
            Input("roi-checklist", "value"),
            Input("layer-checklist", "value"),
        )
        def _update(active_roi_ids, active_layer_names):
            # Normalize inputs to lists and fall back to defaults when None
            a_rois = (
                list(active_roi_ids)
                if active_roi_ids is not None
                else list(self.roi_ids)
            )
            a_layers = (
                list(active_layer_names)
                if active_layer_names is not None
                else [l.name for l in self._layers]
            )
            return self.build_figure(a_rois, a_layers)

        # Run the app per spec
        self.app.run(debug=debug, port=port, jupyter_mode=mode)

    def close(self):
        """Close the Dash app if it's running."""
        if self.app is not None and hasattr(self.app.server, 'stop'):
            self.app.server.stop()
