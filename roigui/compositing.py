"""Additive RGB compositing for multi-layer map overlay."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MapLayerState:
    """Per-map display parameters."""
    enabled: bool = False
    color: str = "w"       # w/r/g/b/c/m/y
    lo: float = 0.0
    hi: float = 1.0
    gamma: float = 0.5     # target geometric mean of normalized image (0.01–0.99)
    intensity: float = 1.0
    gamma_enabled: bool = True


COLOR_MAP: dict[str, np.ndarray] = {
    "w": np.array([1, 1, 1], dtype=np.float32),
    "r": np.array([1, 0, 0], dtype=np.float32),
    "g": np.array([0, 1, 0], dtype=np.float32),
    "b": np.array([0, 0, 1], dtype=np.float32),
    "c": np.array([0, 1, 1], dtype=np.float32),
    "m": np.array([1, 0, 1], dtype=np.float32),
    "y": np.array([1, 1, 0], dtype=np.float32),
}


def compose_rgb(layers: List[Tuple[np.ndarray, MapLayerState, float]]) -> np.ndarray:
    """Additive blend of map layers into RGB image.

    Args:
        layers: list of (raw_image_hw, layer_state, actual_gmean) for enabled layers.
            actual_gmean is the geometric mean of the normalized image (pre-cached).

    Returns:
        (h, w, 3) float32 in [0, 1].
    """
    if not layers:
        return np.zeros((0, 0, 3), dtype=np.float32)

    first_img = layers[0][0]
    h, w = first_img.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for img, state, actual_gmean in layers:
        if img is None:
            continue
        if img.shape != (h, w):
            continue

        lo = float(state.lo)
        hi = float(state.hi)
        intensity = float(state.intensity) if state.intensity is not None else 1.0
        target_gmean = float(state.gamma) if state.gamma is not None else 0.5

        denom = hi - lo
        if denom <= 1e-12:
            continue

        norm = (img.astype(np.float32) - lo) / denom
        np.clip(norm, 0.0, 1.0, out=norm)

        # Gamma correction: if disabled, use exponent=1 (no-op)
        exponent = 1.0
        if getattr(state, "gamma_enabled", True):
            # Compute exponent that maps actual_gmean -> target_gmean
            if actual_gmean > 1e-12 and actual_gmean < (1.0 - 1e-12) and target_gmean > 1e-12:
                exponent = np.log(target_gmean) / np.log(actual_gmean)
                exponent = max(0.05, min(20.0, exponent))
                np.power(norm, exponent, out=norm)
        # If gamma disabled, exponent remains 1 and norm unchanged

        norm *= intensity

        color_vec = COLOR_MAP.get(getattr(state, "color", "w"), COLOR_MAP["w"])
        factor = norm[:, :, None] * color_vec[None, None, :]
        rgb += factor

    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb
