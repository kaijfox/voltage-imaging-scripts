"""ROI manipulation operations: extend, refine, etc."""

import numpy as np
from scipy import ndimage
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .roi import ROI, ROIGeometry
    from .data_source import DataSource

watershed_info = {}

def extend_roi_watershed(roi: "ROI", data_source: "DataSource",
                         expansion_pixels: int = 20,
                         image: np.ndarray = None) -> np.ndarray:
    """Extend ROI using watershed on provided image or correlation map.

    Args:
        roi: Current ROI to extend
        data_source: Data source for shape information
        expansion_pixels: How many pixels to expand search region
        image: Image to use for watershed (e.g., current view). If None,
               uses correlation map from ROI code.

    Returns:
        Extended footprint as (n_pixels, 2) array
    """
    if len(roi.footprint) == 0:
        return roi.footprint.copy()

    # Use provided image or compute correlation map from ROI code
    if image is not None:
        corr_map = image
    else:
        code = data_source.extract_code(roi.footprint, roi.weights)
        corr_map = data_source.correlation_map_from_code(code)

    # Get bounding box with expansion
    r_min, c_min = roi.footprint.min(axis=0)
    r_max, c_max = roi.footprint.max(axis=0)
    h, w = data_source.shape

    r_min = max(0, r_min - expansion_pixels)
    r_max = min(h, r_max + expansion_pixels + 1)
    c_min = max(0, c_min - expansion_pixels)
    c_max = min(w, c_max + expansion_pixels + 1)

    # Extract local region
    elevation = corr_map[r_min:r_max, c_min:c_max]

    # Create marker for watershed: current ROI is marker 1
    markers = np.zeros_like(elevation, dtype=np.int32)
    for r, c in roi.footprint:
        local_r, local_c = r - r_min, c - c_min
        if 0 <= local_r < markers.shape[0] and 0 <= local_c < markers.shape[1]:
            markers[local_r, local_c] = 1

    # Background marker at edges
    markers[0, :] = 2
    markers[-1, :] = 2
    markers[:, 0] = 2
    markers[:, -1] = 2

    # Watershed flows downhill: invert if ROI local high
    roi_vals = elevation[markers == 1]
    roi_is_local_max = np.median(roi_vals) > np.median(elevation)
    if roi_is_local_max:
        elevation = -elevation


    watershed_result = ndimage.watershed_ift(
        (elevation * 255).astype(np.uint8),
        markers
    )

    watershed_info['elevation'] = elevation
    watershed_info['markers'] = markers
    watershed_info['result'] = watershed_result
    

    # Extract pixels that belong to the ROI (marker 1)
    extended_mask = watershed_result == 1

    # Convert back to global coordinates
    local_footprint = np.argwhere(extended_mask)
    global_footprint = local_footprint + np.array([r_min, c_min])

    return global_footprint


def recompute_weights(footprint: np.ndarray, code: np.ndarray,
                      data_source: "DataSource") -> np.ndarray:
    """Recompute pixel weights as projection onto ROI code.

    Weights represent how strongly each pixel correlates with the ROI code.
    Each pixel's spatial code is normalized before the dot product, so weights
    reflect alignment with the ROI code regardless of pixel magnitude.

    Args:
        footprint: (n_pixels, 2) array of (row, col) coordinates
        code: (n_components,) ROI code vector
        data_source: Data source for pixel codes

    Returns:
        weights: (n_pixels,) projection weights (not normalized)
    """
    if len(footprint) == 0:
        return np.array([], dtype=np.float32)

    # Get pixel codes at footprint locations
    pixel_codes = data_source.get_pixel_codes(footprint)  # (n_pixels, n_components)

    # Normalize each pixel's code (spatial normalization)
    pixel_norms = np.linalg.norm(pixel_codes, axis=1, keepdims=True)
    pixel_norms = np.maximum(pixel_norms, 1e-10)
    normalized_codes = pixel_codes / pixel_norms

    # Project normalized codes onto ROI code
    weights = normalized_codes @ code  # (n_pixels,)

    # Clip negative values (pixels anti-correlated with code)
    weights = np.maximum(weights, 0)

    return weights.astype(np.float32)


def compute_pixel_correlations(roi: "ROI", data_source: "DataSource") -> np.ndarray:
    """Compute correlation of each ROI pixel with the ROI's code.

    Args:
        roi: ROI with footprint
        data_source: Data source for code extraction

    Returns:
        correlations: (n_pixels,) array of correlation values
    """
    if len(roi.footprint) == 0:
        return np.array([])

    # Get ROI code
    code = data_source.extract_code(roi.footprint, roi.weights)
    code_norm = np.linalg.norm(code)

    if code_norm < 1e-10:
        return np.ones(len(roi.footprint))

    # Get pixel codes
    pixel_codes = data_source.get_pixel_codes(roi.footprint)
    pixel_norms = np.linalg.norm(pixel_codes, axis=1)
    pixel_norms = np.maximum(pixel_norms, 1e-10)

    # Compute normalized correlation
    correlations = (pixel_codes @ code) / (pixel_norms * code_norm)

    return correlations


def iter_extend(roi: "ROI", data_source: "DataSource",
                max_pixels: int = 10000, threshold_fraction: float = 0.2
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively extend ROI by adding correlated pixels.

    Based on sourcery's iter_extend algorithm. Expands ROI by finding
    neighboring pixels with high correlation to the ROI code.

    Args:
        roi: Starting ROI
        data_source: Data source
        max_pixels: Safety limit for ROI size
        threshold_fraction: Fraction of max correlation to use as threshold

    Returns:
        footprint: Extended (n_pixels, 2) array
        weights: (n_pixels,) normalized weights (correlation values)
    """
    if len(roi.footprint) == 0:
        return roi.footprint.copy(), roi.weights.copy()

    h, w = data_source.shape
    U = data_source.spatial_loadings  # (h, w, n_components)

    # Start with current ROI pixels
    pixel_set = set(map(tuple, roi.footprint))
    code = data_source.extract_code(roi.footprint, roi.weights)

    prev_size = 0
    iteration = 0
    direction = 1  # Track growth direction

    while len(pixel_set) < max_pixels:
        current_size = len(pixel_set)

        # Get neighboring pixels (1-pixel dilation)
        neighbors = set()
        for r, c in pixel_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pixel_set:
                    neighbors.add((nr, nc))

        if not neighbors:
            break

        # Compute correlation of neighbors with current code
        neighbor_arr = np.array(list(neighbors))
        neighbor_codes = U[neighbor_arr[:, 0], neighbor_arr[:, 1], :]
        correlations = neighbor_codes @ code

        # Threshold: keep pixels above threshold_fraction of max
        threshold = max(0, correlations.max() * threshold_fraction)
        above_threshold = correlations > threshold

        if not above_threshold.any():
            break

        # Add pixels above threshold
        for i, (r, c) in enumerate(neighbors):
            if above_threshold[i]:
                pixel_set.add((r, c))

        # Update code based on new pixels
        footprint = np.array(list(pixel_set))
        code = data_source.extract_code(footprint)

        # Check for growth stagnation
        if iteration == 0:
            direction = 1
        new_size = len(pixel_set)
        if direction * (new_size - current_size) <= 0:
            break

        prev_size = current_size
        iteration += 1

    # Build final footprint and weights
    footprint = np.array(list(pixel_set))
    pixel_codes = U[footprint[:, 0], footprint[:, 1], :]
    weights = pixel_codes @ code
    weights = np.maximum(weights, 0)  # Clip negative
    weights = weights / (np.sum(weights ** 2) + 1e-10) ** 0.5  # Normalize

    return footprint, weights
