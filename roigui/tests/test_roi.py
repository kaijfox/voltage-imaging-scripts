"""Tests for ROI geometry utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Import roi module directly to avoid __init__.py pulling in PyQt6
sys.path.insert(0, str(Path(__file__).parent.parent))
from roi import (
    ROI,
    ROIGeometry,
    get_pixel_edges,
    compute_boundary_edges,
    footprint_to_mask,
    mask_to_footprint,
    footprint_to_crop_region,
)


class TestGetPixelEdges:
    def test_returns_four_edges(self):
        edges = get_pixel_edges(5, 10)
        assert len(edges) == 4

    def test_edge_coordinates(self):
        edges = get_pixel_edges(2, 3)
        # Pixel (2,3) has corners (2,3), (2,4), (3,3), (3,4)
        expected = {
            ((2, 3), (2, 4)),    # top
            ((3, 3), (3, 4)),    # bottom
            ((2, 3), (3, 3)),    # left
            ((2, 4), (3, 4)),    # right
        }
        assert set(edges) == expected


class TestComputeBoundaryEdges:
    def test_single_pixel(self):
        pixel_set = {(5, 5)}
        edges = compute_boundary_edges(pixel_set)
        # Single pixel has all 4 edges on boundary
        assert len(edges) == 4
        assert edges == set(get_pixel_edges(5, 5))

    def test_two_adjacent_horizontal(self):
        # Two horizontally adjacent pixels share one edge
        pixel_set = {(5, 5), (5, 6)}
        edges = compute_boundary_edges(pixel_set)
        # 8 total edges - 2 shared = 6 boundary edges
        assert len(edges) == 6

    def test_two_adjacent_vertical(self):
        pixel_set = {(5, 5), (6, 5)}
        edges = compute_boundary_edges(pixel_set)
        assert len(edges) == 6

    def test_2x2_square(self):
        pixel_set = {(0, 0), (0, 1), (1, 0), (1, 1)}
        edges = compute_boundary_edges(pixel_set)
        # 2x2 square has perimeter of 8 unit edges
        assert len(edges) == 8

    def test_empty_set(self):
        edges = compute_boundary_edges(set())
        assert len(edges) == 0


class TestFootprintMaskConversion:
    def test_footprint_to_mask(self):
        footprint = np.array([[1, 2], [3, 4], [5, 6]])
        mask = footprint_to_mask(footprint, (10, 10))
        assert mask.shape == (10, 10)
        assert mask[1, 2] == True
        assert mask[3, 4] == True
        assert mask[5, 6] == True
        assert mask.sum() == 3

    def test_mask_to_footprint(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1, 2] = True
        mask[3, 4] = True
        footprint = mask_to_footprint(mask)
        assert footprint.shape == (2, 2)
        assert set(map(tuple, footprint)) == {(1, 2), (3, 4)}

    def test_roundtrip(self):
        original = np.array([[2, 3], [4, 5], [6, 7]])
        mask = footprint_to_mask(original, (10, 10))
        recovered = mask_to_footprint(mask)
        assert set(map(tuple, original)) == set(map(tuple, recovered))

    def test_empty_footprint(self):
        footprint = np.zeros((0, 2), dtype=int)
        mask = footprint_to_mask(footprint, (10, 10))
        assert mask.sum() == 0


class TestFootprintToCropRegion:
    def test_basic(self):
        footprint = np.array([[5, 10], [7, 12], [6, 11]])
        r_min, r_max, c_min, c_max = footprint_to_crop_region(footprint)
        assert r_min == 5
        assert r_max == 8  # 7 + 1
        assert c_min == 10
        assert c_max == 13  # 12 + 1

    def test_with_padding(self):
        footprint = np.array([[5, 10], [7, 12]])
        r_min, r_max, c_min, c_max = footprint_to_crop_region(footprint, padding=2)
        assert r_min == 3   # 5 - 2
        assert r_max == 10  # 7 + 1 + 2
        assert c_min == 8   # 10 - 2
        assert c_max == 15  # 12 + 1 + 2

    def test_empty_footprint(self):
        footprint = np.zeros((0, 2), dtype=int)
        result = footprint_to_crop_region(footprint)
        assert result == (0, 0, 0, 0)


class TestROI:
    def test_pixel_set_property(self):
        footprint = np.array([[1, 2], [3, 4]])
        roi = ROI(footprint=footprint, weights=np.ones(2), code=np.array([]))
        assert roi.pixel_set == {(1, 2), (3, 4)}

    def test_pixel_set_cached(self):
        footprint = np.array([[1, 2], [3, 4]])
        roi = ROI(footprint=footprint, weights=np.ones(2), code=np.array([]))
        ps1 = roi.pixel_set
        ps2 = roi.pixel_set
        assert ps1 is ps2  # Same object (cached)

    def test_from_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 3:6] = True
        roi = ROI.from_mask(mask)
        assert len(roi.footprint) == 9  # 3x3 region
        assert len(roi.weights) == 9
        assert np.all(roi.weights == 1.0)

    def test_from_mask_with_weights(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        weights = np.arange(100).reshape(10, 10).astype(float)
        roi = ROI.from_mask(mask, weights=weights)
        # Weights should be extracted at mask positions
        assert set(roi.weights) == {0.0, 11.0}

    def test_empty(self):
        roi = ROI.empty()
        assert len(roi.footprint) == 0
        assert len(roi.weights) == 0
        assert len(roi.pixel_set) == 0


class TestROIGeometry:
    def test_initial_boundary(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        roi = ROI.from_mask(mask)
        geom = ROIGeometry(roi)
        assert len(geom.boundary_edges) == 4

    def test_add_pixel_updates_boundary(self):
        roi = ROI.empty()
        geom = ROIGeometry(roi)
        assert len(geom.boundary_edges) == 0

        geom.add_pixel(5, 5)
        assert len(geom.boundary_edges) == 4
        assert (5, 5) in roi.pixel_set

    def test_add_adjacent_pixel_merges_edge(self):
        roi = ROI.empty()
        geom = ROIGeometry(roi)

        geom.add_pixel(5, 5)
        assert len(geom.boundary_edges) == 4

        geom.add_pixel(5, 6)  # Adjacent horizontally
        assert len(geom.boundary_edges) == 6  # Shared edge removed

    def test_remove_pixel_updates_boundary(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        mask[5, 6] = True
        roi = ROI.from_mask(mask)
        geom = ROIGeometry(roi)
        assert len(geom.boundary_edges) == 6

        geom.remove_pixel(5, 6)
        assert len(geom.boundary_edges) == 4
        assert (5, 6) not in roi.pixel_set

    def test_add_duplicate_pixel_no_op(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        roi = ROI.from_mask(mask)
        geom = ROIGeometry(roi)

        edges_before = len(geom.boundary_edges)
        geom.add_pixel(5, 5)  # Already exists
        assert len(geom.boundary_edges) == edges_before

    def test_remove_nonexistent_pixel_no_op(self):
        roi = ROI.empty()
        geom = ROIGeometry(roi)

        geom.remove_pixel(5, 5)  # Doesn't exist
        assert len(geom.boundary_edges) == 0

    def test_contains(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        roi = ROI.from_mask(mask)
        geom = ROIGeometry(roi)

        assert geom.contains(5, 5)
        assert not geom.contains(5, 6)

    def test_rebuild_boundary(self):
        roi = ROI.empty()
        geom = ROIGeometry(roi)

        # Manually modify footprint (simulating bulk operation)
        roi.footprint = np.array([[5, 5], [5, 6]])
        roi.weights = np.array([1.0, 1.0])
        roi.invalidate_cache()

        # Boundary is stale
        geom.rebuild_boundary()
        assert len(geom.boundary_edges) == 6


class TestIncrementalBoundaryCorrectness:
    """Verify incremental updates match full recomputation."""

    def test_random_additions(self):
        roi = ROI.empty()
        geom = ROIGeometry(roi)

        # Add random pixels
        np.random.seed(42)
        pixels = [(np.random.randint(0, 20), np.random.randint(0, 20))
                  for _ in range(50)]

        for r, c in pixels:
            geom.add_pixel(r, c)

        # Verify against full recomputation
        expected = compute_boundary_edges(roi.pixel_set)
        assert geom.boundary_edges == expected

    def test_random_additions_and_removals(self):
        roi = ROI.empty()
        geom = ROIGeometry(roi)

        np.random.seed(123)
        # Add pixels
        for _ in range(30):
            r, c = np.random.randint(0, 10), np.random.randint(0, 10)
            geom.add_pixel(r, c)

        # Remove some
        pixels = list(roi.pixel_set)
        for r, c in pixels[:10]:
            geom.remove_pixel(r, c)

        # Add more
        for _ in range(20):
            r, c = np.random.randint(0, 10), np.random.randint(0, 10)
            geom.add_pixel(r, c)

        expected = compute_boundary_edges(roi.pixel_set)
        assert geom.boundary_edges == expected
