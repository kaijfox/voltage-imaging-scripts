"""Tests for RefineState in roi module."""

import numpy as np
import pytest
from ..roi import ROI, ROIGeometry, RefineState, compute_boundary_edges


class TestRefineState:
    @pytest.fixture
    def simple_roi(self):
        """Create a simple 5x5 square ROI."""
        footprint = np.array([
            [r, c] for r in range(10, 15) for c in range(10, 15)
        ])
        weights = np.ones(len(footprint))
        code = np.zeros(10)
        return ROI(footprint=footprint, weights=weights, code=code)

    @pytest.fixture
    def correlations(self, simple_roi):
        """Fake correlations: higher for pixels closer to center."""
        center = np.array([12, 12])
        dists = np.linalg.norm(simple_roi.footprint - center, axis=1)
        return 1.0 - dists / dists.max()

    @pytest.fixture
    def refine_state(self, simple_roi, correlations):
        return RefineState.from_roi_and_correlations(
            simple_roi, correlations, checkpoint_interval=5
        )

    def test_n_pixels(self, refine_state, simple_roi):
        assert refine_state.n_pixels == len(simple_roi.footprint)

    def test_initial_index_is_all_pixels(self, refine_state):
        assert refine_state.current_index == refine_state.n_pixels

    def test_ranked_pixels_ordered_by_correlation(self, refine_state, correlations):
        """Highest correlation pixels should be first."""
        # The first pixel should have the highest correlation
        # (which is the center pixel in our test case)
        first_pixel = refine_state.ranked_pixels[0]
        assert first_pixel == (12, 12)  # Center of the 10-15 square

    def test_checkpoints_created(self, refine_state):
        """Checkpoints should exist at interval boundaries."""
        assert 0 in refine_state._checkpoints
        assert 5 in refine_state._checkpoints
        assert 10 in refine_state._checkpoints
        assert 15 in refine_state._checkpoints
        assert 20 in refine_state._checkpoints
        assert 25 in refine_state._checkpoints  # n_pixels

    def test_set_index_reduces_pixels(self, refine_state, simple_roi):
        geom = ROIGeometry(simple_roi)
        initial_pixels = len(geom.roi.footprint)

        refine_state.set_index(10, geom)

        assert len(geom.roi.footprint) == 10
        assert refine_state.current_index == 10

    def test_set_index_increases_pixels(self, refine_state, simple_roi):
        geom = ROIGeometry(simple_roi)

        # First reduce
        refine_state.set_index(10, geom)
        assert len(geom.roi.footprint) == 10

        # Then increase
        refine_state.set_index(20, geom)
        assert len(geom.roi.footprint) == 20
        assert refine_state.current_index == 20

    def test_set_index_zero_empties_roi(self, refine_state, simple_roi):
        geom = ROIGeometry(simple_roi)

        refine_state.set_index(0, geom)

        assert len(geom.roi.footprint) == 0
        assert refine_state.current_index == 0

    def test_boundary_consistent_after_index_change(self, refine_state, simple_roi):
        """Boundary should be correct after walking to a new index."""
        geom = ROIGeometry(simple_roi)

        refine_state.set_index(15, geom)

        # Compute expected boundary from scratch
        expected_boundary = compute_boundary_edges(geom.roi.pixel_set)
        assert geom.boundary_edges == expected_boundary

    def test_large_jump_uses_checkpoint(self, refine_state, simple_roi):
        """Large jumps should snap to checkpoint then walk."""
        geom = ROIGeometry(simple_roi)

        # Jump from 25 to 7 (should use checkpoint at 5 or 10)
        refine_state.set_index(7, geom)

        assert refine_state.current_index == 7
        assert len(geom.roi.footprint) == 7

        # Boundary should still be correct
        expected_boundary = compute_boundary_edges(geom.roi.pixel_set)
        assert geom.boundary_edges == expected_boundary

    def test_current_threshold(self, refine_state, correlations):
        """Threshold should correspond to last included pixel's correlation."""
        # At full index, threshold should be 0 (or min correlation)
        assert refine_state.current_threshold == pytest.approx(0.0, abs=0.1)

        # At index 1, threshold should be highest correlation
        # We can't easily test this without setting up the geometry,
        # but we can check the property doesn't crash
        assert refine_state.current_threshold >= 0

    def test_from_roi_and_correlations_classmethod(self, simple_roi, correlations):
        state = RefineState.from_roi_and_correlations(
            simple_roi, correlations, checkpoint_interval=10
        )

        assert state.n_pixels == len(simple_roi.footprint)
        assert len(state.ranked_pixels) == len(simple_roi.footprint)
        assert len(state.correlations) == len(simple_roi.footprint)


class TestRefineStateBoundaryConsistency:
    """Test that boundary is always correct regardless of index changes."""

    def test_random_walks(self):
        """Random walks through index should maintain boundary consistency."""
        # Create a random-ish ROI
        np.random.seed(42)
        footprint = np.array([
            [r, c] for r in range(20, 35) for c in range(20, 35)
        ])
        correlations = np.random.rand(len(footprint))
        roi = ROI(footprint=footprint, weights=np.ones(len(footprint)), code=np.zeros(5))

        state = RefineState.from_roi_and_correlations(roi, correlations, checkpoint_interval=20)
        geom = ROIGeometry(roi)

        # Random walk through indices
        indices = [len(footprint), 100, 50, 10, 80, 5, 150, 0, 200, 225]
        for idx in indices:
            idx = min(idx, state.n_pixels)
            state.set_index(idx, geom)

            expected_boundary = compute_boundary_edges(geom.roi.pixel_set)
            assert geom.boundary_edges == expected_boundary, f"Boundary mismatch at index {idx}"
