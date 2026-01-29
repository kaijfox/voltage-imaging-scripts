"""Tests for operations module."""

import numpy as np
import pytest
from ..roi import ROI
from ..data_source import SVDDataSource
from ..operations import compute_pixel_correlations, extend_roi_watershed


class TestComputePixelCorrelations:
    @pytest.fixture
    def video_with_cell(self):
        """Video with a bright cell in the center."""
        h, w, t = 64, 64, 200
        video = np.random.randn(h, w, t).astype(np.float32) * 0.5

        # Add correlated signal in a circular region
        y, x = np.ogrid[:h, :w]
        cell_mask = ((y - 32) ** 2 + (x - 32) ** 2) < 100
        signal = np.sin(np.arange(t) / 10) * 2
        video[cell_mask] += signal[np.newaxis, :]

        return video

    @pytest.fixture
    def data_source(self, video_with_cell):
        return SVDDataSource(video_with_cell, n_components=20)

    def test_correlations_shape(self, data_source):
        footprint = np.array([[30, 30], [31, 31], [32, 32]])
        roi = ROI(footprint=footprint, weights=np.ones(3), code=np.zeros(20))

        corrs = compute_pixel_correlations(roi, data_source)

        assert corrs.shape == (3,)

    def test_correlations_in_range(self, data_source):
        """Correlations should be in [-1, 1] range."""
        footprint = np.array([[r, c] for r in range(28, 36) for c in range(28, 36)])
        roi = ROI(footprint=footprint, weights=np.ones(len(footprint)), code=np.zeros(20))

        corrs = compute_pixel_correlations(roi, data_source)

        assert np.all(corrs >= -1.0)
        assert np.all(corrs <= 1.0)

    def test_empty_roi(self, data_source):
        roi = ROI.empty()
        corrs = compute_pixel_correlations(roi, data_source)
        assert len(corrs) == 0


class TestExtendROIWatershed:
    @pytest.fixture
    def video_with_cell(self):
        """Video with a distinct cell."""
        np.random.seed(123)
        h, w, t = 64, 64, 100
        video = np.random.randn(h, w, t).astype(np.float32) * 0.2

        # Add strong correlated signal in a circular region
        y, x = np.ogrid[:h, :w]
        cell_mask = ((y - 32) ** 2 + (x - 32) ** 2) < 144  # r=12
        signal = np.sin(np.arange(t) / 8) * 3
        video[cell_mask] += signal[np.newaxis, :]

        return video

    @pytest.fixture
    def data_source(self, video_with_cell):
        return SVDDataSource(video_with_cell, n_components=15)

    def test_extends_small_roi(self, data_source):
        """Starting from a small seed inside the cell should expand."""
        # Small seed ROI in center of cell
        footprint = np.array([[31, 31], [31, 32], [32, 31], [32, 32]])
        roi = ROI(footprint=footprint, weights=np.ones(4), code=np.zeros(15))

        extended = extend_roi_watershed(roi, data_source, expansion_pixels=20)

        # Should have grown
        assert len(extended) > len(footprint)

    def test_empty_roi_returns_empty(self, data_source):
        roi = ROI.empty()
        extended = extend_roi_watershed(roi, data_source)
        assert len(extended) == 0

    def test_returns_valid_coordinates(self, data_source):
        """Extended footprint should have valid image coordinates."""
        footprint = np.array([[32, 32]])
        roi = ROI(footprint=footprint, weights=np.ones(1), code=np.zeros(15))

        extended = extend_roi_watershed(roi, data_source, expansion_pixels=15)

        h, w = data_source.shape
        assert np.all(extended[:, 0] >= 0)
        assert np.all(extended[:, 0] < h)
        assert np.all(extended[:, 1] >= 0)
        assert np.all(extended[:, 1] < w)
