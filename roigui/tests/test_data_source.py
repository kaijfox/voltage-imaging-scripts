"""Tests for data_source module."""

import numpy as np
import pytest
from ..data_source import SVDDataSource, MeanImageDataSource


class TestSVDDataSource:
    @pytest.fixture
    def simple_video(self):
        """Create a simple test video with known structure."""
        h, w, t = 32, 32, 100
        video = np.random.randn(h, w, t).astype(np.float32) * 0.1
        # Add a signal in a specific region
        video[10:20, 10:20, :] += np.sin(np.arange(t) / 10)[np.newaxis, np.newaxis, :]
        return video

    @pytest.fixture
    def data_source(self, simple_video):
        return SVDDataSource(simple_video, n_components=10)

    def test_shape(self, data_source):
        assert data_source.shape == (32, 32)

    def test_n_components(self, data_source):
        assert data_source.n_components == 10

    def test_mean_image_shape(self, data_source):
        mean = data_source.mean_image()
        assert mean.shape == (32, 32)

    def test_spatial_loadings_shape(self, data_source):
        U = data_source.spatial_loadings
        assert U.shape == (32, 32, 10)

    def test_temporal_components_shape(self, simple_video, data_source):
        u = data_source.temporal_components
        assert u.shape[0] == simple_video.shape[2]  # t
        assert u.shape[1] == 10  # n_components

    def test_correlation_map_shape(self, data_source):
        corr = data_source.correlation_map(15, 15)
        assert corr.shape == (32, 32)

    def test_correlation_map_seed_is_one(self, data_source):
        """Correlation of seed with itself should be ~1."""
        corr = data_source.correlation_map(15, 15)
        assert corr[15, 15] == pytest.approx(1.0, abs=0.01)

    def test_correlation_map_from_code(self, data_source):
        # Get code at a pixel
        footprint = np.array([[15, 15]])
        code = data_source.extract_code(footprint)

        # Correlation map from code should match correlation map from pixel
        corr_from_pixel = data_source.correlation_map(15, 15)
        corr_from_code = data_source.correlation_map_from_code(code)

        np.testing.assert_allclose(corr_from_pixel, corr_from_code, rtol=1e-5)

    def test_extract_code_shape(self, data_source):
        footprint = np.array([[10, 10], [11, 11], [12, 12]])
        code = data_source.extract_code(footprint)
        assert code.shape == (10,)

    def test_extract_code_with_weights(self, data_source):
        footprint = np.array([[10, 10], [11, 11]])
        weights = np.array([1.0, 0.0])  # Only first pixel
        code_weighted = data_source.extract_code(footprint, weights)
        code_single = data_source.extract_code(footprint[:1])
        np.testing.assert_allclose(code_weighted, code_single)

    def test_extract_trace_shape(self, simple_video, data_source):
        footprint = np.array([[15, 15]])
        trace = data_source.extract_trace(footprint)
        assert trace.shape == (simple_video.shape[2],)

    def test_get_pixel_codes_shape(self, data_source):
        footprint = np.array([[10, 10], [11, 11], [12, 12]])
        codes = data_source.get_pixel_codes(footprint)
        assert codes.shape == (3, 10)

    def test_empty_footprint(self, data_source):
        empty = np.zeros((0, 2), dtype=int)
        assert data_source.extract_code(empty).shape == (10,)
        assert np.all(data_source.extract_code(empty) == 0)
        assert data_source.get_pixel_codes(empty).shape == (0, 10)

    def test_local_correlation_map_shape(self, data_source):
        lc = data_source.local_correlation_map()
        assert lc.shape == (32, 32)

    def test_local_correlation_map_cached(self, data_source):
        lc1 = data_source.local_correlation_map()
        lc2 = data_source.local_correlation_map()
        assert lc1 is lc2  # Same object, cached

    def test_local_correlation_higher_in_correlated_region(self, simple_video):
        """Signal region should have higher local correlation."""
        ds = SVDDataSource(simple_video, n_components=10)
        lc = ds.local_correlation_map()
        # Region 10:20, 10:20 has correlated signal
        signal_mean = lc[12:18, 12:18].mean()
        noise_mean = lc[0:6, 0:6].mean()
        assert signal_mean > noise_mean


class TestMeanImageDataSource:
    @pytest.fixture
    def mean_source(self):
        image = np.random.randn(64, 64).astype(np.float32)
        return MeanImageDataSource(image)

    def test_shape(self, mean_source):
        assert mean_source.shape == (64, 64)

    def test_n_components(self, mean_source):
        assert mean_source.n_components == 0

    def test_mean_image(self, mean_source):
        mean = mean_source.mean_image()
        assert mean.shape == (64, 64)

    def test_correlation_map_zeros(self, mean_source):
        """Without temporal data, correlation should be zero."""
        corr = mean_source.correlation_map(32, 32)
        assert np.all(corr == 0)

    def test_extract_code_empty(self, mean_source):
        footprint = np.array([[10, 10]])
        code = mean_source.extract_code(footprint)
        assert len(code) == 0
