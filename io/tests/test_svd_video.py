import numpy as np
import pytest
from imaging_scripts.io.svd_video import SVDVideo


@pytest.fixture
def simple_video():
    """Create a simple SVDVideo for testing: U (1000, 50), S (50,), Vt (50, 128, 128)"""
    np.random.seed(42)
    U = np.random.randn(1000, 50)
    S = np.abs(np.random.randn(50)) + 0.1
    Vt = np.random.randn(50, 128, 128)
    return SVDVideo(U, S, Vt)


class TestConvolveTemporal:
    def test_temporal_boxcar_shape(self, simple_video):
        """Temporal convolution preserves shape"""
        kernel = np.ones(5) / 5
        result = simple_video.convolve(kernel, dim='t')

        assert result.U.shape == simple_video.U.shape
        assert result.S.shape == simple_video.S.shape
        assert result.Vt.shape == simple_video.Vt.shape

    def test_temporal_smoothing_reduces_variance(self, simple_video):
        """Temporal smoothing should reduce variance along time axis"""
        kernel = np.ones(11) / 11
        result = simple_video.convolve(kernel, dim='t')

        # Compare variance of first component's temporal trace
        orig_var = np.var(simple_video.U[:, 0])
        smooth_var = np.var(result.U[:, 0])
        assert smooth_var < orig_var

    def test_temporal_identity_kernel(self, simple_video):
        """Convolution with delta function should be identity"""
        kernel = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = simple_video.convolve(kernel, dim='t')

        # Should be approximately equal (some edge effects possible)
        np.testing.assert_allclose(result.U[2:-2], simple_video.U[2:-2], rtol=1e-10)

    def test_temporal_on_array_directly(self, simple_video):
        """Decorator allows calling on array directly"""
        arr = simple_video.U.T  # (50, 1000)
        kernel = np.ones(5) / 5
        result = SVDVideo.convolve(arr, kernel, dim='t')

        assert result.shape == arr.shape


class TestConvolveSpatial:
    def test_spatial_single_kernel_shape(self, simple_video):
        """Spatial convolution with single kernel preserves shape"""
        kernel = np.ones((3, 3)) / 9
        result = simple_video.convolve(kernel, dim='spatial')

        assert result.U.shape == simple_video.U.shape
        assert result.S.shape == simple_video.S.shape
        assert result.Vt.shape == simple_video.Vt.shape

    def test_spatial_multiple_kernels_shape(self, simple_video):
        """Spatial convolution with multiple kernels adds n_kernels dimension"""
        kernels = np.random.randn(4, 7, 7)
        result = simple_video.convolve(kernels, dim='spatial')

        assert result.U.shape == simple_video.U.shape
        assert result.S.shape == simple_video.S.shape
        assert result.Vt.shape == (50, 4, 128, 128)

    def test_spatial_smoothing_reduces_variance(self, simple_video):
        """Spatial smoothing should reduce variance"""
        kernel = np.ones((5, 5)) / 25
        result = simple_video.convolve(kernel, dim='spatial')

        orig_var = np.var(simple_video.Vt[0])
        smooth_var = np.var(result.Vt[0])
        assert smooth_var < orig_var

    def test_spatial_on_array_directly(self, simple_video):
        """Decorator allows calling on array directly"""
        arr = simple_video.Vt  # (50, 128, 128)
        kernel = np.ones((3, 3)) / 9
        result = SVDVideo.convolve(arr, kernel, dim='spatial')

        assert result.shape == arr.shape


class TestAdd:
    def test_add_single_component(self, simple_video):
        """Adding a single component increases rank by 1"""
        temporal = np.random.randn(1000, 1)
        spatial = np.random.randn(1, 128, 128)

        result = simple_video.add(temporal, spatial)

        assert result.U.shape == (1000, 51)
        assert result.S.shape == (51,)
        assert result.Vt.shape == (51, 128, 128)
        assert result.rank == 51

    def test_add_multiple_components(self, simple_video):
        """Adding multiple components increases rank accordingly"""
        temporal = np.random.randn(1000, 3)
        spatial = np.random.randn(3, 128, 128)
        amplitude = np.array([1.0, 2.0, 3.0])

        result = simple_video.add(temporal, spatial, amplitude=amplitude)

        assert result.U.shape == (1000, 53)
        assert result.S.shape == (53,)
        assert result.Vt.shape == (53, 128, 128)
        # Check amplitude is preserved
        np.testing.assert_allclose(result.S[-3:], amplitude)

    def test_add_with_n_filters(self, simple_video):
        """Adding with n_filters broadcasts existing Vt"""
        temporal = np.random.randn(1000, 3)
        spatial = np.random.randn(3, 4, 128, 128)  # 4 filters

        result = simple_video.add(temporal, spatial)

        assert result.U.shape == (1000, 53)
        assert result.S.shape == (53,)
        assert result.Vt.shape == (53, 4, 128, 128)

        # Original components should be broadcast across n_filters dimension
        np.testing.assert_allclose(result.Vt[0, 0], result.Vt[0, 1])
        np.testing.assert_allclose(result.Vt[0, 0], simple_video.Vt[0])

    def test_add_preserves_original(self, simple_video):
        """Original video should be unchanged"""
        original_U = simple_video.U.copy()
        original_S = simple_video.S.copy()
        original_Vt = simple_video.Vt.copy()

        temporal = np.random.randn(1000, 1)
        spatial = np.random.randn(1, 128, 128)
        _ = simple_video.add(temporal, spatial)

        np.testing.assert_array_equal(simple_video.U, original_U)
        np.testing.assert_array_equal(simple_video.S, original_S)
        np.testing.assert_array_equal(simple_video.Vt, original_Vt)

    def test_add_orthonormal_false(self, simple_video):
        """Result of add should have orthonormal=False"""
        temporal = np.random.randn(1000, 1)
        spatial = np.random.randn(1, 128, 128)

        result = simple_video.add(temporal, spatial)

        assert result.orthonormal is False
