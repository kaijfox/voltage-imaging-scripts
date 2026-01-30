import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from imaging_scripts.io.svd_conversion import array_to_svd, slice_svd_file
from imaging_scripts.io.svd import SRSVD


@pytest.fixture
def svd_file():
    """Create a temporary SVD file from a random video array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create random video: 200 frames, 32x32 spatial
        np.random.seed(42)
        video = np.random.randn(200, 32, 32).astype(np.float32)

        svd_path = os.path.join(tmpdir, "test.h5")
        svd = array_to_svd(
            video,
            output_path=svd_path,
            rank=20,
            batch_size=50,
            progress=False,
        )

        yield svd_path, svd


class TestSliceSvdFile:
    def test_slice_temporal(self, svd_file):
        """Slicing temporal dimension produces correct values."""
        svd_path, svd = svd_file
        loaded = svd.to_loaded_svd()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                temporal=slice(50, 150),
                progress=False,
            )

            sliced_svd = SRSVD(out_path)
            sliced = sliced_svd.to_loaded_svd()

            np.testing.assert_allclose(sliced.U, loaded.U[50:150, :])
            np.testing.assert_allclose(sliced.S, loaded.S)
            np.testing.assert_allclose(sliced.Vt, loaded.Vt)

    def test_slice_component(self, svd_file):
        """Slicing component dimension produces correct values."""
        svd_path, svd = svd_file
        loaded = svd.to_loaded_svd()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                component=slice(0, 10),
                progress=False,
            )

            sliced_svd = SRSVD(out_path)
            sliced = sliced_svd.to_loaded_svd()

            np.testing.assert_allclose(sliced.U, loaded.U[:, 0:10])
            np.testing.assert_allclose(sliced.S, loaded.S[0:10])
            np.testing.assert_allclose(sliced.Vt, loaded.Vt[0:10, :, :])

    def test_slice_spatial(self, svd_file):
        """Slicing spatial dimensions produces correct values."""
        svd_path, svd = svd_file
        loaded = svd.to_loaded_svd()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                spatial=(slice(5, 25), slice(10, 30)),
                progress=False,
            )

            sliced_svd = SRSVD(out_path)
            sliced = sliced_svd.to_loaded_svd()

            np.testing.assert_allclose(sliced.U, loaded.U)
            np.testing.assert_allclose(sliced.S, loaded.S)
            np.testing.assert_allclose(sliced.Vt, loaded.Vt[:, 5:25, 10:30])

    def test_slice_all_dimensions(self, svd_file):
        """Slicing all dimensions at once produces correct values."""
        svd_path, svd = svd_file
        loaded = svd.to_loaded_svd()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                temporal=slice(20, 180),
                component=slice(5, 15),
                spatial=(slice(8, 24), slice(4, 28)),
                progress=False,
            )

            sliced_svd = SRSVD(out_path)
            sliced = sliced_svd.to_loaded_svd()

            np.testing.assert_allclose(sliced.U, loaded.U[20:180, 5:15])
            np.testing.assert_allclose(sliced.S, loaded.S[5:15])
            np.testing.assert_allclose(sliced.Vt, loaded.Vt[5:15, 8:24, 4:28])

    def test_slice_negative_indices(self, svd_file):
        """Slicing with negative indices works correctly."""
        svd_path, svd = svd_file
        loaded = svd.to_loaded_svd()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                temporal=slice(-50, None),
                component=slice(-5, None),
                progress=False,
            )

            sliced_svd = SRSVD(out_path)
            sliced = sliced_svd.to_loaded_svd()

            np.testing.assert_allclose(sliced.U, loaded.U[-50:, -5:])
            np.testing.assert_allclose(sliced.S, loaded.S[-5:])
            np.testing.assert_allclose(sliced.Vt, loaded.Vt[-5:, :, :])

    def test_slice_with_step(self, svd_file):
        """Slicing with step produces correct values."""
        svd_path, svd = svd_file
        loaded = svd.to_loaded_svd()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                temporal=slice(0, 100, 2),
                progress=False,
            )

            sliced_svd = SRSVD(out_path)
            sliced = sliced_svd.to_loaded_svd()

            np.testing.assert_allclose(sliced.U, loaded.U[0:100:2, :])
            np.testing.assert_allclose(sliced.S, loaded.S)
            np.testing.assert_allclose(sliced.Vt, loaded.Vt)

    def test_slice_with_streaming(self, svd_file):
        """Slicing with streaming (small batch_size) produces same result."""
        svd_path, svd = svd_file
        loaded = svd.to_loaded_svd()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                temporal=slice(50, 150),
                component=slice(0, 10),
                batch_size=20,
                progress=False,
            )

            sliced_svd = SRSVD(out_path)
            sliced = sliced_svd.to_loaded_svd()

            np.testing.assert_allclose(sliced.U, loaded.U[50:150, 0:10])
            np.testing.assert_allclose(sliced.S, loaded.S[0:10])
            np.testing.assert_allclose(sliced.Vt, loaded.Vt[0:10, :, :])

    def test_metadata_updated(self, svd_file):
        """Metadata is correctly updated after slicing."""
        svd_path, svd = svd_file

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sliced.h5")
            slice_svd_file(
                svd_path, out_path,
                temporal=slice(50, 150),
                spatial=(slice(5, 25), slice(10, 30)),
                progress=False,
            )

            import h5py
            with h5py.File(out_path, "r") as f:
                assert f.attrs["n_rows"] == 100  # 150 - 50
                assert f.attrs["n_inner"] == 20 * 20  # (25-5) * (30-10)
