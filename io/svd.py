from .cli_tools import configure_logging

import os
import numpy as np
import h5py
from contextlib import contextmanager

logger, (error, warn, info, debug) = configure_logging("srsvd", 2)


class SRSVD:
    """
    Streaming randomized SVD.
    Processes batches, builds randomized projections, and writes
    intermediates and final factors to HDF5.

    Two second_pass_type options:
    - "row": Both passes stream over rows. First pass computes sample_col and
      sample_row, does intermediate SVD. Second pass refines with exact_row.
    - "inner": First pass streams over rows (computes sample_col only).
      Second pass streams over inner dimensions, assigning slices to exact_row.
    """

    def __init__(
        self,
        h5_filename,
        n_rows=None,
        n_rand_col=None,
        n_rand_row=None,
        second_pass_type="row",
        seed=None,
        dtype="float32",
        checkpoint_every=1,
    ):
        """
        Initialize configuration, require n_rows at init.

        Parameters
        ----------
        n_rows : int
            Size along the streaming dimension (first dimension of input tensor)
        n_rand_col : int
            Number of random projection vectors used to probe the column space.
            In N-d inputs, the column space is considered to be the span of the
            vectors
                `{data[:, *(i1, i2, ...)
                 for i1, i2, ... in ndindex(data.shape[1:])]}`
        n_rand_row : int, optional
            Number of random vectors used to probe the row space.
            Only used in "row" second pass type.
        second_pass_type : str
            "row" or "inner" - determines second pass streaming behavior
        """
        self.h5_filename = h5_filename
        base, ext = os.path.splitext(h5_filename)
        self.spaces_filename = (
            base + ".spaces.h5" if ext == ".h5" else h5_filename + ".spaces.h5"
        )

        # Validate second_pass_type
        if second_pass_type not in ("row", "inner"):
            raise ValueError(f"second_pass_type must be 'row' or 'inner', got {second_pass_type!r}")
        self.second_pass_type = second_pass_type

        # Warn if n_rand_row provided in inner mode
        if second_pass_type == "inner" and n_rand_row is not None:
            warn(f"n_rand_row is ignored in 'inner' mode")
            n_rand_row = None

        # If existing completed file, load metadata
        try:
            with h5py.File(self.h5_filename, "r") as f:
                f_n_rows = f.attrs["n_rows"]
                f_n_inner = f.attrs["n_inner"]
                f_nrand = f.attrs["n_rand_col"]
                f_nrand_row = f.attrs.get("n_rand_row")
                f_dtype = f.attrs["dtype"]
                f_second_pass_type = f.attrs.get("second_pass_type", "row")
            n_rows = f_n_rows
            self.n_inner = f_n_inner
            n_rand_col = f_nrand
            n_rand_row = f_nrand_row
            dtype = f_dtype
            self.second_pass_type = f_second_pass_type
        except:
            if n_rows is None:
                warn("Pass `n_rows` to avoid forcing load mode.")
                raise

        # Dimensions and randomness
        self.n_rows = int(n_rows)
        self.n_inner = None  # prod(inner_shape), inferred from first batch
        self.inner_shape = None  # shape of non-streaming dimensions
        self.n_rand_col = int(n_rand_col)
        self.n_rand_row = int(n_rand_row) if n_rand_row is not None else int(n_rand_col)
        self.seed = int(seed) if seed is not None else None
        self.dtype = np.dtype(dtype)

        # Streaming state
        self._pass = None  # "first" or "second"
        self._stream_dim_second = None  # set in second_pass()
        self._main = None  # h5py.File for results
        self._spaces = None  # h5py.File for intermediates
        self._cursor = 0  # default index for placement
        self._batch_counter = 0
        self.checkpoint_every = int(checkpoint_every)

        # In-memory accumulators
        self._sample_row_mem = None
        self._exact_row_mem = None
        self._Q = None  # row_space, loaded once at start of second pass

        # Pending writes
        self._pending_sample_col = {}
        self._pending_mask_ranges = []

        self._partially_run = os.path.exists(self.h5_filename) or os.path.exists(
            self.spaces_filename
        )

    @contextmanager
    def first_pass(self):
        """
        Randomized projection phase (always streams over dimension 0).

        On enter:
        - open HDF5 files, create sample_col (n_rows x L), mask_first
        - For "row" mode: also create Ω̃ (n_rows x L_row) and sample_row
        On exit:
        - Orthonormalize sample_col → row_space Q
        - For "row" mode: orthonormalize sample_row → col_space Q̃,
          compute intermediate SVD, write U,S,V
        - For "inner" mode: only store Q, no intermediate SVD
        - Set first_pass_complete flag
        """
        do_post = True
        try:
            self._open_files(pass_kind="first")
            self._pass = "first"
            yield self
        except:
            warn("Error in first pass.")
            do_post = False
            raise
        finally:
            if do_post:
                info("First pass succeeded, computing row space.")

                # Flush any remaining buffered data before finalization
                self._flush_pending("first")

                # Load sample_col and compute Q
                info("loading intermediates")
                Y = self._spaces["sample_col"][:]  # n_rows x L

                info("orthogonalizing sample_col")
                Q, _ = np.linalg.qr(Y, mode="reduced")  # n_rows x L_eff

                # Store row_space (Q)
                if "row_space" in self._spaces:
                    del self._spaces["row_space"]
                self._spaces.create_dataset("row_space", data=Q, chunks=True)

                if self.second_pass_type == "row":
                    # Need sample_row for intermediate SVD
                    if "sample_row" not in self._spaces:
                        raise RuntimeError(
                            "First pass finalization requires sample_row; no batches received or inner_shape not inferred."
                        )

                    Ytilde = self._spaces["sample_row"][:]  # n_inner x L_row

                    info("Orthogonalizing sample_row")
                    Qtilde, _ = np.linalg.qr(Ytilde, mode="reduced")

                    # Store col_space (Q̃)
                    if "col_space" in self._spaces:
                        del self._spaces["col_space"]
                    self._spaces.create_dataset("col_space", data=Qtilde, chunks=True)

                    # Ensure Ω exists
                    if "omega" not in self._spaces:
                        raise RuntimeError(
                            "Omega not found; it should have been created upon first batch."
                        )
                    omega = self._spaces["omega"][:]

                    # Small rectangular SVD
                    info("Computing reduced SVD")
                    QtY = Q.T @ Y
                    QtOmega = Qtilde.T @ omega
                    B = QtY @ np.linalg.pinv(QtOmega)
                    U_tilde, s, Vh_tilde = np.linalg.svd(B, full_matrices=False)
                    info("Expanding SVD")
                    U = Q @ U_tilde
                    V = Qtilde @ Vh_tilde.T
                    S = s
                    Vh = V.T.reshape(V.shape[-1], *self.inner_shape)

                    # Write U,S,V
                    for name, arr in [("U", U), ("S", S), ("Vh", Vh)]:
                        if name in self._main:
                            del self._main[name]
                        if name == "S":
                            self._main.create_dataset(name, data=arr)
                        else:
                            self._main.create_dataset(name, data=arr, chunks=True)

                # Metadata
                self._main.attrs["n_rows"] = self.n_rows
                self._main.attrs["n_inner"] = self.n_inner
                self._main.attrs["n_rand_col"] = self.n_rand_col
                if self.second_pass_type == "row":
                    self._main.attrs["n_rand_row"] = self.n_rand_row
                self._main.attrs["second_pass_type"] = self.second_pass_type
                if self.seed is not None:
                    self._main.attrs["seed"] = self.seed
                self._main.attrs["dtype"] = str(self.dtype)

                # Mark first pass complete
                self._spaces.attrs["first_pass_complete"] = True

            # Flush and close
            info("Cleaning up first pass.")
            self._main.flush()
            self._spaces.flush()
            self._main.close()
            self._spaces.close()
            self._main = None
            self._spaces = None
            self._pass = None
            self._sample_row_mem = None
            self._pending_sample_col.clear()
            self._pending_mask_ranges.clear()

    @contextmanager
    def second_pass(self, stream_dim=0):
        """
        Reduction phase.

        Parameters
        ----------
        stream_dim : int
            Dimension to stream over. For "row" mode, must be 0.
            For "inner" mode, must be >= 1.

        On enter:
        - Validate stream_dim against second_pass_type
        - Check first_pass_complete flag
        - Load Q into memory once
        - Create exact_row if missing
        On exit:
        - SVD(exact_row), expand with Q → final U,S,V
        """
        # Validate stream_dim
        if self.second_pass_type == "row" and stream_dim != 0:
            raise ValueError(
                f"second_pass_type='row' requires stream_dim=0, got {stream_dim}"
            )
        if self.second_pass_type == "inner" and stream_dim < 1:
            raise ValueError(
                f"second_pass_type='inner' requires stream_dim>=1, got {stream_dim}"
            )
        self._stream_dim_second = stream_dim

        try:
            self._open_files(pass_kind="second")

            # Verify first-pass completeness
            if not self._spaces.attrs.get("first_pass_complete", False):
                raise RuntimeError(
                    "Cannot enter second_pass: first_pass_complete flag not set."
                )

            if "row_space" not in self._spaces:
                raise RuntimeError(
                    "row_space (Q) not found; run first_pass to compute it."
                )

            # Load Q into memory once
            self._Q = self._spaces["row_space"][:].copy()
            L_eff = self._Q.shape[1]

            # n_inner must be known from first pass
            if self.n_inner is None and "n_inner" in self._spaces.attrs:
                self.n_inner = int(self._spaces.attrs["n_inner"])
            if self.inner_shape is None and "inner_shape" in self._spaces.attrs:
                self.inner_shape = tuple(self._spaces.attrs["inner_shape"])
            if self.n_inner is None:
                raise RuntimeError(
                    "n_inner unknown; complete first pass before second pass."
                )

            # Create exact_row if missing
            if self.second_pass_type == "inner" and self.inner_shape is not None:
                # For inner mode: exact_row has shape (L_eff, *inner_shape)
                exact_row_shape = (L_eff,) + self.inner_shape
            else:
                # For row mode: exact_row has shape (L_eff, n_inner)
                exact_row_shape = (L_eff, self.n_inner)

            if "exact_row" not in self._spaces:
                self._exact_row_mem = np.zeros(exact_row_shape, dtype=self.dtype)
                self._spaces.create_dataset(
                    "exact_row",
                    data=self._exact_row_mem,
                    dtype=self.dtype,
                    chunks=True,
                )
            elif self._exact_row_mem is None:
                # Resuming: load existing data into memory
                self._exact_row_mem = self._spaces["exact_row"][:].copy()

            # Create mask for second pass streaming dimension
            if self.second_pass_type == "inner":
                mask_size = self.inner_shape[stream_dim - 1]
                if "mask_second" not in self._spaces or self._spaces["mask_second"].shape[0] != mask_size:
                    if "mask_second" in self._spaces:
                        del self._spaces["mask_second"]
                    self._spaces.create_dataset(
                        "mask_second", shape=(mask_size,), dtype="uint8"
                    )
                    self._spaces["mask_second"][:] = 0

            self._pass = "second"
            yield self
        finally:
            try:
                # Flush any remaining buffered data before finalization
                self._flush_pending("second")

                # Final SVD on exact_row
                exact_row = self._spaces["exact_row"][:].astype(np.float64)
                # Flatten to 2D for SVD if needed
                exact_row_2d = exact_row.reshape(exact_row.shape[0], -1)

                info(f"Computing reduced SVD {exact_row_2d.shape}")
                U_tilde, s, Vh = np.linalg.svd(exact_row_2d, full_matrices=False)
                info("Expanding SVD")
                U = (self._Q.astype(np.float64) @ U_tilde)
                S = s
                Vh = Vh.reshape(len(s), *self.inner_shape)

                # Write U,S,V
                for name, arr in [("U", U), ("S", S), ("Vh", Vh)]:
                    if name in self._main:
                        del self._main[name]
                    if name == "S":
                        self._main.create_dataset(name, data=arr)
                    else:
                        self._main.create_dataset(name, data=arr, chunks=True)

                # Success cleanup
                self._main.flush()
                self._spaces.flush()
                self._main.close()
                self._spaces.close()
                self._main = None
                self._spaces = None
                self._pass = None
                self._exact_row_mem = None
                self._Q = None
                self._stream_dim_second = None
                self._pending_mask_ranges.clear()
            except:
                self._main.flush()
                self._spaces.flush()
                self._main.close()
                self._spaces.close()
                self._main = None
                self._spaces = None
                self._pass = None
                self._exact_row_mem = None
                self._Q = None
                self._stream_dim_second = None
                self._pending_mask_ranges.clear()
                raise

    def receive_batch(self, batch, index=None):
        """
        Accept a batch.

        First pass:
        - batch shape: (batch_size, *inner_shape)
        - Flatten to (batch_size, n_inner) for projections
        - Place Y_block = batch_flat @ Ω into sample_col
        - For "row" mode: accumulate sample_row += batch_flat.T @ Ω̃_block

        Second pass ("row" mode, stream_dim=0):
        - batch shape: (batch_size, *inner_shape)
        - Accumulate exact_row += Q[start:end].T @ batch_flat

        Second pass ("inner" mode, stream_dim>=1):
        - batch shape: (n_rows, *inner_shape_with_batch_at_stream_dim)
        - Assign exact_row[:, ..., start:end, ...] = Q.T @ batch_reshaped
        """
        if self._pass not in ("first", "second"):
            raise RuntimeError(
                "receive_batch must be called within first_pass/second_pass context."
            )

        batch = np.asarray(batch, dtype=self.dtype)
        spaces = self._spaces

        if self._pass == "first":
            # First pass always streams over dim 0
            if batch.ndim < 2:
                raise ValueError("batch must be at least 2D")
            N = batch.shape[0]
            batch_inner_shape = batch.shape[1:]

            # Infer inner_shape on first batch
            if self.inner_shape is None:
                self.inner_shape = batch_inner_shape
                self.n_inner = int(np.prod(batch_inner_shape))
                spaces.attrs["inner_shape"] = self.inner_shape
                spaces.attrs["n_inner"] = self.n_inner
                self._initialize_inner_datasets(spaces)

            # Flatten batch for matrix ops
            batch_flat = batch.reshape(N, -1)

            # Determine placement index
            start = int(index) if index is not None else int(self._cursor)
            if start < 0 or start + N > self.n_rows:
                raise ValueError(
                    f"index range [{start}:{start+N}] out of bounds for n_rows={self.n_rows}"
                )
            self._cursor = start + N

            # Compute projections
            omega = spaces["omega"][:]
            Y_block = batch_flat @ omega  # N x L

            # Buffer sample_col for later write
            self._pending_sample_col[(start, N)] = Y_block

            if self.second_pass_type == "row":
                omega_tilde = spaces["omega_tilde"][:]
                OmegaT_block = omega_tilde[start : start + N, :]
                contrib_row = batch_flat.T @ OmegaT_block
                self._sample_row_mem = self._sample_row_mem + contrib_row

            # Check overlap and buffer mask update
            self._check_overlap_and_buffer_mask(
                pass_kind="first", start=start, length=N
            )

        elif self._pass == "second":
            if self.second_pass_type == "row":
                # Row mode: same streaming as first pass
                if batch.ndim < 2:
                    raise ValueError("batch must be at least 2D")
                N = batch.shape[0]
                batch_flat = batch.reshape(N, -1)

                start = int(index) if index is not None else int(self._cursor)
                if start < 0 or start + N > self.n_rows:
                    raise ValueError(
                        f"index range [{start}:{start+N}] out of bounds for n_rows={self.n_rows}"
                    )
                self._cursor = start + N

                # Check overlap and buffer mask update
                self._check_overlap_and_buffer_mask(
                    pass_kind="second", start=start, length=N
                )

                # Accumulate into in-memory exact_row
                Q_block = self._Q[start : start + N, :]
                contrib = Q_block.T @ batch_flat
                self._exact_row_mem = self._exact_row_mem + contrib

            else:
                # Inner mode: streaming over inner dimension
                stream_dim = self._stream_dim_second
                # batch shape: (n_rows, ..., batch_size, ...)
                # where batch_size is at axis stream_dim
                if batch.shape[0] != self.n_rows:
                    raise ValueError(
                        f"batch dim 0 must be n_rows={self.n_rows}, got {batch.shape[0]}"
                    )

                batch_size = batch.shape[stream_dim]
                start = int(index) if index is not None else int(self._cursor)

                # Validate against inner dimension size
                inner_dim_size = self.inner_shape[stream_dim - 1]
                if start < 0 or start + batch_size > inner_dim_size:
                    raise ValueError(
                        f"index range [{start}:{start+batch_size}] out of bounds for inner dim {stream_dim-1} size={inner_dim_size}"
                    )
                self._cursor = start + batch_size

                # Check overlap and buffer mask update
                self._check_overlap_and_buffer_mask(
                    pass_kind="second", start=start, length=batch_size
                )

                # Compute Q.T @ batch_reshaped
                # batch: (n_rows, *batch_inner_shape)
                # Q: (n_rows, L_eff)
                # Result: (L_eff, *batch_inner_shape)
                batch_flat = batch.reshape(self.n_rows, -1)
                result_flat = self._Q.T @ batch_flat
                result = result_flat.reshape(self._Q.shape[1], *batch.shape[1:])

                # Assign to exact_row at the appropriate slice
                # exact_row shape: (L_eff, *inner_shape)
                # We need to assign at axis stream_dim (since exact_row[0] is L_eff)
                idx = [slice(None)] * self._exact_row_mem.ndim
                idx[stream_dim] = slice(start, start + batch_size)
                self._exact_row_mem[tuple(idx)] = result

        # Periodic flush
        self._batch_counter += 1
        if self._batch_counter % self.checkpoint_every == 0:
            self._flush_pending(self._pass)

    def _initialize_inner_datasets(self, spaces):
        """Initialize datasets that depend on inner_shape (called on first batch)."""
        info(f"Initializing datasets for inner_shape={self.inner_shape}")

        # Create Ω (n_inner x L) once
        if "omega" not in spaces:
            rng = np.random.default_rng(self.seed)
            info("Sampling probe vectors (omega)")
            omega = rng.standard_normal((self.n_inner, self.n_rand_col), dtype=self.dtype)
            info(f"Creating dataset: omega {omega.shape}.")
            spaces.create_dataset("omega", data=omega, chunks=True)

        if self.second_pass_type == "row":
            # Create sample_row (n_inner x L_row), initialized to zero
            if "sample_row" not in spaces:
                info("Creating dataset: sample_row.")
                self._sample_row_mem = np.zeros(
                    (self.n_inner, self.n_rand_row), dtype=self.dtype
                )
                spaces.create_dataset(
                    "sample_row",
                    data=self._sample_row_mem,
                    dtype=self.dtype,
                    chunks=True,
                )
            elif self._sample_row_mem is None:
                self._sample_row_mem = spaces["sample_row"][:].copy()

    def _open_files(self, pass_kind):
        """
        Open/create HDF5 files, load available metadata & state, create datasets if missing.
        """
        self._main = h5py.File(self.h5_filename, "a")
        self._spaces = h5py.File(self.spaces_filename, "a")

        # Load attrs if present
        if "n_inner" in self._spaces.attrs:
            self.n_inner = int(self._spaces.attrs["n_inner"])
        if "inner_shape" in self._spaces.attrs:
            self.inner_shape = tuple(self._spaces.attrs["inner_shape"])

        # Always write attrs that we know
        self._spaces.attrs["n_rows"] = self.n_rows
        self._spaces.attrs["n_rand_col"] = self.n_rand_col
        self._spaces.attrs["second_pass_type"] = self.second_pass_type
        if self.second_pass_type == "row":
            self._spaces.attrs["n_rand_row"] = self.n_rand_row
        if self.seed is not None:
            self._spaces.attrs["seed"] = self.seed
        self._spaces.attrs["dtype"] = str(self.dtype)

        # Create mask_first (n_rows,) if missing
        if "mask_first" not in self._spaces:
            self._spaces.create_dataset(
                "mask_first", shape=(self.n_rows,), dtype="uint8"
            )
            self._spaces["mask_first"][:] = 0

        # For row mode, create mask_second (n_rows,) if missing
        if self.second_pass_type == "row" and "mask_second" not in self._spaces:
            self._spaces.create_dataset(
                "mask_second", shape=(self.n_rows,), dtype="uint8"
            )
            self._spaces["mask_second"][:] = 0

        if pass_kind == "first":
            # Load sample_row into memory if resuming
            if "sample_row" in self._spaces and self._sample_row_mem is None:
                self._sample_row_mem = self._spaces["sample_row"][:].copy()

            # Create sample_col (n_rows x L) fixed-size
            if "sample_col" not in self._spaces:
                info(f"Creating dataset: sample_col {(self.n_rows, self.n_rand_col)}")
                self._spaces.create_dataset(
                    "sample_col",
                    data=np.zeros((self.n_rows, self.n_rand_col), self.dtype),
                    dtype=self.dtype,
                    chunks=True,
                )

            # Create Ω̃ (n_rows x L_row) for row mode
            if self.second_pass_type == "row" and "omega_tilde" not in self._spaces:
                rng = np.random.default_rng(2*self.seed+1 if self.seed is not None else None)
                info("sampling probe vectors (omega_tilde)")
                omega_tilde = rng.standard_normal(
                    (self.n_rows, self.n_rand_row),
                    dtype=self.dtype
                )
                info(f"Creating dataset: omega_tilde {omega_tilde.shape}")
                self._spaces.create_dataset(
                    "omega_tilde", data=omega_tilde, chunks=True
                )

        # main attrs for self-describing file
        self._main.attrs["n_rows"] = self.n_rows
        if self.n_inner is not None:
            self._main.attrs["n_inner"] = self.n_inner
        self._main.attrs["n_rand_col"] = self.n_rand_col
        if self.second_pass_type == "row":
            self._main.attrs["n_rand_row"] = self.n_rand_row
        self._main.attrs["second_pass_type"] = self.second_pass_type
        if self.seed is not None:
            self._main.attrs["seed"] = self.seed
        self._main.attrs["dtype"] = str(self.dtype)

    def _check_overlap_and_buffer_mask(self, pass_kind, start, length):
        """
        Check for overlap in mask (disk + pending), then buffer the range.
        """
        end = start + length
        name = "mask_first" if pass_kind == "first" else "mask_second"
        mask = self._spaces[name]

        # Check disk
        current = mask[start:end]
        if np.any(current == 1):
            raise RuntimeError(f"Overlap in indices [{start}:{end}] (already on disk)")

        # Check pending ranges
        for pstart, plength in self._pending_mask_ranges:
            pend = pstart + plength
            if not (end <= pstart or start >= pend):
                raise RuntimeError(
                    f"Overlap in indices [{start}:{end}] with pending [{pstart}:{pend}]"
                )

        self._pending_mask_ranges.append((start, length))

    def _flush_pending(self, pass_kind):
        """
        Flush all pending in-memory data to disk.
        """
        spaces = self._spaces
        if spaces is None:
            return

        if pass_kind == "first":
            sample_row_data = (
                self._sample_row_mem.copy()
                if self._sample_row_mem is not None
                else None
            )
            sample_col_items = list(self._pending_sample_col.items())
            mask_ranges = list(self._pending_mask_ranges)

            if sample_row_data is not None and "sample_row" in spaces:
                spaces["sample_row"][:] = sample_row_data
            for (start, length), data in sample_col_items:
                spaces["sample_col"][start : start + length, :] = data
            for start, length in mask_ranges:
                spaces["mask_first"][start : start + length] = 1

            self._pending_sample_col.clear()
            self._pending_mask_ranges.clear()

        elif pass_kind == "second":
            exact_row_data = (
                self._exact_row_mem.copy() if self._exact_row_mem is not None else None
            )
            mask_ranges = list(self._pending_mask_ranges)

            if exact_row_data is not None and "exact_row" in spaces:
                spaces["exact_row"][:] = exact_row_data
            for start, length in mask_ranges:
                spaces["mask_second"][start : start + length] = 1

            self._pending_mask_ranges.clear()

        spaces.flush()
        if self._main is not None:
            self._main.flush()

    def row_components(self, scaled=False):
        with h5py.File(self.h5_filename, "r") as f:
            U = f["U"][:]
            if scaled:
                return U * f["S"][None, :]
            return U

    def col_components(self, scaled=False):
        with h5py.File(self.h5_filename, "r") as f:
            Vh = f["Vh"][:]
            if scaled:
                return Vh * f["S"][:, *((None,) * (Vh.ndim-1))]
            return Vh

    def to_loaded_svd(self):
        import imaging_scripts.io.svd_video as svd_video 
        with h5py.File(self.h5_filename, "r") as f:
            Vt = f["Vh"][:]
            return svd_video.SVDVideo(f["U"][:], f["S"][:], Vt)





