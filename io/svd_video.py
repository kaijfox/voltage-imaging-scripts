from array_api_compat import array_namespace
from pathlib import Path
import os

def _parse_batch(*operators, n_core_dims):
    """
    Extract batch shape from operator(s), validate consistency, provide output reshaper.

    Parameters
    ----------
    *operators : arrays
        One or more arrays with shape (batch..., core...)
    n_core_dims : int or tuple of int
        Number of trailing "core" dimensions per operator.
        If int, same value used for all operators.

    Returns
    -------
    batch_shape : tuple
        Unified batch shape (broadcast across all operators)
    cores : list of arrays
        Operators reshaped to (flat_batch, *core_shape) for processing
    reshape_out : callable
        reshape_out(arr, insert_at=1) reshapes (dim0, flat_batch, rest...)
        to (dim0, *batch_shape, rest...)
    """
    xp = array_namespace(operators[0])

    if isinstance(n_core_dims, int):
        n_core_dims = (n_core_dims,) * len(operators)

    # Extract batch shapes
    batch_shapes = []
    for op, n_core in zip(operators, n_core_dims):
        if n_core == 0:
            batch_shapes.append(op.shape)
        else:
            batch_shapes.append(op.shape[:-n_core])

    # Compute broadcast batch shape
    batch_shape = ()
    for bs in batch_shapes:
        # Manual broadcast shape computation
        max_len = max(len(batch_shape), len(bs))
        padded_a = (1,) * (max_len - len(batch_shape)) + batch_shape
        padded_b = (1,) * (max_len - len(bs)) + bs
        batch_shape = tuple(
            max(a, b) if a == 1 or b == 1 or a == b else -1  # -1 signals error
            for a, b in zip(padded_a, padded_b)
        )
        if -1 in batch_shape:
            raise ValueError(f"Incompatible batch shapes: {batch_shapes}")

    flat_batch = 1
    for s in batch_shape:
        flat_batch *= s

    # Reshape operators to (flat_batch, *core_shape)
    cores = []
    for op, n_core in zip(operators, n_core_dims):
        if n_core == 0:
            core_shape = ()
        else:
            core_shape = op.shape[-n_core:]
        # Broadcast to full batch shape, then flatten batch dims
        target_shape = batch_shape + core_shape
        op_broadcast = xp.broadcast_to(
            xp.reshape(op, (1,) * (len(batch_shape) - (op.ndim - n_core)) + op.shape),
            target_shape,
        )
        cores.append(xp.reshape(op_broadcast, (flat_batch,) + core_shape))

    def reshape_out(arr, insert_at=1):
        """Reshape (dim0, flat_batch, rest...) to (dim0, *batch_shape, rest...)"""
        pre = arr.shape[:insert_at]
        post = arr.shape[insert_at + 1 :]
        return xp.reshape(arr, pre + batch_shape + post)

    return batch_shape, cores, reshape_out


def _static_optional_filter(f):
    """
    Convert (array, dim) -> array function to SVDVideo -> SVDVideo
    """

    def wrapped(bound, *a, dim="t", **kw):
        # Called on an instance
        if isinstance(bound, SVDVideo):
            # Select U or V, or  as target
            # In either case, shape component is first dim
            #   (component, time/rows) or (component, spatial1, spatial2, ...)
            if SVDVideo._row_axis(dim):
                target = bound.U.T
            else:
                target = bound.Vt

            # Call the actual filter
            filtered = f(target, *a, dim=dim, **kw)

            # Return as a new SVD video
            if SVDVideo._row_axis(dim):
                return SVDVideo(filtered.T, bound.S, bound.Vt)
            else:
                return SVDVideo(bound.U, bound.S, filtered)

        # Called on an array
        else:
            return f(bound, *a, dim=dim, **kw)

    # Format decorator output properly
    wrapped.__name__ = f.__name__
    wrapped.__doc__ = f.__doc__
    return wrapped


def _backend(arr, check_eq=None):
    if check_eq:
        return check_eq in str(type(arr).__module__)
    return str(type(arr).__module__)


class SVDVideo:

    def __init__(self, U, S, Vt, orthonormal=False):
        """
        SVD representation of a video: U @ diag(S) @ Vt.

        Parameters
        ----------
        U : array (time, rank)
            Temporal basis vectors.
        S : array (rank,)
            Singular values.
        Vt : array (rank, spatial...)
            Spatial basis vectors (can have arbitrary spatial dimensions).
        orthonormal : bool
            If True, U and Vt are orthonormal bases (U.T @ U = I, Vt @ Vt.T = I).
            This is a metadata flag for downstream operations that may require
            or benefit from orthonormality (e.g., avoiding reorthogonalization).
        """
        xp = array_namespace(U)
        self.U = U
        self.S = xp.asarray(S)
        self.Vt = xp.asarray(Vt)
        self.orthonormal = orthonormal

        if self.U.ndim != 2:
            raise ValueError("U must be 2D (time x rank)")
        if self.S.ndim != 1:
            raise ValueError("S must be 1D (rank,)")
        if self.Vt.shape[0] != self.S.shape[0] or self.U.shape[1] != self.S.shape[0]:
            raise ValueError("Rank mismatch among U, S, Vt")

        self.rank = self.S.shape[0]
        self.ndim_spatial = self.Vt.ndim - 1

    @classmethod
    def load(cls, path: os.PathLike, backend=None):
        """
        Load SVDVideo from HDF5 file.

        Parameters
        ----------
        path : path-like
            Path to HDF5 file containing U, S, Vh datasets.
        backend : str, optional
            Array backend for loaded data. Options: None (numpy), "jax".

        Returns
        -------
        SVDVideo
        """
        import h5py

        path = str(Path(path))
        with h5py.File(path, 'r') as f:
            U = f['U'][:]
            S = f['S'][:]
            Vh = f['Vh'][:]
            orthonormal = f.attrs.get("orthonormal", False)

        if backend == "jax":
            import jax.numpy as jnp #type: ignore
            U, S, Vh = jnp.asarray(U), jnp.asarray(S), jnp.asarray(Vh)

        return cls(U, S, Vh, orthonormal=orthonormal)
        
    def save(self, path: os.PathLike):
        """
        Save SVDVideo to HDF5 file.

        Parameters
        ----------
        path : path-like
            Output path. Creates file with U, S, Vh datasets and metadata
            attributes (orthonormal, n_rows, n_inner, dtype).
        """
        import h5py
        import numpy as np

        path = str(Path(path))

        # Convert to numpy if needed (e.g., from JAX)
        U = np.asarray(self.U)
        S = np.asarray(self.S)
        Vh = np.asarray(self.Vt)

        with h5py.File(path, 'w') as f:
            f.create_dataset('U', data=U)
            f.create_dataset('S', data=S)
            f.create_dataset('Vh', data=Vh)
            f.attrs['orthonormal'] = self.orthonormal
            f.attrs['n_rows'] = U.shape[0]
            f.attrs['n_inner'] = int(np.prod(Vh.shape[1:]))
            f.attrs['dtype'] = str(U.dtype)

    def __getitem__(self, idx):
        time_idx, spatial_idx = idx[0], idx[1:]
        print(time_idx, spatial_idx)
        Usel = self.U[time_idx]
        Vsel = self.Vt[:, *spatial_idx]

        print(f"Usel {Usel.shape} Vsel {Vsel.shape}")
        if Usel.ndim == 1:
            Uscaled = Usel * self.S
            reconst = (Vsel.T @ Uscaled[None].T).T[0]
        else:
            Uscaled = Usel * self.S[None, :]
            print(f"Uscaled {Uscaled.shape}")
            reconst = (Vsel.T @ Uscaled.T).T

        return reconst

    def backend(self, check_eq=None):
        return _backend(self.U, check_eq)

    @staticmethod
    def _row_axis(dim):
        return dim in ["t", "time", "row"]

    @_static_optional_filter
    def convolve(
        arr, filter, dim="t", axes=None, pad_mode=None, pad_value=None
    ) -> "SVDVideo":
        """
        Convolve array along specified axes.

        Parameters
        ----------
        arr : array (component, dim1, ..., dimN)
        filter : array (batch..., kernel_size1, ..., kernel_sizeM)
            Trailing `len(axes)` dimensions are kernel sizes; leading dimensions
            are batch dimensions that will appear in output.
        dim : str
            "t"/"time"/"row" for temporal (equiv. to axes=(1,)), else spatial.
        axes : tuple of int
            Axes to convolve over (1-indexed into arr). Default: all axes after component.
        pad_mode : str
            Padding mode. Default: 'constant'.
            For numpy backend, supported values are as for
            `scipy.ndimage.convolve`, `reflect`, `constant`, `nearest`,
            `mirror`, `wrap`. For jax backend, only constant zero paddding is used.
        pad_value : float
            Value for constant padding. Default: 0.0.

        Returns
        -------
        array (component, batch..., dim1, ..., dimN)
            Convolved array with batch dimensions inserted after component.
        """
        xp = array_namespace(arr)
        pad_mode = pad_mode or "constant"
        pad_value = pad_value if pad_value is not None else 0.0

        # Temporal is spatial with axes=(1,)
        if SVDVideo._row_axis(dim):
            if filter.ndim != 1:
                raise ValueError("Temporal filter must be 1D")
            axes = (1,)
        elif axes is None:
            axes = tuple(range(1, arr.ndim))
        axes = tuple(axes)

        n_spatial = arr.ndim - 1
        n_conv = len(axes)

        # Use _parse_batch to handle batch dimensions
        batch_shape, [filters], reshape_out = _parse_batch(filter, n_core_dims=n_conv)
        kernel_sizes = filter.shape[-n_conv:] if n_conv > 0 else ()
        n_k = filters.shape[0]

        # Full kernel shape with 1s for non-convolved axes
        axes_set = set(axes)
        full_kernel = tuple(
            (
                kernel_sizes[sum(1 for a in axes if a < i + 1)]
                if (i + 1) in axes_set
                else 1
            )
            for i in range(n_spatial)
        )

        # Padding per spatial dim
        padding = tuple(
            (
                (kernel_sizes[sum(1 for a in axes if a < i + 1)] // 2,) * 2
                if (i + 1) in axes_set
                else (0, 0) 
            )
            for i in range(n_spatial)
        )

        if _backend(arr, "jax"):
            import jax.lax  # type: ignore

            lhs = arr[:, None, ...]  # (C, 1, S...)
            rhs = xp.reshape(filters, (n_k, 1) + full_kernel)

            # Dimension spec with digits for unlimited spatial dims
            dims = "".join(str(i) for i in range(n_spatial))
            spec = (f"NC{dims}", f"OI{dims}", f"NC{dims}")

            out = jax.lax.conv_general_dilated(
                lhs,
                rhs,
                window_strides=(1,) * n_spatial,
                padding=padding,
                dimension_numbers=spec,
            )
            return reshape_out(out)

        elif _backend(arr, "numpy"):
            from scipy.ndimage import convolve  # type: ignore
            import numpy as np  # type: ignore

            conv_axes = tuple(range(1, arr.ndim))
            out = np.stack(
                [
                    convolve(
                        arr,
                        f.reshape(full_kernel),
                        mode=pad_mode,
                        cval=pad_value,
                        axes=conv_axes,
                    )
                    for f in filters
                ],
                axis=1,
            )
            return reshape_out(out)

        else:
            raise RuntimeError("Unsupported backend for `convolve`.")

    @_static_optional_filter
    def gain(arr, weights, dim="t", axes=None) -> "SVDVideo":
        """
        Multiply array by weights along specified axes (element-wise gain).

        Parameters
        ----------
        arr : array (component, dim1, ..., dimN)
        weights : array (batch..., size1, ..., sizeM)
            Trailing `len(axes)` dimensions must match the size of the
            corresponding axes in arr. Leading dimensions are batch.
        dim : str
            "t"/"time"/"row" for temporal (equiv. to axes=(1,)), else spatial.
        axes : tuple of int
            Axes to apply gain (1-indexed into arr). Default: all axes after component.

        Returns
        -------
        array (component, batch..., dim1, ..., dimN)
            Array with gain applied and batch dimensions inserted after component.
        """
        xp = array_namespace(arr)

        # Temporal is spatial with axes=(1,)
        if SVDVideo._row_axis(dim):
            axes = (1,)
        elif axes is None:
            axes = tuple(range(1, arr.ndim))
        axes = tuple(axes)

        n_axes = len(axes)

        # Use _parse_batch to handle batch dimensions
        batch_shape, [weights_flat], reshape_out = _parse_batch(
            weights, n_core_dims=n_axes
        )
        # weights_flat: (flat_batch, size1, ..., sizeM)

        n_spatial = arr.ndim - 1
        n_batch = len(batch_shape)
        flat_batch = weights_flat.shape[0]

        # Build broadcast shape for weights: (flat_batch, dim1, ..., dimN)
        # where dims in axes get their actual size, others get 1
        axes_set = set(axes)
        broadcast_shape = [flat_batch]
        axis_idx = 0
        for i in range(n_spatial):
            if (i + 1) in axes_set:
                broadcast_shape.append(weights_flat.shape[1 + axis_idx])
                axis_idx += 1
            else:
                broadcast_shape.append(1)

        weights_broadcast = xp.reshape(weights_flat, tuple(broadcast_shape))

        # arr: (component, dim1, ..., dimN) -> (component, 1, dim1, ..., dimN)
        arr_expanded = xp.expand_dims(arr, axis=1)

        # Multiply and reshape output
        out = (
            arr_expanded * weights_broadcast
        )  # (component, flat_batch, dim1, ..., dimN)
        return reshape_out(out)

    @_static_optional_filter
    def convolve_separable(
        arr, filters, dim="t", axes=None, pad_mode=None, pad_value=None
    ) -> "SVDVideo":
        """
        Apply separable convolution (1D filters sequentially along each axis).

        Parameters
        ----------
        arr : array (component, dim1, ..., dimN)
        filters : list of arrays [(batch..., k1), (batch..., k2), ...]
            One 1D filter per axis in `axes`. Trailing dim is kernel size;
            leading dims are batch (must broadcast across all filters).
        dim : str
            "t"/"time"/"row" for temporal (equiv. to axes=(1,)), else spatial.
        axes : tuple of int
            Axes to convolve over (1-indexed). Default: all axes after component.
        pad_mode : str
            Padding mode. Default: 'constant'.
        pad_value : float
            Value for constant padding. Default: 0.0.

        Returns
        -------
        array (component, batch..., dim1, ..., dimN)
            Convolved array with batch dimensions inserted after component.
        """
        xp = array_namespace(arr)
        pad_mode = pad_mode or "constant"
        pad_value = pad_value if pad_value is not None else 0.0

        # Temporal is spatial with axes=(1,)
        if SVDVideo._row_axis(dim):
            axes = (1,)
        elif axes is None:
            axes = tuple(range(1, arr.ndim))
        axes = tuple(axes)

        if len(filters) != len(axes):
            raise ValueError(f"Expected {len(axes)} filters, got {len(filters)}")

        # Parse batch shape from all filters (must broadcast)
        batch_shape, filters_flat, reshape_out = _parse_batch(*filters, n_core_dims=1)
        flat_batch = filters_flat[0].shape[0] if batch_shape else 1

        n_spatial = arr.ndim - 1

        # Expand arr for batch dimension: (component, dim...) -> (component, 1, dim...)
        result = xp.broadcast_to(
            xp.expand_dims(arr, axis=1), (arr.shape[0], flat_batch) + arr.shape[1:]
        )
        # Copy to make concrete for in-place-style updates
        if hasattr(xp, "asarray"):
            result = xp.asarray(result, copy=True)
        else:
            result = result.copy()

        # Apply 1D convolution along each axis sequentially
        for ax, filt_flat in zip(axes, filters_flat):
            # filt_flat: (flat_batch, kernel_size)
            kernel_size = filt_flat.shape[1]
            pad_size = kernel_size // 2

            # Build padding spec: ((0,0), (0,0), ..., (pad, pad), ..., (0,0))
            # result shape: (component, flat_batch, dim1, ..., dimN)
            pad_width = [(0, 0)] * result.ndim
            pad_width[ax + 1] = (pad_size, pad_size)  # +1 for batch dim

            if _backend(arr, "jax"):
                import jax.numpy as jnp  # type: ignore
                import jax.lax  # type: ignore

                result_padded = jnp.pad(
                    result, pad_width, mode=pad_mode, constant_values=pad_value
                )

                # Build 1D conv kernel shape: (flat_batch, 1, k) for conv per batch
                # Actually simpler: loop over batch and use 1D conv
                # Or use conv_general_dilated with appropriate dimension spec

                # Reshape for grouped conv: treat (component * flat_batch) as channels
                C = result.shape[0]
                B = flat_batch
                spatial_shape = result.shape[2:]

                # Move axis to convolve to last, do 1D conv, move back
                # Axis in result is ax+1 (0=component, 1=batch, 2+=spatial)
                # In spatial_shape, axis is ax-1 (since spatial starts at index 2 in result)

                # Simpler approach: manual sliding window via vmap or explicit loop
                # For efficiency, use lax.conv_general_dilated on reshaped data

                # Reshape: (C, B, S1, ..., Sn) -> (C*B, 1, S1, ..., Sn)
                result_for_conv = xp.reshape(
                    result_padded, (C * B, 1) + result_padded.shape[2:]
                )

                # Kernel: (B, 1, 1, ..., k, ..., 1) where k is at position ax
                kernel_shape = [B, 1] + [1] * n_spatial
                kernel_shape[ax + 1] = (
                    kernel_size  # ax is 1-indexed, +1 for output channel dim
                )
                kernel = xp.reshape(filt_flat, tuple(kernel_shape))

                # Repeat kernel for each component: (C*B, 1, ...) needs (C*B, 1, ...) kernel
                # Use groups=C*B with kernel (C*B, 1, ...)
                kernel_grouped = xp.tile(kernel, (C, 1) + (1,) * n_spatial)

                dims = "".join(str(i) for i in range(n_spatial))
                spec = (f"NC{dims}", f"OI{dims}", f"NC{dims}")

                padding_spec = [(0, 0)] * n_spatial
                # Already padded, so use 'VALID' equivalent
                conv_result = jax.lax.conv_general_dilated(
                    result_for_conv,
                    kernel_grouped,
                    window_strides=(1,) * n_spatial,
                    padding=padding_spec,
                    dimension_numbers=spec,
                    feature_group_count=C * B,
                )

                result = xp.reshape(conv_result, (C, B) + conv_result.shape[2:])

            elif _backend(arr, "numpy"):
                import numpy as np  # type: ignore
                from scipy.ndimage import convolve1d  # type: ignore

                result_padded = np.pad(
                    result, pad_width, mode=pad_mode, constant_values=pad_value
                )

                # Apply 1D conv along axis for each batch element
                out_slices = []
                for b in range(flat_batch):
                    kernel_1d = filt_flat[b]  # (kernel_size,)
                    slice_b = result_padded[:, b]  # (component, dim1, ..., dimN)
                    # convolve1d along axis ax (in the slice, which has no batch dim)
                    convolved = convolve1d(
                        slice_b, kernel_1d, axis=ax, mode="constant", cval=0.0
                    )
                    # Trim padding
                    slices = [slice(None)] * convolved.ndim
                    slices[ax] = slice(pad_size, -pad_size if pad_size > 0 else None)
                    out_slices.append(convolved[tuple(slices)])

                result = np.stack(out_slices, axis=1)

            else:
                raise RuntimeError("Unsupported backend for `convolve_separable`.")

        return reshape_out(result)

    def add(self, temporal, spatial, amplitude=None) -> "SVDVideo":
        """
        Add components to the SVD: USV' + temporal @ spatial.

        Parameters
        ----------
        temporal : array (batch..., time, new_rank)
            Temporal basis for new components. Trailing 2 dims are core;
            leading dims are batch dimensions.
        spatial : array (batch..., new_rank, spatial0, spatial1, ...)
            Spatial basis for new components. Trailing (1 + ndim_spatial) dims
            are core; leading dims are batch dimensions (must broadcast with temporal).
        amplitude : array (new_rank,), optional
            Singular values for new components. Default: ones.

        Returns
        -------
        SVDVideo
            New video with shape (time, rank + new_rank) for U,
            (rank + new_rank, batch..., spatial...) for Vt.

        Notes
        -----
        Batch dimensions from temporal/spatial are inserted after the rank
        dimension in the output Vt, and the existing Vt is broadcast to match.
        """
        xp = array_namespace(self.U)

        # Core dims: temporal has 2 (time, rank), spatial has 1 + ndim_spatial
        n_core_temporal = 2
        n_core_spatial = 1 + self.ndim_spatial

        batch_shape, [temp_flat, spat_flat], _ = _parse_batch(
            temporal, spatial, n_core_dims=(n_core_temporal, n_core_spatial)
        )

        # temp_flat: (flat_batch, time, new_rank)
        # spat_flat: (flat_batch, new_rank, spatial...)
        new_rank = temp_flat.shape[-1]

        if amplitude is None:
            amplitude = xp.ones(new_rank, dtype=self.S.dtype)
        amplitude = xp.asarray(amplitude)

        # For U: we take temporal from first batch element (all should have same time dim)
        # Actually, temporal should be the same across batch for U concatenation
        # We'll use the mean or just the first - semantically batch means multiple spatial outputs
        # For now, require temporal to not vary across batch (just broadcast)
        temporal_core = temp_flat[0]  # (time, new_rank)

        # Reshape spat_flat: (flat_batch, new_rank, spatial...) -> (new_rank, flat_batch, spatial...)
        # Then reshape to (new_rank, *batch_shape, spatial...)
        spat_transposed = xp.permute_dims(
            spat_flat, (1, 0) + tuple(range(2, spat_flat.ndim))
        )
        spatial_new = xp.reshape(
            spat_transposed, (new_rank,) + batch_shape + spat_flat.shape[2:]
        )

        # Broadcast existing Vt to include batch dimensions
        # Vt: (rank, S1, ..., SN) -> (rank, 1, ..., 1, S1, ..., SN) -> (rank, batch..., S1, ...)
        if batch_shape:
            Vt_expanded = xp.reshape(
                self.Vt, (self.rank,) + (1,) * len(batch_shape) + self.Vt.shape[1:]
            )
            broadcast_target = (self.rank,) + batch_shape + self.Vt.shape[1:]
            Vt_broadcast = xp.broadcast_to(Vt_expanded, broadcast_target)
            # Copy to avoid issues with views during concatenation
            Vt_broadcast = (
                xp.asarray(Vt_broadcast, copy=True)
                if hasattr(xp, "asarray")
                else Vt_broadcast.copy()
            )
        else:
            Vt_broadcast = self.Vt

        # Concatenate
        U_new = xp.concat([self.U, temporal_core], axis=1)
        S_new = xp.concat([self.S, amplitude])
        Vt_new = xp.concat([Vt_broadcast, spatial_new], axis=0)

        return SVDVideo(U_new, S_new, Vt_new, orthonormal=False)

    @staticmethod
    def orthogonal(
        S,
        F,
        Gt,
    ):
        """
        Reorthogonalize SVD after applying linear filters.

        Given original SVD A = U @ diag(S) @ Vt and linear filters X (temporal)
        and Y' (spatial), computes orthonormal SVD of X @ A @ Y' from the
        pre-filtered components F = X @ U and Gt = Vt @ Y'.

        Uses Gramian eigendecomposition for efficiency: O(r^3) where r is rank,
        rather than O(T*M) for full recomputation.

        Parameters
        ----------
        S : array (rank,)
            Original singular values.
        F : array (time, rank)
            Filtered temporal basis (X @ U).
        Gt : array (rank, spatial...)
            Filtered spatial basis (Vt @ Y', can have arbitrary spatial dims).

        Returns
        -------
        U_new : array (time, rank)
            Orthonormal temporal basis.
        S_new : array (rank,)
            New singular values.
        Vt_new : array (rank, spatial_flat)
            Orthonormal spatial basis (flattened spatial dimensions).
        """
        if _backend(S, "numpy"):
            from scipy import linalg  # type: ignore
            import numpy as np  # type: ignore
        elif _backend(S, "jax"):
            from jax.scipy import linalg  # type: ignore
            import jax.numpy as np  # type: ignore

        # 2. Gramian Eigendecomposition (O(Mr^2 + r^3))
        sf2, Qf = linalg.eigh(F.T @ F)
        Gt_flat = Gt.reshape(Gt.shape[0], -1)
        sg2, Qg = linalg.eigh(Gt_flat @ Gt_flat.T)

        sf = np.sqrt(np.maximum(sf2, 0))
        sg = np.sqrt(np.maximum(sg2, 0))

        # 3. Construct Core Matrix B and its SVD (O(r^3))
        # B = Sf @ Qf' @ S @ Qg @ Sg
        B = (sf[:, None] * Qf.T) @ np.diag(S) @ (Qg * sg)
        U_hat, S_filt, Vt_hat = linalg.svd(B)

        # 4. Assembly (O((T+M)r^2))
        # U_filt = F @ (Qf @ Sf^-1) @ U_hat
        U_filt = (F @ Qf / sf) @ U_hat
        # V_filt' = Vt_hat @ (Sg^-1 @ Qg') @ G
        Vt_filt = (Vt_hat @ Qg.T / sg[:, None]) @ Gt_flat

        # Unflatten and return
        Vt_filt = Vt_filt.reshape(Vt_filt.shape[0], *Gt.shape[1:])
        return U_filt, S_filt, Vt_filt


# Make compatible with jitting
try:
    import jax.tree_util  # type: ignore

    jax.tree_util.register_pytree_node(
        SVDVideo,
        lambda v: ((v.U, v.S, v.Vt), {"orthonormal": v.orthonormal}),
        lambda meta, children: SVDVideo(*children, orthonormal=meta["orthonormal"]),
    )
except:
    pass
