from array_api_compat import array_namespace

def _static_optional_filter(f):
    """
    Convert (array, dim) -> array function to SVDVideo -> SVDVideo
    """
    def wrapped(bound, *a, dim = 't', **kw):
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

    def __getitem__(self, *idx):
        time_idx, spatial_idx = idx[0], idx[1:]
        Usel = self.U[time_idx]
        Vsel = self.Vt[:, *spatial_idx]

        if Usel.ndim == 1:
            Uscaled = Usel * self.S
            reconst = (Vsel.T @ Uscaled[None].T).T[0]
        else:
            Uscaled = Usel * self.S[None, :]
            reconst = (Vsel.T @ Uscaled.T).T[0]

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
        arr: (component, time,) or (component, spatial1, ..., spatialN)
        filter: array (n_kernels..., kernel_sizes..,)
            Shape `n_kernels...` can have at most ndim=1 for temporal filter to
            maintain (component, time,) output shape. For spatial filtering,
            `n_kernels` can be any number of dimensions and the spatial shape
            of the resulting video will be higher dimensional.
        axes: tuple of int
            When `dim` indicated spatial convolution, axes indicates the
            dimensions of the video object expected in the filter array.
            If not provided in that case, `kernel.ndim` is expected to be at least
            `ndim_spatial`. Spatial dimensions are 1-indexed, to account for the
            temporal dimension.
        """
        xp = array_namespace(arr)

        pad_mode = pad_mode or 'constant'
        pad_value = pad_value if pad_value is not None else 0.0

        is_temporal = SVDVideo._row_axis(dim)
        ndim_spatial = arr.ndim - 1

        if is_temporal:
            # arr: (C, T), filter: (K,)
            assert filter.ndim == 1, "Temporal filter must be 1D"

            pad_width = filter.shape[0] // 2

            if _backend(arr, "jax"):
                import jax.lax  # type: ignore

                lhs = arr[:, None, :]  # (C, 1, T)
                rhs = filter[None, None, :]  # (1, 1, K)

                filtered = jax.lax.conv_general_dilated(
                    lhs, rhs,
                    window_strides=(1,),
                    padding=((pad_width, pad_width),),
                    dimension_numbers=('NCH', 'OIH', 'NCH')
                )
                return filtered[:, 0, :]  # (C, T)

            elif _backend(arr, "numpy"):
                from scipy.ndimage import convolve  # type: ignore

                kernel = xp.reshape(filter, (1, -1))  # (1, K)
                return convolve(arr, kernel, mode=pad_mode, cval=pad_value)
            else:
                raise RuntimeError("Unsupported backend for `convolve`.")

        else:
            # Spatial convolution
            # arr: (C, S1, ..., SN), filter: (n_k..., k1, ..., kM)

            if axes is None:
                axes = tuple(range(1, arr.ndim))

            n_conv_axes = len(axes)
            n_kernel_dims = filter.ndim - n_conv_axes
            n_kernels_shape = filter.shape[:n_kernel_dims] if n_kernel_dims > 0 else ()
            kernel_sizes = filter.shape[n_kernel_dims:]

            n_k_total = 1
            for s in n_kernels_shape:
                n_k_total *= s

            filter_flat = xp.reshape(filter, (n_k_total,) + kernel_sizes) if n_kernel_dims > 0 else filter[None]

            # Build full kernel shape (1s for non-convolved dims)
            full_kernel_shape = []
            kernel_idx = 0
            for i in range(1, arr.ndim):
                if i in axes:
                    full_kernel_shape.append(kernel_sizes[kernel_idx])
                    kernel_idx += 1
                else:
                    full_kernel_shape.append(1)

            if _backend(arr, "jax"):
                import jax.lax  # type: ignore

                filter_full = xp.reshape(filter_flat, (n_k_total,) + tuple(full_kernel_shape))

                lhs = arr[:, None, ...]  # (C, 1, S1, ..., SN)
                rhs = filter_full[:, None, ...]  # (n_k_total, 1, k1, ..., kN)

                padding = []
                kernel_idx = 0
                for i in range(1, arr.ndim):
                    if i in axes:
                        pad_w = kernel_sizes[kernel_idx] // 2
                        padding.append((pad_w, pad_w))
                        kernel_idx += 1
                    else:
                        padding.append((0, 0))

                spatial_chars = 'HWDXYZ'[:ndim_spatial]
                dim_spec = ('NC' + spatial_chars, 'OI' + spatial_chars, 'NC' + spatial_chars)

                filtered = jax.lax.conv_general_dilated(
                    lhs, rhs,
                    window_strides=(1,) * ndim_spatial,
                    padding=tuple(padding),
                    dimension_numbers=dim_spec
                )
                # (C, n_k_total, S1, ..., SN) -> (C, n_k1, ..., S1, ..., SN)
                out_shape = (arr.shape[0],) + n_kernels_shape + filtered.shape[2:]
                return xp.reshape(filtered, out_shape)

            elif _backend(arr, "numpy"):
                from scipy.ndimage import convolve  # type: ignore
                import numpy as np  # type: ignore

                full_kernel_shape_with_c = [1] + full_kernel_shape

                results = []
                for k in range(n_k_total):
                    kernel = filter_flat[k].reshape(full_kernel_shape_with_c)
                    res = convolve(arr, kernel, mode=pad_mode, cval=pad_value)
                    results.append(res)

                stacked = np.stack(results, axis=1)  # (C, n_k_total, S1, ..., SN)
                out_shape = (arr.shape[0],) + n_kernels_shape + stacked.shape[2:]
                return stacked.reshape(out_shape)
            else:
                raise RuntimeError("Unsupported backend for `convolve`.")
        
    
    def add(self, temporal, spatial, amplitude=None, axes=None) -> 'SVDVideo':
        """
        USV' + temporal @ spatial
        temporal: array (time, rank)
        spatial: array (rank, n_filters..., spatial0, spatial1, ...)
        amplitude: array (rank,)
        axes:
            Dimensions in `temporal` and `spatial` assumed to match the shape of
            the video object.
            If not provided, `spatial.ndim` is assumed to to be at least the
            number of spatial dimensions plus one. If spatial.ndim is greater
            than as indicated by `axes`, dimensions will be prepended to the
            spatial dimensions of the video and the video will be broadcast.

        Rank is added to the SVD by extending U and V with the spatial and
        temporal arrays and S with ones or amplitude.
        """
        xp = array_namespace(self.U)

        new_rank = temporal.shape[1]

        if amplitude is None:
            amplitude = xp.ones(new_rank, dtype=self.S.dtype)
        amplitude = xp.asarray(amplitude)

        # Determine n_filters shape from spatial
        # spatial: (new_rank, n_f1, ..., n_fK, S1, ..., SN)
        if axes is None:
            n_filter_dims = spatial.ndim - 1 - self.ndim_spatial
        else:
            n_filter_dims = spatial.ndim - 1 - len(axes)

        n_filters_shape = spatial.shape[1:1+n_filter_dims] if n_filter_dims > 0 else ()

        # Broadcast existing Vt to include n_filters dimensions
        # Vt: (rank, S1, ..., SN) -> (rank, 1, ..., 1, S1, ..., SN) -> (rank, n_f1, ..., S1, ...)
        if n_filters_shape:
            Vt_expanded = xp.reshape(
                self.Vt,
                (self.rank,) + (1,) * len(n_filters_shape) + self.Vt.shape[1:]
            )
            broadcast_shape = (self.rank,) + n_filters_shape + self.Vt.shape[1:]
            Vt_broadcast = xp.broadcast_to(Vt_expanded, broadcast_shape)
            # Copy to avoid issues with views during concatenation
            Vt_broadcast = xp.asarray(Vt_broadcast, copy=True) if hasattr(xp, 'asarray') else Vt_broadcast.copy()
        else:
            Vt_broadcast = self.Vt

        # Concatenate
        U_new = xp.concat([self.U, temporal], axis=1)
        S_new = xp.concat([self.S, amplitude])
        Vt_new = xp.concat([Vt_broadcast, spatial], axis=0)

        return SVDVideo(U_new, S_new, Vt_new, orthonormal=False)


        
        

    @staticmethod
    def orthogonal(S, F, Gt,):
        """
        Compute orthogonal bases after applying linear filters

        For filters X, Y', form decomposed X A Y' from A = U S Vt
        and from the filtered components F = XU, G' = V'Y'
        """
        if _backend(S, 'numpy'):
            from scipy import linalg #type: ignore
            import numpy as np #type: ignore
        elif _backend(S, 'jax'):
            from jax.scipy import linalg #type: ignore
            import jax.numpy as np #type: ignore
            
        # 2. Gramian Eigendecomposition (O(Mr^2 + r^3))
        sf2, Qf = linalg.eigh(F.T @ F)
        sg2, Qg = linalg.eigh(Gt @ Gt.T)
        
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
        Vt_filt = (Vt_hat @ Qg.T / sg[:, None]) @ Gt.T

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
