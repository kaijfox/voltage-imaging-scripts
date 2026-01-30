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
        Convolve array along specified axes.

        Parameters
        ----------
        arr : array (component, dim1, ..., dimN)
        filter : array (n_kernels..., kernel_size1, ..., kernel_sizeM)
            Last M dimensions are kernel sizes for M convolution axes.
        dim : str
            "t"/"time"/"row" for temporal (equiv. to axes=(1,)), else spatial.
        axes : tuple of int
            Axes to convolve over (1-indexed). Default: all spatial axes.
        """
        xp = array_namespace(arr)
        pad_mode = pad_mode or 'constant'
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

        # Parse filter: (n_kernels..., kernel_sizes...)
        n_extra = filter.ndim - n_conv
        n_kernels_shape = filter.shape[:n_extra] if n_extra > 0 else ()
        kernel_sizes = filter.shape[n_extra:]
        n_k = 1
        for s in n_kernels_shape:
            n_k *= s

        # Flatten kernel batch: (n_k, k1, ..., kM)
        filters = xp.reshape(filter, (n_k,) + kernel_sizes) if n_extra > 0 else filter[None]

        # Full kernel shape with 1s for non-convolved axes
        axes_set = set(axes)
        full_kernel = tuple(
            kernel_sizes[sum(1 for a in axes if a < i + 1)] if (i + 1) in axes_set else 1
            for i in range(n_spatial)
        )

        # Padding per spatial dim
        padding = tuple(
            (kernel_sizes[sum(1 for a in axes if a < i + 1)] // 2,) * 2
            if (i + 1) in axes_set else (0, 0)
            for i in range(n_spatial)
        )

        if _backend(arr, "jax"):
            import jax.lax  # type: ignore

            lhs = arr[:, None, ...]  # (C, 1, S...)
            rhs = xp.reshape(filters, (n_k, 1) + full_kernel)

            # Dimension spec with digits for unlimited spatial dims
            dims = ''.join(str(i) for i in range(n_spatial))
            spec = (f'NC{dims}', f'OI{dims}', f'NC{dims}')

            out = jax.lax.conv_general_dilated(
                lhs, rhs,
                window_strides=(1,) * n_spatial,
                padding=padding,
                dimension_numbers=spec
            )
            return xp.reshape(out, (arr.shape[0],) + n_kernels_shape + out.shape[2:])

        elif _backend(arr, "numpy"):
            from scipy.ndimage import convolve  # type: ignore
            import numpy as np  # type: ignore

            conv_axes = tuple(range(1, arr.ndim))
            out = np.stack([
                convolve(arr, f.reshape(full_kernel), mode=pad_mode, cval=pad_value, axes=conv_axes)
                for f in filters
            ], axis=1)
            return out.reshape((arr.shape[0],) + n_kernels_shape + out.shape[2:])

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
