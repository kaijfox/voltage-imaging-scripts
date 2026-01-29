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
        axes: tuple of int
            When `dim` indicated spatial convolution, axes indicates the
            number of dimensions expected in the filter array. If not provided
            in that case, `kernel.ndim` is expected to be at least
            `ndim_spatial`. Spatial dimensions are 1-indexed, to account for the
            temporal dimension.
        """

        # Add batch dimensions to spatial/temporal target
        # ...

        # Add extra dimensions to the filter to match spatial dimensions
        # and component dimension
        # ...

        if _backend(arr, "jax"):
            import jax.lax #type: ignore

            # convert `mode` and `pad_value` to compatible arguments
            filtered = jax.lax.conv_general_dilated(
                # ...
            )
        elif _backend(arr, "numpy"):
            from scipy.ndimage import convolve #type: ignore

            # convert `mode` to compatible argument
            filtered = convolve(
                # ...
            )
        else:
            raise RuntimeError("Unsupported backend for `convolve`.")
        

    @staticmethod
    def orthogonal(S, F, Gt,):
        """
        Compute orthogonal bases after applying filters

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
