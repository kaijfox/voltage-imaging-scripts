"""
Vectorized functions on ragged arrays
"""

import numpy as np


def ak_infer_shape(arr):
    """
    Infer a batch-style shape from an Awkward array-like object.

    Returns a tuple where each entry is either an int (if that axis has the
    same length for every element) or an Awkward array of counts for that axis
    (if irregular). This mirrors the logic in the original snippets but is
    defensive and easy to reuse.

    Examples
    --------
    - Regular 2D ragged array -> (n_rows, n_cols)
    - Irregular second axis -> (n_rows, counts_per_row)
    """
    import awkward as ak

    # Fast path for numpy-like objects
    if hasattr(arr, "shape") and getattr(arr, "ndim", None) is not None:
        return arr.shape

    shape = ()

    # Determine number of axes; fall back to 1 if unknown
    ndim = 1
    if hasattr(arr, "ndim"):
        ndim = arr.ndim
    elif hasattr(arr, "layout") and hasattr(arr.layout, "minmax_depth"):
        ndim = arr.layout.minmax_depth[0]
    elif hasattr(arr, "minmax_depth"):
        ndim = arr.minmax_depth[0]

    for i in range(int(ndim)):
        counts = ak.num(arr, axis=i)

        # Obtain a scalar representative for comparison (first element)
        first = ak.flatten([counts], axis=None)
        if len(first) == 0:
            # empty shape
            shape = shape + (0,)
            continue
        first = first[0]

        # If every entry equals the first, return an int, else keep the counts array
        try:
            try:
                all_eq = ak.all(counts == first)
            except TypeError:
                all_eq = np.all(counts == first)

            if all_eq:
                shape = shape + (int(first),)
            else:
                shape = shape + (counts,)
        except Exception as e:
            # Conservative fallback when equality cannot be vectorized
            shape = shape + (counts,)

    return shape


def ak_flatten_n_times(array, n, axis=-2, highlevel=False):
    """
    Flatten an Awkward array `n` times along `axis`.

    This encapsulates the repeated flattening pattern from your snippet and
    handles trivial cases (n <= 0) safely.
    """
    import awkward as ak

    if n <= 0:
        return array

    out = array
    for _ in range(int(n)):
        out = ak.flatten(out, axis=axis, highlevel=highlevel)
    return out


def ak_unflatten_by_batch_shape(sampled_flat, ns, batch_shape, axis=0, highlevel=False):
    """
    Reconstruct a nested Awkward structure from a flattened `sampled_flat`.

    - First unflatten by `ns` (total windows per outer product).
    - Then unflatten repeatedly by entries of `batch_shape` (from the end
      towards the front) so the final layout matches the original batch nesting.

    Elements of `batch_shape` may be integers or Awkward arrays (irregular shapes).
    """
    import awkward as ak

    sampled = ak.unflatten(sampled_flat, ns, axis=axis, highlevel=highlevel)

    # Walk batch_shape from last (closest to window dimension) to first
    # and unflatten progressively. Skip the last element because ns already
    # accounts for the innermost split.
    for tgt in reversed(batch_shape[:-1]):
        # If target is an Awkward array, ensure it's flattened when needed
        if hasattr(tgt, "ndim") and getattr(tgt, "ndim") > 0:
            tgt_un = ak.flatten(tgt, axis=None, highlevel=highlevel)
        else:
            tgt_un = tgt

        sampled = ak.unflatten(sampled, tgt_un, axis=axis, highlevel=highlevel)

    return sampled


def _nplike_is_jax(nplike):
    try:
        from awkward._nplikes.jax import Jax
        from awkward._nplikes.array_module import ArrayModuleNumpyLike

        if not isinstance(nplike, ArrayModuleNumpyLike):
            raise ImportError
        return isinstance(nplike, Jax)
    except ImportError:
        return "jax" in nplike.__name__


def _debug_search(data, starts, lengths, bins, low, high, mid, old_low, old_high, mask):
    for i in range(bins.size):
        # Display range & midpoint, e.g. [  0 ,  0 , (1), *2*, (2), (3) ]
        s = data[starts[0] : starts[0] + lengths[0]]
        print(f"val {bins[i]}: [ ", end="")
        for j, c in enumerate(s):
            if j == mid[0, i]:
                print(f"*{c}*, ", end="")
            elif j >= old_low[0, i] and j <= old_high[0, i]:
                print(f"({c}), ", end="")
            else:
                print(f" {c} , ", end="")
        print(" ]")

        # Display mask & new bounds: [  0 ,  0 , (1), *2*, (2), (3) ]
        #                            [  . ,  . , (.),  . ,  . ,  .  ]
        print(f"val {bins[i]}: [ ", end="")
        for j, c in enumerate(s):
            if j >= low[0, i] and j < high[0, i]:
                print(f"(.), ", end="")
            else:
                print(f" . , ", end="")
        print(" ]")

        print(f"val {bins[i]}: mask={mask[0, i]}")


def _minimum(a, b, nplike):
    return nplike.where(a < b, a, b)


def boundsorted(data, offsets, bins, nplike=np, count=False):
    """
    data : 1d arraylike
        Flattened sorted segments, sorted *within each segment*.
    offsets : 1d arraylike
        Indices of segment bounds in data.
        Segment i is data[offsets[i]:offsets[i+1]].
    bins : 1d arraylike
        Values to search for within each segment.
    count : bool, optional
        If True, return counts of occurrences instead of boundary indices.

    Returns
    -------
    result : 2d array
        Array of shape (n_segments, len(bins)).
    """

    data = nplike.asarray(data)
    offsets = nplike.asarray(offsets)
    bins = nplike.asarray(bins)
    n_val = bins.size
    n_segments = offsets.size - 1

    if bins.ndim != 1:
        raise ValueError("bins must be a 1-D array")
    if offsets.ndim != 1:
        raise ValueError("offsets must be a 1-D array")
    if data.ndim != 1:
        raise ValueError("data must be a 1-D array")

    starts = offsets[:-1]  # shape (n_segments,)
    lengths = offsets[1:] - offsets[:-1]  # shape (n_segments,)

    # Indices local to each segment
    low = nplike.zeros((n_segments, n_val), dtype=int)
    high = nplike.zeros((n_segments, n_val), dtype=int)
    if _nplike_is_jax(nplike):
        high = high.at[:].set(lengths[:, None])
    else:
        high[:] = lengths[:, None]

    not_done = nplike.ones((n_segments, n_val), dtype=bool)

    # vectorized binary-search loop
    while nplike.any(not_done):

        # Gather new midpoint indices of the current ranges
        mid = (low + high) // 2  # shape (n_segments, n_val)

        # Convert local midpoints to indices into data
        mid = _minimum(mid, lengths[:, None] - 1, nplike)
        idx = starts[:, None] + mid  # shape (n_segments, n_val)
        sample = data[idx]  # shape (n_segments, n_val)

        # For debugging
        # old_low = low.copy()
        # old_high = high.copy()

        # Update bounds based on comparison of value to midpoint
        mask = sample < bins[None, :]  # shape (n_segments, n_val)
        if _nplike_is_jax(nplike):
            low.at[mask].set(mid[mask] + 1)
            high.at[~mask].set(mid[~mask])
        else:
            low[mask] = mid[mask] + 1
            high[~mask] = mid[~mask]

        not_done = low < high

        # For debugging
        # _debug_search(
        #     data, starts, lengths, bins, low, high, mid, old_low, old_high,
        #     mask
        # )

    # For empty rows, every index is zero
    empty_rows = lengths == 0  # shape (n_segments,)
    low[empty_rows, :] = 0
    high[empty_rows, :] = 0

    left = low[:, :-1]
    right = high[:, 1:]
    if count:
        return right - left
    return left, right


def create_action(
    func,
):
    import awkward as ak
    from awkward._nplikes.numpy_like import NumpyMetadata

    np = NumpyMetadata.instance()

    def action(layout, **kwargs):
        if layout.branch_depth == (False, 1):
            if layout.is_indexed:
                layout = layout.project()

            if (
                layout.parameter("__array__") == "string"
                or layout.parameter("__array__") == "bytestring"
            ):
                nextcontent, _ = func(ak.highlevel.Array(layout), None)
                return ak.contents.NumpyArray(nextcontent)

            if layout.is_unknown:
                layout = layout.to_NumpyArray(np.float64)
            elif not layout.is_numpy:
                raise NotImplementedError("run_lengths on " + type(layout).__name__)

            nextcontent, _ = func(layout.data, None)
            return ak.contents.NumpyArray(nextcontent)

        elif layout.branch_depth == (False, 2):
            if layout.is_indexed:
                layout = layout.project()

            if not layout.is_list:
                raise NotImplementedError("run_lengths on " + type(layout).__name__)

            if (
                layout.content.parameter("__array__") == "string"
                or layout.content.parameter("__array__") == "bytestring"
            ):
                # We also want to trim the _upper_ bound of content,
                # so we manually convert the list type to zero-based
                listoffsetarray = layout.to_ListOffsetArray64(False)
                content = listoffsetarray.content[
                    listoffsetarray.offsets[0] : listoffsetarray.offsets[-1]
                ]

                if content.is_indexed:
                    content = content.project()

                offsets = listoffsetarray.offsets.data
                nextcontent, nextoffsets = func(
                    ak.highlevel.Array(content), offsets - offsets[0]
                )
                return ak.contents.ListOffsetArray(
                    ak.index.Index64(nextoffsets), ak.contents.NumpyArray(nextcontent)
                )

            listoffsetarray = layout.to_ListOffsetArray64(False)
            content = listoffsetarray.content[
                listoffsetarray.offsets[0] : listoffsetarray.offsets[-1]
            ]

            if content.is_indexed:
                content = content.project()

            if content.is_unknown:
                content = content.to_NumpyArray(np.float64)
            elif not content.is_numpy:
                raise NotImplementedError(
                    "run_lengths on "
                    + type(layout).__name__
                    + " with content "
                    + type(content).__name__
                )

            offsets = listoffsetarray.offsets.data
            nextcontent, nextoffsets = func(content.data, offsets - offsets[0])
            return ak.contents.ListOffsetArray(
                ak.index.Index64(nextoffsets), ak.contents.NumpyArray(nextcontent)
            )
        else:
            return None

    return action


def ak_boundsorted(
    array,
    bins=None,
    max_val=None,
    return_count=True,
    side="left",
    highlevel=True,
    behavior=None,
    attrs=None,
):
    import awkward as ak
    from awkward._layout import HighLevelContext
    from awkward._nplikes.jax import Jax
    from awkward._nplikes.numpy_like import NumpyMetadata
    from awkward._nplikes.shape import unknown_length

    np = NumpyMetadata.instance()

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    def compute_bounds(data, offsets):
        backend = layout.backend

        if bins is not None:
            compute_bins = bins
        else:
            compute_bins = backend.nplike.arange(0, max_val + 2, dtype=np.int64)

        if backend.nplike.is_own_array(data):
            size = data.size
        else:
            size = ak.to_layout(data).length

        if size is not unknown_length and size == 0:
            return backend.nplike.empty(0, dtype=np.int64), offsets

        # Compute boundaries
        lefts, rights = boundsorted(
            data,
            offsets,
            compute_bins,
            nplike=backend.nplike,
        )

        if return_count:
            nextcontent = rights - lefts
        else:
            nextcontent = lefts if side == "left" else rights

        # Boundsorted return shape (n_segments, len(bins)-1)
        nextcontent = backend.nplike.reshape(nextcontent, (-1,))
        nextoffsets = backend.nplike.arange(
            0, nextcontent.size + 1, compute_bins.size - 1, dtype=np.int64
        )

        return nextcontent, nextoffsets

    action = create_action(compute_bounds)
    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)


def digitize(data, bins, right=False, nplike=np):

    data = nplike.asarray(data)
    bins = nplike.asarray(bins)

    if bins.ndim != 1 or bins.size < 1:
        raise ValueError("bins must be a 1-D array")

    low = nplike.zeros(data.shape, dtype=int)
    high = nplike.full(data.shape, bins.size, dtype=int)
    not_done = nplike.ones(data.shape, dtype=bool)

    while nplike.any(not_done):

        # Compute midpoint and sample value
        mid = (low + high) // 2
        mid = nplike.minimum(mid, bins.size - 1)
        sample = bins[mid]

        # Sample comparison, with bins left right- or left-inclusive
        if right:
            mask = sample <= data
        else:
            mask = sample < data

        # Update bounds based on comparison of value to midpoint
        if nplike.any(mask):
            low = nplike.where(mask, mid + 1, low)
        if nplike.any(~mask):
            high = nplike.where(~mask, mid, high)

        not_done = low < high

    return low


def ak_digitize(
    array, bins=None, side="left", highlevel=True, behavior=None, attrs=None
):
    import awkward as ak
    from awkward._layout import HighLevelContext
    from awkward._nplikes.jax import Jax
    from awkward._nplikes.numpy_like import NumpyMetadata
    from awkward._nplikes.shape import unknown_length

    np = NumpyMetadata.instance()

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    def compute_digitize(data, offsets):
        backend = layout.backend

        if backend.nplike.is_own_array(data):
            size = data.size
        else:
            size = ak.to_layout(data).length

        if size is not unknown_length and size == 0:
            return backend.nplike.empty(0, dtype=np.int64), offsets

        if hasattr(backend.nplike._module, "digitize"):
            func = backend.nplike._module.digitize
        else:
            func = digitize

        # Compute elementwise and return
        nextcontent = func(
            data,
            bins,
            right=(side == "right"),
            nplike=backend.nplike,
        )
        return nextcontent, offsets

    action = create_action(compute_digitize)
    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)


def _ak_unique_array(array, highlevel=True, behavior=None, attrs=None):
    """
    Compress leaf nodes to unique values.
    """
    import awkward as ak
    from awkward._layout import HighLevelContext
    from awkward._nplikes.numpy_like import NumpyMetadata
    from awkward._nplikes.shape import unknown_length

    np = NumpyMetadata.instance()

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    def compute_unique(data, offsets):
        backend = layout.backend
        nextcontent = backend.nplike._module.unique(data)
        nextoffsets = backend.nplike.asarray([0, nextcontent.size], dtype=np.int64)
        return nextcontent, nextoffsets

    action = create_action(compute_unique)
    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)


def ak_unique(array, behavior=None, attrs=None):
    import awkward as ak

    compressed = _ak_unique_array(
        array.layout, highlevel=False, behavior=behavior, attrs=attrs
    )
    flat = ak.flatten(compressed, axis=None, highlevel=False)
    nplike = compressed.backend.nplike

    if not hasattr(nplike._module, "unique"):
        raise ValueError(f"{nplike} does not support unique")
    if not hasattr(flat, "data"):
        raise ValueError(
            f"unique unsupported: {type(flat)} does not have data attribute."
        )

    unique_vals = nplike._module.unique(flat.data)
    return unique_vals


def ak_apply_1d(array, func, highlevel=True, behavior=None, attrs=None):
    """
    Apply a function to each segment of a ragged array.

    Func must preserve its arguments' shape.

    ak_apply_1d([[1, 2, 3], [3, 5]], demean) -> [[-1, 0, 1], [1, 1]]
    """
    import awkward as ak
    from awkward._layout import HighLevelContext
    from awkward._nplikes.numpy_like import NumpyMetadata

    np = NumpyMetadata.instance()

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    def _apply_to_padded(start_and_stop, data, max_length, dtype, func, nplike):

        # Apply a function to each segment of data, padding to max_length.

        # compute_apply will then remove padding by masking elements past `start-stop`

        # Parameters:
        # - start_and_stop: A pair of start and stop indices for the segment.
        # - data: The flattened array containing all segments.
        # - max_length: The maximum length to pad each segment to.
        # - dtype: fallback dtype if cannot be inferred from processed data
        # - func: The function to apply to each segment.
        # - nplike: The numpy-like backend for array operations.

        start, stop = start_and_stop
        segment = data[start:stop]

        # Apply the function to the segment
        processed = func(segment)

        out_dtype = getattr(processed, "dtype", dtype)
        result = nplike.zeros((max_length,), dtype=out_dtype)

        # Create a padded row with NaN and insert the processed values
        if _nplike_is_jax(nplike) and not hasattr(processed, "layout"):
            result = result.at[: len(processed)].set(processed)
        else:
            result[: len(processed)] = processed

        return result

    def compute_apply(layout, **kwargs):

        # Compute the application of the function to the ragged array.

        # Logic:
        # - Extract the data and offsets from the layout.
        # - Compute the start and stop indices for each segment.
        # - Apply the function to each segment, padding the results.
        # - Remove the padding to reconstruct the valid data.
        # - Compute new offsets based on the valid lengths of each segment.

        if layout.is_indexed:
            layout = layout.project()  # Resolve any indexing in the layout

        nplike = (
            layout.backend.nplike
        )  # Get the numpy-like backend for array operations

        data, offsets, dtype, content_type = _extract_layout_info(layout, nplike)
        if data is None:
            return None  # Unsupported layout depth

        # Compute the start and stop indices for each segment
        starts = offsets[:-1]
        stops = offsets[1:]
        lengths = stops - starts

        # Determine the maximum length of any segment for padding
        max_length = nplike.max(lengths) if len(starts) > 0 else 0

        # Apply the function, returning a rectangular raw array with padding
        if hasattr(nplike._module, "apply_along_axis"):
            result = nplike._module.apply_along_axis(
                _apply_to_padded,
                0,
                nplike.stack([starts, stops]),
                data,
                max_length,
                dtype,
                func,
                nplike,
            )  # shape (max_length, n_segments)
            result = result.T

        else:
            raise NotImplementedError(
                "ak_apply_1d requires nplike with apply_along_axis support."
            )

        # create mask for valid results & excise padding
        ixs = nplike.arange(max_length)[None, :]  # (n_segments, max_length)
        mask = ixs < lengths[:, None]  # True indicates valid data
        result_masked = result[mask]

        # Return a new ListOffsetArray with the flattened data and updated offsets
        return ak.contents.ListOffsetArray(
            ak.index.Index64(offsets),
            content_type(result_masked),  # Match original dtype
        )

    if ak.highlevel.Array(layout).ndim == 1:
        # For 1d arrays, wrap and apply
        return ak_apply_1d([layout], func)[0]

    out = ak._do.recursively_apply(layout, compute_apply)
    return ctx.wrap(out, highlevel=highlevel)


def _extract_layout_info(layout, nplike):
    """
    Helper function to extract data, offsets, dtype, and content_type based on layout.branch_depth.
    """
    import awkward as ak

    if layout.branch_depth == (False, 1):

        if len(layout) == 0:
            print("Handling empty content depth 1")
            # For empty arrays, create empty data & offsets
            data = nplike.asarray([])
            offsets = nplike.asarray([0, 0])
            dtype = data.dtype
            content_type = lambda *a, **kw: type(layout)()

        else:

            print("Handling 1D array depth 1")
            # For 1D arrays, use the entire data as a single segment
            data = layout.data
            offsets = nplike.asarray([0, layout.data.size])
            dtype = data.dtype
            content_type = type(layout)

    elif layout.branch_depth == (False, 2):
        listoffsetarray = layout.to_ListOffsetArray64(False)

        if (
            layout.content.parameter("__array__") == "string"
            or layout.content.parameter("__array__") == "bytestring"
        ):

            # For strings, form highlevel Array (see create_action)
            content = listoffsetarray.content[
                listoffsetarray.offsets[0] : listoffsetarray.offsets[-1]
            ]
            offsets = listoffsetarray.offsets.data
            data = ak.highlevel.Array(content).to_numpy()
            offsets = offsets - offsets[0]
            dtype = "uint8"
            content_type = lambda x: ak.with_parameter(
                ak.contents.ListOffsetArray(
                    ak.index.Index64(nplike.arange(x.size + 1, dtype=np.int64)),
                    ak.with_parameter(
                        type(content.content)(x),
                        "__array__",
                        content.content.parameter("__array__"),
                        highlevel=False,
                    ),
                ),
                "__array__",
                layout.content.parameter("__array__"),
                highlevel=False,
            )

            def content_type(x):
                return ak.from_numpy(x).layout

        elif len(listoffsetarray.content) == 0:

            print("Handling empty content depth 2")

            # For empty arrays, create empty data & offsets
            data = nplike.asarray([])
            offsets = listoffsetarray.offsets.data
            dtype = data.dtype
            content_type = lambda *a, **kw: ak.contents.ListOffsetArray(
                listoffsetarray.offsets, type(listoffsetarray.content)()
            )

        else:
            print("Handling normal content")
            # For normal 2D arrays, extract the content and offsets
            data = listoffsetarray.content.data
            offsets = listoffsetarray.offsets.data
            content_type = type(listoffsetarray.content)
            dtype = data.dtype

    else:
        return None, None, None, None  # Unsupported layout depth

    return data, offsets, dtype, content_type


def ak_reduce_1d(array, func, highlevel=True, behavior=None, attrs=None):
    """
    Reduce segments of a ragged array to scalars,

    ak_reduce_1d([[1, 2, 3], [4, 5]], min) -> [1, 4]
    """

    # Approach
    #
    # _apply_to_padded(start, stop, data):
    # index into data, apply func
    #
    # compute_apply(layout):
    # construct offsets/data depending on depth
    # construct starts, stops from offsets
    # apply along axis (_apply_to_padded, [starts, stops])
    # compute offsets indicating single segment in result
    import awkward as ak
    from awkward._layout import HighLevelContext
    from awkward._nplikes.numpy_like import NumpyMetadata

    np = NumpyMetadata.instance()

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    def _apply_to_padded(start_and_stop, data, max_length, dtype, func, nplike):

        # Apply a function to each segment of data, padding to max_length.

        # compute_apply will then remove padding by masking elements past `start-stop`

        # Parameters:
        # - start_and_stop: A pair of start and stop indices for the segment.
        # - data: The flattened array containing all segments.
        # - max_length: The maximum length to pad each segment to.
        # - dtype: fallback dtype if cannot be inferred from processed data
        # - func: The function to apply to each segment.
        # - nplike: The numpy-like backend for array operations.

        start, stop = start_and_stop
        segment = data[start:stop]

        if segment.size == 0:
            return

        # Apply the function to the segment
        return func(segment)

    def compute_apply(layout, **kwargs):

        # Compute the application of the function to the ragged array.

        # Logic:
        # - Extract the data and offsets from the layout.
        # - Compute the start and stop indices for each segment.
        # - Apply the function to each segment, padding the results.
        # - Remove the padding to reconstruct the valid data.
        # - Compute new offsets based on the valid lengths of each segment.

        if layout.is_indexed:
            layout = layout.project()  # Resolve any indexing in the layout

        nplike = (
            layout.backend.nplike
        )  # Get the numpy-like backend for array operations

        data, offsets, dtype, content_type = _extract_layout_info(layout, nplike)
        if data is None:
            return None  # Unsupported layout depth

        # Compute the start and stop indices for each segment
        starts = offsets[:-1]
        stops = offsets[1:]
        lengths = stops - starts

        # Determine the maximum length of any segment for padding
        max_length = nplike.max(lengths) if len(starts) > 0 else 0

        # Apply the function, returning a rectangular raw array with padding
        if hasattr(nplike._module, "apply_along_axis"):
            result = nplike._module.apply_along_axis(
                _apply_to_padded,
                0,
                nplike.stack([starts, stops]),
                data,
                max_length,
                dtype,
                func,
                nplike,
            )  # shape (max_length, n_segments)
            result = result

        else:
            raise NotImplementedError(
                "ak_apply_1d requires nplike with apply_along_axis support."
            )

        # Return a new ListOffsetArray with the flattened data and updated offsets
        return content_type(result)

    if ak.highlevel.Array(layout).ndim == 1:
        # For 1d arrays, wrap and apply
        return ak_reduce_1d([layout], func)[0]

    out = ak._do.recursively_apply(layout, compute_apply)
    return ctx.wrap(out, highlevel=highlevel)


def slice_by_events(trace_data, event_frames, n_pre, n_post):
    """
    trace_data: (*batch_shape, n_frames)
    event_frames: (*batch_shape, <n_events>)
    n_pre, n_post: int"""

    import awkward as ak
    import numpy as np

    trace_data = ak.Array(trace_data)
    events = ak.Array(event_frames)
    if trace_data.ndim != events.ndim:
        raise ValueError(
            "trace_data and event_frames must have the same number of dimensions"
        )

    # Ensure batch shapes are same
    trace_data, events = ak.broadcast_arrays(
        trace_data, events, depth_limit=trace_data.ndim - 1
    )

    # Extract generalized shape of trace (and therefore of batch dimensions)
    shape = ()
    if hasattr(trace_data, "shape"):
        shape = trace_data.shape
    else:
        for i in range(trace_data.ndim):
            num = ak.num(trace_data, axis=i)
            num_first = ak.flatten([num], axis=None)[0]
            if np.all(num == num_first):
                shape = shape + (num_first,)
            else:
                shape = shape + (num,)

    n_frames = shape[-1]
    batch_shape = shape[:-1]

    # (*batch_shape, <n_events>, frames_per_window)
    rel_times = np.arange(-n_pre, n_post + 1)
    rel_times = rel_times[(None,) * len(batch_shape) + (None, slice(None))]
    sample_ixs = events[..., None] + rel_times

    # (*batch_shape, n_frames)
    x = ak.from_regular(trace_data, highlevel=False)
    # Flatten to only one batch dimension in x
    for i in range(len(batch_shape) - 1):
        x = ak.from_regular(ak.flatten(x, axis=-2, highlevel=False), highlevel=False)

    starts = x.offsets.data[:-1]
    ends = x.offsets.data[1:]
    ns = ak.flatten(ak.num(sample_ixs, axis=-2), axis=None)
    starts, ends = np.concatenate(
        [np.broadcast_to([[s], [e]], (2, n)) for s, e, n in zip(starts, ends, ns)],
        axis=1,
    )
    # ( prod(batch_shape) * n_total_windows, n_frames )
    x = ak.contents.ListArray(
        starts=ak.index.Index64(starts),
        stops=ak.index.Index64(ends),
        content=x.content,
    )

    # (batch_shape, <n_windows>, frames_per_window)
    window_ixs = sample_ixs[:]
    # ( prod(batch_shape) * n_total_windows, frames_per_window )
    for i in range(len(batch_shape)):
        window_ixs = ak.flatten(window_ixs, axis=-2, highlevel=False)
    # window_ixs = ak.flatten(window_ixs, axis=-2, highlevel=False)
    # convert window_ixs to ListOffsetArray
    # for some reason indexing fails with regular layout
    if isinstance(window_ixs, ak.contents.RegularArray):
        window_ixs = ak.from_regular(window_ixs, axis=1, highlevel=False)

    # Mark invalid indices (will be nan'ed post-hoc)
    valid_ixs = (
        (ak.highlevel.Array(window_ixs) >= 0)
        & (ak.highlevel.Array(window_ixs) < n_frames)
    ).layout
    window_ixs = ak.where(valid_ixs, window_ixs, 0)

    # Finally perform the indexing
    sampled_flat = x[window_ixs]

    # Remove samples from outside the window
    # from_iter([None]) and [..., 0] required to avoid 'none propomotion disabled' error
    sampled = ak.where(valid_ixs[..., None], sampled_flat, ak.from_iter([None]))[..., 0]

    # Unflatten to original shapes
    sampled = ak.unflatten(sampled, ns, axis=0, highlevel=False)
    for i in range(len(batch_shape) - 1):
        tgt_shape = batch_shape[-i - 1]
        # Flatten target shape up to last axis if irregular, to match flat `sampled`
        if hasattr(tgt_shape, "ndim") and tgt_shape.ndim > 0:
            tgt_shape = ak.flatten(tgt_shape, axis=None)
        sampled = ak.unflatten(sampled, tgt_shape, axis=0, highlevel=False)
    sampled = ak.highlevel.Array(sampled)

    return sampled
