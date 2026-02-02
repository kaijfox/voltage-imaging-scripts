import click
from pathlib import Path


from .common import (
    input_output_options,
    batch_progress_options,
    configure_logging,
)


@click.command()
@input_output_options
@click.option(
    "--rois",
    required=True,
    type=click.Path(exists=True),
    help="Path to ROI collection file (.mat, .h5, or .npz).",
)
@click.option(
    "--neuropil-range",
    required=True,
    type=str,
    help=(
        "Neuropil annulus dilation range as 'inner,outer' (e.g., '2,15'). "
        "Use inner < 0 to disable neuropil subtraction."
    ),
)
@batch_progress_options(1000)
@click.option(
    "--fs", type=float, default=None, help="Sampling rate in Hz (stored in output)."
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Output dF/F as percentage instead of raw fluorescence.",
)
@click.option(
    "--bg-smooth-size",
    type=int,
    default=0,
    help="Savitzky-Golay window for neuropil smoothing (must be odd, 0 to disable).",
)
@click.option(
    "--exclusion-threshold",
    type=float,
    default=0.4,
    help="Fraction threshold for excluding other ROIs from neuropil annulus.",
)
@click.option(
    "--no-ols",
    is_flag=True,
    help="Use simple weighted summation instead of OLS regression.",
)
@click.option(
    "--unweighted",
    is_flag=True,
    help="Use uniform weights instead of ROI collection weights.",
)
def extract_traces_cmd(
    input_path,
    output_path,
    rois,
    neuropil_range,
    batch_size,
    no_progress,
    fs,
    normalize,
    bg_smooth_size,
    exclusion_threshold,
    no_ols,
    unweighted,
):
    """Extract ROI traces from video (SVD or raw H5)."""
    import h5py
    from ..timeseries import extract_traces
    from ..timeseries.rois import ROICollection

    logger, (error, warning, info, debug) = configure_logging("extract")

    # Parse neuropil range
    try:
        parts = neuropil_range.split(",")
        if len(parts) != 2:
            raise ValueError()
        np_range = (int(parts[0]), int(parts[1]))
    except ValueError:
        raise click.BadParameter(
            "--neuropil-range must be 'inner,outer' (e.g., '2,15')"
        )

    # Validate bg_smooth_size
    if bg_smooth_size < 0:
        raise click.BadParameter("--bg-smooth-size must be >= 0")
    if bg_smooth_size > 0 and bg_smooth_size % 2 == 0:
        raise click.BadParameter("--bg-smooth-size must be odd when nonzero")

    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    # Load ROI collection
    info(f"Loading ROIs from {rois}")
    roi_collection = ROICollection.load(rois)
    n_rois = len(roi_collection.rois)
    info(f"Loaded {n_rois} ROIs")

    # Check if we need to infer image_shape from video
    image_shape = roi_collection.image_shape
    if image_shape is None:
        # Infer from video file
        with h5py.File(input_path, "r") as f:
            if "video" in f:
                _, H, W = f["video"].shape
            elif "Vh" in f:
                shape = f["Vh"].shape
                if len(shape) == 3:
                    _, H, W = shape
                else:
                    raise click.ClickException(
                        "Cannot infer image shape from SVD file with flattened Vh. "
                        "Provide ROI file with image_shape or use --shape option."
                    )
            else:
                raise click.ClickException("Cannot infer image shape from input file.")
            image_shape = (H, W)
            warning(f"Inferred image shape {image_shape} from video file")
            roi_collection.image_shape = image_shape

    # Detect source type for logging
    with h5py.File(input_path, "r") as f:
        source_type = "SVD" if ("U" in f and "S" in f and "Vh" in f) else "raw H5"
    info(f"Extracting traces from {source_type} video: {input_path}")

    # Extract traces
    try:
        traces = extract_traces(
            source=input_path,
            rois=roi_collection,
            neuropil_range=np_range,
            normalize=normalize,
            bg_smooth_size=bg_smooth_size,
            exclusion_threshold=exclusion_threshold,
            fs=fs,
            batch_size=batch_size,
            ols=not no_ols,
            weighted=not unweighted,
        )
    except Exception as exc:
        error(f"Extraction failed: {exc}")
        raise

    # Save output
    info(
        f"Saving {traces.data.shape[0]} traces ({traces.data.shape[1]} frames) to {output_path}"
    )
    traces.save_mat(output_path)
    info("Done")
