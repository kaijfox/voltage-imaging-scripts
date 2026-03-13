"""CLI entry points for imaging_scripts."""

import click
from .common import enable_logging
from .io_commands import convert, convert_guided, slice_cmd, h5_convert, divisive_lowpass_cmd, sub_lowpass_cmd, motion_correct_cmd, svd_info, build_artifact_cmd, edit_rois_cmd
from .ts_commands import extract_traces_cmd, trace_dff, detect_spikes_cmd, spatial_survey
from .inspect_commands import inspect_filters_cmd, inspect_mc_cmd

@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.pass_context
def cli(ctx, verbose):
    """Imaging scripts CLI."""

    if verbose > 3:
        raise click.BadParameter("Verbose level must be 0-3")
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    enable_logging(verbose)


# Register commands
cli.add_command(convert,                "convert")
cli.add_command(convert_guided,         "convert-guided")
cli.add_command(h5_convert,             "h5-convert")
cli.add_command(motion_correct_cmd,     "motion-correct")
cli.add_command(slice_cmd,              "slice")
cli.add_command(divisive_lowpass_cmd,   "video-dff")
cli.add_command(sub_lowpass_cmd,        "video-sub-dff")
cli.add_command(svd_info,               "svd-info")
cli.add_command(build_artifact_cmd,     "build-artifact")
cli.add_command(edit_rois_cmd,          "edit-rois")
cli.add_command(extract_traces_cmd,     "extract-traces")
cli.add_command(trace_dff,              "trace-dff")
cli.add_command(detect_spikes_cmd,      "detect-spikes")
cli.add_command(spatial_survey,         "spatial-survey")
cli.add_command(inspect_filters_cmd,    "inspect-filters")
cli.add_command(inspect_mc_cmd,         "inspect-mc")

# Allow calling as module
if __name__ == "__main__":
    cli()
