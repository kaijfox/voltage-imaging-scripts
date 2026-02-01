"""CLI entry points for imaging_scripts."""

import click
from .common import enable_logging
from .io_commands import convert, slice_cmd, h5_convert, divisive_lowpass_cmd, svd_info

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


# IO commmands
cli.add_command(convert, "convert")
cli.add_command(h5_convert, "h5-convert")
cli.add_command(slice_cmd, "slice")
cli.add_command(divisive_lowpass_cmd, "video-dff")
cli.add_command(svd_info, "svd-info")

# Allow calling as module
if __name__ == "__main__":
    cli()
