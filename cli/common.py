"""Shared arguments and utilities for CLI commands."""

import click
from functools import wraps
import logging

def configure_logging(name, verbosity=None):
    # map int -> logging level
    logger = logging.getLogger(name)
    if verbosity is not None:
        level_map = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
        level = level_map.get(verbosity, logging.INFO)
        logger.setLevel(level)
    return logger, (logger.error, logger.warning, logger.info, logger.debug)

def enable_logging(verbosity):
    level_map = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    level = level_map.get(verbosity, logging.INFO)
    logging.basicConfig(
        level=level,
        format='[%(name)-10s %(levelname).1s] %(message)s'
    )



def input_output_options(f):
    """Common -i/--input and -o/--output options."""
    @click.option(
        "-i", "--input", "input_path", required=True,
        type=click.Path(exists=True), help="Input file path."
    )
    @click.option(
        "-o", "--output", "output_path", required=True,
        type=click.Path(), help="Output file path."
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def batch_progress_options(batch_default=1000):
    """Common --batch-size and --no-progress options."""
    def decorator(f):
        @click.option(
            "-b", "--batch-size", type=int, default=batch_default,
            help=f"Batch size (default: {batch_default})."
        )
        @click.option(
            "--no-progress", is_flag=True, help="Disable progress bars."
        )
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator
