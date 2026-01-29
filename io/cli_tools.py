import logging

def configure_logging(name, verbosity):
    # map int -> logging level
    level_map = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    level = level_map.get(verbosity, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger, (logger.error, logger.warning, logger.info, logger.debug)

def enable_logging(verbosity):
    level_map = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    level = level_map.get(verbosity, logging.INFO)
    logging.basicConfig(
        level=level,
        format='[%(name)-10s %(levelname).1s] %(message)s'
    )


def add_stream_conversion_args(parser):
    # Mode + paths
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to input video/source.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Output file path",
    )

    # Core SVD parameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1000,
        help="Number of frames to read per batch (default: 1000).",
    )
    
    # Logging/outputs
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bars."
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='increase verbosity (use -v, -vv, -vvv)'
    )