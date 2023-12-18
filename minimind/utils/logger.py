import logging
import os
from datetime import datetime

from rich.logging import RichHandler


def setup_logger(output_dir="logs"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(__name__)

    # Configure the RichHandler with the desired formatting
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=True,
        tracebacks_show_locals=True,
    )

    # Add the RichHandler to the logger
    logger.addHandler(rich_handler)

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a FileHandler with the specified output directory and set the formatter
    file_handler = logging.FileHandler(os.path.join(output_dir, f"output_{current_datetime}.log"))

    # Add the FileHandler to the logger
    logger.addHandler(file_handler)

    # Set the logging level
    logger.setLevel(logging.INFO)

    return logger
