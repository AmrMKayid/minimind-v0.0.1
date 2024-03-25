import logging
import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=16"  # isort:skip
import jax

jax.config.update("jax_debug_nans", True)

from rich.logging import RichHandler

from minimind.config import Config
from minimind.core.cortex import Cortex
from minimind.data.hf import HuggingFaceDataset
from minimind.utils.parser import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=True,
            tracebacks_show_locals=True,
        )
    ],
)


def main() -> None:
    logging.info(f"Total devices: {jax.device_count()}, " f"Devices per task: {jax.local_device_count()}")

    args = parse_args()
    config = Config.read_config_from_yaml(args.config_path)
    logging.info(f"{config=}")

    dataset = HuggingFaceDataset(config)

    cortex = Cortex(config)
    cortex.train(dataset)


if __name__ == "__main__":
    main()
