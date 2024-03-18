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

    # # Using LaBSE since vocab size is 500k
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "hf-internal-testing/llama-tokenizer",
    #     bos_token="<|BOS_TEXT_TOKEN|>",
    #     eos_token="<|EOS_TEXT_TOKEN|>",
    #     pad_token="<|PAD_TOKEN|>",
    #     unk_token="<|UNK_TOKEN|>",
    #     additional_special_tokens=[
    #         "<|BOS_VISION_TOKEN|>",
    #         "<|EOS_VISION_TOKEN|>",
    #     ],
    #     extra_ids=0,
    # )
    # logging.info(f"{tokenizer=}")
    # dataset = prepare_data(config, tokenizer)

    dataset = HuggingFaceDataset(config)

    cortex = Cortex(config)
    # inspect(minimind)

    cortex.train(dataset)


if __name__ == "__main__":
    main()
