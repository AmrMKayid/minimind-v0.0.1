from minimind.config import Config
from minimind.cortex import Cortex
from minimind.utils import parse_args


def main() -> None:
    args = parse_args()
    config = Config.read_config_from_yaml(args.config_path)

    cortex = Cortex(config)
    cortex.run()


if __name__ == "__main__":
    main()
