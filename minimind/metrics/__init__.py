from typing import Dict


class Metric:
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def __call__(self, *args: Dict, **kwargs: Dict):
        raise NotImplementedError
