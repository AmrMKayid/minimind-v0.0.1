from typing import Dict, Iterator, Literal, Mapping, TypeVar

import jax.random
import numpy as np
from jaxtyping import PRNGKeyArray

from minimind.data import ShardableDataset

T = TypeVar("T")


class MixtureDataset(ShardableDataset[T]):
    """MixtureDataset supports loading data from multiple datasets. It takes a
    list of datasets and yields from them according to the weights."""

    def __init__(
        self,
        datasets: Mapping[str, ShardableDataset[T]],
        weights: Dict[str, float],
        key: int | PRNGKeyArray = 0,
    ) -> None:
        self.datasets = datasets
        self.weights = MixtureDataset._normalize_weights(weights)

        if not isinstance(key, int):
            key = jax.random.randint(key, (), 0, 2**20).item()

        self.key = key

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        """Normalize the weights to sum to 1."""
        total: float | Literal[0] = sum(weights.values())
        if total == 0:
            raise ValueError(f"Datasets' weights cannot sum to 0, got {weights}")
        return {name: weight / total for name, weight in weights.items() if weight > 0}

    def shard(self, shard_id: int, num_shards: int) -> "MixtureDataset":
        """Return a MixtureDataset with the sharded datasets."""
        sharded = {name: dset.shard(shard_id, num_shards) for name, dset in self.datasets.items()}
        return MixtureDataset(sharded, self.weights)

    def __iter__(self) -> Iterator[np.ndarray]:
        iterators = {name: iter(dataset) for name, dataset in self.datasets.items()}
        current_weights = self._normalize_weights(self.weights)
        rng = np.random.default_rng(self.key)

        while True:
            dataset_name = rng.choice(list(current_weights.keys()), p=list(current_weights.values()))
            try:
                item = next(iterators[dataset_name])
                yield item
            except StopIteration:
                del iterators[dataset_name]
                del current_weights[dataset_name]
                if len(current_weights) == 0:
                    break
                current_weights = self._normalize_weights(current_weights)
