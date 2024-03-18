from typing import Optional

from flax import linen as nn

from minimind.config import Config


class SimpleModel(nn.Module):
    config: Config
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x, deterministic=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        x = nn.Embed(self.config.architecture.vocab_size, self.config.architecture.embedding_dim)(x)
        x = nn.Sequential(
            [
                nn.Dense(self.config.architecture.embedding_dim, name="FirstDense"),
                nn.relu,
                nn.Dense(self.config.architecture.vocab_size, name="SecondDense"),
            ],
            name="FirstSequential",
        )(x)

        return x
