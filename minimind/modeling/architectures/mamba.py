import math
from typing import Dict

import flax.linen as nn
import jax

from minimind.config import Config
from minimind.modeling.architectures import register_architecture
from minimind.modeling.modules.mamba_block import MambaBlock


@register_architecture
class Mamba(nn.Module):
    """MAMBA is an advanced ML model renowned for its exceptional linear-time
    processing efficiency, which notably enhances its inference speed to
    outperform traditional Transformer models by up to five times in
    throughput.

    Unlike conventional models that struggle with long sequence lengths,
    MAMBA demonstrates a linear scalability with sequence length,
    maintaining or even improving its performance with sequences that
    extend up to a million elements. This attribute makes MAMBA a highly
    versatile and efficient backbone for a variety of sequence modeling
    tasks across different domains, including but not limited to
    language processing, audio analysis, and genomic studies.
    """

    config: Config
    d_conv: int = 3
    d_state: int = 8
    expand: int = 2
    start_token: int = 0
    end_token: int = 64
    bias: bool = True
    conv_bias: bool = True
    dt_rank: int = "auto"

    def setup(self):
        self.d_inner = int(self.expand * self.config.arch.embedding_dim)

        if self.dt_rank == "auto":
            dt_rank = math.ceil(self.config.arch.embedding_dim / 16)
        else:
            dt_rank = self.dt_rank

        self.embedding = nn.Embed(self.config.arch.vocab_size, self.config.arch.embedding_dim)

        self.mamba_blocks: list[MambaBlock] = [
            MambaBlock(
                d_inner=self.d_inner,
                d_conv=self.d_conv,
                dt_rank=dt_rank,
                d_state=self.d_state,
                d_model=self.config.arch.embedding_dim,
                seq_len=self.config.arch.max_sequence_length,
                bias=self.bias,
                conv_bias=self.conv_bias,
                name=f"mamba_block_{idx}",
            )
            for idx in range(self.config.arch.n_layers)
        ]

        self.norm_f = nn.RMSNorm(self.config.arch.embedding_dim)
        self.dropout1 = nn.Dropout(self.config.arch.residual_dropout_rate)
        self.lm_head = nn.Dense(features=self.config.arch.vocab_size, use_bias=False)

    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool = False,
    ) -> Dict[str, jax.Array]:
        inputs = batch.get("inputs")
        x = self.embedding(inputs)
        for block in self.mamba_blocks:
            x = self.dropout1(block(x), deterministic=not training)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        batch.update({"logits": logits})
        return batch
