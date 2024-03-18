from typing import Dict

import flax.linen as nn
import jax

from minimind.config import Config
from minimind.modeling.modules.emb import Embedding, PatchEmbedding
from minimind.modeling.modules.transformer_block import TransformerBlock


class CLIPTextEncoderBlock(TransformerBlock):
    "Clip Text Encoder Block"


class CLIPTextEncoder(nn.Module):
    """Implements the text encoder component of the CLIP model."""

    config: Config

    def setup(self):
        self.text_embedding = Embedding(config=self.config, name="text_embedding")

        self.text_encoder_blocks: list[CLIPTextEncoderBlock] = [
            CLIPTextEncoderBlock(self.config, name=f"text_encoder_block_{idx}")
            for idx in range(self.config.arch.n_layers)
        ]

    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool = False,
    ) -> Dict[str, jax.Array]:

        batch = self.text_embedding(batch, training)
        for idx in range(self.config.arch.n_layers):
            batch = self.text_encoder_blocks[idx](batch, training)
        return batch


class CLIPImageEncoder(nn.Module):
    """Implements the image encoder component of the CLIP model."""

    config: Config

    def setup(self):
        self.image_embedding = PatchEmbedding(config=self.config, name="image_embedding")

        self.image_encoder_blocks: list[CLIPTextEncoderBlock] = [
            CLIPTextEncoderBlock(self.config, name=f"image_encoder_block_{idx}")
            for idx in range(self.config.arch.n_layers)
        ]

    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool = False,
    ) -> Dict[str, jax.Array]:

        batch = self.image_embedding(batch, training)
        for idx in range(self.config.arch.n_layers):
            batch = self.image_encoder_blocks[idx](batch, training)
        return batch
