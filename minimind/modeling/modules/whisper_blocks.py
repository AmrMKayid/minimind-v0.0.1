from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp

from minimind.config import Config
from minimind.modeling.modules.attention import AddNorm, MultiHeadAttention, PositionWiseFFN
from minimind.modeling.modules.emb import Embedding, SpeechEmbedding
from minimind.modeling.modules.transformer_block import TransformerBlock


class WhisperSpeechEncoderBlock(TransformerBlock):
    """Implements a single encoder block for the Whisper Speech model, combining self-attention with a feed-forward network."""


class WhisperSpeechEncoder(nn.Module):
    """Implements the encoder component of the Whisper Speech model."""

    config: Config

    def setup(self):
        self.embedding = SpeechEmbedding(config=self.config, name="speech_embedding")

        self.encoder_blocks: list[WhisperSpeechEncoderBlock] = [
            WhisperSpeechEncoderBlock(self.config, name=f"whisper_encoder_block_{idx}")
            for idx in range(self.config.arch.n_layers)
        ]

    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool = False,
    ) -> Dict[str, jax.Array]:

        batch = self.embedding(batch, training)
        for idx in range(self.config.arch.n_layers):
            batch = self.encoder_blocks[idx](batch, training)
        return batch


class WhisperTextDecoderBlock(nn.Module):
    """
    Implements a single decoder block for the Transformer model, combining self-attention, encoder-decoder attention, and a feed-forward network.

    This block first processes the input through self-attention, allowing each position to attend to all positions up to and including itself. Then, it applies encoder-decoder attention, integrating information from the encoder's output. Finally, a position-wise feed-forward network is applied.
    """

    config: Config

    def setup(self):
        self.attention = MultiHeadAttention(hidden_dim=self.config.arch.emb_dim, num_heads=self.config.arch.n_heads)
        self.cross_attention = MultiHeadAttention(
            hidden_dim=self.config.arch.emb_dim, num_heads=self.config.arch.n_heads
        )
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.config.arch.emb_dim)
        self.add_norm1 = AddNorm(self.self.config.arch.residual_dropout_rate)
        self.add_norm2 = AddNorm(self.self.config.arch.residual_dropout_rate)
        self.add_norm3 = AddNorm(self.self.config.arch.residual_dropout_rate)

    def causal_mask(self, batch_size: int, destination_dim: int, source_dim: int) -> jnp.ndarray:

        # Create index tensors for the source and destination dimensions
        idx_source = jnp.arange(destination_dim)[:, None]
        idx_destination = jnp.arange(source_dim)
        mask = idx_source >= idx_destination - source_dim + destination_dim
        mask = mask.astype(jnp.int32)

        # Expand dimensions to match the required output shape
        mask = mask[None, None, :, :]
        return jnp.broadcast_to(mask, (batch_size, self.num_heads, destination_dim, source_dim))

    def __call__(self, x: jnp.ndarray, context: jnp.ndarray, training: bool = False) -> tuple:

        mask = self.causal_mask(x.shape[0], x.shape[1], context.shape[1])

        attended_x, attention = self.attention(x, x)
        x = self.add_norm1(x, attended_x, training)

        attended_x, cross_attention = self.cross_attention(x, context, mask=mask)
        x = self.add_norm2(x, attended_x, training)

        linear_output = self.feed_forward(x)
        x = self.add_norm3(x, linear_output, training)

        return x, jnp.array(attention), jnp.array(cross_attention)


class WhisperTextDecoder(nn.Module):
    """
    Implements the decoder component of the Transformer model.

    The Transformer decoder generates output sequences by processing input through multiple layers of TransformerDecoderBlocks. It incorporates context from the encoder at each layer to generate predictions.
    """

    config: Config

    def setup(self):
        self.embedding = Embedding(config=self.config)

        self.decoder_blocks: list[WhisperTextDecoderBlock] = [
            WhisperTextDecoderBlock(config=self.config, name=f"whisper_decoder_block_{idx}")
            for idx in range(self.config.arch.n_layers)
        ]

        self.output_layer = nn.Dense(self.config.arch.vocab_size, name="output_layer")

    def __call__(self, x: jnp.ndarray, context: jnp.ndarray, training: bool = False) -> tuple:

        attention_maps = []
        x = self.embedding(x, training)
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, context, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)
        return self.outputs(x), jnp.array(attention_maps), jnp.array(cross_attention_maps)
