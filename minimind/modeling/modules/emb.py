from typing import Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from minimind.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """Implements the positional encoding layer for adding positional
    information to embeddings in a transformer model."""

    config: Config

    def setup(self):
        self.num_embeddings: int = self.config.arch.max_pos_emb_length  # type: ignore
        self.embedding_dim: int = self.config.arch.embedding_dim  # type: ignore

        positional_encoding = jnp.zeros((self.embedding_dim, self.num_embeddings))
        position = jnp.arange(0, self.embedding_dim, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.num_embeddings, 2) * (-jnp.log(10000.0) / self.num_embeddings))
        positional_encoding = positional_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        positional_encoding = positional_encoding.at[:, 1::2].set(jnp.cos(position * div_term))
        self.positional_encoding = positional_encoding.T

    def __call__(self, x: jax.Array):
        x = x + self.positional_encoding[: x.shape[1]]
        return x


class RotaryPositionalEncoding(nn.Module):
    """Implements rotary positional encoding (RoPE) for transformers, enhancing
    their ability to capture sequence order."""

    config: Config

    def setup(self):
        self.embedding_dim = self.config.arch.embedding_dim
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.embedding_dim, 2, dtype=jnp.float32) / self.embedding_dim))
        self.inv_freq = inv_freq
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x: jax.Array, seq_dimension: int = 1):
        seq_len = x.shape[seq_dimension]

        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = jnp.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = jnp.outer(t, self.inv_freq)
            emb = jnp.concatenate((freqs, freqs), axis=-1)
            self._cos_cached = jnp.cos(emb)[None, None, :, :]
            self._sin_cached = jnp.sin(emb)[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def rotate_half(self, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, :, : x.shape[-2], :]
        sin = sin[:, :, : x.shape[-2], :]
        return (x * cos) + (self.rotate_half(x) * sin)

    def __call__(self, q, k):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)[0],
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)[0],
        )


class Embedding(nn.Module):
    """A module that represents an embedding layer.

    Args:
        config (Config): The configuration object.
        dtype (jnp.dtype, optional): The data type of the layer's parameters. Defaults to jnp.float32.
        param_dtype (jnp.dtype, optional): The data type of the layer's parameters. Defaults to jnp.float32.
        precision (Optional[Union[jax.lax.Precision, str]], optional): The precision of the layer's computation. Defaults to None.

    Attributes:
        wte (nn.Embed): The word token embedding layer.
        positional_embedding_type (str): The type of positional embedding.
        wpe (nn.Embed or PositionalEncodingEmbed): The positional embedding layer.

    Methods:
        setup(): Sets up the embedding layer.
        __call__(batch: Dict, training: bool, attend: bool = False) -> Dict[str, jax.Array]: Computes the forward pass of the embedding layer.
    """

    config: Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.wte = nn.Embed(
            num_embeddings=self.config.arch.vocab_size,
            features=self.config.arch.embedding_dim,
            embedding_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=self.config.arch.initializer_range),
                ((self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis), self.config.mesh.tensor_axis),
            ),
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            name="wte",
        )

        self.positional_embedding_type = self.config.arch.positional_embedding_type
        self.wpe = None
        match self.positional_embedding_type:
            case "learned":
                self.wpe = nn.Embed(
                    num_embeddings=self.config.arch.max_pos_emb_length,
                    features=self.config.arch.embedding_dim,
                    embedding_init=nn.with_partitioning(
                        jax.nn.initializers.normal(stddev=self.config.arch.initializer_range),
                        (self.config.mesh.fsdp_axis, self.config.mesh.tensor_axis),
                    ),
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                    name="wpe",
                )
            case "sinusoidal":
                self.wpe = SinusoidalPositionalEncoding(
                    config=self.config,
                    name="wpe",
                )
            case _:
                raise ValueError(f"Unknown positional embedding type: {self.positional_embedding_type}")

    def __call__(
        self,
        batch: Dict,
        training: bool,
        attend: bool = False,
    ) -> Dict[str, jax.Array]:
        """Computes the forward pass of the embedding layer.

        Args:
            batch (Dict): The input batch.
            training (bool): Whether the model is in training mode.
            attend (bool, optional): Whether to compute attention logits. Defaults to False.

        Returns:
            Dict[str, jax.Array]: The output batch.
        """
        if attend:
            inputs = batch.pop("x")  # [batch, seq, emb_dim]
            logits = self.wte.attend(inputs)  # [batch, seq, vocab_size]
            logits = jax.lax.with_sharding_constraint(
                logits, P(self.config.mesh.data_mesh, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis)
            )
            batch.update({"logits": logits})
        else:
            inputs = batch.get("inputs").astype("i4")  # [batch, seq]
            inputs = jnp.minimum(inputs, self.config.arch.vocab_size - 1)  # just for safety/debugging

            embeddings = self.wte(inputs)  # [batch, seq, emb_dim]
            match self.positional_embedding_type:
                case "learned":
                    positions_embeddings = self.wpe(jnp.arange(0, stop=inputs.shape[-1], step=1))  # type: ignore
                    embeddings += positions_embeddings  # [batch, seq, emb_dim]
                case "sinusoidal":
                    embeddings += self.wpe(embeddings)  # type: ignore
                case _:
                    raise ValueError(f"Unknown positional embedding type: {self.positional_embedding_type}")

            embeddings = jax.lax.with_sharding_constraint(
                embeddings, P(self.config.mesh.data_mesh, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis)
            )
            batch.update({"x": embeddings})
        return batch


class PatchEmbedding(nn.Module):
    """Implements patch embedding for vision transformers.

    This module extracts patches from input images, flattens them, and projects them to a specified embedding dimension. Optionally, learned position embeddings can be added to the patch embeddings.

    Attributes:
        patch_size (tuple): Size (height, width) of the patches to extract from input images.
        embed_dim (int): Dimension of the embeddings for the patches.

    Methods:
        __call__(x: jnp.ndarray): Extracts patches from the input images and applies patch embedding.
        extract_patches(images: jnp.ndarray): Extracts and flattens patches from input images.
    """

    config: Config

    def setup(self) -> None:
        self.patch_size: Tuple[int, int] = self.config.arch.patch_size
        self.embedding_dim: int = self.config.arch.embedding_dim

    @nn.compact
    def __call__(
        self,
        batch: Dict,
        training: bool,
    ) -> Dict[str, jax.Array]:
        x = batch.get("inputs")
        x = nn.Dense(self.embedding_dim)(self.extract_patches(x))
        x = x + nn.Embed(num_embeddings=x.shape[1], features=x.shape[2])(jnp.arange(x.shape[1]))
        batch.update({"x": x})
        return batch

    def extract_patches(self, images: jnp.ndarray) -> jnp.ndarray:
        if len(images.shape) != 4:
            raise ValueError(f"Input images {images.shape=} should have shape (batch_size, H, W, C)")

        batch_size, h, w, c = images.shape
        ph, pw = self.patch_size

        if h % ph != 0 or w % pw != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")

        # Calculate the number of patches in each dimension
        num_patches_h = h // ph
        num_patches_w = w // pw

        # Reshape the images into patches and flatten each patch
        patches = jnp.reshape(images, (batch_size, num_patches_h, ph, num_patches_w, pw, c))
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = jnp.reshape(patches, (batch_size, -1, ph * pw * c))
        return patches


class SpeechEmbedding(nn.Module):
    """Implements a speech embedding layer for processing audio signals.

    This layer applies two convolutional operations followed by GELU activations to the input audio signals. The first convolution maintains the sequence length, while the second halves it. Additionally, it adds sinusoidal embeddings to capture positional information within the audio sequence.

    Methods:
        __call__(x): Processes the input audio tensor through the convolutional layers and adds sinusoidal embeddings.
        sinusoidal_embedding(x, max_position): Generates sinusoidal embeddings based on the sequence length and hidden dimension of the input.
    """

    config: Config

    @nn.compact
    def __call__(self, batch: Dict, training: bool = False) -> jnp.ndarray:
        x = batch.get("inputs")
        x = nn.gelu(nn.Conv(features=x.shape[-1], kernel_size=(3,), padding="SAME")(x))
        x = nn.gelu(nn.Conv(features=x.shape[-1], kernel_size=(3,), strides=(2,), padding="SAME")(x))
        x = jnp.concatenate((x, self.sinusoidal_embedding(x)), axis=-2)
        batch.update({"x": x})
        return batch

    def sinusoidal_embedding(self, x: jnp.ndarray, max_position: int = 10000) -> jnp.ndarray:
        batch_size, seq_len, hidden_dim = x.shape
        positions = jnp.arange(seq_len)[:, None]
        angles = (jnp.arange(hidden_dim) / hidden_dim)[None, :]
        encodings = jnp.sin(positions / jnp.power(max_position, angles))[None, :, :]
        encodings = jnp.repeat(encodings, batch_size, axis=0)
        return x + encodings
