from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from minimind.config import Config
from minimind.modeling.architectures import register_architecture
from minimind.modeling.modules.clib_blocks import CLIPImageEncoder, CLIPTextEncoder


@register_architecture
class CLIP(nn.Module):
    """CLIP (Contrastive Language-Image Pretraining) is designed to understand
    and connect vision and language."""

    config: Config

    def setup(self):
        self.text_encoder = CLIPTextEncoder(config=self.config, name="text_encoder")
        self.image_encoder = CLIPImageEncoder(config=self.config, name="image_encoder")
        self.text_pooler = nn.Dense(self.config.arch.embedding_dim, name="text_pooler")
        self.image_pooler = nn.Dense(self.config.arch.embedding_dim, name="image_pooler")
        self.temperature = self.param("temperature", nn.initializers.zeros, ())

    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        texts, images = batch.get("inputs"), batch.get("images")
        text_latents = self.text_encoder({"inputs": texts}, training=training)["x"]
        image_latents = self.image_encoder({"inputs": images}, training=training)["x"]
        text_embedding = self.text_pooler(jnp.mean(text_latents, axis=1))
        image_embedding = self.image_pooler(jnp.mean(image_latents, axis=1))
        return text_embedding, image_embedding, self.clip_loss(text_embedding, image_embedding)

    def clip_loss(self, text_embeddings: jnp.ndarray, image_embeddings: jnp.ndarray) -> float:
        def l2_normalise(x: jnp.ndarray) -> jnp.ndarray:
            return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        def cross_entropy(preds: jax.Array, targets: jax.Array) -> jax.Array:
            return (-targets * jax.nn.log_softmax(preds)).sum(axis=1).mean()

        text_embeddings = l2_normalise(text_embeddings)
        image_embeddings = l2_normalise(image_embeddings)
        similarity_matrix = image_embeddings @ text_embeddings.T / (self.temperature + 0.00001)
        labels = jnp.arange(similarity_matrix.shape[0])
        image_loss = cross_entropy(similarity_matrix, labels)
        text_loss = cross_entropy(similarity_matrix.T, labels)

        return (image_loss + text_loss) / 2

    def get_attention_maps(self, texts: jnp.ndarray, images: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        _, text_attention = self.text_encoder(texts, training=False)
        _, image_attention = self.image_encoder(images, training=False)
        return text_attention, image_attention

    def encode_text(self, texts: jnp.ndarray) -> jnp.ndarray:
        return self.text_encoder(texts)[0]

    def encode_image(self, images: jnp.ndarray) -> jnp.ndarray:
        return self.image_encoder(images)[0]

    def embed_text(self, texts: jnp.ndarray) -> jnp.ndarray:
        return self.text_pooler(jnp.mean(self.text_encoder(texts)[0], axis=1))

    def embed_image(self, images: jnp.ndarray) -> jnp.ndarray:
        return self.image_pooler(jnp.mean(self.image_encoder(images)[0], axis=1))
