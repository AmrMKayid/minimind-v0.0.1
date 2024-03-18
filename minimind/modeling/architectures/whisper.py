"""Whisper model building blocks."""
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp

from minimind.config import Config
from minimind.modeling.architectures import register_architecture
from minimind.modeling.modules.whisper_blocks import WhisperSpeechEncoder, WhisperTextDecoder


@register_architecture
class Whisper(nn.Module):
    """Implements the Whisper model for speech-to-text tasks, such as speech
    recognition and transcription."""

    config: Config

    def setup(self):
        self.encoder = WhisperSpeechEncoder(config=self.config)
        self.decoder = WhisperTextDecoder(config=self.config)

    def __call__(self, batch: Dict[str, jax.Array], training: bool = False) -> jnp.ndarray:
        batch.get("audio", jnp.ones((8, 32, 8)))
        y = batch.get("text", jnp.ones((8, 32, 8)))
        batch = self.encoder(batch=batch, training=training)  # TODO:
        return self.decoder(x=y, context=batch["x"], training=training)[0]  # TODO:
