import jax
import optax

from minimind.metrics import Metric


class CrossEntropyLoss(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, logits: jax.Array, labels: jax.Array) -> jax.Array:
        labels = jax.nn.one_hot(labels, logits.shape[-1])  # (batch, seq, vocab_size)
        return optax.softmax_cross_entropy(logits, labels)
