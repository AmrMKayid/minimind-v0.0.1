from typing import Iterable, Iterator, List, TypeVar

import datasets
import numpy as np

T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


def prepare_data(config, tokenizer) -> datasets.Dataset:
    """Prepare wikitext-103 dataset for training."""
    dataset = datasets.load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train[:1%]",
        trust_remote_code=True,
    )

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.arch.max_sequence_length,
            return_tensors="np",
        )

        # Shift inputs to the right by one position
        targets = np.roll(tokenized["input_ids"], shift=-1, axis=1)
        targets[:, -1] = tokenizer.pad_token_id

        tokenized["targets"] = targets
        tokenized["inputs"] = tokenized.pop("input_ids")
        tokenized["mask"] = tokenized.pop("attention_mask")

        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset.with_format("jax").iter(batch_size=config.data.batch_size)
