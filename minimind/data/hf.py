from typing import Iterator

import datasets
import numpy as np

from minimind.config import Config
from minimind.data.dataset import Dataset
from minimind.data.processors import BatchProcessor
from minimind.data.tokenizers import get_tokenizer


class HuggingFaceDataset(Dataset):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self._tokenizer = get_tokenizer(config)
        self._processor = BatchProcessor(config, tokenizer=self._tokenizer)
        self._dataset = datasets.load_dataset(
            path=self.config.data.path,
            name=self.config.data.name,
            split=self.config.data.split,
            streaming=self.config.data.streaming,
            trust_remote_code=True,
        )

    def __iter__(self) -> Iterator:
        chunk_size = self.config.data.batch_size * self.config.arch.max_sequence_length
        total_tokens = 0
        while True:
            token_buffer = []
            attention_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, attention_mask = self._processor(example)
                token_buffer.extend(tokens)
                attention_mask_buffer.extend(attention_mask)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    batch = {
                        "inputs": np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.data.batch_size, -1
                        ),
                        "targets": np.array(token_buffer[1 : chunk_size + 1], dtype=np.int32).reshape(
                            self.config.data.batch_size, -1
                        ),
                        "mask": np.array(attention_mask_buffer[1 : chunk_size + 1], dtype=np.float32).reshape(
                            self.config.data.batch_size, -1
                        ),
                        "dataset_example_index": index,
                        "dataset_total_tokens": total_tokens,
                    }
                    yield batch
                    token_buffer = token_buffer[chunk_size:]
                    attention_mask_buffer = attention_mask_buffer[chunk_size:]

    @property
    def sequence_length(self):
        return self.config.data.sequence_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)
