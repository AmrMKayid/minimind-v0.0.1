from typing import Optional, Tuple

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

from minimind.config import Config

_TOKENIZERS: dict[str, PreTrainedTokenizer] = {}


def register_tokenizer(cls) -> None:
    _TOKENIZERS[cls.__name__.lower()] = cls
    return cls


@register_tokenizer
class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, config: Config, vocab_size: int = 1024, **kwargs):
        self.config = config
        self._vocab = {i: i for i in range(vocab_size)}
        self._vocab_size = vocab_size
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self):
        return self._vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        tokens = np.fromstring(text, dtype=int, sep=" ")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)


def get_tokenizer(config: Config) -> PreTrainedTokenizer:
    tokenizer_name = config.data.tokenizer.lower()
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    except:
        assert (
            tokenizer_name in _TOKENIZERS
        ), f"Tokenizer: {tokenizer_name=} is not currently supported\nSupported Tokenizers are: {list(_TOKENIZERS.keys())}"
        tokenizer = _TOKENIZERS[tokenizer_name](config=config)
    return tokenizer
