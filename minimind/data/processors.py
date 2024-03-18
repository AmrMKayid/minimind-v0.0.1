class BatchProcessor:
    """Example processor that converts a dictionary of texts into tokens."""

    def __init__(self, config, tokenizer):
        self.config = config
        assert (
            self.config.data.processor.fields != "" or self.config.data.processor.fields_from_example != ""
        ), "Either fields or fields_from_example must be specified."
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False) -> tuple[list[int], list[float], tuple]:
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.data.processor.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.data.processor.fields_from_example != "":
            fields = example[self.config.data.processor.fields_from_example].split(",")
        else:
            fields = self.config.data.processor.fields.split(",")

        for i, field in enumerate(fields):
            if field.startswith("[") and field.endswith("]"):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == "<|bos|>":
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == "<|eos|>":
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split("+")
                text = self.config.data.processor.subfield_separator.join([example[subfield] for subfield in subfields])
                if i == 0:
                    text = self.config.data.processor.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.data.processor.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux
