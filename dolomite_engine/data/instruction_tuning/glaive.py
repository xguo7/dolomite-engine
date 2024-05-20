from typing import List

from datasets import load_dataset

from ...enums import DatasetSplit
from .base import BaseInstructionDataset


class GlaiveCodeAssistantDataset(BaseInstructionDataset):
    def construct_input_from_format(self, input: str) -> List[int]:
        input_text = f"input: {input}\noutput:"
        return input_text

    def prepare_examples(self) -> List[dict]:
        if self.split != DatasetSplit.train:
            return []

        data = load_dataset("glaiveai/glaive-code-assistant-v3")["train"]

        examples = []
        for raw_example in data:
            input = self.construct_input_from_format(raw_example["question"].strip())
            output = self.construct_output_from_format(raw_example["answer"].strip())

            example = self.get_input_output_token_ids(input, output)
            examples.append(example)

        return examples
