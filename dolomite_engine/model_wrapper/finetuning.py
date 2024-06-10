from typing import Union

import torch
import torch.distributed

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..enums import Mode
from ..utils import ProcessGroupManager
from .base import ModelWrapper


class ModelWrapperForFinetuning(ModelWrapper):
    def __init__(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs], mode: Mode):
        super().__init__(args, mode)

        assert not self.reset_attention_mask, "reset_attention_mask is only supported with pretraining"
        assert not self.reset_position_ids, "reset_position_ids is only supported with pretraining"

    def forward(self, batch: dict) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        if not self.use_padding_free_transformer:
            if self.tp_world_size > 1:
                keys = ["input_ids", "attention_mask", "labels"]

                batch_shape = [batch[keys[0]].shape if self.tp_rank == 0 else None]
                torch.distributed.broadcast_object_list(
                    batch_shape,
                    src=ProcessGroupManager.get_tensor_parallel_first_rank(),
                    group=ProcessGroupManager.get_tensor_parallel_group(),
                )
                batch_shape = batch_shape[0]

                if self.tp_rank == 0:
                    for key in keys:
                        batch[key] = batch[key].to(self.input_device)
                else:
                    batch = {
                        key: torch.empty(batch_shape, dtype=torch.long, device=torch.cuda.current_device())
                        for key in keys
                    }

                torch.distributed.broadcast
            else:
                for key in batch:
                    batch[key] = batch[key].to(self.input_device)

        model_outputs = self.model(**batch)

        loss = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs.loss
        return loss
