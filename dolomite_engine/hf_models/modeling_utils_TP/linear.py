import torch
import torch.distributed
import torch.nn as nn

from ...utils import ProcessGroupManager, SafeTensorsWeightsManager
from ..modeling_utils import ParameterizedLinear
from .TP import (
    CopyToTensorParallelRegion,
    ReduceFromTensorParallelRegion,
    tensor_parallel_all_gather,
    tensor_parallel_split_safetensor_slice,
)


class ColumnParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        assert (
            out_features % tp_world_size == 0
        ), f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})"

        self.out_features_per_device = out_features // tp_world_size

        super().__init__(
            in_features=in_features,
            out_features=self.out_features_per_device,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = CopyToTensorParallelRegion.apply(input)
        input = super().forward(input)
        return input

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=0)
        state_dict = {"weight": weight}

        if self.bias is not None:
            bias = safetensors_weight_manager.get_slice(prefix + "bias")
            bias = tensor_parallel_split_safetensor_slice(bias, dim=0)
            state_dict["bias"] = bias

        self.load_state_dict(state_dict)

    def extra_repr(self) -> str:
        return "in_features={}, out_features_per_device={}, bias={}".format(
            self.in_features, self.out_features_per_device, self.bias is not None
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = tensor_parallel_all_gather(self.weight, dim=0)
        destination[prefix + "bias"] = tensor_parallel_all_gather(self.bias, dim=0)


class RowParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
    ) -> None:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        assert (
            in_features % self.tp_world_size == 0
        ), f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})"

        self.in_features_per_device = in_features // self.tp_world_size
        self.out_features = out_features

        super().__init__(
            in_features=self.in_features_per_device,
            out_features=out_features,
            bias=False,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.tp_bias = None
        if bias:
            self.tp_bias = nn.Parameter(torch.empty(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = super().forward(input)
        input = ReduceFromTensorParallelRegion.apply(input)
        if self.tp_bias is not None:
            input = input + self.tp_bias
        return input

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=1)
        state_dict = {"weight": weight}

        if self.tp_bias is not None:
            state_dict["tp_bias"] = safetensors_weight_manager.get_tensor(prefix + "bias")

        self.load_state_dict(state_dict)

    def extra_repr(self) -> str:
        return "in_features_per_device={}, out_features={}, bias={}".format(
            self.in_features_per_device, self.out_features, self.tp_bias is not None
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = tensor_parallel_all_gather(self.weight, dim=1)
        destination[prefix + "bias"] = self.bias
