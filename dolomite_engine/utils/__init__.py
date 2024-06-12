import logging

import torch
import torch.distributed

from .hf_hub import download_repo
from .logging import log_rank_0, print_rank_0, print_ranks_all, set_logger
from .mixed_precision import normalize_dtype_string, string_to_torch_dtype, torch_dtype_to_string
from .packages import (
    is_apex_available,
    is_deepspeed_available,
    is_flash_attention_available,
    is_ms_amp_available,
    is_transformer_engine_available,
    is_triton_available,
)
from .parallel import ProcessGroupManager, run_rank_n
from .pydantic import BaseArgs
from .random import CUDA_RNGStatesTracker, get_cuda_rng_tracker, set_cuda_rng_tracker
from .safetensors import SafeTensorsWeightsManager
from .tracking import ExperimentsTracker, ProgressBar, RunningMean
from .wrapper import get_module_class_from_name
from .yaml import load_yaml


def init_distributed(
    tensor_parallel_size: int, data_parallel_size: int, zero_hpz_partition_size: int, timeout_minutes: int = None
) -> None:
    """intialize distributed

    Args:
        tensor_parallel_size (int): tensor parallel size
        data_parallel_size (int): data parallel size
        zero_hpz_partition_size (int): HSDP size
        timeout_minutes (int, optional): distributed timeout in minutes. Defaults to None.
    """

    process_group_manager = ProcessGroupManager(
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        zero_hpz_partition_size=zero_hpz_partition_size,
        timeout_minutes=timeout_minutes,
    )

    log_rank_0(logging.INFO, process_group_manager)


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
