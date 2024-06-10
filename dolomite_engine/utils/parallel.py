import os
from datetime import timedelta
from typing import Callable

import torch
import torch.distributed
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


# general
_MESH: DeviceMesh = None
_GLOBAL_RANK: int = None
_LOCAL_RANK: int = None
_WORLD_SIZE: int = None

# tensor parallel
_TENSOR_PARALLEL_MESH: DeviceMesh = None
_TENSOR_PARALLEL_GROUP: ProcessGroup = None
_TENSOR_PARALLEL_RANK: int = None
_TENSOR_PARALLEL_WORLD_SIZE: int = None
_TENSOR_PARALLEL_FIRST_RANK: int = None

# data parallel
_DATA_PARALLEL_MESH: DeviceMesh = None
_DATA_PARALLEL_GROUP: ProcessGroup = None
_DATA_PARALLEL_RANK: int = None
_DATA_PARALLEL_WORLD_SIZE: int = None

_ZERO_HPZ_PARTITION_SIZE: int = None


class ProcessGroupManager:
    def __init__(
        self,
        tensor_parallel_size: int = None,
        data_parallel_size: int = None,
        zero_hpz_partition_size: int = None,
        timeout_minutes: int = None,
    ) -> None:
        if timeout_minutes is not None:
            timeout_minutes = timedelta(timeout_minutes)

        torch.distributed.init_process_group(
            "nccl",
            rank=ProcessGroupManager.get_global_rank(),
            world_size=ProcessGroupManager.get_world_size(),
            timeout=timeout_minutes,
        )

        if tensor_parallel_size is None:
            tensor_parallel_size = 1

        total_gpus = int(os.getenv("WORLD_SIZE", 1))

        if data_parallel_size is None:
            data_parallel_size = total_gpus // tensor_parallel_size

        assert tensor_parallel_size * data_parallel_size == total_gpus

        global _MESH, _ZERO_HPZ_PARTITION_SIZE

        _MESH = init_device_mesh(
            "cuda",
            (tensor_parallel_size, data_parallel_size),
            mesh_dim_names=("tp", "dp"),
        )

        _ZERO_HPZ_PARTITION_SIZE = zero_hpz_partition_size

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    @staticmethod
    def get_mesh() -> int:
        global _MESH
        return _MESH

    @staticmethod
    def get_global_rank() -> int:
        global _GLOBAL_RANK

        if _GLOBAL_RANK is None:
            _GLOBAL_RANK = int(os.getenv("RANK", 0))
        return _GLOBAL_RANK

    @staticmethod
    def get_local_rank() -> int:
        global _LOCAL_RANK

        if _LOCAL_RANK is None:
            _LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
        return _LOCAL_RANK

    @staticmethod
    def get_world_size() -> int:
        global _WORLD_SIZE

        if _WORLD_SIZE is None:
            _WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
        return _WORLD_SIZE

    # tensor parallel
    @staticmethod
    def get_tensor_parallel_mesh() -> DeviceMesh:
        global _TENSOR_PARALLEL_MESH

        if _TENSOR_PARALLEL_MESH is None:
            global _MESH
            _TENSOR_PARALLEL_MESH = _MESH["tp"]
        return _TENSOR_PARALLEL_MESH

    @staticmethod
    def get_tensor_parallel_group() -> ProcessGroup:
        global _TENSOR_PARALLEL_GROUP

        if _TENSOR_PARALLEL_GROUP is None:
            _TENSOR_PARALLEL_GROUP = ProcessGroupManager.get_tensor_parallel_mesh().get_group()
        return _TENSOR_PARALLEL_GROUP

    @staticmethod
    def get_tensor_parallel_rank() -> int:
        global _TENSOR_PARALLEL_RANK

        if _TENSOR_PARALLEL_RANK is None:
            _TENSOR_PARALLEL_RANK = ProcessGroupManager.get_tensor_parallel_mesh().get_local_rank()
        return _TENSOR_PARALLEL_RANK

    @staticmethod
    def get_tensor_parallel_world_size() -> int:
        global _TENSOR_PARALLEL_WORLD_SIZE

        if _TENSOR_PARALLEL_WORLD_SIZE is None:
            _TENSOR_PARALLEL_WORLD_SIZE = ProcessGroupManager.get_tensor_parallel_mesh().size()
        return _TENSOR_PARALLEL_WORLD_SIZE

    @staticmethod
    def get_tensor_parallel_first_rank() -> int:
        global _TENSOR_PARALLEL_FIRST_RANK

        if _TENSOR_PARALLEL_FIRST_RANK is None:
            group = ProcessGroupManager.get_tensor_parallel_group()
            ranks = torch.distributed.get_process_group_ranks(group)
            _TENSOR_PARALLEL_FIRST_RANK = ranks[0]
        return _TENSOR_PARALLEL_FIRST_RANK

    # data parallel
    @staticmethod
    def get_data_parallel_mesh() -> DeviceMesh:
        global _DATA_PARALLEL_MESH

        if _DATA_PARALLEL_MESH is None:
            global _MESH
            _DATA_PARALLEL_MESH = _MESH["dp"]
        return _DATA_PARALLEL_MESH

    @staticmethod
    def get_data_parallel_group() -> ProcessGroup:
        global _DATA_PARALLEL_GROUP

        if _DATA_PARALLEL_GROUP is None:
            _DATA_PARALLEL_GROUP = ProcessGroupManager.get_data_parallel_mesh().get_group()
        return _DATA_PARALLEL_GROUP

    @staticmethod
    def get_data_parallel_rank() -> int:
        global _DATA_PARALLEL_RANK

        if _DATA_PARALLEL_RANK is None:
            _DATA_PARALLEL_RANK = ProcessGroupManager.get_data_parallel_mesh().get_local_rank()
        return _DATA_PARALLEL_RANK

    @staticmethod
    def get_data_parallel_world_size() -> int:
        global _DATA_PARALLEL_WORLD_SIZE

        if _DATA_PARALLEL_WORLD_SIZE is None:
            _DATA_PARALLEL_WORLD_SIZE = ProcessGroupManager.get_data_parallel_mesh().size()
        return _DATA_PARALLEL_WORLD_SIZE

    @staticmethod
    def get_data_parallel_mesh_for_hsdp() -> DeviceMesh:
        group = ProcessGroupManager.get_data_parallel_group()
        ranks = get_process_group_ranks(group)
        ranks = torch.tensor(ranks).view(
            (_ZERO_HPZ_PARTITION_SIZE, ProcessGroupManager.get_data_parallel_world_size() // _ZERO_HPZ_PARTITION_SIZE)
        )
        return DeviceMesh("cuda", mesh=ranks, mesh_dim_names=("zero_dp", "ddp"))


def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> Callable:
    """wraps a function to run on a single rank, returns a no-op for other ranks

    Args:
        func (Callable): function to wrap
        rank (int, optional): rank on which function should run. Defaults to 0.
        barrier (bool, optional): whether to synchronize the processes at the end of function execution. Defaults to False.

    Returns:
        Callable: wrapped function
    """

    # wrapper function for the rank to execute on
    def func_rank_n(*args, **kwargs):
        output = func(*args, **kwargs)
        if barrier:
            torch.distributed.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            torch.distributed.barrier()

    global_rank = ProcessGroupManager.get_global_rank()

    if global_rank == rank:
        wrapped_func = func_rank_n
    elif global_rank is None:
        # distributed is not initialized
        wrapped_func = func
    else:
        wrapped_func = func_rank_other

    return wrapped_func
