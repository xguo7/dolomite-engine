import os
from typing import Callable

import torch
from torch.distributed import ProcessGroup, barrier, get_rank, get_world_size
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


_LOCAL_RANK: int = None
_GLOBAL_RANK: int = None
_WORLD_SIZE: int = None
_DEVICE_MESH: DeviceMesh = None


class ProcessGroupManager:
    def __init__(self, tensor_parallel_size: int = None, data_parallel_size: int = None) -> None:
        assert get_rank() == int(os.getenv("RANK", 0))

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        if tensor_parallel_size is None:
            tensor_parallel_size = 1

        if data_parallel_size is None:
            data_parallel_size = get_world_size() // tensor_parallel_size

        global _DEVICE_MESH
        _DEVICE_MESH = init_device_mesh(
            "cuda",
            (tensor_parallel_size, data_parallel_size),
            mesh_dim_names=("tp", "dp"),
        )

    @staticmethod
    def barrier() -> None:
        barrier()

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

    @staticmethod
    def get_tensor_parallel_group() -> ProcessGroup:
        return _DEVICE_MESH["tp"]

    @staticmethod
    def get_tensor_parallel_rank() -> int:
        group = ProcessGroupManager.get_tensor_parallel_group()
        return get_rank(group)

    @staticmethod
    def get_tensor_parallel_world_size() -> int:
        group = ProcessGroupManager.get_tensor_parallel_group()
        return get_world_size(group)

    @staticmethod
    def get_data_parallel_group() -> ProcessGroup:
        return _DEVICE_MESH["dp"]

    @staticmethod
    def get_data_parallel_rank() -> int:
        group = ProcessGroupManager.get_data_parallel_group()
        return get_rank(group)

    @staticmethod
    def get_data_parallel_world_size() -> int:
        group = ProcessGroupManager.get_data_parallel_group()
        return get_world_size(group)


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
            ProcessGroupManager.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            ProcessGroupManager.barrier()

    global_rank = ProcessGroupManager.get_global_rank()

    if global_rank == rank:
        return func_rank_n
    elif global_rank is None:
        # distributed is not initialized
        return func
    else:
        return func_rank_other
