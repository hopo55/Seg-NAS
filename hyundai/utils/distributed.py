"""
Distributed training utilities for DistributedDataParallel (DDP).

This module provides helper functions for initializing and managing
distributed training across multiple GPUs.
"""

import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """
    Initialize distributed training mode.

    Automatically detects if running under torchrun by checking environment
    variables (RANK, WORLD_SIZE, LOCAL_RANK). If detected, initializes the
    distributed process group. Otherwise, sets up for single-GPU or CPU training.

    Args:
        args: Argument namespace. Will be modified to add:
            - distributed (bool): Whether distributed training is enabled
            - rank (int): Global rank of current process
            - world_size (int): Total number of processes
            - local_rank (int): Local rank on current node
            - gpu (int): GPU device ID for current process
    """
    # Check for torchrun environment variables
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running under torchrun - distributed mode
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.gpu = args.local_rank
        args.distributed = True

        print(f"Distributed training: rank {args.rank}/{args.world_size}, local_rank {args.local_rank}")
    else:
        # Not running under torchrun - single GPU or CPU mode
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.gpu = 0 if hasattr(args, 'gpu_idx') and len(args.gpu_idx) > 0 else 0
        args.distributed = False

        print("Single-process training mode")
        return

    # Initialize distributed process group
    if args.distributed:
        # Set device for this process
        torch.cuda.set_device(args.local_rank)

        # Get backend from args or use default
        backend = getattr(args, 'dist_backend', 'nccl')
        init_method = getattr(args, 'dist_url', 'env://')

        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=args.world_size,
            rank=args.rank
        )

        # Wait for all processes to initialize
        dist.barrier()

        if args.rank == 0:
            print(f"Distributed process group initialized:")
            print(f"  Backend: {backend}")
            print(f"  World size: {args.world_size}")
            print(f"  Init method: {init_method}")


def is_distributed():
    """
    Check if distributed training is enabled and properly initialized.

    Returns:
        bool: True if distributed training is active, False otherwise
    """
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """
    Get the rank of the current process.

    Returns:
        int: Rank of current process (0 if not distributed)
    """
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Get the total number of processes in the distributed training.

    Returns:
        int: Number of processes (1 if not distributed)
    """
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank():
    """
    Get the local rank of the current process on this node.

    Returns:
        int: Local rank (0 if not distributed or from LOCAL_RANK env var)
    """
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return 0


def is_main_process():
    """
    Check if this is the main process (rank 0).

    Used to ensure only one process performs I/O operations like
    saving checkpoints, logging to wandb, or printing progress.

    Returns:
        bool: True if rank 0 or not distributed, False otherwise
    """
    return get_rank() == 0


def cleanup_distributed():
    """
    Clean up the distributed process group.

    Should be called at the end of training when using DDP.
    """
    if is_distributed():
        dist.destroy_process_group()


def reduce_dict(input_dict, average=True):
    """
    Reduce a dictionary of tensors across all processes.

    Useful for aggregating metrics (loss, accuracy, etc.) computed
    independently on each GPU.

    Args:
        input_dict (dict): Dictionary with tensor values to reduce
        average (bool): If True, average the values. If False, sum them.

    Returns:
        dict: Dictionary with reduced values
    """
    if not is_distributed():
        return input_dict

    world_size = get_world_size()
    if world_size == 1:
        return input_dict

    # Ensure all processes reach this point
    dist.barrier()

    # Reduce each value in the dictionary
    reduced_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            # Clone to avoid modifying original
            reduced_value = value.clone()
            dist.all_reduce(reduced_value)
            if average:
                reduced_value = reduced_value / world_size
            reduced_dict[key] = reduced_value
        else:
            # For non-tensors, just copy
            reduced_dict[key] = value

    return reduced_dict


def synchronize():
    """
    Synchronize all processes.

    Helper function to ensure all processes reach the same point
    before continuing.
    """
    if not is_distributed():
        return

    world_size = get_world_size()
    if world_size == 1:
        return

    dist.barrier()
