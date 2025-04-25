"""
Helpers for distributed training.
"""

import io
import os
import socket
import torch
import torch.distributed as dist

import blobfile as bf


def setup_dist(local_rank=0):
    """
    Setup a distributed process group for multi-GPU training.
    """
    if dist.is_initialized():
        return

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # 设置环境变量以初始化分布式训练
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    else:
        # 单 GPU 或 CPU 模式，不初始化分布式
        pass


def setup_single():
    """
    Setup for single GPU or CPU training (no distributed initialization).
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def dev():
    """
    Get the device to use.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks (no-op for single GPU).
    """
    if dist.is_initialized():
        for p in params:
            with torch.no_grad():
                dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()