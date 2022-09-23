import torch.distributed as dist
import torch

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size


def average_params(params, is_distributed):
    """ parameter averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= size


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def gather(tensor, nprocs):
    tensor_gather = [torch.zeros_like(tensor) for _ in range(nprocs)]
    dist.all_gather(tensor_gather, tensor)
    return tensor_gather