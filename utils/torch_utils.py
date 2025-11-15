"""
References:
- torchscatter: https://github.com/rusty1s/pytorch_scatter/tree/master
"""

from functools import partial
from typing import Iterable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils
import torch.utils.checkpoint
from contextlib import nullcontext

__all__ = [
    # torch training
    'autocasting_disable_decorator',
    'adaptive_gradient_clipping',
    'lambda_warmup_linear',
    
    # torch tensor op
    'expand_at_dim',
    'permute_final_dims',
    'flatten_final_dims',
    'pad_at_dim',
    'reshape_at_dim',
    'move_final_dim_to_dim',
    'loss_reduction',
    'scatter'
]


def adaptive_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    clip_factor: float = 0.01,
    eps: float = 1e-3
):
    """Adaptive Gradient Clipping (AGC) as used in NFNet.
    """
    for p in parameters:
        if p.grad is None: continue
        param_norm = p.data.norm(2).clamp_(min=eps)
        
        # |grad| <= |param| * clip_factor
        grad_norm = p.grad.data.norm(2)
        max_grad_norm = param_norm * clip_factor
        
        # clip gradient if necessary
        trigger = grad_norm / max_grad_norm
        if trigger > 1:
            p.grad.data.div_(trigger)

def lambda_warmup_linear(curr_step: int, warmup_steps: int) -> float:
    if curr_step < warmup_steps:
        return float(curr_step) / float(max(1, warmup_steps))
    return 1.0

def autocasting_disable_decorator(disable_casting):
    # Force torch.float32 if autocasting is disabled
    def func_wrapper(func):
        def new_func(*args, **kwargs):
            _amp_context = (
                torch.autocast(device_type="cuda", enabled=False)
                if disable_casting
                else nullcontext()
            )
            dtype = torch.float32 if disable_casting else None
            with _amp_context:
                return func(
                    *(
                        v.to(dtype=dtype) if isinstance(v, torch.Tensor) else v
                        for v in args
                    ),
                    **{
                        k: v.to(dtype=dtype) if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items()
                    },
                )
        return new_func
    return func_wrapper

def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)

def permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    """Permute final dims of tensor

    Args:
        tensor (torch.Tensor): the input tensor
            [...]
        inds (List[int]): the dim to permute

    Returns:
        torch.Tensor: the permuted tensor
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Flatten final dims of tensor

    Args:
        t (torch.Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        torch.Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))

def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[Tuple[int, ...], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)

def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[Tuple[int, ...], list[int]]
) -> torch.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)

def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)

def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)

def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    if reduce == "sum" or reduce == "add":
        return _scatter_sum(src, index, dim, out, dim_size)
    if reduce == "mul":
        return _scatter_mul(src, index, dim, out, dim_size)
    elif reduce == "mean":
        return _scatter_mean(src, index, dim, out, dim_size)
    elif reduce == "min":
        return _scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == "max":
        return _scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError

def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def _scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def _scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = _scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = _scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out

def _scatter_mul(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size) # type: ignore

def _scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size) # type: ignore

def _scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size) # type: ignore
