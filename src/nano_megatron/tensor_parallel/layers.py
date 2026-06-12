from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nano_megatron.distributed.parallel_state import (
    get_tensor_parallel_rank,
    get_tensor_parallel_size,
)


def _ensure_divisibility(numerator: int, denominator: int) -> None:
    if numerator % denominator != 0:
        raise ValueError(
            f"{numerator} is not divisible by {denominator}. "
            f"Cannot shard dimension evenly across tensor parallel ranks."
        )


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        tp_group: Any | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = (
            get_tensor_parallel_size() if (dist.is_available() and dist.is_initialized()) else 1
        )
        self.tp_rank = (
            get_tensor_parallel_rank() if (dist.is_available() and dist.is_initialized()) else 0
        )
        self.gather_output = gather_output

        if self.tp_size > 1:
            _ensure_divisibility(out_features, self.tp_size)
            self.out_features_per_partition = out_features // self.tp_size
        else:
            self.out_features_per_partition = out_features

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self._tp_group = tp_group

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        from nano_megatron.tensor_parallel.mappings import copy_to_tensor_model_parallel_region

        input_parallel = copy_to_tensor_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output and self.tp_size > 1:
            output = _gather_along_last_dim(output_parallel, self._tp_group)
        else:
            output = output_parallel

        return output


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        tp_group: Any | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = (
            get_tensor_parallel_size() if (dist.is_available() and dist.is_initialized()) else 1
        )
        self.tp_rank = (
            get_tensor_parallel_rank() if (dist.is_available() and dist.is_initialized()) else 0
        )
        self.input_is_parallel = input_is_parallel

        if self.tp_size > 1:
            _ensure_divisibility(in_features, self.tp_size)
            self.in_features_per_partition = in_features // self.tp_size
        else:
            self.in_features_per_partition = in_features

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self._tp_group = tp_group

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        from nano_megatron.tensor_parallel.mappings import all_reduce

        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = _scatter_along_last_dim(input_, self._tp_group)

        output_parallel = F.linear(input_parallel, self.weight)

        output = all_reduce(output_parallel) if self.tp_size > 1 else output_parallel

        if self.bias is not None:
            output = output + self.bias

        return output


def _gather_along_last_dim(
    tensor: torch.Tensor,
    tp_group: Any | None = None,
) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    if tp_group is None:
        from nano_megatron.distributed.parallel_state import get_tensor_parallel_group

        tp_group = get_tensor_parallel_group()

    tp_size = dist.get_world_size(group=tp_group)
    if tp_size == 1:
        return tensor

    gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(gathered, tensor, group=tp_group)
    return torch.cat(gathered, dim=-1)


def _scatter_along_last_dim(
    tensor: torch.Tensor,
    tp_group: Any | None = None,
) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    if tp_group is None:
        from nano_megatron.distributed.parallel_state import get_tensor_parallel_group

        tp_group = get_tensor_parallel_group()

    tp_size = dist.get_world_size(group=tp_group)
    tp_rank = dist.get_rank(group=tp_group)
    if tp_size == 1:
        return tensor

    chunk_size = tensor.shape[-1] // tp_size
    return tensor[..., tp_rank * chunk_size : (tp_rank + 1) * chunk_size].contiguous()
