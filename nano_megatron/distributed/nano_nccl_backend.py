from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nano_megatron.distributed.routed_backend import RoutedCommBackend
    from nano_megatron.parallel.context import ParallelContext


_DTYPES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}
_REDUCE_OPS = {"sum": 0, "avg": 1, "max": 2, "min": 3}
_TRANSPORTS = {
    "auto": 0,
    "shm": 1,
    "p2p": 2,
    "socket": 3,
    "rdma": 4,
}
_TRANSPORT_NAMES = {
    0: "auto",
    1: "shm",
    2: "p2p",
    3: "socket",
    4: "rdma",
    5: "mixed",
}
_GROUP_NAMES = ("tp", "dp", "cp", "dp-cp")
ParallelGroupName = Literal["tp", "dp", "cp", "dp-cp"]


class _MpiSubgroupConfig(ctypes.Structure):
    _fields_ = [
        ("color", ctypes.c_int),
        ("key", ctypes.c_int),
        ("device", ctypes.c_int),
        ("transport", ctypes.c_int32),
    ]


class _AllReduceArgs(ctypes.Structure):
    _fields_ = [
        ("send_buffer", ctypes.c_void_p),
        ("recv_buffer", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        ("dtype", ctypes.c_int32),
        ("redop", ctypes.c_int32),
    ]


class _ReduceScatterArgs(ctypes.Structure):
    _fields_ = [
        ("send_buffer", ctypes.c_void_p),
        ("recv_buffer", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),
        ("recv_count", ctypes.c_size_t),
        ("dtype", ctypes.c_int32),
        ("redop", ctypes.c_int32),
    ]


class _AllGatherArgs(ctypes.Structure):
    _fields_ = [
        ("send_buffer", ctypes.c_void_p),
        ("recv_buffer", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),
        ("send_count", ctypes.c_size_t),
        ("dtype", ctypes.c_int32),
    ]


class _RecvBufferPool:
    """Reuse out-of-place and packing storage after its work is consumed."""

    def __init__(self) -> None:
        self._available: dict[
            tuple[torch.device, torch.dtype, tuple[int, ...]], list[Tensor]
        ] = {}

    @staticmethod
    def _key(tensor: Tensor) -> tuple[torch.device, torch.dtype, tuple[int, ...]]:
        return tensor.device, tensor.dtype, tuple(tensor.shape)

    def acquire(self, template: Tensor) -> Tensor:
        return self.acquire_shape(template.shape, template.dtype, template.device)

    def acquire_shape(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        normalized_shape = tuple(int(size) for size in shape)
        key = (device, dtype, normalized_shape)
        available = self._available.get(key)
        if available:
            return available.pop()
        return torch.empty(normalized_shape, dtype=dtype, device=device)

    def acquire_flat(self, template: Tensor, count: int) -> Tensor:
        return self.acquire_shape((count,), template.dtype, template.device)

    def release(self, buffer: Tensor) -> None:
        self._available.setdefault(self._key(buffer), []).append(buffer)

    def clear(self) -> None:
        self._available.clear()

    @property
    def cached_buffer_count(self) -> int:
        return sum(len(buffers) for buffers in self._available.values())


class _NanoNcclWork:
    def __init__(
        self,
        backend: NanoNcclBackend,
        event: torch.cuda.Event,
        release_buffers: Iterable[Tensor] = (),
    ) -> None:
        self._backend = backend
        self._event = event
        self._release_buffers = list(release_buffers)
        self._waited = False

    def wait(self) -> bool:
        if not self._waited:
            torch.cuda.current_stream(self._backend.device).wait_event(self._event)
            self._backend.check_async_error()
            for buffer in self._release_buffers:
                self._backend._buffer_pool.release(buffer)
            self._release_buffers.clear()
            self._waited = True
        return True

    def is_completed(self) -> bool:
        return bool(self._event.query())


class NanoNcclBackend:
    """One-process-per-GPU adapter for Nano NCCL C ABI version 2.

    One instance owns one Nano communicator and therefore accepts exactly one
    PyTorch process-group object. Collectives enqueue on a dedicated CUDA stream;
    the returned work inserts the consumer-stream dependency at ``wait()``.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        subgroup_color: int,
        subgroup_rank: int,
        subgroup_size: int,
        group: Any,
        group_name: str,
        transport: str = "rdma",
        device: torch.device | str | int | None = None,
        expected_channels: int = 4,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NanoNcclBackend requires CUDA")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"NanoNcclBackend requires a CUDA device, got {self.device}"
            )
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        assert self.device.index is not None
        if not 0 <= self.device.index < torch.cuda.device_count():
            raise ValueError(
                f"configured CUDA device {self.device.index} is not visible; "
                f"visible device count={torch.cuda.device_count()}"
            )
        if transport not in _TRANSPORTS:
            raise ValueError(f"unsupported Nano NCCL transport: {transport}")
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Nano NCCL C adapter not found: {path}")

        self._lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self._configure_signatures()
        abi_version = self._lib.nano_nccl_abi_version()
        if abi_version != 2:
            raise RuntimeError(
                f"Nano NCCL C ABI version mismatch: expected 2, got {abi_version}"
            )
        self._handle = ctypes.c_void_p()
        self._group = group
        self.group_name = group_name
        self._closed = False
        self._buffer_pool = _RecvBufferPool()
        # Preserve the DP adapter's test/debug attribute while broadening usage.
        self._recv_buffer_pool = self._buffer_pool
        self._stream = torch.cuda.Stream(device=self.device)

        config = _MpiSubgroupConfig(
            subgroup_color,
            subgroup_rank,
            self.device.index,
            _TRANSPORTS[transport],
        )
        status = self._lib.nano_nccl_create_mpi_subgroup_communicator(
            ctypes.byref(config), ctypes.byref(self._handle)
        )
        self._raise_native_error(status, f"create {group_name} communicator")
        try:
            local_size = self._query(
                self._lib.nano_nccl_local_rank_count, self._handle
            )
            if local_size != 1:
                raise RuntimeError(
                    f"Nano NCCL process owns {local_size} ranks, expected one"
                )
            native_size = self._query(
                self._lib.nano_nccl_global_rank_count, self._handle
            )
            if native_size != subgroup_size:
                raise RuntimeError(
                    f"Nano NCCL library has {native_size} ranks, {group_name} "
                    f"subgroup has {subgroup_size}; rebuild with "
                    f"-DNANO_NCCL_NRANKS={subgroup_size}"
                )
            channels = self._query(self._lib.nano_nccl_mpi_channel_count)
            if channels != expected_channels:
                raise RuntimeError(
                    f"Nano NCCL library has {channels} channels, expected "
                    f"{expected_channels}"
                )
            resolved_value = self._query(
                self._lib.nano_nccl_transport, self._handle
            )
            self.transport = _TRANSPORT_NAMES.get(
                resolved_value, f"unknown({resolved_value})"
            )
            self.edge_transports = tuple(
                _TRANSPORT_NAMES.get(
                    self._query(
                        self._lib.nano_nccl_edge_transport,
                        self._handle,
                        edge,
                    ),
                    "unknown",
                )
                for edge in range(native_size)
            )
            self.channel_count = channels
            self.group_size = native_size
        except Exception:
            self.close()
            raise

    @classmethod
    def from_parallel_context(
        cls,
        ctx: ParallelContext,
        library_path: str | Path,
        *,
        group_name: ParallelGroupName = "dp-cp",
        transport: str = "rdma",
        expected_channels: int = 4,
    ) -> NanoNcclBackend:
        from nano_megatron.parallel.rank_generator import RankGenerator

        if group_name not in _GROUP_NAMES:
            raise ValueError(
                f"unknown parallel group {group_name!r}; expected one of "
                f"{_GROUP_NAMES}"
            )
        group_by_name = {
            "tp": ctx.tensor_parallel_group,
            "dp": ctx.data_parallel_group,
            "cp": ctx.context_parallel_group,
            "dp-cp": ctx.data_context_parallel_group,
        }
        generator = RankGenerator(
            tp=ctx.tensor_parallel_size,
            dp=ctx.data_parallel_size,
            pp=ctx.pipeline_parallel_size,
            cp=ctx.context_parallel_size,
            order=ctx.config.order,
        )
        groups = generator.get_ranks(group_name)
        for color, ranks in enumerate(groups):
            if ctx.rank in ranks:
                return cls(
                    library_path,
                    subgroup_color=color,
                    subgroup_rank=ranks.index(ctx.rank),
                    subgroup_size=len(ranks),
                    group=group_by_name[group_name],
                    group_name=group_name,
                    transport=transport,
                    device=torch.device("cuda", torch.cuda.current_device()),
                    expected_channels=expected_channels,
                )
        raise RuntimeError(
            f"rank {ctx.rank} is missing from {group_name} groups {groups}"
        )

    def _configure_signatures(self) -> None:
        handle = ctypes.c_void_p
        status = ctypes.c_int32
        int_output = ctypes.POINTER(ctypes.c_int)
        self._lib.nano_nccl_abi_version.argtypes = []
        self._lib.nano_nccl_abi_version.restype = ctypes.c_uint32
        self._lib.nano_nccl_get_last_error.argtypes = []
        self._lib.nano_nccl_get_last_error.restype = ctypes.c_char_p
        self._lib.nano_nccl_create_mpi_subgroup_communicator.argtypes = [
            ctypes.POINTER(_MpiSubgroupConfig),
            ctypes.POINTER(handle),
        ]
        self._lib.nano_nccl_create_mpi_subgroup_communicator.restype = status
        self._lib.nano_nccl_all_reduce.argtypes = [
            handle,
            ctypes.POINTER(_AllReduceArgs),
        ]
        self._lib.nano_nccl_all_reduce.restype = status
        self._lib.nano_nccl_reduce_scatter.argtypes = [
            handle,
            ctypes.POINTER(_ReduceScatterArgs),
        ]
        self._lib.nano_nccl_reduce_scatter.restype = status
        self._lib.nano_nccl_all_gather.argtypes = [
            handle,
            ctypes.POINTER(_AllGatherArgs),
        ]
        self._lib.nano_nccl_all_gather.restype = status
        self._lib.nano_nccl_check_async_error.argtypes = [handle]
        self._lib.nano_nccl_check_async_error.restype = status
        self._lib.nano_nccl_local_rank_count.argtypes = [handle, int_output]
        self._lib.nano_nccl_local_rank_count.restype = status
        self._lib.nano_nccl_global_rank_count.argtypes = [handle, int_output]
        self._lib.nano_nccl_global_rank_count.restype = status
        self._lib.nano_nccl_mpi_channel_count.argtypes = [int_output]
        self._lib.nano_nccl_mpi_channel_count.restype = status
        self._lib.nano_nccl_transport.argtypes = [handle, int_output]
        self._lib.nano_nccl_transport.restype = status
        self._lib.nano_nccl_edge_transport.argtypes = [
            handle,
            ctypes.c_int,
            int_output,
        ]
        self._lib.nano_nccl_edge_transport.restype = status
        self._lib.nano_nccl_destroy_communicator.argtypes = [handle]
        self._lib.nano_nccl_destroy_communicator.restype = status

    def _raise_native_error(self, status: int, operation: str) -> None:
        if status == 0:
            return
        detail = self._lib.nano_nccl_get_last_error()
        message = detail.decode(errors="replace") if detail else "unknown native error"
        raise RuntimeError(f"Nano NCCL {operation} failed: {message}")

    def _query(self, function: Any, *args: Any) -> int:
        value = ctypes.c_int()
        status = function(*args, ctypes.byref(value))
        self._raise_native_error(status, "communicator query")
        return value.value

    def _require_open(self) -> None:
        if self._closed or not self._handle.value:
            raise RuntimeError("NanoNcclBackend is closed")

    def _validate_group(self, group: Any | None) -> None:
        if group is not self._group:
            raise ValueError(
                f"NanoNcclBackend for {self.group_name} may only use its "
                "configured process group"
            )

    def _validate_tensor(self, tensor: Tensor, *, name: str) -> int:
        if not tensor.is_cuda or tensor.device != self.device:
            raise ValueError(
                f"{name} must be on configured device {self.device}, "
                f"got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        dtype = _DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"unsupported Nano NCCL dtype for {name}: {tensor.dtype}")
        return dtype

    def _redop(self, op: str) -> int:
        key = op.lower()
        if key not in _REDUCE_OPS:
            raise ValueError(
                f"unsupported Nano NCCL reduce op {op!r}; expected one of "
                f"{tuple(_REDUCE_OPS)}"
            )
        return _REDUCE_OPS[key]

    def _begin_launch(self, tensors: Iterable[Tensor]) -> None:
        producer = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(producer)
        for tensor in tensors:
            tensor.record_stream(self._stream)

    def _work(self, release_buffers: Iterable[Tensor] = ()) -> _NanoNcclWork:
        event = torch.cuda.Event()
        event.record(self._stream)
        return _NanoNcclWork(self, event, release_buffers)

    @staticmethod
    def _complete(
        work: _NanoNcclWork,
        result: Any,
        *,
        async_op: bool,
    ) -> Any:
        if async_op:
            return work
        work.wait()
        return result

    def check_async_error(self) -> None:
        self._require_open()
        status = self._lib.nano_nccl_check_async_error(self._handle)
        self._raise_native_error(status, "asynchronous check")

    def all_reduce(
        self,
        tensor: Tensor,
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | _NanoNcclWork:
        self._require_open()
        self._validate_group(group)
        dtype = self._validate_tensor(tensor, name="all-reduce tensor")
        redop = self._redop(op)
        if tensor.numel() == 0:
            return tensor

        output = self._buffer_pool.acquire(tensor)
        self._begin_launch((tensor, output))
        args = _AllReduceArgs(
            tensor.data_ptr(),
            output.data_ptr(),
            self._stream.cuda_stream,
            tensor.numel(),
            dtype,
            redop,
        )
        try:
            status = self._lib.nano_nccl_all_reduce(
                self._handle, ctypes.byref(args)
            )
            self._raise_native_error(status, "all-reduce launch")
        except Exception:
            self._buffer_pool.release(output)
            raise
        with torch.cuda.stream(self._stream):
            tensor.copy_(output)
        work = self._work((output,))
        return self._complete(work, tensor, async_op=async_op)

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> list[Tensor] | _NanoNcclWork:
        self._require_open()
        self._validate_group(group)
        dtype = self._validate_tensor(tensor, name="all-gather input")
        if len(tensor_list) != self.group_size:
            raise ValueError(
                f"all-gather output list has {len(tensor_list)} tensors, "
                f"expected {self.group_size}"
            )
        for rank, output in enumerate(tensor_list):
            output_dtype = self._validate_tensor(
                output, name=f"all-gather output[{rank}]"
            )
            if output_dtype != dtype or output.shape != tensor.shape:
                raise ValueError(
                    f"all-gather output[{rank}] must match input shape and dtype; "
                    f"input={tuple(tensor.shape)}/{tensor.dtype}, "
                    f"output={tuple(output.shape)}/{output.dtype}"
                )
        if tensor.numel() == 0:
            return tensor_list

        flat_output = self._buffer_pool.acquire_flat(
            tensor, tensor.numel() * self.group_size
        )
        self._begin_launch((tensor, flat_output, *tensor_list))
        args = _AllGatherArgs(
            tensor.data_ptr(),
            flat_output.data_ptr(),
            self._stream.cuda_stream,
            tensor.numel(),
            dtype,
        )
        try:
            status = self._lib.nano_nccl_all_gather(
                self._handle, ctypes.byref(args)
            )
            self._raise_native_error(status, "all-gather launch")
        except Exception:
            self._buffer_pool.release(flat_output)
            raise
        with torch.cuda.stream(self._stream):
            for rank, output in enumerate(tensor_list):
                start = rank * tensor.numel()
                output.copy_(
                    flat_output.narrow(0, start, tensor.numel()).view_as(output)
                )
        work = self._work((flat_output,))
        return self._complete(work, tensor_list, async_op=async_op)

    def all_gather_into_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> Tensor | _NanoNcclWork:
        self._require_open()
        self._validate_group(group)
        dtype = self._validate_tensor(input, name="all-gather input")
        output_dtype = self._validate_tensor(output, name="all-gather output")
        if output_dtype != dtype or output.numel() != input.numel() * self.group_size:
            raise ValueError(
                "all-gather output must have the input dtype and "
                f"{self.group_size}x its elements; input={input.numel()}/{input.dtype}, "
                f"output={output.numel()}/{output.dtype}"
            )
        if input.numel() == 0:
            return output

        self._begin_launch((input, output))
        args = _AllGatherArgs(
            input.data_ptr(),
            output.data_ptr(),
            self._stream.cuda_stream,
            input.numel(),
            dtype,
        )
        status = self._lib.nano_nccl_all_gather(
            self._handle, ctypes.byref(args)
        )
        self._raise_native_error(status, "all-gather launch")
        work = self._work()
        return self._complete(work, output, async_op=async_op)

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | _NanoNcclWork:
        self._require_open()
        self._validate_group(group)
        dtype = self._validate_tensor(output, name="reduce-scatter output")
        redop = self._redop(op)
        if len(input_list) != self.group_size:
            raise ValueError(
                f"reduce-scatter input list has {len(input_list)} tensors, "
                f"expected {self.group_size}"
            )
        for rank, input_tensor in enumerate(input_list):
            input_dtype = self._validate_tensor(
                input_tensor, name=f"reduce-scatter input[{rank}]"
            )
            if input_dtype != dtype or input_tensor.shape != output.shape:
                raise ValueError(
                    f"reduce-scatter input[{rank}] must match output shape and "
                    f"dtype; output={tuple(output.shape)}/{output.dtype}, "
                    f"input={tuple(input_tensor.shape)}/{input_tensor.dtype}"
                )
        if output.numel() == 0:
            return output

        flat_input = self._buffer_pool.acquire_flat(
            output, output.numel() * self.group_size
        )
        self._begin_launch((output, flat_input, *input_list))
        with torch.cuda.stream(self._stream):
            for rank, input_tensor in enumerate(input_list):
                start = rank * output.numel()
                flat_input.narrow(0, start, output.numel()).copy_(
                    input_tensor.reshape(-1)
                )
        args = _ReduceScatterArgs(
            flat_input.data_ptr(),
            output.data_ptr(),
            self._stream.cuda_stream,
            output.numel(),
            dtype,
            redop,
        )
        try:
            status = self._lib.nano_nccl_reduce_scatter(
                self._handle, ctypes.byref(args)
            )
            self._raise_native_error(status, "reduce-scatter launch")
        except Exception:
            self._buffer_pool.release(flat_input)
            raise
        work = self._work((flat_input,))
        return self._complete(work, output, async_op=async_op)

    def reduce_scatter_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | _NanoNcclWork:
        self._require_open()
        self._validate_group(group)
        dtype = self._validate_tensor(output, name="reduce-scatter output")
        input_dtype = self._validate_tensor(input, name="reduce-scatter input")
        redop = self._redop(op)
        if input_dtype != dtype or input.numel() != output.numel() * self.group_size:
            raise ValueError(
                "reduce-scatter input must have the output dtype and "
                f"{self.group_size}x its elements; input={input.numel()}/{input.dtype}, "
                f"output={output.numel()}/{output.dtype}"
            )
        if output.numel() == 0:
            return output

        self._begin_launch((input, output))
        args = _ReduceScatterArgs(
            input.data_ptr(),
            output.data_ptr(),
            self._stream.cuda_stream,
            output.numel(),
            dtype,
            redop,
        )
        status = self._lib.nano_nccl_reduce_scatter(
            self._handle, ctypes.byref(args)
        )
        self._raise_native_error(status, "reduce-scatter launch")
        work = self._work()
        return self._complete(work, output, async_op=async_op)

    def close(self) -> None:
        if self._closed:
            return
        if not self._handle.value:
            self._closed = True
            return
        self._stream.synchronize()
        check_status = self._lib.nano_nccl_check_async_error(self._handle)
        check_detail = self._lib.nano_nccl_get_last_error()
        destroy_status = self._lib.nano_nccl_destroy_communicator(self._handle)
        destroy_detail = self._lib.nano_nccl_get_last_error()
        self._handle = ctypes.c_void_p()
        self._buffer_pool.clear()
        self._closed = True
        if check_status != 0:
            detail = check_detail.decode(errors="replace") if check_detail else "unknown"
            raise RuntimeError(f"Nano NCCL asynchronous check failed: {detail}")
        if destroy_status != 0:
            detail = (
                destroy_detail.decode(errors="replace")
                if destroy_detail
                else "unknown"
            )
            raise RuntimeError(f"Nano NCCL destroy communicator failed: {detail}")

    def __enter__(self) -> NanoNcclBackend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def create_nano_nccl_training_backend(
    ctx: ParallelContext,
    library_path: str | Path,
    *,
    group_names: Sequence[ParallelGroupName] = ("tp", "dp-cp"),
    transport: str = "rdma",
    expected_channels: int = 4,
) -> RoutedCommBackend:
    """Replace active TP/SP and DP collectives while retaining Torch P2P.

    All world ranks must call this factory with the same ``group_names`` order
    because every communicator creation performs an MPI_COMM_WORLD split.
    Groups of size one are skipped.
    """
    from nano_megatron.distributed.routed_backend import (
        CollectiveRoute,
        RoutedCommBackend,
    )

    group_size_by_name = {
        "tp": ctx.tensor_parallel_size,
        "dp": ctx.data_parallel_size,
        "cp": ctx.context_parallel_size,
        "dp-cp": ctx.data_parallel_size * ctx.context_parallel_size,
    }
    requested = tuple(group_names)
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate Nano NCCL group names: {requested}")
    unknown = [name for name in requested if name not in _GROUP_NAMES]
    if unknown:
        raise ValueError(
            f"unknown Nano NCCL groups {unknown}; expected values from {_GROUP_NAMES}"
        )

    routes: list[CollectiveRoute] = []
    try:
        for group_name in requested:
            if group_size_by_name[group_name] <= 1:
                continue
            backend = NanoNcclBackend.from_parallel_context(
                ctx,
                library_path,
                group_name=group_name,
                transport=transport,
                expected_channels=expected_channels,
            )
            routes.append(
                CollectiveRoute(
                    name=group_name,
                    group=backend._group,
                    backend=backend,
                )
            )
    except Exception:
        for route in reversed(routes):
            close = getattr(route.backend, "close", None)
            if close is not None:
                close()
        raise

    return RoutedCommBackend(ctx.backend, routes, own_routes=True)
