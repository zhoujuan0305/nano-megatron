from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nano_megatron.parallel.context import ParallelContext


_DTYPES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}
_TRANSPORTS = {"auto": 0, "socket": 3, "rdma": 4}
_TRANSPORT_NAMES = {
    0: "auto",
    1: "shm",
    2: "p2p",
    3: "socket",
    4: "rdma",
    5: "mixed",
}


class _MpiSubgroupConfig(ctypes.Structure):
    _fields_ = [
        ("color", ctypes.c_int),
        ("key", ctypes.c_int),
        ("device", ctypes.c_int),
        ("transport", ctypes.c_int32),
    ]


class _AllReduceArgs(ctypes.Structure):
    _fields_ = [
        ("send_buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("recv_buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("streams", ctypes.POINTER(ctypes.c_void_p)),
        ("count", ctypes.c_size_t),
        ("dtype", ctypes.c_int32),
        ("redop", ctypes.c_int32),
    ]


class _RecvBufferPool:
    """Reuse out-of-place receive storage by tensor shape.

    Gradient buckets without persistent views may get a new input address each
    step. Keying receive buffers by that address retains one full set per step,
    so the pool instead returns storage after the associated work is waited.
    """

    def __init__(self) -> None:
        self._available: dict[
            tuple[torch.device, torch.dtype, tuple[int, ...]], list[Tensor]
        ] = {}

    @staticmethod
    def _key(tensor: Tensor) -> tuple[torch.device, torch.dtype, tuple[int, ...]]:
        return tensor.device, tensor.dtype, tuple(tensor.shape)

    def acquire(self, template: Tensor) -> Tensor:
        available = self._available.get(self._key(template))
        if available:
            return available.pop()
        return torch.empty_like(template)

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
        output: Tensor,
    ) -> None:
        self._backend = backend
        self._event = event
        self._output: Tensor | None = output
        self._waited = False

    def wait(self) -> bool:
        if not self._waited:
            torch.cuda.current_stream(self._backend.device).wait_event(self._event)
            self._backend.check_async_error()
            assert self._output is not None
            self._backend._recv_buffer_pool.release(self._output)
            self._output = None
            self._waited = True
        return True

    def is_completed(self) -> bool:
        return bool(self._event.query())


class NanoNcclBackend:
    """DP-only adapter for Nano NCCL's external-buffer C ABI.

    The native library is compiled for the DP group size. Each training process
    owns one visible GPU and joins one MPI_COMM_WORLD subgroup. Nano NCCL remains
    out-of-place internally; the reduced buffer is copied back on the dedicated
    communication stream before the completion event is recorded.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        subgroup_color: int,
        subgroup_rank: int,
        subgroup_size: int,
        group: Any,
        transport: str = "rdma",
        device: torch.device | str | int = "cuda:0",
        expected_channels: int = 4,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NanoNcclBackend requires CUDA")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"NanoNcclBackend requires a CUDA device, got {self.device}")
        if torch.cuda.device_count() != 1 or (self.device.index or 0) != 0:
            raise RuntimeError(
                "NanoNcclBackend requires one visible GPU per process at cuda:0"
            )
        if transport not in _TRANSPORTS:
            raise ValueError(f"unsupported Nano NCCL transport: {transport}")
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Nano NCCL C adapter not found: {path}")

        self._lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self._configure_signatures()
        if self._lib.nano_nccl_abi_version() != 1:
            raise RuntimeError("Nano NCCL C ABI version mismatch")
        self._handle = ctypes.c_void_p()
        self._group = group
        self._closed = False
        self._recv_buffer_pool = _RecvBufferPool()
        self._stream = torch.cuda.Stream(device=self.device)

        config = _MpiSubgroupConfig(
            subgroup_color, subgroup_rank, 0, _TRANSPORTS[transport]
        )
        status = self._lib.nano_nccl_create_mpi_subgroup_communicator(
            ctypes.byref(config), ctypes.byref(self._handle)
        )
        self._raise_native_error(status, "create communicator")
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
                    f"Nano NCCL library has {native_size} ranks, DP subgroup has "
                    f"{subgroup_size}; rebuild with -DNANO_NCCL_NRANKS={subgroup_size}"
                )
            channels = self._query(self._lib.nano_nccl_mpi_channel_count)
            if channels != expected_channels:
                raise RuntimeError(
                    f"Nano NCCL library has {channels} channels, expected "
                    f"{expected_channels}"
                )
            resolved_value = self._query(self._lib.nano_nccl_transport, self._handle)
            resolved = _TRANSPORT_NAMES.get(resolved_value, f"unknown({resolved_value})")
            if transport != "auto" and resolved not in (transport, "mixed"):
                raise RuntimeError(
                    f"Nano NCCL resolved transport {resolved!r}, requested {transport!r}"
                )
            self.transport = resolved
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
        transport: str = "rdma",
        expected_channels: int = 4,
    ) -> NanoNcclBackend:
        from nano_megatron.parallel.rank_generator import RankGenerator

        generator = RankGenerator(
            tp=ctx.tensor_parallel_size,
            dp=ctx.data_parallel_size,
            pp=ctx.pipeline_parallel_size,
            cp=ctx.context_parallel_size,
            order=ctx.config.order,
        )
        groups = generator.get_ranks("dp-cp")
        for color, ranks in enumerate(groups):
            if ctx.rank in ranks:
                return cls(
                    library_path,
                    subgroup_color=color,
                    subgroup_rank=ranks.index(ctx.rank),
                    subgroup_size=len(ranks),
                    group=ctx.data_context_parallel_group,
                    transport=transport,
                    device=torch.device("cuda", ctx.local_rank),
                    expected_channels=expected_channels,
                )
        raise RuntimeError(f"rank {ctx.rank} is missing from DP×CP groups {groups}")

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
        if group is not self._group:
            raise ValueError("NanoNcclBackend may only reduce its configured DP group")
        if op.lower() != "sum":
            raise ValueError("NanoNcclBackend DP adapter currently supports sum only")
        if not tensor.is_cuda or tensor.device != self.device:
            raise ValueError(
                f"tensor must be on the configured device {self.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError("NanoNcclBackend requires a contiguous tensor")
        dtype = _DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"unsupported Nano NCCL dtype: {tensor.dtype}")
        if tensor.numel() == 0:
            return tensor

        output = self._recv_buffer_pool.acquire(tensor)

        producer = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(producer)
        tensor.record_stream(self._stream)
        output.record_stream(self._stream)
        send_buffers = (ctypes.c_void_p * 1)(tensor.data_ptr())
        recv_buffers = (ctypes.c_void_p * 1)(output.data_ptr())
        streams = (ctypes.c_void_p * 1)(self._stream.cuda_stream)
        args = _AllReduceArgs(
            send_buffers,
            recv_buffers,
            streams,
            tensor.numel(),
            dtype,
            0,
        )
        status = self._lib.nano_nccl_all_reduce(
            self._handle, ctypes.byref(args)
        )
        self._raise_native_error(status, "all-reduce launch")
        with torch.cuda.stream(self._stream):
            tensor.copy_(output)
        event = torch.cuda.Event()
        event.record(self._stream)
        work = _NanoNcclWork(self, event, output)
        if async_op:
            return work
        work.wait()
        return tensor

    def close(self) -> None:
        if self._closed:
            return
        if not self._handle.value:
            self._closed = True
            return
        torch.cuda.synchronize(self.device)
        check_status = self._lib.nano_nccl_check_async_error(self._handle)
        check_detail = self._lib.nano_nccl_get_last_error()
        destroy_status = self._lib.nano_nccl_destroy_communicator(self._handle)
        destroy_detail = self._lib.nano_nccl_get_last_error()
        self._handle = ctypes.c_void_p()
        self._recv_buffer_pool.clear()
        self._closed = True
        if check_status != 0:
            detail = check_detail.decode(errors="replace") if check_detail else "unknown"
            raise RuntimeError(f"Nano NCCL asynchronous check failed: {detail}")
        if destroy_status != 0:
            detail = destroy_detail.decode(errors="replace") if destroy_detail else "unknown"
            raise RuntimeError(f"Nano NCCL destroy communicator failed: {detail}")

    def __enter__(self) -> NanoNcclBackend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
