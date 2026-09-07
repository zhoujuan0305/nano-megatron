#!/usr/bin/env python3
"""Fail fast unless distributed traffic is pinned to the requested NIC/HCA."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import socket
import struct

import torch
import torch.distributed as dist


def _exact_name(value: str) -> str:
    return value.removeprefix("=").split(":", 1)[0]


def _ipv4_address(interface: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        packed = struct.pack("256s", interface.encode()[:15])
        result = fcntl.ioctl(sock.fileno(), 0x8915, packed)
    return socket.inet_ntoa(result[20:24])


def _validate_fabric() -> tuple[str, str, str]:
    socket_interface = _exact_name(os.environ["NCCL_SOCKET_IFNAME"])
    hca = _exact_name(os.environ["NCCL_IB_HCA"])
    expected_address = os.environ["NANO_EXPECTED_NODE_ADDR"]

    interfaces = {name for _, name in socket.if_nameindex()}
    if socket_interface not in interfaces:
        raise RuntimeError(
            f"socket interface {socket_interface!r} is absent; found {sorted(interfaces)}"
        )
    actual_address = _ipv4_address(socket_interface)
    if actual_address != expected_address:
        raise RuntimeError(
            f"{socket_interface} has {actual_address}, expected {expected_address}"
        )
    mapped_interfaces = Path(f"/sys/class/infiniband/{hca}/device/net")
    if not (mapped_interfaces / socket_interface).exists():
        found = sorted(path.name for path in mapped_interfaces.iterdir())
        raise RuntimeError(
            f"HCA {hca} maps to {found}, expected {socket_interface}"
        )
    return socket_interface, hca, actual_address


def main() -> None:
    socket_interface, hca, address = _validate_fabric()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    torch.cuda.synchronize()
    if value.item() != expected:
        raise RuntimeError(
            f"rank={rank}: all-reduce value={value.item()} expected={expected}"
        )
    print(
        f"fabric-smoke rank={rank} host={socket.gethostname()} "
        f"local_rank={local_rank} address={address} interface={socket_interface} "
        f"hca={hca} value={value.item()}"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
