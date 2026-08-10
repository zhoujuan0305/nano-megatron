from __future__ import annotations

from dataclasses import dataclass

# CP sequence packing modes (Megatron DualChunkSwap = zigzag).
_VALID_CONTEXT_PARALLEL_PACK = frozenset({"contiguous", "zigzag"})


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int | None = None
    sequence_parallel: bool = False
    order: str = "tp-cp-dp-pp"
    # CP sequence pack: "contiguous" (default) or "zigzag" (Megatron DualChunkSwap).
    # Default is contiguous for pack-only history (run-20260810-007). With
    # prefix-concat multi-block FA (NANO_CP_FA_CHUNK=prefix, run-20260810-001),
    # zigzag launch tax is much lower; pack default stays contiguous pending
    # a dedicated pack-default revisit.
    context_parallel_pack: str = "contiguous"

    def product_without_dp(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.context_parallel_size
        )

    def resolved_data_parallel_size(self, world_size: int) -> int:
        self._check_positive_sizes()
        base = self.product_without_dp()
        if world_size % base != 0:
            raise ValueError(
                f"world_size ({world_size}) not divisible by tp*pp*cp ({base})"
            )
        inferred = world_size // base
        if self.data_parallel_size is None:
            return inferred
        if self.data_parallel_size != inferred:
            raise ValueError(
                f"data_parallel_size ({self.data_parallel_size}) inconsistent with "
                f"world_size ({world_size}) / (tp*pp*cp={base}) = {inferred}"
            )
        return self.data_parallel_size

    def validate(self, world_size: int) -> None:
        self._check_positive_sizes()
        self._check_context_parallel_pack()
        if self.sequence_parallel and self.context_parallel_size > 1:
            raise ValueError(
                "sequence_parallel is not supported with context_parallel_size > 1"
            )
        if self.pipeline_parallel_size > 1 and self.context_parallel_size > 1:
            raise ValueError(
                "pipeline_parallel is not supported with context_parallel_size > 1"
            )
        dp = self.resolved_data_parallel_size(world_size)
        product = self.product_without_dp() * dp
        if product != world_size:
            raise ValueError(
                f"world_size ({world_size}) != tp*cp*dp*pp ({product})"
            )

    def _check_positive_sizes(self) -> None:
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.data_parallel_size is not None and self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")

    def _check_context_parallel_pack(self) -> None:
        if self.context_parallel_pack not in _VALID_CONTEXT_PARALLEL_PACK:
            raise ValueError(
                "context_parallel_pack must be 'contiguous' or 'zigzag', "
                f"got {self.context_parallel_pack!r}"
            )
