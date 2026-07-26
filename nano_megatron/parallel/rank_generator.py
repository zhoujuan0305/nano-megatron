from __future__ import annotations


def prefix_product(values: list[int], init: int = 1) -> list[int]:
    out = [init]
    cur = init
    for v in values:
        cur *= v
        out.append(cur)
    return out


def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: list[int],
    mask: list[bool],
) -> list[list[int]]:
    def inner_product(a: list[int], b: list[int]) -> int:
        return sum(x * y for x, y in zip(a, b))

    def decompose(index: int, shape: list[int], stride: list[int] | None = None) -> list[int]:
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        assert inner_product(idx, stride[:-1]) == index
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]
    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]
    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size
    ranks: list[list[int]] = []
    for group_index in range(num_of_group):
        # Decompose in the unmasked subspace's own layout; map via global strides.
        group_indices = decompose(group_index, unmasked_shape)
        rank_group: list[int] = []
        for rank_in_group in range(group_size):
            rank_indices = decompose(rank_in_group, masked_shape)
            rank = inner_product(rank_indices, masked_stride) + inner_product(
                group_indices, unmasked_stride
            )
            rank_group.append(rank)
        ranks.append(rank_group)
    return ranks


class RankGenerator:
    def __init__(
        self,
        tp: int,
        dp: int,
        pp: int,
        cp: int = 1,
        order: str = "tp-cp-dp-pp",
    ) -> None:
        if min(tp, dp, pp, cp) < 1:
            raise ValueError("all parallel sizes must be >= 1")
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp
        self.name_to_size = {"tp": tp, "pp": pp, "dp": dp, "cp": cp}
        order = order.lower()
        for name, size in self.name_to_size.items():
            if name not in order:
                if size != 1:
                    raise RuntimeError(
                        f"size of ({name}) is ({size}), but order ({order}) omits it"
                    )
                order = f"{order}-{name}"
        self.order = order
        self.ordered_names = order.split("-")
        self.ordered_size = [self.name_to_size[n] for n in self.ordered_names]

    def get_mask(self, token: str) -> list[bool]:
        tokens = token.split("-")
        mask = [False] * len(self.ordered_names)
        for t in tokens:
            mask[self.ordered_names.index(t)] = True
        return mask

    def get_ranks(self, token: str) -> list[list[int]]:
        return generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, self.get_mask(token)
        )

    def decode(self, global_rank: int) -> dict[str, int]:
        if not 0 <= global_rank < self.world_size:
            raise ValueError(f"rank {global_rank} out of range [0, {self.world_size})")
        stride = prefix_product(self.ordered_size)
        parts: dict[str, int] = {}
        for name, size, s in zip(self.ordered_names, self.ordered_size, stride):
            parts[name] = (global_rank // s) % size
        return parts

    def encode(self, ranks: dict[str, int]) -> int:
        stride = prefix_product(self.ordered_size)
        total = 0
        for name, s in zip(self.ordered_names, stride):
            total += ranks[name] * s
        return total
