# nano-megatron

面向研究与复现的紧凑分布式训练框架，覆盖 Megatron 风格大模型并行技术。

English README: [README.md](README.md).

## ReferenceGPT（数值正确性基准 / oracle）

`nano_megatron.reference` 是单卡、全链路 FP32 的 Megatron 风格 GPT，用作后续并行与优化实现的**数值对照基准**。

在相同 seed、配置与输入下，后续 DP/TP/PP/ZeRO 等实现应与 reference 对齐：

- logits
- loss
- 关键中间激活
- 全部参数梯度
- 优化器状态
- 多步参数更新轨迹

（CPU FP32 下优先严格相等，或约定极严容差。）

### 公共 API

```python
from nano_megatron.reference import (
    AdamW,
    CaptureLevel,
    ReferenceGPT,
    ReferenceGPTConfig,
    StepResult,
    reference_train_loop,
    reference_train_step,
    seed_all,
    shifted_cross_entropy,
    snapshot_grads,
    snapshot_optimizer,
    snapshot_params,
)
```

### 导出训练轨迹

```bash
python scripts/run_reference_gpt.py \
  --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

保存结果为 `list[dict]`（每步一条）。字段：`step`（int），以及 CPU 上的
`loss`、`logits`、`params`、`grads`、`activations`、`optimizer_state`。

### 与候选实现对齐

1. 固定 `seed_all(seed)` 与相同的 `ReferenceGPTConfig` / `input_ids`。
2. 跑 reference loop，或加载 CLI 导出的轨迹。
3. 用相同输入与优化器超参跑候选实现。
4. 对 loss、logits、grads、opt state、params 做相等或严格容差断言。

不要为了让测试通过而放宽数值容差。

## 并行上下文与 CommBackend

`nano_megatron.parallel` 负责进程级拓扑（TP/DP/PP/CP 进程组）。
`nano_megatron.distributed` 提供精简通信后端抽象，模型代码不直接调用
`torch.distributed`。默认实现为 `TorchDistBackend`（PyTorch collective）；
后续可接入 `nano-nccl`。

### 公共 API

```python
from nano_megatron.parallel import (
    ParallelConfig,
    ParallelContext,
    RankGenerator,
    destroy_parallel,
    get_parallel_context,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.distributed import CommBackend, TorchDistBackend
```

### 初始化方式

```python
from nano_megatron.distributed import TorchDistBackend
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
)

# world_size 必须等于 tp * cp * dp * pp（dp 可由 world_size 推断）。
# 默认 rank 序：tp-cp-dp-pp。每个 rank 都必须参与每一次 new_group 调用。
cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2)
ctx = initialize_parallel(cfg, backend=TorchDistBackend())
# ctx.tensor_parallel_group, ctx.data_parallel_group, ...
# ctx.backend.all_reduce(tensor, group=ctx.data_parallel_group)
destroy_parallel()
```

单进程（CPU/gloo）即可跑单元测试。多卡 NCCL 需按常规使用 `torchrun` / 环境变量 rank。

### 安装与测试

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=. python -m pytest tests/unit/reference -v
PYTHONPATH=. python -m pytest tests/unit/parallel tests/unit/distributed -v
```

NCCL 多卡（需 ≥4 张 CUDA；控制面走 loopback）：

```bash
PYTHONPATH=. python -m pytest tests/distributed -v --tb=short
# 或直接：
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_parallel_context_nccl.py -v
```
