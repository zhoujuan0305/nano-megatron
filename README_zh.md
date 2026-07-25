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

## 安装与测试

```bash
python -m pip install -e .
python -m pytest tests/unit/reference -v
```
