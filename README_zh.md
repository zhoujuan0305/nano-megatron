# nano-megatron

[English](README.md)

## 背景

一个基于PyTorch/CUDA的轻量训练框架，通过独立实现并与Megatron对照，理解大模型并行训练机制。
已实现DP、TP、SP、PP和CP，包括自定义梯度分桶、非交错1F1B流水线及通信计算重叠。
PyTorch提供tensor、autograd和分布式原语；Nano的并行层和DDP不使用Megatron实现。

FlashAttention为可选依赖。默认使用PyTorch/NCCL通信，也可接入
[nano-nccl](https://github.com/zhoujuan0305/nano-nccl)。

## 性能

双节点各4张RTX A6000，TP2 × PP2 × DP2并开启SP，1.424B GPT，
BF16 + FlashAttention，共同FP32-master AdamW：

| 实现 | 完整训练step |
| --- | ---: |
| nano-megatron，开启wgrad累加融合 | 266.57 ms |
| Megatron-Core + Transformer Engine | 257.38 ms |

本次单次对照中，Nano吞吐为Megatron的 **96.55%**。每组50步预热、100步计时；
每台机器使用共享的10逻辑CPU池，SMT竞争随测试变化。这不是独占环境下的稳定性能保证。
环境和计时口径见 [performance.md](performance.md)。

## 使用

先安装适合CUDA环境的PyTorch，再安装项目：

```bash
python -m pip install -e ".[dev]"
# 可选；下面的FlashAttention示例需要安装。
python -m pip install flash-attn --no-build-isolation
```

运行小规模CPU参考模型：

```bash
python scripts/run_reference_gpt.py --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

TP2 × SP × PP2 × DP2需要两台机器、每台4张GPU。两台机器均执行下列命令，
`NODE_RANK`分别设为0和1，`MASTER_ADDR`和端口保持一致且两机可达。

```bash
export MASTER_ADDR="<node-0-address>"
export MASTER_PORT=29500
export NODE_RANK=0  # 第二台设为1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

torchrun --nnodes=2 --nproc-per-node=4 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  scripts/benchmark_pp.py --framework nano \
  --tp-size 2 --pp-size 2 --dp-size 2 --sequence-parallel \
  --hidden-size 2048 --num-layers 24 --num-heads 16 --ffn-hidden-size 5504 \
  --batch-size 4 --num-microbatches 4 --seq-len 1024 \
  --precision bf16 --attn-backend flash --optimizer adamw \
  --warmup-steps 50 --benchmark-steps 100 \
  --fuse-rope --rope-fp32-phase --fuse-swiglu --vocab-ce-fp32 \
  --overlap-grad-reduce --overlap-tp-dgrad \
  --disable-bucketing-after-first-stage --fuse-wgrad-accum
```

`--batch-size`是每个DP副本的batch：上述配置global batch为8，microbatch为1。
CPU亲和性和网卡应按自己的机器配置；该命令不会预留CPU，也不自动复现实测的绑核条件。

Wgrad累加融合默认关闭，只支持显式训练反向入口；不支持共享/重复使用权重、同参数并发、
autocast、高阶梯度和完整训练步CUDA Graph。真实文本训练与checkpoint见
[TEXT_TRAINING.md](TEXT_TRAINING.md)。

单元测试：`python -m pytest tests/unit`。运行Megatron对照还需安装Megatron-Core和
Transformer Engine；Nano本身不依赖TE。

## License

[MIT](LICENSE)。
