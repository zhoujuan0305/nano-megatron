from __future__ import annotations

import gc

import pytest
import torch
import torch.nn as nn

from scripts.benchmark_pp import (
    _make_input_ids,
    _mean_next_token_loss,
    _optimizer_step,
    _time_loop,
)


def test_optimizer_step_uses_grad_and_megatron_main_grad() -> None:
    module = nn.Module()
    module.regular = nn.Parameter(torch.tensor([2.0]))
    module.megatron = nn.Parameter(torch.tensor([3.0]))
    module.regular.grad = torch.tensor([4.0])
    module.megatron.main_grad = torch.tensor([5.0])

    _optimizer_step(module, "sgd", 0.1)

    torch.testing.assert_close(module.regular, torch.tensor([1.6]))
    torch.testing.assert_close(module.megatron, torch.tensor([2.5]))


def test_benchmark_input_ids_use_reproducible_independent_generator() -> None:
    state = torch.random.get_rng_state()
    first = _make_input_ids(vocab_size=32, shape=(2, 5), device=torch.device("cpu"))
    second = _make_input_ids(vocab_size=32, shape=(2, 5), device=torch.device("cpu"))

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_megatron_next_token_loss_omits_wrapped_last_target() -> None:
    per_token = torch.tensor([[1.0, 3.0, 1000.0]], requires_grad=True)
    loss = _mean_next_token_loss(per_token)
    assert loss.item() == 2.0
    loss.backward()
    torch.testing.assert_close(per_token.grad, torch.tensor([[0.5, 0.5, 0.0]]))


def test_time_loop_collects_after_warmup_validation_before_measurement(monkeypatch):
    from scripts import benchmark_pp

    events = []
    monkeypatch.setattr(
        gc, "collect", lambda generation: events.append(("gc", generation))
    )
    monkeypatch.setattr(
        benchmark_pp.time,
        "perf_counter",
        lambda: events.append(("timer",)) or 1.0,
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: events.append(("sync",))
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda: events.append(("reset",)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda _name: events.append(("range_push",)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_pop", lambda: events.append(("range_pop",))
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    def step():
        events.append(("step",))

    def validate():
        events.append(("validate",))

    _time_loop(
        step,
        warmup=1,
        steps=1,
        device=torch.device("cuda"),
        after_warmup=validate,
    )

    assert events.index(("validate",)) < events.index(("gc", 2))
    assert events.index(("gc", 2)) < events.index(("sync",))
    assert events.index(("reset",)) < events.index(("timer",))
    last_step = max(i for i, event in enumerate(events) if event == ("step",))
    assert events.index(("timer",)) < last_step
    assert events.count(("step",)) == 2


def test_time_loop_collects_without_callback_or_warmup(monkeypatch):
    from scripts import benchmark_pp

    events = []
    monkeypatch.setattr(
        gc, "collect", lambda generation: events.append(("gc", generation))
    )
    monkeypatch.setattr(
        benchmark_pp.time,
        "perf_counter",
        lambda: events.append(("timer",)) or 1.0,
    )
    _time_loop(
        lambda: events.append(("step",)),
        warmup=0,
        steps=0,
        device=torch.device("cpu"),
    )
    assert events == [("gc", 2), ("timer",), ("timer",)]


@pytest.mark.parametrize("enabled", [True, False])
def test_time_loop_preserves_gc_settings_and_allows_timed_collection(monkeypatch, enabled):
    from scripts import benchmark_pp

    original_enabled = gc.isenabled()
    original_thresholds = gc.get_threshold()
    thresholds = (1000, 11, 13)
    events = []

    def collect(generation=2):
        events.append(("gc", generation))
        return 0

    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(benchmark_pp.time, "perf_counter", lambda: 1.0)
    try:
        (gc.enable if enabled else gc.disable)()
        gc.set_threshold(*thresholds)
        _time_loop(
            lambda: gc.collect(2), warmup=0, steps=1, device=torch.device("cpu")
        )
        assert events == [("gc", 2), ("gc", 2)]
        assert gc.isenabled() is enabled
        assert gc.get_threshold() == thresholds
    finally:
        gc.set_threshold(*original_thresholds)
        (gc.enable if original_enabled else gc.disable)()


def test_time_loop_callback_error_prevents_collection_and_timing(monkeypatch):
    from scripts import benchmark_pp

    monkeypatch.setattr(gc, "collect", lambda *_args: pytest.fail("unexpected GC"))
    monkeypatch.setattr(
        benchmark_pp.time,
        "perf_counter",
        lambda: pytest.fail("unexpected timer"),
    )

    def fail():
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        _time_loop(
            lambda: None,
            warmup=0,
            steps=1,
            device=torch.device("cpu"),
            after_warmup=fail,
        )


@pytest.mark.parametrize("device, ids", [("cpu", None), ("cuda:3", [3]), ("cuda", [2])])
def test_time_loop_rendezvous_once_after_gc_before_timer(monkeypatch, device, ids):
    from scripts import benchmark_pp as bp

    events = []
    monkeypatch.setattr(gc, "collect", lambda generation: events.append("gc"))
    monkeypatch.setattr(bp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(bp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(bp.dist, "barrier", lambda **kw: events.append(("barrier", kw)))
    monkeypatch.setattr(bp.time, "perf_counter", lambda: events.append("timer") or 1.0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    _time_loop(lambda: events.append("step"), warmup=0, steps=3, device=torch.device(device))
    barrier = ("barrier", {} if ids is None else {"device_ids": ids})
    assert events.count(barrier) == 1
    assert events.index("gc") < events.index(barrier) < events.index("timer")
    assert events.count("step") == 3
    if ids is not None:
        assert events[:5] == ["gc", "sync", barrier, "sync", "reset"]


def test_time_loop_skips_rendezvous_for_initialized_single_rank(monkeypatch):
    from scripts import benchmark_pp as bp

    monkeypatch.setattr(gc, "collect", lambda _generation: 0)
    monkeypatch.setattr(bp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(bp.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(bp.dist, "barrier", lambda **_kw: pytest.fail("unexpected barrier"))
    _time_loop(lambda: None, warmup=0, steps=1, device=torch.device("cpu"))


def test_nano_channel_cli_keeps_four_default_and_accepts_eight(monkeypatch):
    from scripts.benchmark_pp import parse_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py"])
    assert parse_args().nano_nccl_channels == 4
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--nano-nccl-channels", "8"])
    assert parse_args().nano_nccl_channels == 8


def test_fused_wgrad_cli_is_opt_in_tp_only_and_rejects_graphs(monkeypatch):
    from scripts.benchmark_pp import parse_args, _validate_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py"])
    assert parse_args().fuse_wgrad_accum is False
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--tp-size", "2", "--fuse-wgrad-accum"])
    args = parse_args()
    _validate_args(args)
    args.cuda_graphs = True
    with pytest.raises(ValueError, match="CUDA Graph"):
        _validate_args(args)
    args.cuda_graphs = False
    args.framework = "megatron"
    with pytest.raises(ValueError, match="requires Nano"):
        _validate_args(args)


def test_tp_attention_views_cli_is_opt_in_and_nano_only(monkeypatch):
    from scripts.benchmark_pp import parse_args, _validate_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py"])
    args = parse_args()
    assert not args.tp_attention_views
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--tp-attention-views"])
    args = parse_args()
    assert args.tp_attention_views
    _validate_args(args)
    args.framework = "megatron"
    with pytest.raises(ValueError, match="supports nano only"):
        _validate_args(args)


def test_stage_bucketing_cli_is_opt_in_and_only_applies_after_stage_zero(monkeypatch):
    from scripts.benchmark_pp import parse_args, _disable_stage_bucketing, _validate_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py"])
    args = parse_args()
    assert not args.disable_bucketing_after_first_stage
    assert not any(_disable_stage_bucketing(args, rank) for rank in range(4))
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--disable-bucketing-after-first-stage"])
    args = parse_args()
    _validate_args(args)
    assert [_disable_stage_bucketing(args, rank) for rank in range(4)] == [False, True, True, True]
    args.framework = "megatron"
    with pytest.raises(ValueError, match="supports nano only"):
        _validate_args(args)


def test_graph_cli_defaults_off_and_validated_configuration(monkeypatch):
    from scripts.benchmark_pp import parse_args, _validate_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py"])
    args = parse_args()
    assert not args.cuda_graphs and not args.deterministic
    assert args.cuda_graph_warmup_steps == 50
    args = _graph_args(monkeypatch)
    _validate_args(args)


def _graph_args(monkeypatch):
    from scripts.benchmark_pp import parse_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--cuda-graphs", "--deterministic", "--tp-size", "2", "--pp-size", "2", "--dp-size", "1", "--sequence-parallel", "--precision", "bf16", "--attn-backend", "flash", "--optimizer", "adamw"])
    return parse_args()


@pytest.mark.parametrize("field,value", [
    ("device", "cpu"), ("tp_size", 1), ("pp_size", 1), ("dp_size", 2),
    ("sequence_parallel", False), ("precision", "fp32"), ("attn_backend", "unfused"),
    ("optimizer", "sgd"), ("collective_backend", "nano-nccl"),
    ("overlap_p2p_comm", True), ("overlap_tp_dgrad", True), ("overlap_grad_reduce", True),
])
def test_graph_cli_rejects_unvalidated_combinations(monkeypatch, field, value):
    from scripts.benchmark_pp import _validate_args
    args = _graph_args(monkeypatch)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="CUDA graphs currently require"):
        _validate_args(args)


def test_graph_cli_requires_determinism_and_positive_extra_warmup(monkeypatch):
    from scripts.benchmark_pp import _validate_args
    args = _graph_args(monkeypatch)
    args.deterministic = False
    with pytest.raises(ValueError, match="requires --deterministic"):
        _validate_args(args)
    args.deterministic = True
    args.cuda_graph_warmup_steps = 0
    with pytest.raises(ValueError, match="must be positive"):
        _validate_args(args)


def test_graph_result_discloses_extra_updates_without_counting_tokens(monkeypatch, tmp_path):
    from scripts.benchmark_pp import _make_result, write_markdown
    args = _graph_args(monkeypatch)
    result = _make_result(framework="nano+cuda-graph", args=args, dp_size=1, elapsed=2.0, memory_mb=10.0)
    assert result.extra_warmup_updates == 50
    assert result.tokens_per_sec_global == args.benchmark_steps * args.batch_size * args.seq_len / 2.0
    path = tmp_path / "results.md"
    write_markdown([result], str(path))
    assert "50 additional real warmup updates" in path.read_text()
    assert "CUDA_GRAPH_PREPARATION" in path.read_text()


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("workspace", [None, ":16:8"])
def test_determinism_scope_restores_flags_and_environment(monkeypatch, raises, workspace):
    import os
    from types import SimpleNamespace
    from scripts.benchmark_pp import _scoped_determinism
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    @_scoped_determinism
    def call(args):
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == (workspace or ":4096:8")
        if raises:
            raise RuntimeError("deliberate")
    if raises:
        with pytest.raises(RuntimeError, match="deliberate"):
            call(SimpleNamespace(deterministic=True))
    else:
        call(SimpleNamespace(deterministic=True))
    assert torch.are_deterministic_algorithms_enabled() == enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warning
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace


def test_optimizer_validation_rejects_cpu_only_step_increment():
    from scripts.benchmark_optimizer import make_benchmark_adamw
    from scripts.benchmark_pp import _validate_training_step
    module = nn.Linear(2, 2)
    optimizer = make_benchmark_adamw(module, learning_rate=1e-4)
    module(torch.ones(1, 2)).square().mean().backward()
    optimizer.step()
    _validate_training_step(optimizer, [], "real")
    optimizer.step_count += 1  # A replay that forgot its GPU update must fail.
    with pytest.raises(RuntimeError, match="state step counts disagree"):
        _validate_training_step(optimizer, [], "false-replay")


def test_nano_channel_cli_rejects_nonpositive_value(monkeypatch):
    from scripts.benchmark_pp import parse_args, _validate_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--nano-nccl-channels", "0"])
    with pytest.raises(ValueError, match="nano-nccl-channels must be positive"):
        _validate_args(parse_args())
