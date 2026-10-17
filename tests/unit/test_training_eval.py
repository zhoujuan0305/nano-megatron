from types import SimpleNamespace

import torch
import scripts.train_gpt as driver


def test_sp_heldout_uses_global_positions_on_nonfirst_stage(monkeypatch):
    class Stage(torch.nn.Module):
        is_first_stage = False
        is_last_stage = True
        config = SimpleNamespace(hidden_size=8)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, hidden, *, positions):
            assert hidden.shape == (1, 16, 8)
            assert torch.equal(positions, torch.arange(32).unsqueeze(0))
            assert not torch.is_grad_enabled()
            self.calls += 1
            return hidden

        def shifted_cross_entropy(self, output, labels):
            assert labels.shape == (1, 32)
            return torch.tensor(2.0)

    sampler = SimpleNamespace(sequence_length=32, batch=lambda index, size: (torch.zeros(size, 32, dtype=torch.long), {}))
    stage = Stage()
    ctx = SimpleNamespace(tensor_parallel_size=2, sequence_parallel=True, tensor_parallel_rank=0)
    monkeypatch.setattr(driver, "recv_forward", lambda ctx, **kwargs: torch.zeros(kwargs["shape"]))
    loss = driver.evaluate(stage, ctx, sampler, batch_size=4, microbatches=4, batches=2, device=torch.device("cpu"))
    assert loss == 2.0 and stage.calls == 8 and stage.training


def test_heldout_real_pp1_matches_reference_cpu(tmp_path):
    from nano_megatron.parallel import ParallelConfig, initialize_parallel, destroy_parallel
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
    from nano_megatron.model import build_pipeline_stage_from_reference
    from nano_megatron.reference.loss import shifted_cross_entropy
    import torch.distributed as dist
    destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    ctx = initialize_parallel(ParallelConfig(), dist_backend="gloo", rank=0, world_size=1,
                              init_method=f"file://{tmp_path / 'rendezvous'}")
    try:
        cfg = ReferenceGPTConfig(vocab_size=64, max_seq_len=8, hidden_size=32,
                                 num_layers=2, num_heads=4, ffn_hidden_size=64,
                                 position_embedding_type="rope", rotary_dim=8)
        reference = ReferenceGPT(cfg)
        stage = build_pipeline_stage_from_reference(reference, ctx)
        ids = torch.arange(32).reshape(4, 8)
        sampler = SimpleNamespace(sequence_length=8, batch=lambda index, size: (ids, {}))
        result = driver.evaluate(stage, ctx, sampler, batch_size=4, microbatches=4, batches=2,
                                 device=torch.device("cpu"))
        with torch.no_grad():
            expected = shifted_cross_entropy(reference(ids), ids).item()
        assert abs(result - expected) < 1e-6
        assert all(p.grad is None for p in stage.parameters())
    finally:
        destroy_parallel()
