import torch


def test_compile_cache_has_static_code_identity_without_global_config(monkeypatch):
    import nano_megatron.parallel.fused_pointwise as helper
    observed = []
    def compile_stub(fn, **options):
        observed.append((fn, options))
        return fn
    helper._compiled.cache_clear()
    monkeypatch.setattr(torch, 'compile', compile_stub)
    try:
        signatures = [helper._signature(torch.empty(1, 2, 3, d)) for d in range(2, 26, 2)]
        functions = [helper._compiled('rotary', (s, True)) for s in signatures]
        assert len({id(f.__code__) for f in functions}) == len(signatures) > 8
        assert helper._compiled('rotary', (signatures[0], True)) is functions[0]
        assert len(observed) == len(signatures)
        assert all(opts == dict(fullgraph=True, dynamic=False, backend='inductor',
                               options={'triton.cudagraphs': False}) for _, opts in observed)
        x, freq = torch.randn(1, 2, 3, 2), torch.randn(1, 3, 1)
        torch.testing.assert_close(functions[0](x, freq), helper.rotary_reference(x, freq))
        torch.testing.assert_close(functions[0](x, freq, fp32_phase=True),
                                   helper.rotary_reference(x, freq, fp32_phase=True))
    finally:
        helper._compiled.cache_clear()
