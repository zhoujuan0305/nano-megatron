"""Metadata-only flash call test; no CUDA context or optional extension needed."""
import sys
from types import SimpleNamespace

import pytest
import torch
from nano_megatron.parallel import attention_backend as backend


@pytest.mark.parametrize('deterministic', [False, True])
def test_flash_cp1_honors_process_determinism(monkeypatch, deterministic):
    calls=[]
    result=object()
    def flash(*args,**kwargs):
        calls.append(kwargs)
        return result
    monkeypatch.setitem(sys.modules,'flash_attn',SimpleNamespace(flash_attn_func=flash))
    monkeypatch.setattr(torch,'are_deterministic_algorithms_enabled',lambda:deterministic)
    monkeypatch.setattr(backend,'_to_bshd',lambda x:x)
    monkeypatch.setattr(backend,'_to_bhsd',lambda x:x)
    q=SimpleNamespace(dim=lambda:4,shape=(1,2,8,16),device=torch.device('cuda'),dtype=torch.bfloat16)
    assert backend.flash_causal_attention(q,q,q,scale=0.25) is result
    if deterministic:assert calls[0]['deterministic'] is True
    else:assert 'deterministic' not in calls[0]  # Default optional-version compatibility.
