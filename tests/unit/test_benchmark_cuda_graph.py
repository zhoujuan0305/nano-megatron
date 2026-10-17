"""CPU mocks test bookkeeping only, never CUDA or numerical equivalence."""
from contextlib import contextmanager
from types import SimpleNamespace
import gc

import pytest
import torch
from scripts.benchmark_cuda_graph import BenchmarkCudaGraphStep


@pytest.fixture
def setup(monkeypatch):
    step = torch.tensor(0.)
    master = object()
    param = SimpleNamespace(grad=None)
    optimizer = SimpleNamespace(fused=True, step_count=0, _masters=[master], _model_params=[param],
        optimizer=SimpleNamespace(state={master: {'step': step}}, defaults={}, param_groups=[{}]))
    loss, grad, losses = torch.tensor(0.), torch.tensor([2.]), []
    plan = SimpleNamespace(capturing=False, replays=0, eager_calls=0, capture_updates=False, count_capture=True)
    class Graph:
        def replay(self):
            plan.replays += 1
            step.add_(1)
            loss.copy_(step + 10)
    @contextmanager
    def capture(_):
        plan.capturing=True
        try:yield
        finally:plan.capturing=False
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
    monkeypatch.setattr(torch.cuda, 'graph', capture)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    def body():
        if not plan.capturing or plan.count_capture:optimizer.step_count += 1
        if not plan.capturing:
            plan.eager_calls += 1
            step.add_(1)
        elif plan.capture_updates:step.add_(1)
        param.grad=grad
        losses[:]=[loss]
    runner=BenchmarkCudaGraphStep(body, optimizer, losses, torch.device('cuda'))
    # Explicitly mock the hardware guard; counter math still uses real CPU tensors.
    monkeypatch.setattr(runner, '_state_steps', lambda: [step])
    return runner, optimizer, step, plan, losses, loss, grad


def test_warmup_capture_and_replay_counters(setup):
    runner,opt,step,plan,*_=setup
    assert opt.optimizer.defaults['capturable'] is True
    assert opt.optimizer.param_groups[0]['capturable'] is True
    runner.run()
    assert opt.step_count == step.item() == 1
    info=runner.prepare(2)
    assert opt.step_count == step.item() == 3
    runner.run()
    assert opt.step_count == step.item() == 4
    assert plan.eager_calls == 1 and plan.replays == 3
    assert info['extra_warmup_updates']==2 and info['capture_gpu_update_delta']==0


def test_loss_and_gradient_addresses_retained(setup):
    runner,_,_,_,losses,loss,grad=setup
    runner.prepare(1)
    assert runner.captured_losses[0] is loss
    assert runner.captured_gradients[0] is grad
    losses.clear();runner.run()
    assert losses[0] is loss
    runner.close()
    assert runner.captured_losses == runner.captured_gradients == ()
    with pytest.raises(RuntimeError,match='closed'):runner.run()


def test_capture_gpu_update_rejected_without_fallback(setup):
    runner,_,_,plan,*_=setup
    plan.capture_updates=True
    with pytest.raises(RuntimeError,match='unexpectedly executed'):runner.prepare(1)
    with pytest.raises(RuntimeError,match='failed'):runner.run()
    with pytest.raises(RuntimeError,match='failed'):runner.prepare(1)


def test_wrong_cpu_delta_rejected(setup):
    runner,_,_,plan,*_=setup
    plan.count_capture=False
    with pytest.raises(RuntimeError,match='CPU step counter'):runner.prepare(1)


def test_prepare_guards_and_gc_unchanged(setup):
    runner,*_=setup
    enabled,threshold=gc.isenabled(),gc.get_threshold()
    with pytest.raises(ValueError,match='positive'):runner.prepare(0)
    runner.prepare(1)
    with pytest.raises(RuntimeError,match='already prepared'):runner.prepare(1)
    assert gc.isenabled()==enabled and gc.get_threshold()==threshold


def test_real_hardware_guard_rejects_cpu_optimizer_state(setup):
    runner,*_=setup
    with pytest.raises(RuntimeError,match='CUDA step tensors'):
        BenchmarkCudaGraphStep._state_steps(runner)


def test_non_cuda_and_unfused_rejected(setup):
    runner,opt,*_=setup
    with pytest.raises(ValueError,match='CUDA fused'):
        BenchmarkCudaGraphStep(runner.step_fn,opt,[],torch.device('cpu'))
    opt.fused=False
    with pytest.raises(ValueError,match='CUDA fused'):
        BenchmarkCudaGraphStep(runner.step_fn,opt,[],torch.device('cuda'))
