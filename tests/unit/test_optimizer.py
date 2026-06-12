from nano_megatron.config import GPTConfig
from nano_megatron.model.gpt import GPTModel
from nano_megatron.training.optimizer import create_adam_optimizer


class TestCreateAdamOptimizer:
    def test_optimizer_groups(self) -> None:
        config = GPTConfig()
        model = GPTModel(config)
        optimizer = create_adam_optimizer(model)
        assert len(optimizer.param_groups) == 2

    def test_bias_no_weight_decay(self) -> None:
        config = GPTConfig()
        model = GPTModel(config)
        optimizer = create_adam_optimizer(model)
        bias_group = optimizer.param_groups[1]
        assert bias_group["weight_decay"] == 0.0

    def test_weight_decay_group(self) -> None:
        config = GPTConfig()
        model = GPTModel(config)
        optimizer = create_adam_optimizer(model, weight_decay=0.05)
        weight_group = optimizer.param_groups[0]
        assert weight_group["weight_decay"] == 0.05
