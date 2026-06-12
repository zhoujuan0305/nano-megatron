import torch

from nano_megatron.model.loss import cross_entropy_loss


class TestCrossEntropy:
    def test_loss_shape(self) -> None:
        logits = torch.randn(4, 10)
        labels = torch.tensor([1, 2, 3, 4])
        loss = cross_entropy_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_correct_prediction_lower_loss(self) -> None:
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        labels = torch.tensor([0, 1])
        loss = cross_entropy_loss(logits, labels)
        assert loss.item() < 0.01
