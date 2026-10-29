import torch
import torch.nn as nn


class TotalLoss(nn.Module):
    def __init__(self, loss_param, num_samples_per_classes, cuda_id):
        super().__init__()
        self.loss_param = loss_param
        self.mse = nn.MSELoss()

    @staticmethod
    def ccc_loss(logits, targets, eps=1e-8):
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

        pred_mean = logits.mean(dim=0)
        target_mean = targets.mean(dim=0)
        pred_centered = logits - pred_mean
        target_centered = targets - target_mean

        covariance = (pred_centered * target_centered).mean(dim=0)
        pred_var = pred_centered.square().mean(dim=0)
        target_var = target_centered.square().mean(dim=0)

        ccc = 2 * covariance / (
            pred_var + target_var
            + (pred_mean - target_mean).square() + eps
        )
        return (1 - ccc).mean()

    def forward(self, logits, targets, emb=None, emb_norm=None,
                step=None, summary_writer=None):
        loss = logits.new_zeros(())

        if "MSE" in self.loss_param:
            loss = loss + self.loss_param["MSE"]["w"] * self.mse(logits, targets)

        if "CCC" in self.loss_param:
            loss = loss + self.loss_param["CCC"]["w"] * self.ccc_loss(logits, targets)

        return loss

