import torch
import torch.nn as nn


def G1(z, W):
    return 0


def pG2(z, q):
    return torch.log(z)


def G2(z, q):
    return 1 / z


def a(z, W, q):
    return 0


def general_loss_function(X1, X2, Y, q, W, G1=G1, G2=G2, pG2=pG2, a=a):
    return ((Y>X1)*1) * (-G1(X1, W) + G1(Y, W) - G2(X2, q) * (X1 - Y)) + \
    (1-q) * (G1(X1, W) - G2(X2, q) * (X2 - X1) + pG2(X2, q)) + a(Y, W, q)


class QuantileLossCTE(nn.Module):
    def __init__(self, quantiles, w=100, loss_scale=1000):
        super().__init__()
        self.quantiles = quantiles
        self.w = w
        self.loss_scale = loss_scale

    def forward(self, preds, target, weights=None):
        preds_q = preds[:, :len(self.quantiles)]
        preds_cte = preds[:, len(self.quantiles):]
        assert not target.requires_grad
        assert preds_q.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds_q[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        if weights is not None:
            losses_tensor = torch.cat(losses, axis=1) * torch.Tensor(weights).unsqueeze(1)
        else:
            losses_tensor = torch.cat(losses, axis=1)
        loss_q = torch.sum(torch.nansum(losses_tensor, dim=0))

        losses_cte = []
        for i, q in enumerate(self.quantiles):

            cte_loss = general_loss_function(preds_q[:, i],
                                             preds_q[:, i]+preds_cte[:, i],
                                             target,
                                             q,
                                             self.w).unsqueeze(1)  # Unsqueeze for proper tensor format

            losses_cte.append(cte_loss / self.loss_scale)
        if weights is not None:
            losses_cte_tensor = torch.cat(losses_cte, axis=1) * torch.Tensor(weights).unsqueeze(1)
        else:
            losses_cte_tensor = torch.cat(losses_cte, axis=1)
        loss_cte = torch.sum(torch.nansum(losses_cte_tensor, dim=0))

        return loss_cte, loss_cte.detach().item()
