import torch
import torch.nn as nn
import torch.nn.functional as F

class QModelCteNonCrossing(nn.Module):
    def __init__(self, n_features, n_quantiles, n_hidden=64):
        super(QModelCteNonCrossing, self).__init__()
        self.n_quantiles = n_quantiles
        self.qfc1 = nn.Linear(n_features, n_hidden)
        self.qfc2 = nn.Linear(n_hidden, n_hidden)
        self.qfc_last_layer = nn.ModuleList()
        for i in range(self.n_quantiles):
            self.qfc_last_layer.append(nn.Linear(n_hidden, 1))

        self.ctefc1 = nn.Linear(n_features, n_hidden)
        self.ctefc2 = nn.Linear(n_hidden, n_hidden)
        self.ctefc3 = nn.Linear(n_hidden, self.n_quantiles)
        self.ctefc_last_layer = nn.ModuleList()
        for i in range(self.n_quantiles):
            self.ctefc_last_layer.append(nn.Linear(n_hidden, 1))

    def forward(self, x):
        qx = F.relu(self.qfc1(x))
        qx = F.relu(self.qfc2(qx))
        if self.n_quantiles > 0:
            qxout = torch.zeros([qx.shape[0], self.n_quantiles])
            qxout[:, 0] = self.qfc_last_layer[0](qx).squeeze(1)
            for i in range(1, self.n_quantiles):
                qxout[:, i] = qxout[:, i - 1] + torch.sigmoid(self.qfc_last_layer[i](qx)).squeeze(1)
        else:
            raise ValueError
        ctex = F.relu(self.ctefc1(x))
        ctex = F.relu(self.ctefc2(ctex))
        if self.n_quantiles > 0:
            ctexout = torch.zeros([qx.shape[0], self.n_quantiles])
            ctexout[:, 0] = torch.sigmoid(self.qfc_last_layer[0](ctex).squeeze(1))
            for i in range(1, self.n_quantiles):
                ctexout[:, i] = torch.max(torch.zeros([ctexout.shape[0]]),
                                          (qxout[:, i - 1] + ctexout[:, i - 1] - qxout[:, i])) + \
                                torch.sigmoid(self.qfc_last_layer[i](ctex)).squeeze(1)
        else:
            raise ValueError
        x_out = torch.concat([qxout, ctexout], axis=1)
        return x_out
