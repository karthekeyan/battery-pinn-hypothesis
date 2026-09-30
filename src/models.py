"""Model arms for the hypothesis test.

Every arm has the same interface:
    fit_<arm>(train_lab, train_all, seed) -> predict(frame) -> SOH fraction array

train_lab : training rows that have a capacity label (depends on the budget)
train_all : every training row, labelled or not (usage data is always available)

Arm 1, ML alone:
    ml_mlp  neural network trained on labelled points only
    ml_gbm  gradient boosting, a strong non-neural reference

The PINN and PINN + ML arms will reuse SOHNet and train_net with an extra
physics loss, so the only difference between arms is the physics term.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from common import FEATURE_COLS, TARGET_COL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared training settings, identical for every neural arm
WIDTH = 64
EPOCHS = 3000
LR = 3e-3
WEIGHT_DECAY = 1e-4


class SOHNet(nn.Module):
    """Small MLP: history features -> SOH fraction. Tanh keeps derivatives smooth for the physics term."""

    def __init__(self, n_in, width=WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return 1.0 + self.net(x).squeeze(-1)  # starts near SOH = 1 (new battery)


def to_tensor(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=DEVICE)


def train_net(x_lab, y_lab, seed, extra_loss=None):
    """Full-batch Adam on the labelled points, plus an optional extra loss term."""
    torch.manual_seed(seed)
    net = SOHNet(x_lab.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xl, yl = to_tensor(x_lab), to_tensor(y_lab)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = torch.mean((net(xl) - yl) ** 2)
        if extra_loss is not None:
            loss = loss + extra_loss(net)
        loss.backward()
        opt.step()
    net.eval()
    return net


def fit_ml_mlp(train_lab, train_all, seed):
    scaler = StandardScaler().fit(train_all[FEATURE_COLS])
    net = train_net(scaler.transform(train_lab[FEATURE_COLS]), train_lab[TARGET_COL], seed)

    def predict(frame):
        with torch.no_grad():
            return net(to_tensor(scaler.transform(frame[FEATURE_COLS]))).cpu().numpy()
    return predict


def fit_ml_gbm(train_lab, train_all, seed):
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    model.fit(train_lab[FEATURE_COLS], train_lab[TARGET_COL])

    def predict(frame):
        return model.predict(frame[FEATURE_COLS])
    return predict


ARMS = {
    "ml_mlp": fit_ml_mlp,
    "ml_gbm": fit_ml_gbm,
}
