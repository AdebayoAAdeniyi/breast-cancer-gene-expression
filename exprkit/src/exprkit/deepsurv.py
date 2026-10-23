"""Optional neural Cox model (DeepSurv style; Katzman et al. 2018, BMC
Medical Research Methodology 18:24), written so its behaviour can be tested.

Replaces the hazard part of model_survival_2.py and trainer_survival.py.

A common bug in hand written Cox training loops is an ad hoc sign "repair":
negate the scores, compute a C index, and flip back if it looks low. When the
negation already reversed a good ordering, such a rule fires exactly when the
model is doing well. Here the sign is fixed once:
the network outputs a log hazard, and ``cox_ph_loss`` is the Breslow negative
log partial likelihood for that log hazard. tests/test_survival.py checks it
against a slow reference implementation and checks that training on
simulated data recovers the true direction.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn

from .metrics import harrell_c


def cox_ph_loss(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Breslow negative log partial likelihood, averaged over events.

    Exact Breslow handling of tied times: every patient whose time equals the
    event time is in the risk set.
    """
    log_risk = log_risk.reshape(-1)
    order = torch.argsort(time, descending=True)
    lr, t, e = log_risk[order], time[order], event[order].to(log_risk.dtype)
    log_cum = torch.logcumsumexp(lr, dim=0)
    # For tied times take the cumulative sum up to the last patient with that time.
    neg_t = -t
    last = torch.searchsorted(neg_t.contiguous(), neg_t.contiguous(), right=True) - 1
    log_cum = log_cum[last]
    return -((lr - log_cum) * e).sum() / e.sum().clamp(min=1.0)


class DeepSurvNet(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (64,), dropout: float = 0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_deepsurv(X_train, t_train, e_train, X_val, t_val, e_val, hidden=(64,), dropout=0.3,
                   lr=1e-3, weight_decay=1e-3, max_epochs=500, patience=30, seed=0):
    """Full batch training with early stopping on validation C index.

    Full batch (a few thousand patients fit in memory) keeps every risk set
    complete, which mini batches of 100 do not. The weights from the best
    validation epoch are restored before returning.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    Xt, tt, et = f(X_train), f(t_train), f(e_train)
    Xv = f(X_val)
    model = DeepSurvNet(Xt.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, wait, history = -np.inf, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = cox_ph_loss(model(Xt), tt, et)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            c_val = harrell_c(t_val, e_val, model(Xv).numpy())
        history.append({"epoch": epoch + 1, "train_loss": loss.item(), "val_c": c_val})
        if c_val > best + 1e-4:
            best, best_state, wait = c_val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_risk(model: DeepSurvNet, X) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()


class DeepSurvEstimator(BaseEstimator):
    """scikit-learn style wrapper so the neural Cox model fits in a Pipeline.

    ``fit(X, y)`` takes a scikit-survival structured array ``y`` with fields
    ``event`` and ``time``. A validation fraction of the training rows,
    stratified on the event indicator, is held back for early stopping.
    ``predict`` returns log hazard scores; ``score`` returns Harrell's C.
    """

    def __init__(self, hidden=(64,), dropout=0.3, lr=1e-3, weight_decay=1e-3, max_epochs=500,
                 patience=30, val_fraction=0.2, seed=0):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed


    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        X = np.asarray(X, dtype=np.float32)
        e, t = np.asarray(y["event"]).astype(float), np.asarray(y["time"], dtype=float)
        tr, va = train_test_split(np.arange(len(t)), test_size=self.val_fraction, stratify=e,
                                  random_state=self.seed)
        self.model_, self.history_ = train_deepsurv(
            X[tr], t[tr], e[tr], X[va], t[va], e[va], hidden=tuple(self.hidden), dropout=self.dropout,
            lr=self.lr, weight_decay=self.weight_decay, max_epochs=self.max_epochs, patience=self.patience,
            seed=self.seed)
        return self

    def predict(self, X):
        return predict_risk(self.model_, np.asarray(X, dtype=np.float32))

    def score(self, X, y):
        return harrell_c(y["time"], y["event"], self.predict(X))
