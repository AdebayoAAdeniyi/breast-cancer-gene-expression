"""Figures written by train and evaluate. Drawn only from results, never at import."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def roc(y_true, scores_by_run: list[np.ndarray], positive, path: str | Path, title: str):
    y = (np.asarray(y_true).astype(str) == str(positive)).astype(int)
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, s in enumerate(scores_by_run):
        m = ~np.isnan(s)
        fpr, tpr, _ = roc_curve(y[m], s[m])
        ax.plot(fpr, tpr, color="tab:blue", alpha=0.25 if len(scores_by_run) > 1 else 1, lw=1 if len(scores_by_run) > 1 else 2)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel=f"True positive rate ({positive})", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion(cm: list[list[int]], labels: list[str], path: str | Path, title: str):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def km_by_group(time, event, groups, path: str | Path, title: str, time_label: str = "Time") -> dict:
    from sksurv.compare import compare_survival
    from sksurv.nonparametric import kaplan_meier_estimator
    time, event, groups = np.asarray(time, float), np.asarray(event).astype(bool), np.asarray(groups).astype(str)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    order = [g for g in ["low", "middle", "high"] if g in set(groups)] or sorted(set(groups))
    # ordered groups: one hue, light (low risk) to dark (high risk)
    ramp = ["#6da7ec", "#2a78d6", "#104281"] if len(order) == 3 else [None] * len(order)
    for g, col in zip(order, ramp):
        m = groups == g
        x, s = kaplan_meier_estimator(event[m], time[m])
        ax.step(x, s, where="post", color=col, linewidth=2,
                label=f"{g} risk (n={m.sum()}, events={event[m].sum()})")
    y = np.array(list(zip(event, time)), dtype=[("event", bool), ("time", float)])
    chisq, p = compare_survival(y, groups)
    ax.set(xlabel=time_label, ylabel="Survival probability", title=f"{title}\nlog rank p = {p:.3g}")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"logrank_chisq": float(chisq), "logrank_p": float(p)}


def scatter(y_true, y_pred, path: str | Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo, hi = np.nanmin([y_true, y_pred]), np.nanmax([y_true, y_pred])
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
    ax.set(xlabel="Observed", ylabel="Predicted", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
