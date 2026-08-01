from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from ._save import finalize, save_figure


def plot_y0_convergence(iterations, values, reference=None, ylabel="$Y_0$", save=None, show=True):
    values = np.asarray(values, dtype=float)
    mean, sd = values.mean(axis=0), values.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(iterations, mean, color="tab:blue")
    ax.fill_between(iterations, mean - sd, mean + sd, color="tab:blue", alpha=0.25)
    if reference is not None:
        ax.axhline(reference, color="black", ls="--", lw=1.0, label="reference")
        ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    save_figure(fig)
    return finalize(fig, save, show)
