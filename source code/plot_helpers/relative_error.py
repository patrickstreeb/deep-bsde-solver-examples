
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from ._save import finalize, save_figure

def plot_relative_error(iterations, errors, save=None, show=True):
    errors = np.asarray(errors, dtype=float)
    mean, sd = errors.mean(axis=0), errors.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(iterations, mean, color="tab:blue")
    ax.fill_between(iterations, np.maximum(mean - sd, 1e-6), mean + sd, color="tab:blue", alpha=0.25)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative approximation error")
    ax.margins(x=0)
    save_figure(fig)
    return finalize(fig, save, show)
