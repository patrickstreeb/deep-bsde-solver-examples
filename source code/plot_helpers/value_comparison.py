
from __future__ import annotations
import matplotlib.pyplot as plt
from ._save import finalize, save_figure
_STYLES = ["o-", "s--", "^:", "d-.", "v-", "*--"]


def plot_value_comparison(x, series, xlabel="", ylabel="", save=None, show=True):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for style, (label, values) in zip(_STYLES, series.items()):
        ax.plot(x, values, style, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_figure(fig)
    return finalize(fig, save, show)
