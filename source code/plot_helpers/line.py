
import numpy as np
import matplotlib.pyplot as plt
from ._save import resolve, save_figure


def plot_line(x, y, xlabel="", ylabel="", color="#0519EF",
              vline=None, vline_label=None, hline=None, hline_label=None,
              ax=None, save=None):
    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color=color)
    if vline is not None:
        for i, v in enumerate(np.atleast_1d(vline)):
            ax.axvline(v, color="#ff2600", ls="--", label=vline_label if i == 0 else None)
    if hline is not None:
        for i, h in enumerate(np.atleast_1d(hline)):
            ax.axhline(h, color="#ff2600", ls="--", label=hline_label if i == 0 else None)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.margins(x=0)
    if vline_label or hline_label:
        ax.legend()
    if created:
        save_figure(ax.get_figure())
    if save is not None:
        ax.get_figure().savefig(resolve(save), dpi=200, bbox_inches="tight")
    return ax
