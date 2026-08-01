

""" helpers for saving images """


from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


def resolve(save):
    path = Path(save)
    if not path.is_absolute():
        IMAGES_DIR.mkdir(exist_ok=True)
        path = IMAGES_DIR / path
    return path


def finalize(fig, save=None, show=True, dpi=200):
    global _last_fig
    _last_fig = fig
    fig.tight_layout()
    if save is not None:
        fig.savefig(resolve(save), dpi=dpi)
    if show:
        plt.show()
    return fig

_example_name = None
_figure_count = 0
_last_fig = None

def set_example(name):
    global _example_name, _figure_count
    _example_name = name
    _figure_count = 0


def save_figure(fig=None):
    global _figure_count
    if _example_name is None:
        return
    fig = fig if fig is not None else (_last_fig if _last_fig is not None else plt.gcf())
    _figure_count += 1
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / f"{_example_name}_{_figure_count}.png", dpi=150, bbox_inches="tight")
