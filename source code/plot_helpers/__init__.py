"""Plot functions used by deep_bsde_pnas.ipynb, copied from the examples repository."""
from ._save import (
    IMAGES_DIR,
    finalize,
    save_figure,
    set_example,
)
from .line import plot_line
from .relative_error import plot_relative_error
from .value_comparison import plot_value_comparison
from .y0_convergence import plot_y0_convergence

__all__ = [
    "IMAGES_DIR",
    "finalize",
    "plot_line",
    "plot_relative_error",
    "plot_value_comparison",
    "plot_y0_convergence",
    "save_figure",
    "set_example",
]
