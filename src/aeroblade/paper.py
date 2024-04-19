import re
from typing import Literal, Optional

import matplotlib.pyplot as plt
import torch

DATASET_ORDER = ["SD1.1", "SD1.5", "SD2.1", "KD2.1", "MJ4", "MJ5", "MJ5.1", "Real"]


def get_nice_name(input: str) -> str:
    # datasets
    if "data/raw/" in input:
        if "real" in input:
            return "Real"
        elif "stable-diffusion" in input:
            version = re.search("(\d)-(\d)", input)
            return f"SD{version.group(1)}.{version.group(2)}"
        elif "kandinsky" in input:
            version = re.search("(\d)-(\d)", input)
            return f"KD{version.group(1)}.{version.group(2)}"
        elif "midjourney" in input:
            version = re.search("-v(.*)", input)
            return f"MJ{version.group(1).replace('-', '.')}"
    # AEs
    elif input == "CompVis/stable-diffusion-v1-1":
        return "SD1"
    elif input == "stabilityai/stable-diffusion-2-base":
        return "SD2"
    elif input == "kandinsky-community/kandinsky-2-1":
        return "KD2.1"

    # distance metrics
    elif input.startswith("lpips"):
        if "alex" in input:
            net = " (AlexNet)"
        elif "squeeze" in input:
            net = " (SqueezeNet)"
        else:
            net = ""
        if input.endswith("0"):
            return "LPIPS" + net
        else:
            return f"LPIPS$_{input[-1]}$" + net
    elif input in (
        metric_dict := {
            "dists": "DISTS",
            "psnr": "PSNR",
            "ssimc": "SSIM",
            "ms_ssim": "MS-SSIM",
        }
    ):
        return metric_dict[input]
    else:
        return input


def configure_mpl() -> None:
    plt.rcdefaults()
    params = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath}",
        "font.family": "serif",
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "legend.handlelength": 1.0,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.labelpad": 2.0,
        "xtick.major.pad": 1.0,
        "ytick.major.pad": 1.0,
        "lines.linewidth": 0.75,
        "lines.markersize": 2,
    }

    plt.rcParams.update(params)

    figure_params = {
        "figure.dpi": 300,
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.0,
    }

    plt.rcParams.update(figure_params)

    set_figsize()


def set_figsize(
    format: Literal["single", "double"] = "single",
    ratio: float = 2 / (1 + 5**0.5),
    factor: float = 1.0,
    nrows: int = 1,
    ncols: int = 1,
) -> None:
    """
    Set width and height of figure.

    :param width: Width of figure, single or double column.
    :param ratio: Ratio between width and height (height = width * ratio).
        Defaults to golden ratio.
    :param factor: Scaling factor for both width and height.
    :param nrows, ncols: Number of rows/columns if subplots are used.
    """
    if format == "single":
        width = 237.13594
    elif format == "double":
        width = 496.85625
    else:
        raise ValueError

    height = width * ratio * (nrows / ncols)
    factor = 100 / 7227 * factor
    plt.rcParams["figure.figsize"] = width * factor, height * factor


def colorbar(mappable):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


@plt.rc_context({"figure.dpi": 600, "axes.grid": False})
def plot_tensor(
    image: torch.Tensor,
    overlay: Optional[torch.Tensor] = None,
    alpha: float = 0.9,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_cbar: bool = True,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    img = ax.imshow(image.permute(1, 2, 0))
    cbar = colorbar(img)
    cbar.remove()
    if overlay is not None:
        ol = ax.imshow(overlay, alpha=alpha, vmin=vmin, vmax=vmax)
        if show_cbar:
            colorbar(ol)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
