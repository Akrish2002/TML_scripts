# AI Assistance Declaration:
# The entire script was generated with the assistance of OpenAI ChatGPT (GPT-5.2).
# The generated code has only been lightly tested for basic functionality.
# Any errors, numerical inaccuracies, or unintended behavior must be rigorously
# investigated and validated by the user. None of the sections have undergone
# manual review or verification beyond minimal execution.

# =========================
# SCRIPT 2: plot_eps_compare.py
# Loads saved vectors, interpolates to a common grid, plots with paper-style aesthetic.
# Example:
#python3 plot_terms_to_compare_v2.py --inputs eps_n256_ts5500.npz eps_n512_ts12000.npz eps_n1024_ts108000.npz --labels "M1 (256³)" "M2 (512³)" "M3 (1024³)" --out eps_compare.png --zoom_pi --pi_window 0.6
# =========================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


def y_cell_centers(ny: int, y_min: float = 0.0, y_max: float = 2 * np.pi) -> np.ndarray:
    dy = (y_max - y_min) / ny
    return y_min + (np.arange(ny) + 0.5) * dy


def interp_periodic_2pi(y_src: np.ndarray, f_src: np.ndarray, y_tgt: np.ndarray) -> np.ndarray:
    L = 2.0 * np.pi
    y_src = np.mod(np.asarray(y_src).ravel(), L)
    f_src = np.asarray(f_src).ravel()
    y_tgt = np.mod(np.asarray(y_tgt).ravel(), L)

    idx = np.argsort(y_src)
    y_sorted = y_src[idx]
    f_sorted = f_src[idx]

    y_ext = np.concatenate([y_sorted[-1:] - L, y_sorted, y_sorted[:1] + L])
    f_ext = np.concatenate([f_sorted[-1:],     f_sorted, f_sorted[:1]])

    return np.interp(y_tgt, y_ext, f_ext)


def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    case = str(d["case"])
    ts = int(d["time_step"])
    ny = int(d["ny"])
    y = np.asarray(d["y"], dtype=np.float64).squeeze()
    eps = np.asarray(d["eps"], dtype=np.float64).squeeze()
    if y.shape != (ny,) or eps.shape != (ny,):
        raise ValueError(f"{path}: expected y, eps shape ({ny},), got {y.shape}, {eps.shape}")
    return case, ts, ny, y, eps


def apply_paper_style(ax):
    # light dotted grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="0.55")

    # black frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("k")

    # tick style
    ax.tick_params(direction="out", length=4, width=1.0, colors="k")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="list of .npz files from compute_eps_save.py")
    p.add_argument("--labels", nargs="*", default=None, help="optional labels (same count as inputs)")
    p.add_argument("--out", default="eps_compare.png")
    p.add_argument("--zoom_pi", action="store_true", help="zoom around y=pi")
    p.add_argument("--pi_window", type=float, default=0.5, help="half-width around pi (radians)")
    p.add_argument("--common_ny", type=int, default=0, help="override common ny (0 => use max ny)")
    #p.add_argument("--title", default="Dissipation", help="title text shown inside plot")
    p.add_argument("--title", help="title text shown inside plot")
    p.add_argument("--ylabel", default="Dissipation", help="y-axis label")
    p.add_argument("--scale", type=float, default=1.0, help="multiply eps by this factor before plotting")
    p.add_argument("--ylim", nargs=2, type=float, default=None, help="y-limits: ymin ymax (optional)")
    p.add_argument("--figsize", nargs=2, type=float, default=(7.2, 5.0), help="figure size in inches: W H")
    args = p.parse_args()

    entries = [load_npz(f) for f in args.inputs]

    if args.labels is not None and len(args.labels) not in (0, len(entries)):
        raise ValueError("--labels must be omitted or have the same length as --inputs")

    ny_common = args.common_ny if args.common_ny > 0 else max(ny for _, _, ny, _, _ in entries)
    y_common = y_cell_centers(ny_common)

    if args.zoom_pi:
        x1 = np.pi - args.pi_window
        x2 = np.pi + args.pi_window
        mask = (y_common >= x1) & (y_common <= x2)
    else:
        mask = slice(None)

    # --- paper-style plot ---
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)
    ax = fig.add_subplot(111)

    # all-red lines with different dashes (paper-like)
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]
    for idx, (case, ts, ny, y, eps) in enumerate(entries):
        lab = (
            args.labels[idx]
            if args.labels and len(args.labels) == len(entries)
            else f"{Path(case).name} (ny={ny}, ts={ts})"
        )
        eps_common = interp_periodic_2pi(y, eps, y_common) * args.scale
        ax.plot(y_common[mask], eps_common[mask], color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                linewidth=1.2, label=lab)

    # labels
    ax.set_xlabel("y")
    ax.set_ylabel(args.ylabel)

    if args.zoom_pi:
        ax.set_xlim(np.pi - args.pi_window, np.pi + args.pi_window)

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])

    apply_paper_style(ax)

    # title inside plot area (paper-like)
    ax.text(0.50, 0.93, args.title, transform=ax.transAxes,
            ha="center", va="top", fontsize=14, color="k")

    # legend in upper-right, no box
    #ax.legend(loc="upper right", frameon=False)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

