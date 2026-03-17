import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    case = str(d["case"])
    ts = int(d["time_step"])
    ny = int(d["ny"])
    y = np.asarray(d["y"], dtype=np.float64).squeeze()
    #y = np.asarray(d["zeta"], dtype=np.float64).squeeze()
    u_avg_normalized = np.asarray(d["u_avg_normalized"], dtype=np.float64).squeeze()
    if y.shape != (ny,) or u_avg_normalized.shape != (ny,):
        raise ValueError(f"{path}: expected y, u_avg_normalized shape ({ny},), got {y.shape}, {u_avg_normalized.shape}")
    return case, ts, ny, y, u_avg_normalized


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
    p.add_argument("--inputs", nargs="+", required=True, help="List of .npz files")
    p.add_argument("--labels", nargs="*", default=None, help="optional labels (same count as inputs)")
    p.add_argument("--out", default="OUTPUT.png")
    p.add_argument("--zoom_pi", action="store_true", help="zoom around y=pi")
    p.add_argument("--pi_window", type=float, default=0.5, help="half-width around pi (radians)")
    p.add_argument("--common_ny", type=int, default=0, help="override common ny (0 => use max ny)")
    #p.add_argument("--title", default="Dissipation", help="title text shown inside plot")
    p.add_argument("--title", help="title text shown inside plot")
    p.add_argument("--ylabel", default="u_bar/Ug", help="y-axis label")
    #p.add_argument("--scale", type=float, default=1.0, help="multiply eps by this factor before plotting")
    p.add_argument("--ylim", nargs=2, type=float, default=None, help="y-limits: ymin ymax (optional)")
    p.add_argument("--figsize", nargs=2, type=float, default=(7.2, 5.0), help="figure size in inches: W H")
    args = p.parse_args()

    entries = [load_npz(f) for f in args.inputs]

    if args.labels is not None and len(args.labels) not in (0, len(entries)):
        raise ValueError("--labels must be omitted or have the same length as --inputs")

    # --- paper-style plot ---
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)
    ax = fig.add_subplot(111)

    # all-red lines with different dashes (paper-like)
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]
    for idx, (case, ts, ny, y, u_avg_normalized) in enumerate(entries):
        lab = (
            args.labels[idx]
            if args.labels and len(args.labels) == len(entries)
            else f"{Path(case).name} (ny={ny}, ts={ts})"
        )

        #y = interp_periodic_2pi(y, u_avg_normalized, y_common) * args.scale
        y = np.asarray(y).squeeze()
        u_avg_normalized = np.asarray(u_avg_normalized).squeeze()

        #No sorting with respect to y

        #Zoom mask in this
        if args.zoom_pi:
            #x1 = np.pi - args.pi_window
            #x2 = np.pi + args.pi_window

            x1 = 0 - args.pi_window
            x2 = 0 + args.pi_window
            m = (y >= x1) & (y <= x2)
            ax.plot(y[m], u_avg_normalized[m], color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                    linewidth=1.2, label=lab)
        else:
            ax.plot(y, u_avg_normalized, color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                    linewidth=1.2, label=lab)

    #Labels
    ax.set_xlabel("y")
    ax.set_ylabel(args.ylabel)

    if args.zoom_pi:
        #ax.set_xlim(np.pi - args.pi_window, np.pi + args.pi_window)
        ax.set_xlim(0 - args.pi_window, 0 + args.pi_window)

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
