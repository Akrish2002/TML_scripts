import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


def apply_paper_style(ax):
    # light dotted grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="0.55")

    # black frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("k")

    # tick style
    ax.tick_params(direction="out", length=4, width=1.0, colors="k")


def get_grid_name(filename):
    if filename is None:
        return None

    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    return basename.split("_")[-1]


def load_csv_thickness(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 4:
        raise ValueError(
            f"{filename} must have at least 4 columns: "
            "Time, Momentum_Thickness, VolFrac_Thickness, Mixing_Thickness"
        )

    time_steps     = data[:, 0]
    delta_theta    = data[:, 1]
    delta_alpha    = data[:, 2]
    delta_mixing   = data[:, 3]

    return time_steps, delta_theta, delta_alpha, delta_mixing


def plot_one_quantity(entries, quantity_idx, ylabel, out_name, args):
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)
    ax = fig.add_subplot(111)

    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for idx, entry in enumerate(entries):
        filename = entry["filename"]
        label    = entry["label"]

        time_steps, delta_theta, delta_alpha, delta_mixing = load_csv_thickness(filename)

        quantities = [
            delta_theta,
            delta_alpha,
            delta_mixing,
        ]

        y = quantities[quantity_idx]

        ax.plot(
            time_steps,
            y,
            color="r",
            linestyle=dash_cycle[idx % len(dash_cycle)],
            linewidth=1.2,
            label=label,
        )

        # file path text at bottom-right, like TKE script
        p = Path(filename)
        short = Path(*p.parts[-2:]) if len(p.parts) >= 2 else p

        fig.text(
            0.98,
            0.01 + 0.025 * idx,
            str(short),
            ha="right",
            va="bottom",
            fontsize=5,
        )

    ax.set_xlabel(r"$t^*$", fontsize=16)
    ax.set_ylabel(ylabel)

    apply_paper_style(ax)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout(pad=1.0)

    out_path = Path(args.out_dir) / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"--Saved {out_path}")


def perform_plotting(args):
    dataset_files = [
        args.dataset_1,
        args.dataset_2,
        args.dataset_3,
        args.dataset_4,
        args.dataset_5,
        args.dataset_6,
    ]

    dataset_names = [
        get_grid_name(args.dataset_1),
        get_grid_name(args.dataset_2),
        get_grid_name(args.dataset_3),
        get_grid_name(args.dataset_4),
        get_grid_name(args.dataset_5),
        get_grid_name(args.dataset_6),
    ]

    entries = []

    for idx, filename in enumerate(dataset_files):
        if filename is None:
            continue

        if not os.path.exists(filename):
            print(f"--File {filename} not found, skipping.")
            continue

        label = dataset_names[idx]
        if label is None:
            label = f"dataset_{idx + 1}"

        entries.append(
            {
                "filename": filename,
                "label": label,
            }
        )

    if len(entries) == 0:
        raise RuntimeError("--No valid datasets found.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    plot_one_quantity(
        entries,
        quantity_idx=0,
        ylabel=r"$\delta_\theta$",
        out_name="delta_theta_overlay.png",
        args=args,
    )

    plot_one_quantity(
        entries,
        quantity_idx=1,
        ylabel=r"$\delta_\alpha$",
        out_name="delta_alpha_overlay.png",
        args=args,
    )

    plot_one_quantity(
        entries,
        quantity_idx=2,
        ylabel=r"$\delta_{\mathrm{mixing}}$",
        out_name="delta_mixing_overlay.png",
        args=args,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot thickness datasets")

    parser.add_argument("--dataset_1", type=str, default=None, help="CSV file for dataset 1")
    parser.add_argument("--dataset_2", type=str, default=None, help="CSV file for dataset 2")
    parser.add_argument("--dataset_3", type=str, default=None, help="CSV file for dataset 3")
    parser.add_argument("--dataset_4", type=str, default=None, help="CSV file for dataset 4")
    parser.add_argument("--dataset_5", type=str, default=None, help="CSV file for dataset 5")
    parser.add_argument("--dataset_6", type=str, default=None, help="CSV file for dataset 6")

    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--figsize", type=float, nargs=2, default=[6.0, 4.0], help="Figure size")

    args = parser.parse_args()

    perform_plotting(args)


if __name__ == "__main__":
    main()
