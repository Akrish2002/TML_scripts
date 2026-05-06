import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FILES = {
    "1pct": {
        "noise": "integrand_data_noise_1%_n256.csv",
        "k4": "integrand_data_k4_1%_n256.csv",
    },
    "5pct": {
        "noise": "integrand_data_noise_5%_n256.csv",
        "k4": "integrand_data_k4_5%_n256.csv",
    },
    "10pct": {
        "noise": "integrand_data_noise_10%_n256.csv",
        "k4": "integrand_data_k4_10%_n256.csv",
    },
}

LABELS = {
    "1pct": {
        "noise": "n256_initialization_noise_1%",
        "k4": "n256_initialization_k4_1%",
    },
    "5pct": {
        "noise": "n256_initialization_noise_5%",
        "k4": "n256_initialization_k4_5%",
    },
    "10pct": {
        "noise": "n256_initialization_noise_10%",
        "k4": "n256_initialization_k4_10%",
    },
}


def load_curve(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)

    if data.ndim != 2 or data.shape[1] <= 1:
        raise ValueError(f"Unexpected data shape in {filename}: {data.shape}")

    time_values = data[:, 0]
    momentum_values = data[:, 1]

    return time_values, momentum_values


def get_time_from_point_number(time_values, point_number, label):
    if point_number < 1:
        raise ValueError(f"Point number for {label} must be >= 1")

    if point_number > len(time_values):
        raise ValueError(
            f"Point number for {label} exceeds data length: {point_number} > {len(time_values)}"
        )

    index = point_number - 1
    return time_values[index]


def shift_and_trim_curve(time_values, momentum_values, point_number, reference_time, extra_shift, label):
    if point_number < 1:
        raise ValueError(f"Point number for {label} must be >= 1")

    if point_number > len(time_values):
        raise ValueError(
            f"Point number for {label} exceeds data length: {point_number} > {len(time_values)}"
        )

    index = point_number - 1
    current_time = time_values[index]
    delta_t = reference_time - current_time + extra_shift

    shifted_time_values = time_values + delta_t

    trimmed_time_values = []
    trimmed_shifted_time_values = []
    trimmed_momentum_values = []

    for i in range(index, len(time_values)):
        trimmed_time_values.append(time_values[i])
        trimmed_shifted_time_values.append(shifted_time_values[i])
        trimmed_momentum_values.append(momentum_values[i])

    trimmed_time_values = np.asarray(trimmed_time_values)
    trimmed_shifted_time_values = np.asarray(trimmed_shifted_time_values)
    trimmed_momentum_values = np.asarray(trimmed_momentum_values)

    return trimmed_time_values, trimmed_shifted_time_values, trimmed_momentum_values, delta_t


def compute_slope(x_values, y_values, slope_start, slope_end, label):
    if slope_start < 1:
        raise ValueError(f"slope start for {label} must be >= 1")

    if slope_end < 1:
        raise ValueError(f"slope end for {label} must be >= 1")

    if slope_start >= slope_end:
        raise ValueError(f"slope start must be smaller than slope end for {label}")

    if slope_end > len(x_values):
        raise ValueError(
            f"slope end for {label} exceeds visible clipped curve length: {slope_end} > {len(x_values)}"
        )

    start_index = slope_start - 1
    end_index = slope_end - 1

    x_segment = x_values[start_index:end_index + 1]
    y_segment = y_values[start_index:end_index + 1]

    coefficients = np.polyfit(x_segment, y_segment, 1)
    slope_value = coefficients[0]

    return slope_value, start_index, end_index


def add_short_marker_lines(ax, x_start, x_end, color):
    ax.axvline(
        x=x_start,
        ymin=0.03,
        ymax=0.12,
        color=color,
        linestyle=":",
        linewidth=2.0,
        alpha=0.95,
    )
    ax.axvline(
        x=x_end,
        ymin=0.03,
        ymax=0.12,
        color=color,
        linestyle=":",
        linewidth=2.0,
        alpha=0.95,
    )

def add_marker_on_curve(ax, x_value, y_value, half_height, color):
    y1 = y_value - half_height
    y2 = y_value + half_height

    ax.plot(
        [x_value, x_value],
        [y1, y2],
        color=color,
        linestyle=":",
        linewidth=2.0,
        alpha=0.95,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot shifted 5 percent and 10 percent curves on top of shifted 1 percent reference curves in one subplot window."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory containing the CSV files.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--noise-1pct-point",
        type=int,
        required=True,
        help="1-based point number used to shift the noise 1 percent curve.",
    )
    parser.add_argument(
        "--k4-1pct-point",
        type=int,
        required=True,
        help="1-based point number used as the reference on the k4 1 percent curve.",
    )
    parser.add_argument(
        "--noise-5pct-point",
        type=int,
        required=True,
        help="1-based point number used to shift the noise 5 percent curve.",
    )
    parser.add_argument(
        "--k4-5pct-point",
        type=int,
        required=True,
        help="1-based point number used to shift the k4 5 percent curve.",
    )
    parser.add_argument(
        "--noise-10pct-point",
        type=int,
        required=True,
        help="1-based point number used to shift the noise 10 percent curve.",
    )
    parser.add_argument(
        "--k4-10pct-point",
        type=int,
        required=True,
        help="1-based point number used to shift the k4 10 percent curve.",
    )
    parser.add_argument(
        "--noise-1pct-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the noise 1 percent curve.",
    )
    parser.add_argument(
        "--k4-1pct-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the k4 1 percent curve.",
    )
    parser.add_argument(
        "--noise-5pct-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the noise 5 percent curve.",
    )
    parser.add_argument(
        "--k4-5pct-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the k4 5 percent curve.",
    )
    parser.add_argument(
        "--noise-10pct-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the noise 10 percent curve.",
    )
    parser.add_argument(
        "--k4-10pct-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the k4 10 percent curve.",
    )
    parser.add_argument(
        "--slope-start",
        type=int,
        required=True,
        help="1-based start point for slope fitting on the clipped visible curve.",
    )
    parser.add_argument(
        "--slope-end",
        type=int,
        required=True,
        help="1-based end point for slope fitting on the clipped visible curve.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    curves = {}
    strengths = ["1pct", "5pct", "10pct"]
    initializations = ["noise", "k4"]

    for strength in strengths:
        curves[strength] = {}
        for initialization in initializations:
            filename = data_path / FILES[strength][initialization]
            time_values, momentum_values = load_curve(filename)
            curves[strength][initialization] = {
                "time": time_values,
                "momentum": momentum_values,
            }

    reference_time = get_time_from_point_number(
        curves["1pct"]["k4"]["time"],
        args.k4_1pct_point,
        "k4_1pct"
    )

    point_numbers = {
        "1pct": {
            "noise": args.noise_1pct_point,
            "k4": args.k4_1pct_point,
        },
        "5pct": {
            "noise": args.noise_5pct_point,
            "k4": args.k4_5pct_point,
        },
        "10pct": {
            "noise": args.noise_10pct_point,
            "k4": args.k4_10pct_point,
        },
    }

    extra_shifts = {
        "1pct": {
            "noise": args.noise_1pct_shift,
            "k4": args.k4_1pct_shift,
        },
        "5pct": {
            "noise": args.noise_5pct_shift,
            "k4": args.k4_5pct_shift,
        },
        "10pct": {
            "noise": args.noise_10pct_shift,
            "k4": args.k4_10pct_shift,
        },
    }

    for strength in strengths:
        for initialization in initializations:
            raw_time_values, shifted_time_values, visible_momentum_values, delta_t = shift_and_trim_curve(
                curves[strength][initialization]["time"],
                curves[strength][initialization]["momentum"],
                point_numbers[strength][initialization],
                reference_time,
                extra_shifts[strength][initialization],
                f"{initialization}_{strength}"
            )
            curves[strength][initialization]["raw_visible_time"] = raw_time_values
            curves[strength][initialization]["shifted_time"] = shifted_time_values
            curves[strength][initialization]["shifted_momentum"] = visible_momentum_values
            curves[strength][initialization]["delta_t"] = delta_t

    for strength in strengths:
        for initialization in initializations:
            raw_slope, start_index, end_index = compute_slope(
                curves[strength][initialization]["raw_visible_time"],
                curves[strength][initialization]["shifted_momentum"],
                args.slope_start,
                args.slope_end,
                f"{initialization}_{strength} raw"
            )
            shifted_slope, _, _ = compute_slope(
                curves[strength][initialization]["shifted_time"],
                curves[strength][initialization]["shifted_momentum"],
                args.slope_start,
                args.slope_end,
                f"{initialization}_{strength} shifted"
            )

            curves[strength][initialization]["raw_slope"] = raw_slope
            curves[strength][initialization]["shifted_slope"] = shifted_slope
            curves[strength][initialization]["slope_start_index"] = start_index
            curves[strength][initialization]["slope_end_index"] = end_index

            print(f"--{initialization}_{strength} delta_t: {curves[strength][initialization]['delta_t']}")
            print(f"--{initialization}_{strength} raw slope: {raw_slope}")
            print(f"--{initialization}_{strength} shifted slope: {shifted_slope}")

    targets = [
        ("5pct", "noise"),
        ("10pct", "noise"),
        ("5pct", "k4"),
        ("10pct", "k4"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    for plot_index in range(len(targets)):
        strength, initialization = targets[plot_index]
        ax = axes[plot_index]

        line_1_noise, = ax.plot(
            curves["1pct"]["noise"]["shifted_time"],
            curves["1pct"]["noise"]["shifted_momentum"],
            "-",
            label=LABELS["1pct"]["noise"],
        )
        line_1_k4, = ax.plot(
            curves["1pct"]["k4"]["shifted_time"],
            curves["1pct"]["k4"]["shifted_momentum"],
            "-",
            label=LABELS["1pct"]["k4"],
        )
        line_target, = ax.plot(
            curves[strength][initialization]["shifted_time"],
            curves[strength][initialization]["shifted_momentum"],
            "-",
            label=LABELS[strength][initialization],
        )

        x_start = curves["1pct"]["noise"]["shifted_time"][curves["1pct"]["noise"]["slope_start_index"]]
        y_start = curves["1pct"]["noise"]["shifted_momentum"][curves["1pct"]["noise"]["slope_start_index"]]
        
        x_end = curves["1pct"]["noise"]["shifted_time"][curves["1pct"]["noise"]["slope_end_index"]]
        y_end = curves["1pct"]["noise"]["shifted_momentum"][curves["1pct"]["noise"]["slope_end_index"]]
        
        add_marker_on_curve(ax, x_start, y_start, 0.2, line_1_noise.get_color())
        add_marker_on_curve(ax, x_end, y_end, 0.2, line_1_noise.get_color())
        
        
        x_start = curves["1pct"]["k4"]["shifted_time"][curves["1pct"]["k4"]["slope_start_index"]]
        y_start = curves["1pct"]["k4"]["shifted_momentum"][curves["1pct"]["k4"]["slope_start_index"]]
        
        x_end = curves["1pct"]["k4"]["shifted_time"][curves["1pct"]["k4"]["slope_end_index"]]
        y_end = curves["1pct"]["k4"]["shifted_momentum"][curves["1pct"]["k4"]["slope_end_index"]]
        
        add_marker_on_curve(ax, x_start, y_start, 0.2, line_1_k4.get_color())
        add_marker_on_curve(ax, x_end, y_end, 0.2, line_1_k4.get_color())
        
        
        x_start = curves[strength][initialization]["shifted_time"][curves[strength][initialization]["slope_start_index"]]
        y_start = curves[strength][initialization]["shifted_momentum"][curves[strength][initialization]["slope_start_index"]]
        
        x_end = curves[strength][initialization]["shifted_time"][curves[strength][initialization]["slope_end_index"]]
        y_end = curves[strength][initialization]["shifted_momentum"][curves[strength][initialization]["slope_end_index"]]
        
        add_marker_on_curve(ax, x_start, y_start, 0.2, line_target.get_color())
        add_marker_on_curve(ax, x_end, y_end, 0.2, line_target.get_color())

        slope_text = (
            f"1% noise = {curves['1pct']['noise']['shifted_slope']:.5f}\n"
            f"1% k4 = {curves['1pct']['k4']['shifted_slope']:.5f}\n"
            f"{strength} {initialization} = {curves[strength][initialization]['shifted_slope']:.5f}"
        )

        ax.text(
            0.03,
            0.97,
            slope_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
        )

        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel("Momentum_Thickness")
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_title(f"{strength} {initialization}")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.tight_layout()

    output_file = output_path / "Momentum_Thickness_shifted_1pct_with_5pct_10pct_subplots.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"--Reference curve: k4 1%")
    print("--Reference time:", reference_time)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
