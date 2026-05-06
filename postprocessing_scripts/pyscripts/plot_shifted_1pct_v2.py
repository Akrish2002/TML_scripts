import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FILES_1PCT = {
    "noise": "integrand_data_noise_1%_n256.csv",
    "k4": "integrand_data_k4_1%_n256.csv",
    "constant": "integrand_data_constant_1%_n256.csv",
}

LABELS_1PCT = {
    "noise": "n256_initialization_noise_1%",
    "k4": "n256_initialization_k4_1%",
    "constant": "n256_initialization_constant_1%",
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
        ymax=0.13,
        color=color,
        linestyle=":",
        linewidth=1.8,
        alpha=0.95,
    )
    ax.axvline(
        x=x_end,
        ymin=0.03,
        ymax=0.13,
        color=color,
        linestyle=":",
        linewidth=1.8,
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
        description="Shift 1 percent momentum-thickness curves in time to overlap visually."
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
        "--noise-point",
        type=int,
        required=True,
        help="1-based point number used to shift the noise 1 percent curve.",
    )
    parser.add_argument(
        "--k4-point",
        type=int,
        required=True,
        help="1-based point number used as the reference on the k4 1 percent curve.",
    )
    parser.add_argument(
        "--constant-point",
        type=int,
        required=True,
        help="1-based point number used to shift the constant 1 percent curve.",
    )
    parser.add_argument(
        "--noise-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the noise 1 percent curve.",
    )
    parser.add_argument(
        "--k4-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the k4 1 percent curve.",
    )
    parser.add_argument(
        "--constant-shift",
        type=float,
        default=0.0,
        help="Additional floating x-shift applied to the constant 1 percent curve.",
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

    noise_time, noise_momentum = load_curve(data_path / FILES_1PCT["noise"])
    k4_time, k4_momentum = load_curve(data_path / FILES_1PCT["k4"])
    constant_time, constant_momentum = load_curve(data_path / FILES_1PCT["constant"])

    reference_time = get_time_from_point_number(k4_time, args.k4_point, "k4")

    noise_time_raw, noise_time_shifted, noise_momentum_visible, noise_delta_t = shift_and_trim_curve(
        noise_time,
        noise_momentum,
        args.noise_point,
        reference_time,
        args.noise_shift,
        "noise",
    )
    k4_time_raw, k4_time_shifted, k4_momentum_visible, k4_delta_t = shift_and_trim_curve(
        k4_time,
        k4_momentum,
        args.k4_point,
        reference_time,
        args.k4_shift,
        "k4",
    )
    constant_time_raw, constant_time_shifted, constant_momentum_visible, constant_delta_t = shift_and_trim_curve(
        constant_time,
        constant_momentum,
        args.constant_point,
        reference_time,
        args.constant_shift,
        "constant",
    )

    noise_raw_slope, noise_start_index, noise_end_index = compute_slope(
        noise_time_raw,
        noise_momentum_visible,
        args.slope_start,
        args.slope_end,
        "noise",
    )
    noise_shifted_slope, _, _ = compute_slope(
        noise_time_shifted,
        noise_momentum_visible,
        args.slope_start,
        args.slope_end,
        "noise shifted",
    )

    k4_raw_slope, k4_start_index, k4_end_index = compute_slope(
        k4_time_raw,
        k4_momentum_visible,
        args.slope_start,
        args.slope_end,
        "k4",
    )
    k4_shifted_slope, _, _ = compute_slope(
        k4_time_shifted,
        k4_momentum_visible,
        args.slope_start,
        args.slope_end,
        "k4 shifted",
    )

    constant_raw_slope, constant_start_index, constant_end_index = compute_slope(
        constant_time_raw,
        constant_momentum_visible,
        args.slope_start,
        args.slope_end,
        "constant",
    )
    constant_shifted_slope, _, _ = compute_slope(
        constant_time_shifted,
        constant_momentum_visible,
        args.slope_start,
        args.slope_end,
        "constant shifted",
    )

    print("--Reference curve: k4 1%")
    print("--Reference point number:", args.k4_point)
    print("--Reference time:", reference_time)
    print("--Noise shift delta_t:", noise_delta_t)
    print("--K4 shift delta_t:", k4_delta_t)
    print("--Constant shift delta_t:", constant_delta_t)

    print("--Noise raw slope:", noise_raw_slope)
    print("--Noise shifted slope:", noise_shifted_slope)
    print("--K4 raw slope:", k4_raw_slope)
    print("--K4 shifted slope:", k4_shifted_slope)
    print("--Constant raw slope:", constant_raw_slope)
    print("--Constant shifted slope:", constant_shifted_slope)

    fig = plt.figure(figsize=(6.4, 4.8), dpi=400)
    ax = fig.add_subplot(111)

    line_noise, = ax.plot(noise_time_shifted, noise_momentum_visible, "-", label=LABELS_1PCT["noise"])
    line_k4, = ax.plot(k4_time_shifted, k4_momentum_visible, "-", label=LABELS_1PCT["k4"])
    line_constant, = ax.plot(constant_time_shifted, constant_momentum_visible, "-", label=LABELS_1PCT["constant"])

    #add_short_marker_lines(
    #    ax,
    #    noise_time_shifted[noise_start_index],
    #    noise_time_shifted[noise_end_index],
    #    line_noise.get_color(),
    #)
    #add_short_marker_lines(
    #    ax,
    #    k4_time_shifted[k4_start_index],
    #    k4_time_shifted[k4_end_index],
    #    line_k4.get_color(),
    #)
    #add_short_marker_lines(
    #    ax,
    #    constant_time_shifted[constant_start_index],
    #    constant_time_shifted[constant_end_index],
    #    line_constant.get_color(),
    #)

    x_start = noise_time_shifted[noise_start_index]
    x_end = noise_time_shifted[noise_end_index]
    y_start = noise_momentum_visible[noise_start_index]
    y_end = noise_momentum_visible[noise_end_index]
    add_marker_on_curve(ax, x_start, y_start, 0.1, line_noise.get_color())
    add_marker_on_curve(ax, x_end, y_end, 0.1, line_noise.get_color())

    x_start = k4_time_shifted[k4_start_index]
    x_end = k4_time_shifted[k4_end_index]
    y_start = k4_momentum_visible[k4_start_index]
    y_end = k4_momentum_visible[k4_end_index]
    add_marker_on_curve(ax, x_start, y_start, 0.1, line_k4.get_color())
    add_marker_on_curve(ax, x_end, y_end, 0.1, line_k4.get_color())

    x_start = constant_time_shifted[constant_start_index]
    x_end = constant_time_shifted[constant_end_index]
    y_start = constant_momentum_visible[constant_start_index]
    y_end = constant_momentum_visible[constant_end_index]
    add_marker_on_curve(ax, x_start, y_start, 0.1, line_constant.get_color())
    add_marker_on_curve(ax, x_end, y_end, 0.1, line_constant.get_color())

    slope_text = (
        f"noise slope = {noise_shifted_slope:.5f}\n"
        f"k4 slope = {k4_shifted_slope:.5f}\n"
        f"constant slope = {constant_shifted_slope:.5f}"
    )

    ax.text(
        0.03,
        0.97,
        slope_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
    )

    ax.set_xlabel(r"$t^*$")
    ax.set_ylabel("Momentum_Thickness")
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    output_file = output_path / "Momentum_Thickness_1pct_shifted_overlay.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
