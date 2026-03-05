# =========================
# SCRIPT 2: plot_eps_compare.py
# Run once (serial is fine). Loads the saved vectors, interpolates to a common grid, plots.
# Example:
#   python plot_eps_compare.py --inputs eps_n256_ts12000.npz eps_n512_ts7000.npz eps_n1024_ts3500.npz --out eps_compare.png
# =========================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def y_cell_centers(ny: int, y_min: float = 0.0, y_max: float = 2*np.pi) -> np.ndarray:
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

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="list of .npz files from compute_eps_save.py")
    p.add_argument("--labels", nargs="*", default=None, help="optional labels (same count as inputs)")
    p.add_argument("--out", default="eps_compare.png")
    p.add_argument("--zoom_pi", action="store_true", help="zoom around y=pi")
    p.add_argument("--pi_window", type=float, default=0.5, help="half-width around pi (radians)")
    p.add_argument("--common_ny", type=int, default=0, help="override common ny (0 => use max ny)")
    args = p.parse_args()

    entries = [load_npz(f) for f in args.inputs]

    if args.labels is not None and len(args.labels) not in (0, len(entries)):
        raise ValueError("--labels must be omitted or have the same length as --inputs")

    # choose common grid
    ny_common = args.common_ny if args.common_ny > 0 else max(ny for _, _, ny, _, _ in entries)
    y_common = y_cell_centers(ny_common)

    # optional zoom mask around pi
    if args.zoom_pi:
        mask = (y_common >= (np.pi - args.pi_window)) & (y_common <= (np.pi + args.pi_window))
    else:
        mask = slice(None)

    plt.figure()
    for idx, (case, ts, ny, y, eps) in enumerate(entries):
        lab = args.labels[idx] if args.labels and len(args.labels) == len(entries) else f"{Path(case).name} (ny={ny}, ts={ts})"
        eps_common = interp_periodic_2pi(y, eps, y_common)
        plt.plot(y_common[mask], eps_common[mask], label=lab)

    plt.title("Dissipation (plane-averaged) comparison" + (" (zoom near pi)" if args.zoom_pi else ""))
    plt.xlabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

