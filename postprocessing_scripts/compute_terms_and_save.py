# AI Assistance Declaration:
# The entire script was generated with the assistance of OpenAI ChatGPT (GPT-5.2).
# The generated code has only been lightly tested for basic functionality.
# Any errors, numerical inaccuracies, or unintended behavior must be rigorously
# investigated and validated by the user. None of the sections have undergone
# manual review or verification beyond minimal execution.

# =========================
# SCRIPT 1: compute_terms_and_save.py
# Run this separately for EACH grid/case with the correct MPI layout for that case.
# Example:
#   srun -n <N> python compute_eps_save.py --case ../n512 --ts 7000 --out eps_n512_ts7000.npz
# =========================

from mpi4py import MPI
import numpy as np
from pathlib import Path

from test_TKE_v8 import TKE_Budget

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, type=str, help="case directory (contains incompressible_tml.ctr)")
    p.add_argument("--ts", required=True, type=int, help="time_step to compute")
    p.add_argument("--std", required=True, type=int, help="stack direction")
    p.add_argument("--out", required=True, type=str, help="output .npz file (written by rank 0)")
    args = p.parse_args()

    T = TKE_Budget(args.case)
    T._time_step      = args.ts
    T._stackdirection = args.std

    T.common_terms()
    T.dissipation()

    if T._case.rank == 0:
        ny = T._ny_g
        y = (np.arange(ny) + 0.5) * (2*np.pi / ny)  # cell centers on [0,2pi)
        eps = np.asarray(T._dissipation_global).squeeze()
        if eps.ndim != 1 or eps.shape[0] != ny:
            raise ValueError(f"dissipation shape mismatch: got {eps.shape}, expected ({ny},)")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_path,
            case=str(Path(args.case).resolve()),
            time_step=int(args.ts),
            ny=int(ny),
            y=y.astype(np.float64),
            eps=eps.astype(np.float64),
        )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.ts})")

if __name__ == "__main__":
    main()

