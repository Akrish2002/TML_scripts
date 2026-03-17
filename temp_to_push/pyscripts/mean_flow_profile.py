import numpy as np
from mpi4py import MPI
from pathlib import Path

from pyscripts.test_TKE_vGPT_v1 import TKE_Budget

def mean_flow_profile(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.ts
    T._stackdirection = args.std
    T._option         = 1                                                       #For mean flow profile
    
    T.common_terms()
    T.miscellaneous()

    if T._case.rank == 0:
        U_l = 0.
        U_g = 3.1830988618379066
        ny = T._ny_g
        u_avg = T._u_avg_global
        u_avg_normalized = np.asarray((u_avg/U_g)).squeeze()

        #HAVE TO FIND A BETTER METHOD
        step = 2 * np.pi /  ny
        y_grid = np.arange(0, 2 * np.pi, step)

        #Finding U(x, y_0.1&0.9, z) 
        U_01 = U_l + 0.1 * (U_g - U_l)
        U_09 = U_l + 0.9 * (U_g - U_l)

        idx = np.where((u_avg >= U_01) & (u_avg <= U_09))

        y_01 = y_grid[idx][0]
        y_09 = y_grid[idx][-1]

        delta = y_09 - y_01
        y_bar = 0.5 * (y_09 + y_01)
        #Zeta forms my new y
        zeta = (y_grid - y_bar) / delta

        #Debug
        print("--zeta shape: ",     zeta.shape)
        print("--delta : ",    delta)
        print("--y01 : ",      y_01)
        print("--y09 : ",      y_09)
        print("--y_bar : ",    y_bar)
        print("--y_grid shape: ",   y_grid.shape)
    
        #Converts to numpy array and removes any dimension of lenght 1

        if u_avg_normalized.ndim != 1 or u_avg_normalized.shape[0] != ny:
            raise ValueError(f"u_avg_normalized shape mismatch: got {u_avg_normalized.shape}, expected ({ny},)")
    
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
        np.savez(
                    out_path,
                    case=str(Path(args.case).resolve()),
                    time_step=int(args.ts),
                    ny=int(ny),
                    y=zeta.astype(np.float64),
                    u_avg_normalized=u_avg_normalized.astype(np.float64),
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.ts})")


