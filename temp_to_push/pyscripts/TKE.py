import numpy as np
from mpi4py import MPI
from pathlib import Path

from pyscripts.test_TKE_vGPT_v1 import TKE_Budget

def TKE(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.ts
    T._stackdirection = args.std
    T._option         = 2                                                       
    
    T.common_terms()
    T.miscellaneous()

    if T._case.rank == 0:
        TKE = T._TKE_global
        ny  = T._ny_g
        #Generate y at the cell centers, hardcoded for 2pi
        y = (np.arange(ny) + 0.5) * (2*np.pi / ny)  

        #Converts to numpy array and removes any dimension of length 1
        #TKE = np.asarray(TKE).squeeze()

        if TKE.ndim != 1 or TKE.shape[0] != ny:
            raise ValueError(f"TKE shape mismatch: got {TKE.shape}, expected ({ny},)")
    
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
        np.savez(
                    out_path,
                    case=str(Path(args.case).resolve()),
                    time_step=int(args.ts),
                    ny=int(ny),
                    y=y.astype(np.float64),
                    TKE=TKE.astype(np.float64),
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.ts})")
