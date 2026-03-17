import numpy as np
from mpi4py import MPI
from pathlib import Path

from pyscripts.test_TKE_vGPT_v1 import TKE_Budget
from pyscripts.mean_flow_profile import mean_flow_profile
from pyscripts.TKE import TKE 
#from pyscripts.total_dissipation_of_TKE import total_dissipation_of_TKE
from total_dissipation_of_TKE import total_dissipation_of_TKE

def vorticity_z():
    print("-- Not yet implemented")

#def mean_flow_profile():
#    import argparse
#    p = argparse.ArgumentParser()
#    p.add_argument("--case", required=True, type=str, help="case directory (contains incompressible_tml.ctr)")
#    p.add_argument("--ts", required=True, type=int, help="time_step to compute")
#    p.add_argument("--std", required=True, type=int, help="stack direction")
#    p.add_argument("--out", required=True, type=str, help="output .npz file (written by rank 0)")
#    args = p.parse_args()
#    
#    T = TKE_Budget(args.case)
#    T._time_step      = args.ts
#    T._stackdirection = args.std
#    T._option         = 1                                                       #For mean flow profile
#    
#    T.common_terms()
#    T.miscellaneous()
#
#    if T._case.rank == 0:
#        #y_org = T._y_org_global
#        u_avg = T._u_avg_global
#        #(x, y, z) --> Shape
#        #u_org = T._u_org_global
#
#        #Debug
#        #print(u_org[0, :, 0])
#        #print(u_org[125, :, 125])
#        #print(u_org[250, :, 250])
#        #print(u_org[375, :, 375])
#        #print(u_org[511, :, 511])
#
#        #print(u_org[250, :, :])
#        #print(np.count_nonzero(u_org))
#        #idx = (np.nonzero(u_org))
#        #print(u_org.shape)
#        #print(u_avg.shape)
#        #print(idx[1])
#        #exit()
#
#        U_l = 0.
#        U_g = 3.1830988618379066
#
#        #HAVE TO FIND A BETTER METHOD
#        step = 2 * np.pi / 1024
#        y_grid = np.arange(0, 2 * np.pi, step)
#
#        #Finding U(x, y_0.1&0.9, z) 
#        U_01 = U_l + 0.1*(U_g - U_l)
#        U_09 = U_l + 0.9*(U_g - U_l)
#
#        #scale = 1e-8
#        #epsilon = (U_g - U_l) * scale
#
#        idx = np.where((u_avg >= U_01) & (u_avg <= U_09))
#
#        y_01 = y_grid[idx][0]
#        y_09 = y_grid[idx][-1]
#
#        delta = y_09 - y_01
#        y_bar = 0.5 * (y_09 + y_01)
#        zeta = (y_grid - y_bar) / delta
#
#        #Debug
#        print("--zeta shape: ",     zeta.shape)
#        print("--delta : ",    delta)
#        print("--y01 : ",      y_01)
#        print("--y09 : ",      y_09)
#        print("--y_bar : ",    y_bar)
#        print("--y_grid shape: ",   y_grid.shape)
#    
#
#        #print(y_01)
#        #print(y_09)
#        #print(delta)
#        #exit()
#
#        ny = T._ny_g
#        #Generate y at the cell centers, hardcoded for 2pi
#        y = (np.arange(ny) + 0.5) * (2*np.pi / ny)  
#
#        #Converts to numpy array and removes any dimension of lenght 1
#        u_avg_normalized = np.asarray((u_avg/U_g)).squeeze()
#
#        if u_avg_normalized.ndim != 1 or u_avg_normalized.shape[0] != ny:
#            raise ValueError(f"u_avg_normalized shape mismatch: got {u_avg_normalized.shape}, expected ({ny},)")
#    
#        out_path = Path(args.out)
#        out_path.parent.mkdir(parents=True, exist_ok=True)
#    
#        np.savez(
#            out_path,
#            case=str(Path(args.case).resolve()),
#            time_step=int(args.ts),
#            ny=int(ny),
#            #y=y.astype(np.float64),
#            y=zeta.astype(np.float64),
#            u_avg_normalized=u_avg_normalized.astype(np.float64),
#        )
#        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.ts})")


#def TKE():
#    import argparse
#    p = argparse.ArgumentParser()
#    p.add_argument("--case", required=True, type=str, help="case directory (contains incompressible_tml.ctr)")
#    p.add_argument("--ts", required=True, type=int, help="time_step to compute")
#    p.add_argument("--std", required=True, type=int, help="stack direction")
#    p.add_argument("--out", required=True, type=str, help="output .npz file (written by rank 0)")
#    args = p.parse_args()
#    
#    T = TKE_Budget(args.case)
#    T._time_step      = args.ts
#    T._stackdirection = args.std
#    T._option         = 2                                                       
#    
#    T.common_terms()
#    T.miscellaneous()
#
#    if T._case.rank == 0:
#        #TKE = T._TKE_global
#        #exit()
#
#        TKE = T._TKE_global
#        ny = T._ny_g
#        #Generate y at the cell centers, hardcoded for 2pi
#        y = (np.arange(ny) + 0.5) * (2*np.pi / ny)  
#
#        #Converts to numpy array and removes any dimension of length 1
#        #TKE = np.asarray(TKE).squeeze()
#
#        if TKE.ndim != 1 or TKE.shape[0] != ny:
#            raise ValueError(f"TKE shape mismatch: got {TKE.shape}, expected ({ny},)")
#    
#        out_path = Path(args.out)
#        out_path.parent.mkdir(parents=True, exist_ok=True)
#    
#        np.savez(
#            out_path,
#            case=str(Path(args.case).resolve()),
#            time_step=int(args.ts),
#            ny=int(ny),
#            y=y.astype(np.float64),
#            TKE=TKE.astype(np.float64),
#        )
#        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.ts})")

if __name__ == "__main__" :                                                     
    import argparse                                                             
    p = argparse.ArgumentParser()                                               
    p.add_argument("--case", required=True, type=str, help="case directory (contains incompressible_    tml.ctr)")
    #p.add_argument("--resolution", required=True, type=int, help="Grid resolution")
    p.add_argument("--compute_all", action='store_true')                        
    p.add_argument("--ts", required=False, type=int, help="time_step to compute")
    p.add_argument("--std", required=True, type=int, help="stack direction")    
    p.add_argument("--out", required=True, type=str, help="output .npz file (written by rank 0)")
    args = p.parse_args()                                                       
                                                                                
    if(args.compute_all):                                                       
        total_dissipation_of_TKE(args)

    #mean_flow_profile(args)
    #TKE(args)
