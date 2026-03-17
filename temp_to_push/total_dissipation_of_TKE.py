import numpy as np                                                              
from mpi4py import MPI                                                          
from pathlib import Path                                                        
import re, pathlib
                                                                                
from pyscripts.test_TKE_vGPT_v1 import TKE_Budget                                         

def grep_timestep(path = "."):                                                  
    """ Grepping time steps to calculate first step, step and last step         
                                                                                
    Args:                                                                       
                                                                                
    Return:                                                                     
                                                                                
    """                                                                         
                                                                                
    root = Path(path)                                                           
    nums = []                                                                   
                                                                                
    for p in root.iterdir():                                                    
        m = re.fullmatch(r"time_step-(\d+)", p.name)                            
        if m:                                                                   
            nums.append(int(m.group(1)))                                        
                                                                                
    nums.sort()                                                                 
    if nums:                                                                    
        fs, step, ls = min(nums), nums[1] - nums[0], max(nums)                  
                                                                                
    return fs, step, ls
                                                                                
def total_dissipation_of_TKE(args):                                             
                                                                                
    (start_ts, step_ts, end_ts) = grep_timestep(args.case)                      
    total_dissipation = []                                                      
    time_steps = [i for i in range(start_ts, end_ts, step_ts)]                  
                                                                                
    for time_step in time_steps:                                                
        T                   = TKE_Budget(args.case)                             
        args.ts             = time_step                                         
        T._time_step        = args.ts                                           
        T._stackdirection   = args.std                                          
                                                                                
        T.common_terms()                                                        
        T.dissipation()                                                         
                                                                                
        ny                       = T._ny_g                                      
        dy                       = 2 * np.pi / ny                               
        total_dissipation_of_TKE = []                                           
        if T._case.rank == 0:                                                   
            integrand_dissipation_of_TKE = T._dissipation_global                
            #total_dissipation     = np.trapezoid(integrand_dissipation, dx=dy) 
            total_dissipation_of_TKE.append(np.trapezoid(integrand_dissipation_of_TKE, dx=dy))
                                                                                
    if T._case.rank == 0:                                                       
        if total_dissipation_of_TKE.ndim != 1 or total_dissipation_of_TKE.shape[0] != ny:
            raise ValueError(f"total_dissipation_of_TKE shape mismatch: got {total_dissipation_of_TKE.shape}, expected ({ny},)")
        if total_dissipation_of_TKE.size != time_steps.size
            raise ValueError("Mismatch between number of total dissipations computed and time_steps to be plotted against")
                                                                                
        y = (np.arange(ny) + 0.5) * (2 * np.pi / ny)                            
        out_path = Path(args.out)                                               
        out_path.parent.mkdir(parents=True, exist_ok=True)                      
                                                                                
        np.savez(                                                               
                    out_path,                                                   
                    case=str(Path(args.case).resolve()),                        
                    #Check if this right for the timesteps to be stored
                    time_steps=time_steps.astype(np.float64),                                    
                    #I do not need these to plot against time
                    #ny=int(ny),                                                 
                    #y=y.astype(np.float64),                                     
                    total_dissipation_of_TKE=total_dissipation_of_TKE.astype(np.float64),
                )                                                               
        print(f"[rank0] wrote {out_path} (ny={ny})")                            

