import numpy as np                                                              
from mpi4py import MPI                                                          
from pathlib import Path                                                        
import re, pathlib
import csv
import matplotlib.pyplot as plt
import os
                                                                                
from pyscripts.test_TKE_vGPT_v3 import TKE_Budget                                         

def grep_ctr(st, ctr_file="incompressible_tml.ctr"):
    """ To grep all the required data from the CTR file
    
    Args:
        st (string) : The variable whose data is to be grep-ed
    
    Return:
        The variable's value
    """
    
    text = pathlib.Path(ctr_file).read_text()
    pat = re.compile(rf"\b{re.escape(st)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    m = pat.search(text)
    n = float(m.group(1)) if m else None

    return n

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
                                                                                
def compute_total_dissipation_of_TKE(args):                                             
                                                                                
    (start_ts, step_ts, end_ts) = grep_timestep(args.case)                      

    ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
    dt              = grep_ctr('dt', ctr_file)
    U_l             = 0.
    U_g             = 3.1830988618379066
    delta_ts        = (2. * np.pi) / 100.

    time_steps      = [i for i in range(start_ts, end_ts+step_ts, step_ts)]                  
    t_normalized    = [(time_step * dt * U_g) / delta_ts for time_step in time_steps]

    #Debug
    T = TKE_Budget(args.case)                             
    #if(T._case.rank == 0):
    #    print(start_ts)
    #    print(step_ts)
    #    print(end_ts)

    #This is for computing dy for performing integration 
    ny                       = T._ny_g                                      
    dy                       = 2 * np.pi / ny                               

    if T._case.rank == 0:
        fname = os.path.join(args.output_path, f"total_dissipation_rate_{ny}.csv")
        write_header = not os.path.exists(fname) or os.path.getsize(fname) == 0
        f = open(fname, "a", newline="")
        w = csv.writer(f)
        if write_header:
            w.writerow(["TimeStep", "NormalizedTime", "TotalDissipation"])
    else:
        f = w = None

    del(T)
    total_dissipation_of_TKE = []

    for i, time_step in enumerate(time_steps):                                                
        T                   = TKE_Budget(args.case)                             
        T._time_step        = time_step                                           
        T._stackdirection   = args.std                                          
                                                                                
        T.common_terms()                                                        
        T.dissipation()                                                         
                                                                                
        if T._case.rank == 0:                                                   
            #integrand_dissipation_of_TKE = T._dissipation_global                
            #total_dissipation_of_TKE.append(np.trapezoid(integrand_dissipation_of_TKE, dx=dy))

            integrand_dissipation_of_TKE = T._dissipation_global                
            total_dissipation_of_TKE     = (np.trapezoid(integrand_dissipation_of_TKE, dx=dy))
            w.writerow([time_step, t_normalized[i], total_dissipation_of_TKE])
            f.flush()
            os.fsync(f.fileno())

    #time_steps = np.asarray(time_steps)
    #total_dissipation_of_TKE = np.asarray(total_dissipation_of_TKE)

    #if T._case.rank == 0:                                                       
    #    if total_dissipation_of_TKE.ndim != 1:
    #        raise ValueError(f"total_dissipation_of_TKE has more dimensions than one!, that is incorrect!")

    #    if total_dissipation_of_TKE.shape != time_steps.shape: 
    #        raise ValueError("Mismatch between number of total dissipations computed and time_steps to be plotted against")

    #    out_path = Path(args.output_path)                                               
    #    out_path.parent.mkdir(parents=True, exist_ok=True)                      

    #    for tnorm, eps_int in zip(t_normalized, total_dissipation_of_TKE):
    #        w.writerow([tnorm, eps_int])

    #    f.flush()
    #    os.fsync(f.fileno())
    #    f.close()

        #print(f"[rank0] wrote {fname} (ny={ny})")                                                                                

        #np.savez(                                                               
        #            out_path,                                                   
        #            case            =   str(Path(args.case).resolve()),                        

        #            ny              =   int(ny),
        #            time_steps      =   time_steps.astype(np.float64),                                    
        #            t_normalized    =   t_normalized.astype(np.float64), 

        #            #Custom
        #            total_dissipation_of_TKE    =   total_dissipation_of_TKE.astype(np.float64)
        #        )                                                               
        #print(f"[rank0] wrote {out_path} (ny={ny})")                            

    if(T._case.rank == 0):
        f.close()
    del(T)
#------------------------------------------------------------------------------

#Plotting
def apply_paper_style(ax):
    # light dotted grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="0.55")

    # black frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("k")

    # tick style
    ax.tick_params(direction="out", length=4, width=1.0, colors="k")

def load_npz_total_dissipation_of_TKE(path: str):
    d               = np.load(path, allow_pickle=True)
    case            = str(d["case"])

    ny              = int(d["ny"])
    t_normalized    = float(d["t_normalized"])

    total_dissipation_of_TKE    = d["total_dissipation_of_TKE"].astype(np.float64)

    print("t_normalized shape: ", xi.shape)
    print("total_dissipation_of_TKE shape: ", uprime_uprime.shape)

    if(total_dissipation_of_TKE.ndim != 1 or total_dissipation_of_TKE.shape != t_normalized.shape):
        raise ValueError(f"Size error while loading dataset, please check the generated dataset!")

    return case, ny, t_normalized, total_dissipation_of_TKE

def plot_total_dissipation_of_TKE(args):
    entries = [load_npz_total_dissipation_of_TKE(f) for f in args.inputs]                                
    case    = [entry[0] for entry in entries]
                                                                                
    #Paper-style plot                                                           
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)       
    ax = fig.add_subplot(111)                                                   
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for idx, (case, ny, t_normalized, total_dissipation_of_TKE) in enumerate(entries):

        lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                else f"{ny}$^3$"
              )

        #Zoom mask in this
        x = t_normalized
        y = total_dissipation_of_TKE
        if args.zoom:
            x1 = 0 - args.zoom_window
            x2 = 0 + args.zoom_window
            m = (x >= x1) & (x <= x2)
            ax.plot(x[m], y[m], color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                    linewidth=1.2, label=lab)

        else:
            ax.plot(x, y, color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                    linewidth=1.2, label=lab)


    #Labels
    ax.set_ylabel(r"$\int_{-\infty}^{\infty} \epsilon \, dy$", fontsize=16)
    ax.set_xlabel(r"$\t*$", fontsize=16)

    #To have path of run being used
    p = Path(case)
    short = Path(*p.parts[-2:])
    fig.text(
        0.98, 0.01, short,
        ha="right",
        va="bottom",
        fontsize=5
    )

    if args.zoom:
        ax.set_xlim(0 - args.zoom_window, 0 + args.zoom_window)

    apply_paper_style(ax)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)

