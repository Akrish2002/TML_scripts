import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pathlib import Path
import re, pathlib
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
    n   = float(m.group(1)) if m else None

    return n

def mean_flow_profile(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.time_step
    T._stackdirection = args.stackdirection
    
    T.common_terms()
    T._option = 1                                                       #For mean flow profile
    T.miscellaneous()

    if T._case.rank == 0:
        print("--Computing mean flow profile!")

        U_l              = 0.
        U_g              = 3.1830988618379066
        print("-- Using hardcoded U_g and U_l values!")
        ny               = T._ny_g
        u_avg            = T._u_avg_global
        u_avg_normalized = np.asarray((u_avg/U_g)).squeeze()

        #Computing normalized time
        ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
        dt              = grep_ctr('dt', ctr_file)
        delta_ts        = (2. * np.pi) / 100.
        ts              = args.time_step
        t_normalized    = (ts * dt * U_g)/delta_ts

        #Generating y_grid
        step   = 2 * np.pi / ny
        y_grid = np.arange(0, 2 * np.pi, step)

        #Finding U(x, y_0.1 & 0.9, z) 
        print("-- The choice of constructing the similarity variable has to be justified")
        U_01 = U_l + 0.1 * (U_g - U_l)
        U_09 = U_l + 0.9 * (U_g - U_l)

        idx  = np.where((u_avg >= U_01) & (u_avg <= U_09))

        y_01 = y_grid[idx][0]
        y_09 = y_grid[idx][-1]

        delta = y_09 - y_01
        y_bar = 0.5 * (y_09 + y_01)
        #xi forms my new y
        xi    = (y_grid - y_bar) / delta

        #Debug
        print("-- xi shape: ",       xi.shape)
        print("-- delta : ",         delta)
        print("-- y01 : ",           y_01)
        print("-- y09 : ",           y_09)
        print("-- y_bar : ",         y_bar)
        print("-- y_grid shape: ",   y_grid.shape)
    
        if u_avg_normalized.ndim != 1 or u_avg_normalized.shape[0] != ny:
            raise ValueError(f"u_avg_normalized shape mismatch: got {u_avg_normalized.shape}, expected ({ny},)")
    
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
        np.savez(
                    out_path,
                    case             =   str(Path(args.case).resolve()),

                    time_step        =   int(args.time_step),
                    t_normalized     =   np.float64(t_normalized),
                    ny               =   int(ny),

                    y                =   xi.astype(np.float64),
                    u_avg_normalized =   u_avg_normalized.astype(np.float64),
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.time_step})")

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

def load_npz_mean_flow_profile(path: str):
    d               = np.load(path, allow_pickle=True)
    case            = str(d["case"])

    time_step       = int(d["time_step"])
    t_normalized    = float(d["t_normalized"])
    ny              = int(d["ny"])

    xi               = d["y"].astype(np.float64)
    u_avg_normalized = d["u_avg_normalized"].astype(np.float64)

    print("xi shape: ", xi.shape)
    print("u_avg_normalized shape: ", u_avg_normalized.shape)

    if(u_avg_normalized.ndim != 1 or u_avg_normalized.shape != xi.shape):
        raise ValueError(f"Size error while loading dataset, please check the generated dataset!")

    return case, t_normalized, ny, xi, u_avg_normalized

def plot_mean_flow_profile(args):
    entries = [load_npz_mean_flow_profile(f) for f in args.inputs]                                
    case    = [entry[0] for entry in entries]
                                                                                
    #Paper-style plot                                                           
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)       
    ax = fig.add_subplot(111)                                                   
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for idx, (case, t_normalized, ny, xi, u_avg_normalized) in enumerate(entries):

        lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                else f"{ny}$^3$, t*={t_normalized:.2f}"
              )

        #Zoom mask in this
        x = xi
        y = u_avg_normalized
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
    ax.set_ylabel(r"$\overline{U} / U_g$", fontsize=16)
    ax.set_xlabel(r"$\xi$", fontsize=16)

    #To have path of run in the bottom of the screen
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
