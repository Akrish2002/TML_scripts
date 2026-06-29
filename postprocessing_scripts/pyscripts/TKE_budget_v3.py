import numpy as np
from mpi4py import MPI
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import re, pathlib
import os

from pyscripts.test_TKE_vGPT_v4 import TKE_Budget

mpl.rcParams.update({
        # Figure
        "figure.figsize": (3.5, 2.625),

        #Font
        "font.family"                   : "serif",
        "font.serif"                    : ["STIXGeneral"],
        "axes.formatter.use_mathtext"   : True,
        "mathtext.fontset"              : "cm",
        #"font.size"                     : 10,

        #Ticks
        "xtick.direction"       : "in",
        "xtick.major.size"      : 3,
        "xtick.major.width"     : 0.5,
        "xtick.minor.size"      : 1.5,
        "xtick.minor.width"     : 0.5,
        "xtick.minor.visible"   : True,
        "xtick.top"             : True,

        "ytick.direction"       : "in",
        "ytick.major.size"      : 3,
        "ytick.major.width"     : 0.5,
        "ytick.minor.size"      : 1.5,
        "ytick.minor.width"     : 0.5,
        "ytick.minor.visible"   : True,
        "ytick.right"           : True,

        #Linewidth
        "lines.linewidth"   : 1.0,

        #Axis
        "axes.linewidth"    : 1.2,
        "axes.labelsize"    : 10,     
        "axes.titlesize"    : 8,     
        "axes.grid"         : True,
        "axes.axisbelow"    : True,
        "axes.edgecolor"    : "k",

        #Grid
        "grid.linewidth"    : 0.5,
        "grid.linestyle"    : "--",
        "grid.color"        : "0.45",
        "grid.alpha"        : 0.25,

        #Legend
        "legend.frameon"    : True,
        "legend.framealpha" : 1.0, 
        "legend.fancybox"   : False,
        "legend.edgecolor"  : "none",
        "legend.numpoints"  : 1,
        "legend.loc"        : "best",
        "legend.fontsize"   : 8,

        #Saving
        "savefig.bbox"      : "tight",
        "savefig.pad_inches": 0.05,
    })


colors = {
    "Advection"                : "#7B3294",    # Purple
    "Turb. diff."              : "#4DAF4A",    # Green
    "Production"               : "#E69F00",    # Orange
    "Dissipation"              : "#D62728",    # Red
    "Pres. diff."              : "#1F78B4",    # Blue
    "Visc. diff."              : "#000000",    # Black
    "Surface tension"          : "#d47264",    # Med Red
    "Sum terms"                : "#ededed",    # Gray
}


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


#Computing
def TKE_budget(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.time_step
    T._stackdirection = args.stackdirection
    
    T.common_terms()
    T.advection()
    T.pressure_diffusion()
    T.turbulent_diffusion()
    T.viscous_diffusion()
    T.dissipation()
    T.production()
    if(not args.singlephase):
        T.surface_tension()
    else:
        T.surface_tension() == np.zeros_like(T._dissipation_global)
    T.sum_all_terms()

    if T._case.rank == 0:
        print("-- Computing TKE budgets!")

        ny = T._ny_g
        y_max = T._ymax
        y = (np.arange(ny) + 0.5) * (y_max / ny)

        U_l              = 0.
        U_g              = 3.1830988618379066
        print("-- Using hardcoded U_g and U_l values!")

        #Computing normalized time
        ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
        dt              = grep_ctr('dt', ctr_file)
        delta_ts        = (2. * np.pi) / 100.
        ts              = args.time_step
        t_normalized    = (ts * dt * U_g)/delta_ts

        TKE_advection            = np.asarray(T._advection_global).squeeze()
        TKE_pressure_diffusion   = np.asarray(T._pressure_diffusion_global).squeeze()
        TKE_turbulent_diffusion  = np.asarray(T._turbulent_diffusion_global).squeeze()
        TKE_viscous_diffusion    = np.asarray(T._viscous_diffusion_global).squeeze()
        TKE_dissipation          = np.asarray(T._dissipation_global).squeeze()
        if(not args.singlephase):
            TKE_surface_tension  = np.asarray(T._surface_tension_global).squeeze()
        else:
            TKE_surface_tension  = np.zeros_like(TKE_dissipation)
        TKE_production           = np.asarray(T._production_global).squeeze()
        TKE_sum_terms            = np.asarray(T._sum_all_terms_global).squeeze()

        arrays = [
            TKE_advection,
            TKE_pressure_diffusion,
            TKE_turbulent_diffusion,
            TKE_viscous_diffusion,
            TKE_dissipation,
            TKE_surface_tension,
            TKE_production,
            TKE_sum_terms,
        ]

        for arr in arrays:
            if arr.ndim != 1 or arr.shape[0] != ny:
                raise ValueError(f"TKE budget shape mismatch: got {arr.shape}, expected ({ny},)")
    
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"TKE_Budget_n{ny}_ts{int(args.time_step)}.npz"
    
        np.savez(
                    out_path,
                    case                    =   str(Path(args.case).resolve()),

                    time_step               =   int(args.time_step),
                    t_normalized            =   np.float64(t_normalized),
                    ny                      =   int(ny),

                    y                       =   y.astype(np.float64),
                    TKE_advection           =   TKE_advection.astype(np.float64),
                    TKE_pressure_diffusion  =   TKE_pressure_diffusion.astype(np.float64),
                    TKE_turbulent_diffusion =   TKE_turbulent_diffusion.astype(np.float64),
                    TKE_viscous_diffusion   =   TKE_viscous_diffusion.astype(np.float64),
                    TKE_dissipation         =   TKE_dissipation.astype(np.float64),
                    TKE_surface_tension     =   TKE_surface_tension.astype(np.float64),
                    TKE_production          =   TKE_production.astype(np.float64),
                    TKE_sum_terms           =   TKE_sum_terms.astype(np.float64),
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.time_step})")


#Alias if you decide to call it as TKE from the wrapper.
def TKE(args):
    TKE_budget(args)


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


def load_npz_TKE_budget(path: str, args):
    d               = np.load(path, allow_pickle=True)
    case            = str(d["case"])

    time_step       = int(d["time_step"])
    t_normalized    = float(d["t_normalized"])
    ny              = int(d["ny"])
    y               = d["y"].astype(np.float64)

    TKE_advection            = d["TKE_advection"].astype(np.float64)
    TKE_pressure_diffusion   = d["TKE_pressure_diffusion"].astype(np.float64)
    TKE_turbulent_diffusion  = d["TKE_turbulent_diffusion"].astype(np.float64)
    TKE_viscous_diffusion    = d["TKE_viscous_diffusion"].astype(np.float64)
    TKE_dissipation          = d["TKE_dissipation"].astype(np.float64)
    TKE_production           = d["TKE_production"].astype(np.float64)
    if(not args.singlephase):
        TKE_surface_tension  = d["TKE_surface_tension"].astype(np.float64)
    else:
        TKE_surface_tension  = np.zeros_like(TKE_dissipation)
    TKE_sum_terms            = d["TKE_sum_terms"].astype(np.float64)

    print("y shape: ", y.shape)
    print("TKE_advection shape: ", TKE_advection.shape)
    print("TKE_pressure_diffusion shape: ", TKE_pressure_diffusion.shape)
    print("TKE_turbulent_diffusion shape: ", TKE_turbulent_diffusion.shape)
    print("TKE_viscous_diffusion shape: ", TKE_viscous_diffusion.shape)
    print("TKE_dissipation shape: ", TKE_dissipation.shape)
    print("TKE_surface_tension shape: ", TKE_surface_tension.shape)
    print("TKE_production shape: ", TKE_production.shape)
    print("TKE_sum_terms shape: ", TKE_sum_terms.shape)

    arrays = [
        TKE_advection,
        TKE_pressure_diffusion,
        TKE_turbulent_diffusion,
        TKE_viscous_diffusion,
        TKE_dissipation,
        TKE_surface_tension,
        TKE_production,
        TKE_sum_terms,
    ]

    if y.ndim != 1:
        raise ValueError(f"Size error while loading dataset, please check the generated dataset!")

    for arr in arrays:
        if arr.ndim != 1 or arr.shape != y.shape:
            raise ValueError(f"Size error while loading dataset, please check the generated dataset!")

    return case, time_step, t_normalized, ny, y, \
           TKE_advection,                         \
           TKE_pressure_diffusion,                \
           TKE_turbulent_diffusion,               \
           TKE_viscous_diffusion,                 \
           TKE_dissipation,                       \
           TKE_surface_tension,                   \
           TKE_production,                        \
           TKE_sum_terms


def plot_TKE_budget(args):
    entries = [load_npz_TKE_budget(f, args) for f in args.inputs]

    #Paper-style plot
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)
    ax = fig.add_subplot(111)

    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    #Keep the old component behavior unchanged.
    component = getattr(args, "TKE_Budget_components", None)
    if isinstance(component, list):
        component = component[0]
    if component is not None:
        term_labels = [
            "Advection",
            "Pres. diff.",
            "Turb. diff.",
            "Visc. diff.",
            "Dissipation",
            "Surface tension",
            "Production",
            "Sum terms",
        ]

        y_label_list = [
            r"Advection",
            r"Pressure diffusion",
            r"Turbulent diffusion",
            r"Viscous diffusion",
            r"Dissipation",
            r"Surface tension",
            r"Production",
            r"Sum terms",
        ]

        for idx, (case, time_step, t_normalized, ny, y,
                  TKE_advection,
                  TKE_pressure_diffusion,
                  TKE_turbulent_diffusion,
                  TKE_viscous_diffusion,
                  TKE_dissipation,
                  TKE_surface_tension,
                  TKE_production,
                  TKE_sum_terms) in enumerate(entries):

            lab = (
                    args.labels[idx]
                    if args.labels and len(args.labels) == len(entries)
                    else f"{ny}$^3$, t*={t_normalized:.2f}"
                  )

            term_list = [
                TKE_advection,
                TKE_pressure_diffusion,
                TKE_turbulent_diffusion,
                TKE_viscous_diffusion,
                TKE_dissipation,
                TKE_surface_tension,
                TKE_production,
                TKE_sum_terms,
            ]

            y_plot = term_list[component - 1]
            if args.zoom:
                x1 = args.zoom - args.zoom_window
                x2 = args.zoom + args.zoom_window
                mask = (y >= x1) & (y <= x2)
                ax.plot(y[mask], y_plot[mask], color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                        linewidth=1.2, label=lab)
            else:
                ax.plot(y, y_plot, color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                        linewidth=1.2, label=lab)

        ax.set_ylabel(y_label_list[component - 1])

    else:
        terms = [
            ("TKE_advection", "Advection"),
            ("TKE_turbulent_diffusion", "Turb. diff."),
            ("TKE_production", "Production"),
            ("TKE_dissipation", "Dissipation"),
            ("TKE_pressure_diffusion", "Pres. diff."),
            ("TKE_viscous_diffusion", "Visc. diff."),
            ("TKE_surface_tension", "Surface Tension"),
            ("TKE_sum_terms", "Sum"),
        ]

        plotted_any = False

        for idx, (case, time_step, t_normalized, ny, y,
                  TKE_advection,
                  TKE_pressure_diffusion,
                  TKE_turbulent_diffusion,
                  TKE_viscous_diffusion,
                  TKE_dissipation,
                  TKE_surface_tension,
                  TKE_production,
                  TKE_sum_terms) in enumerate(entries):

            term_values = {
                "TKE_advection"           : TKE_advection,
                "TKE_pressure_diffusion"  : TKE_pressure_diffusion,
                "TKE_turbulent_diffusion" : TKE_turbulent_diffusion,
                "TKE_viscous_diffusion"   : TKE_viscous_diffusion,
                "TKE_dissipation"         : TKE_dissipation,
                "TKE_surface_tension"     : TKE_surface_tension,
                "TKE_production"          : TKE_production,
                "TKE_sum_terms"           : TKE_sum_terms,
            }

            if args.zoom:
                x1 = args.zoom - args.zoom_window
                x2 = args.zoom + args.zoom_window
                mask = (y >= x1) & (y <= x2)
            else:
                mask = slice(None)

            for i, (key, lab) in enumerate(terms):
                if((key == "TKE_surface_tension" and args.singlephase) or key == "TKE_sum_terms"):
                    continue
                arr = np.asarray(term_values[key]).squeeze()
                if arr.ndim != 1 or arr.shape[0] != ny:
                    raise ValueError(f"{key} must be (ny,), got {np.asarray(term_values[key]).shape}")

                if len(entries) == 1:
                    term_lab = lab
                else:
                    case_lab = (
                        args.labels[idx]
                        if args.labels and len(args.labels) == len(entries)
                        else f"{ny}$^3$, t*={t_normalized:.2f}"
                    )
                    term_lab = f"{case_lab}, {lab}"

                #ax.plot(
                #    y[mask],
                #    arr[mask],
                #    color=colors.get(lab, "k"),
                #    linestyle=dash_cycle[i % len(dash_cycle)],
                #    linewidth=1.2,
                #    label=term_lab,
                #)
                ax.plot(
                    y[mask],
                    arr[mask],
                    color=colors.get(lab, "k"),
                    linestyle=dash_cycle[i % len(dash_cycle)],
                    label=term_lab,
                )
                plotted_any = True

        if not plotted_any:
            raise RuntimeError("No TKE term arrays were available for plotting.")

        ax.set_ylabel("TKE budget")

        title = getattr(args, "title", None)
        if title is None:
            title = "TKE budget terms (plane-averaged)"

        ax.text(
            0.50, 0.93, title,
            transform=ax.transAxes,
            ha="center", va="top",
            color="k",
        )

    ax.set_xlabel("y")

    if args.zoom:
        ax.set_xlim(args.zoom - args.zoom_window, args.zoom + args.zoom_window)

    ylim = getattr(args, "ylim", None)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    #apply_paper_style(ax)
    #ax.legend(loc="best", frameon=False)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
