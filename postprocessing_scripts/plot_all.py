import argparse

from mean_flow_profile import plot_mean_flow_profile
from TKE import plot_TKE
from reynolds_stresses import plot_reynolds_stresses
from total_dissipation_of_TKE import plot_total_dissipation_of_TKE
from spectra import plot_spectra
from reynolds_number import plot_reynolds_number

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--function", required=True,
                        choices=[
                            "mean_flow_profile",
                            "dissipation",
                            "TKE",
                            "reynolds_stresses",
                            "total_dissipation_of_TKE",
                            "spectra",
                            "reynolds_number",
                        ])
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--figsize", nargs=2, type=float, default=[6.0, 4.0])
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--zoom-window", type=float, default=1.0)

    args = parser.parse_args()

    args.out = args.output_path

    if args.function == "mean_flow_profile":
        plot_mean_flow_profile(args)

    elif args.function == "TKE":
        plot_TKE(args)

    elif args.function == "dissipation":
        plot_dissipation(args)

    elif args.function == "reynolds_stresses":
        plot_reynolds_stresses(args)

    elif args.function == "total_dissipation_of_TKE":
        plot_total_dissipation_of_TKE(args)

    elif args.function == "spectra":
        plot_spectra(args)

    elif args.function == "reynolds_number":
        plot_reynolds_number(args)

    else:
        print("-- Nothing has been specified")
