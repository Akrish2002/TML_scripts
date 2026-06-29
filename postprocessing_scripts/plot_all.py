import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--function", required=True,
                        choices=[
                            "mean_flow_profile",
                            "dissipation",
                            "TKE",
                            "TKE_Budget",
                            "reynolds_stresses",
                            "total_dissipation_of_TKE",
                            "reynolds_number",
                            "kx_spectra",
                            "kz_spectra",
                            "streamwise_two_point_correlation",
                            "spanwise_two_point_correlation",
                        ])
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--reynolds_stress_components",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Reynolds stress component to plot: 1=uu, 2=vv, 3=ww, 4=uv"
    )
    parser.add_argument(
        "--spectra_components",
        nargs="+",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Spectra components to plot: spectra 1=Euu, 2=Evv, 3=Eww, 4=ETKE"
    )
    parser.add_argument(
        "--twopoint_correlation_components",
        nargs="+",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Correlation components to plot: 1=Ruu, 2=Rvv, 3=Rww, 4=Rphi2phi2"
    )
    parser.add_argument(
        "--TKE_Budget_components",
        nargs="+",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="asdf"
    )
    parser.add_argument(
        "--y-delta-multiples",
        nargs="+",
        type=float,
        default=None,
        help="y locations to plot as multiples of initial momentum thickness away from the centerplane"
    )
    parser.add_argument("--zoom", type=float)
    parser.add_argument("--zoom-window", type=float, default=1.0)
    parser.add_argument("--singlephase", action="store_true")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--figsize", nargs=2, type=float, default=[6.0, 4.0])
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()

    args.out = args.output_path

    if args.function == "mean_flow_profile":
        if args.singlephase:
            from pyscripts.mean_flow_profile_singlephase import plot_mean_flow_profile
            plot_mean_flow_profile(args)
        else:
            from pyscripts.mean_flow_profile import plot_mean_flow_profile
            plot_mean_flow_profile(args)

    elif args.function == "TKE":
        from pyscripts.TKE import plot_TKE
        plot_TKE(args)

    elif args.function == "TKE_Budget":
        from pyscripts.TKE_budget_v3 import plot_TKE_budget
        plot_TKE_budget(args)

    elif args.function == "dissipation":
        from pyscripts.dissipation import plot_dissipation
        plot_dissipation(args)

    elif args.function == "reynolds_stresses":
        if args.singlephase:
            from pyscripts.reynolds_stresses_singlephase import plot_reynolds_stresses
            plot_reynolds_stresses(args)
        else:
            from pyscripts.reynolds_stresses import plot_reynolds_stresses
            plot_reynolds_stresses(args)

    elif args.function == "total_dissipation_of_TKE":
        from pyscripts.total_dissipation_of_TKE_v2 import plot_total_dissipation_of_TKE
        plot_total_dissipation_of_TKE(args)

    elif args.function == "reynolds_number":
        from pyscripts.reynolds_number import plot_reynolds_number
        plot_reynolds_number(args)

    elif args.function == "kx_spectra":
        from pyscripts.kx_spectra import plot_spectra as plot_kx_spectra
        plot_kx_spectra(args)

    elif args.function == "kz_spectra":
        from pyscripts.kz_spectra import plot_spectra as plot_kz_spectra
        plot_kz_spectra(args)

    elif args.function == "streamwise_two_point_correlation":
        from pyscripts.streamwise_two_point_correlation import plot_streamwise_two_point_correlation
        plot_streamwise_two_point_correlation(args)

    elif args.function == "spanwise_two_point_correlation":
        from pyscripts.spanwise_two_point_correlation import plot_spanwise_two_point_correlation
        plot_spanwise_two_point_correlation(args)

    else:
        print("-- Nothing has been specified")
