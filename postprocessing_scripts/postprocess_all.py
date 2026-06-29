import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--function", required=True,
                        choices=[
                            "mean_flow_profile",
                            "TKE",
                            "TKE_budget",
                            "dissipation",
                            "reynolds_stresses",
                            "total_dissipation_of_TKE",
                            "kx_spectra",
                            "kz_spectra",
                            "spanwise_two_point_correlation",
                            "streamwise_two_point_correlation",
                            "reynolds_number",
                        ])
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--singlephase", action="store_true")
    parser.add_argument("--time-step", type=int, default=None)
    parser.add_argument("--stackdirection", type=int, default=1)
    parser.add_argument("--std", type=int, default=None)
    parser.add_argument("--start_ts", type=int, default=0)
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()

    args.case = args.input_path

    if args.std is None:
        args.std = args.stackdirection

    if args.function == "mean_flow_profile":
        if args.time_step is None:
            parser.error("--time_step is required for mean_flow_profile")
        if args.singlephase:
            from pyscripts.mean_flow_profile_singlephase import mean_flow_profile
            mean_flow_profile(args)
        else:
            from pyscripts.mean_flow_profile import mean_flow_profile
            mean_flow_profile(args)

    elif args.function == "TKE_budget":
        from pyscripts.TKE_budget_v3 import TKE_budget
        if args.time_step is None:
            parser.error("--time_step is required for TKE")
        TKE_budget(args)

    elif args.function == "TKE":
        from pyscripts.TKE import TKE
        if args.time_step is None:
            parser.error("--time_step is required for TKE")
        TKE(args)

    elif args.function == "dissipation":
        from pyscripts.dissipation import dissipation
        if args.time_step is None:
            parser.error("--time_step is required for dissipation")
        dissipation(args)

    elif args.function == "reynolds_stresses":
        if args.time_step is None:
            parser.error("--time_step is required for mean_flow_profile")
        if args.singlephase:
            from pyscripts.reynolds_stresses_singlephase import compute_reynolds_stresses
            compute_reynolds_stresses(args)
        else:
            from pyscripts.reynolds_stresses import compute_reynolds_stresses
            compute_reynolds_stresses(args)

    elif args.function == "total_dissipation_of_TKE":
        from pyscripts.total_dissipation_of_TKE_v2 import compute_total_dissipation_of_TKE
        compute_total_dissipation_of_TKE(args)

    elif args.function == "kx_spectra":
        from pyscripts.kx_spectra import compute_spectra
        if args.time_step is None:
            parser.error("--time_step is required for spectra")
        compute_spectra(args)

    elif args.function == "kz_spectra":
        from pyscripts.kz_spectra import compute_spectra
        if args.time_step is None:
            parser.error("--time_step is required for spectra")
        compute_spectra(args)

    elif args.function == "spanwise_two_point_correlation":
        from pyscripts.spanwise_two_point_correlation import compute_spanwise_two_point_correlation
        if args.time_step is None:
            parser.error("--time_step is required for two point correlation")
        compute_spanwise_two_point_correlation(args)

    elif args.function == "streamwise_two_point_correlation":
        from pyscripts.streamwise_two_point_correlation import compute_streamwise_two_point_correlation
        if args.time_step is None:
            parser.error("--time_step is required for two point correlation")
        compute_streamwise_two_point_correlation(args)

    elif args.function == "reynolds_number":
        from pyscripts.reynolds_number import compute_reynolds_number
        compute_reynolds_number(args)

    else:
        print("-- Nothing has been specified")
