import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--function", required=True,
                        choices=[
                            "mean_flow_profile",
                            "TKE",
                            "reynolds_stresses",
                            "total_dissipation_of_TKE",
                            "spectra",
                            "reynolds_number",
                        ])
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--time-step", type=int, default=None)
    parser.add_argument("--stackdirection", type=int, default=1)
    parser.add_argument("--std", type=int, default=None)

    args = parser.parse_args()

    args.case = args.input_path

    if args.std is None:
        args.std = args.stackdirection

    if args.function == "mean_flow_profile":
        from pyscripts.mean_flow_profile import mean_flow_profile
        if args.time_step is None:
            parser.error("--time_step is required for mean_flow_profile")
        mean_flow_profile(args)

    elif args.function == "TKE":
        from pyscripts.TKE import TKE
        if args.time_step is None:
            parser.error("--time_step is required for TKE")
        TKE(args)

    elif args.function == "reynolds_stresses":
        from pyscripts.reynolds_stresses import compute_reynolds_stresses
        if args.time_step is None:
            parser.error("--time_step is required for reynolds_stresses")
        compute_reynolds_stresses(args)

    elif args.function == "total_dissipation_of_TKE":
        from pyscripts.total_dissipation_of_TKE import compute_total_dissipation_of_TKE
        compute_total_dissipation_of_TKE(args)

    elif args.function == "spectra":
        from pyscripts.spectra import compute_spectra
        if args.time_step is None:
            parser.error("--time_step is required for spectra")
        compute_spectra(args)

    elif args.function == "reynolds_number":
        from pyscripts.reynolds_number import compute_reynolds_number
        compute_reynolds_number(args)

    else:
        print("-- Nothing has been specified")
