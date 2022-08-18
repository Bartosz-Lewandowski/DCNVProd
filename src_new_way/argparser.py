import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sim",
        "--prepare_sim_data",
        action="store_true",
        dest="sim",
        help="Simulate bam file, create features and prepare data for ML algorithm",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-real",
        "--prepare_real_data",
        action="store_true",
        dest="real",
        help="Create features for real data and prepare it for ML algorithm",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-train",
        "--train_data",
        action="store_true",
        dest="train",
        help="Train ML algorithms",
        required=False,
    )
    parser.add_argument(
        "-predict" "--predict",
        type=str,
        dest="predict",
        help="File to predict data on",
        required=False,
    )
    parser.add_argument(
        "-cpus",
        type=int,
        dest="cpus",
        help="Number of cpus to make calculations",
        required=False,
        default=10,
    )

    return parser
