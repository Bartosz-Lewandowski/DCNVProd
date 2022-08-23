import argparse

CHRS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "all",
]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cpus",
        type=int,
        dest="cpus",
        help="Number of cpus to make calculations",
        required=False,
        default=10,
    )
    parser.add_argument(
        "-w",
        "--window_size",
        dest="window_size",
        help="Genomic window size",
        required=False,
        default=50,
    )
    subparsers = parser.add_subparsers(dest="command")
    sim = subparsers.add_parser(
        "sim",
        help="Simulate data with duplication and deletion and generate features for ML algorithm.",
    )
    real = subparsers.add_parser(
        "real",
        help="Prepare target for real data and generate features for ML algorithm.",
    )
    train = subparsers.add_parser("train", help="Train ML algorithms.")
    predict = subparsers.add_parser(
        "predict", help="Using best model predict data on given file."
    )

    sim.add_argument(
        "--new_data",
        action="store_true",
        dest="new_data",
        help="Generate new duplication and deletion coordinates and create BAM file.",
        required=False,
        default=False,
    )
    sim.add_argument(
        "--new_features",
        action="store_true",
        dest="new_features",
        help="Calculate new features for simulated data",
        required=False,
        default=False,
    )
    sim.add_argument(
        "--chr",
        dest="chrs",
        help=f"Pick chromosomes to build model on. Chromosomes to pick: {CHRS} or 'all' for all of them",
        nargs="+",
        choices=CHRS,
        required=False,
        default="1",
        action="store",
    )
    real.add_argument(
        "--new_features",
        action="store_true",
        dest="new_features",
        help="Calculate new features for real data",
        required=False,
        default=False,
    )
    real.add_argument(
        "--chr",
        dest="chrs",
        help=f"Pick chromosomes to build model on. Chromosomes to pick: {CHRS} or 'all' for all of them",
        nargs="+",
        choices=CHRS,
        required=False,
        default="1",
        action="store",
    )
    train.add_argument(
        "-EDA",
        help="Perform EDA and save plots.",
        dest="EDA",
        action="store_true",
        default=False,
        required=False,
    )
    predict.add_argument(
        "-file",
        dest="file",
        help="Path to file to make predictions",
        required=True,
    )
    return parser
