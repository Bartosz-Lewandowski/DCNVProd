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
    subparsers = parser.add_subparsers(dest="command")
    sim = subparsers.add_parser(
        "sim",
        help="Simulate data with duplication and deletion and generate features for ML algorithm.",
    )
    train = subparsers.add_parser("train", help="Train ML algorithms.")

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
    sim.add_argument(
        "--window_size",
        dest="window_size",
        help="Genomic window size",
        required=False,
        type=int,
        default=50,
    )
    sim.add_argument(
        "--max_cnv_length",
        dest="max_cnv_length",
        help="Maximum length of CNV",
        type=int,
        required=False,
        default=100000,
    )
    sim.add_argument(
        "--min_cnv_gap",
        dest="min_cnv_gap",
        help="Minimum gap between CNVs",
        type=int,
        required=False,
        default=50,
    )
    sim.add_argument(
        "--n_percentage",
        dest="N_percentage",
        help="Percentage threshold for N amino acids from which mutation is not created",
        type=float,
        required=False,
        default=0.7,
    )
    sim.add_argument(
        "--cov",
        dest="cov",
        help="Coverage for simulating reads",
        type=int,
        required=False,
        default=10,
    )
    train.add_argument(
        "-EDA",
        help="Perform EDA and save plots.",
        dest="EDA",
        action="store_true",
        default=False,
        required=False,
    )
    return parser
