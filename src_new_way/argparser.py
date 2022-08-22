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
    parser.add_argument(
        "-w",
        "--window_size",
        dest="window_size",
        help="Genomic window size",
        required=False,
        default=50,
    )
    parser.add_argument(
        "-chr",
        "--chromosomes",
        dest="chrs",
        help=f"Pick chromosomes to build model on. Chromosomes to pick: {CHRS} or 'all' for all of them",
        nargs="+",
        choices=CHRS,
        required=False,
        default="1",
        action="store",
    )
    parser.add_argument(
        "-new",
        "--new_data",
        help="Create new simulated data and compute all features again. By default data is taken from GCP bucket.",
        dest="new",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-EDA",
        help="Perform EDA and save plots.",
        dest="EDA",
        action="store_true",
        default=False,
        required=False,
    )
    return parser
