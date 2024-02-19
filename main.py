import logging
import os
from subprocess import call

from src.argparser import CHRS, arg_parser
from src.cnv_generator import CNVGenerator
from src.dnn import Train as DNNTrain
from src.generate_features import Stats
from src.paths import (
    REF_FASTA_FILE,
    SIM_BAM_FILE_NAME,
    SIM_DATA_PATH,
    SIM_READS_FOLDER,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.sim_reads import SimReads
from src.train_basic_model import Train
from src.utils import (
    combine_and_cleanup_reference_genome,
    download_reference_genome,
    prepare_data,
)

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)


def create_sim_bam(
    chrs: list,
    cpus: int,
    window_size: int,
    min_cnv_gap: int,
    max_cnv_length: int,
    N_percentage: float,
    cov: int,
) -> None:
    """
    Creates simulated bam file. First it downloads reference genome, then it creates CNV's and then it simulates reads.
    At the end it aligns reads to reference genome and creates bam file.

    Args:
        chrs (list): list of chromosomes to simulate
        cpus (int): number of cpus to use
        window_size (int): window size for CNV's

    Returns:
        None
    """
    download_reference_genome(chrs)
    combine_and_cleanup_reference_genome(
        "reference_genome", "reference_genome/ref_genome.fa"
    )
    cnv_gen = CNVGenerator(window_size, max_cnv_length, min_cnv_gap, N_percentage)
    os.makedirs(SIM_DATA_PATH, exist_ok=True)
    total = cnv_gen.generate_cnv()
    cnv_gen.modify_fasta_file(total)
    sim_reads = SimReads(cov, cpu=cpus)
    sim_reads.sim_reads_genome()
    r1, r2 = f"{SIM_READS_FOLDER}/{cov}_R1.fastq", f"{SIM_READS_FOLDER}/{cov}_R2.fastq"
    call(f"bwa index {REF_FASTA_FILE}", shell=True)  # index reference genome
    bwa_command = f"bwa mem -t {cpus} {REF_FASTA_FILE} \
                    {r1} {r2} | samtools view -Shb - | \
                    samtools sort - > {SIM_DATA_PATH}/{SIM_BAM_FILE_NAME}"
    call(bwa_command, shell=True)


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    if args.command == "sim":
        if args.chrs == ["all"]:
            args.chrs = CHRS[:-1]

        if args.new_data:
            create_sim_bam(
                args.chrs,
                args.cpus,
                args.window_size,
                args.min_cnv_gap,
                args.max_cnv_length,
                args.N_percentage,
                args.cov,
            )

        if args.new_features:
            logging.info("Generating new features")
            stats = Stats(cpus=args.cpus)
            calculated_stats = stats.generate_stats(
                chrs=args.chrs, window_size=args.window_size
            )
            stats.combine_into_one_big_file(calculated_stats)
        else:
            logging.warning(
                "WARNING##: You need to create new features aswell, make sure to run with --new_features flag."
            )
    if args.command == "train":
        if args.DL:
            logging.info("Starting training DNN model")
            classifier = DNNTrain(args.EDA)
            if (
                not os.path.exists(TRAIN_PATH)
                or not os.path.exists(TEST_PATH)
                or not os.path.exists(VAL_PATH)
            ):
                prepare_data()
            classifier.train()
        else:
            logging.info("Starting training basic ML model")
            ml_classifier = Train(args.EDA)
            if (
                not os.path.exists(TRAIN_PATH)
                or not os.path.exists(TEST_PATH)
                or not os.path.exists(VAL_PATH)
            ):
                prepare_data()
            ml_classifier.train()
            ml_classifier.evaluate_on_test_data()
