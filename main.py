import glob
import logging
import os
from subprocess import call

from src.argparser import CHRS, arg_parser
from src.cnv_generator import CNVGenerator
from src.config import (
    REF_GEN_FILE_NAME,
    REF_GEN_PATH,
    SIM_BAM_FILE_NAME,
    SIM_DATA_PATH,
)
from src.generate_features import Stats
from src.sim_reads import SimReads
from src.train import Train
from src.utils import (
    combine_and_cleanup_reference_genome,
    download_reference_genome,
    get_number_of_individuals,
)

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)


def create_sim_bam(
    chrs: list,
    cpus: int,
    window_size: int,
    min_cnv_gap: int,
    max_cnv_length: int,
    N_percentage: float,
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
    sim_reads = SimReads(10, cpu=cpus)
    r1, r2 = sim_reads.sim_reads_genome()
    r1 = f"{SIM_DATA_PATH}/10_R1.fastq"
    r2 = f"{SIM_DATA_PATH}/10_R2.fastq"
    call(
        f"bwa index {REF_GEN_PATH}/{REF_GEN_FILE_NAME}", shell=True
    )  # index reference genome
    bwa_command = f"bwa mem -t {cpus} {REF_GEN_PATH}/{REF_GEN_FILE_NAME} \
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
            )

        if args.new_features:
            logging.info("Generating new features")
            stats = Stats(cpus=args.cpus)
            df_with_stats = stats.generate_stats(
                chrs=args.chrs, window_size=args.window_size
            )
            stats.combine_into_one_big_file()
        else:
            logging.warning(
                "WARNING##: You need to create new features aswell, make sure to run with --new_features flag."
            )
    if args.command == "train":
        classifier = Train(args.EDA)
        if not os.path.exists("train/sim.csv"):
            classifier.prepare_sim()
        if not os.path.exists("train/real.csv"):
            bam_real_files: list[str] = sorted(
                glob.glob(os.path.join("real_data/", "*.bam"))
            )
            indv = get_number_of_individuals(bam_real_files)
            indv.sort(key=int)
            train_indv = indv[:-1]
            test_indv = [indv[-1]]
            classifier.prepare_real(train_indv, test_indv)
        if args.train:
            classifier.train()
