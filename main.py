import glob
import os
from subprocess import call

from src.argparser import CHRS, arg_parser
from src.cnv_generator import CNVGenerator
from src.generate_features import Stats
from src.sim_reads import SimReads
from src.train import Train
from src.utils import download_reference_genom, get_number_of_individuals


def create_sim_bam(cpus: int, window_size: int) -> None:
    download_reference_genom()
    call("src/combine_ref_gens.sh", shell=True)
    ref_genome_fasta = "reference_genome/ref_genome.fa"
    CNV = CNVGenerator(ref_genome_fasta, window_size)
    total = CNV.generate_cnv()
    fasta_modified = CNV.modify_fasta_file(total)
    sim_reads = SimReads(fasta_modified, 10, cpu=cpus)
    r1, r2 = sim_reads.sim_reads_genome()
    r1 = "train_sim/10_R1.fastq"
    r2 = "train_sim/10_R2.fastq"
    call(f"bwa index {ref_genome_fasta}", shell=True)
    os.makedirs("sim_data/", exist_ok=True)
    bwa_command = f"bwa mem -t {cpus} {ref_genome_fasta} \
                    {r1} {r2} | samtools view -Shb - | \
                    samtools sort - > sim_data/sim_data.bam"
    call(bwa_command, shell=True)


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    if args.command == "sim":
        if args.chrs == ["all"]:
            args.chrs = CHRS[:-1]
        if args.new_data and not args.new_features:
            raise Exception(
                "Creating new data make sure to calculate new features as well!"
            )
        if args.new_data:
            create_sim_bam(args.cpus, args.window_size)
        if args.new_features:
            print("##Creating features for simulated data...")
            stats = Stats("sim_data/sim_data.bam", output_folder="sim", cpus=args.cpus)
            df_with_stats = stats.generate_stats(
                chrs=args.chrs, window_size=args.window_size, step=args.window_size
            )
            stats.combine_into_one_big_file("sim_target/sim_target.csv")

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
